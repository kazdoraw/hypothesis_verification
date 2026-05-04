"""Генерация датасета D1 v6 из seed-фраз через Grok LLM вариации.

Процесс:
1. Загрузка seeds из d1/data/d1_v6_seeds.yaml
2. Для каждого seed → Grok генерирует вариации (hurried + anxious + normal)
3. Фильтрация: только кириллица, удаление пустых, domain check
4. Дедупликация: cosine similarity внутри домена (sentence-transformers)
5. Сохранение в d1/data/d1_v6_full.csv

Resume: после каждого seed результат записывается в checkpoint JSON.
Можно прервать и продолжить — уже обработанные seeds пропускаются.

Запуск:
    cd study && python -m d1.scripts.generate_dataset [--limit N] [--resume] [--sample]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Добавляем корень study в sys.path для импорта utils
_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import (
    COSINE_DEDUP_THRESHOLD,
    CSV_COLUMNS,
    DATA_DIR,
    EMBEDDING_MODEL_PRIMARY,
    MAX_TOKENS_VARIATION,
    MIN_VARIATIONS_WARN,
    PROMPTS_DIR,
    SEED_AUDIT_FILE,
    SEEDS_FILE,
    TEMPERATURE_VARIATION,
    VARIATION_MODEL,
    VARIATIONS_PER_SEED,
    resolve_model_path,
)
from utils.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_CHECKPOINT_FILE = DATA_DIR / "_generation_checkpoint.json"
_OUTPUT_FILE = DATA_DIR / "d1_v6_full.csv"
_VARIATION_PROMPT_FILE = PROMPTS_DIR / "d1_variation.md"

# Фильтр: только кириллица + разрешённые символы (из D4 query_generator.py)
_ALLOWED_QUERY_RE = re.compile(
    r"^[\u0400-\u04FF\s\d\.\,\!\?\-\:\;\(\)\"\'«»\…\—\–₽%\+\@\#\&\*\/]+$",
)
# Дополнительный фильтр: украинские символы (і, ї, є, ґ) — исключаем
_UKRAINIAN_CHARS_RE = re.compile(r"[іїєґІЇЄҐ]")

# Подтипы, для которых LLM-генерация не нужна (конечные множества из 1-3 слов)
_SKIP_LLM_SUBTYPES = {"greeting_only", "farewell_only", "unclear_short"}

# Маркеры доменов для domain check (из онтологии route_domain.yaml)
_DOMAIN_MARKERS: dict[str, re.Pattern] = {
    "booking": re.compile(
        r"\b(запис\w*|запиш\w*|перенес\w*|отмен\w*|отказ\w* от запис|"
        r"когда \w* попасть|свободн\w* врем\w*|перезапиш\w*)\b",
        re.IGNORECASE,
    ),
    "anamnesis": re.compile(
        r"\b(бол\w+ зуб|но\w+ десн|кровоточ\w*|опух\w*|выпал\w* пломб|"
        r"отколол\w*|треснул\w*|шата\w*|гно\w* ид\w*)\b",
        re.IGNORECASE,
    ),
    "faq": re.compile(
        r"\b(сколько сто\w*|как\w* цен|прайс|где \w* наход|"
        r"график|рассрочк|парковк|документ\w*)\b",
        re.IGNORECASE,
    ),
}


def _check_domain_consistency(text: str, expected_domain: str, seed_id: str) -> bool:
    """Проверка, что вариация не содержит сильных маркеров чужого домена.

    Проверяет каждый домен ≠ expected. Если найден маркер — вариация
    подозрительна, логируем и отбрасываем.

    Returns:
        True если текст консистентен с expected_domain.
    """
    for domain, pattern in _DOMAIN_MARKERS.items():
        if domain == expected_domain:
            continue
        if pattern.search(text):
            logger.debug(
                "Domain leak: seed=%s (expected=%s) содержит маркер '%s': %s",
                seed_id, expected_domain, domain, text[:80],
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Cosine дедупликация внутри домена
# ---------------------------------------------------------------------------

def deduplicate_by_similarity(
    rows: list[dict[str, Any]],
    threshold: float = COSINE_DEDUP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Удаление дублей по cosine similarity внутри (domain, style).

    Гарантии:
    - seeds (source=seed) всегда сохраняются
    - первая вариация каждого style для каждого seed_id сохраняется
      (style diversity guarantee)
    - дедупликация только внутри одного (domain, style) сегмента

    Returns:
        Отфильтрованный список rows (порядок сохранён).
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    if not rows:
        return rows

    model = SentenceTransformer(resolve_model_path(EMBEDDING_MODEL_PRIMARY))
    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Трекинг: гарантируем min 1 variation/style/seed
    seen_seed_style: set[tuple[str, str]] = set()

    # Группируем по (domain, style)
    segment_indices: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        key = (row["route_domain"], row.get("style", ""))
        segment_indices.setdefault(key, []).append(idx)

    kept_mask = [False] * len(rows)
    total_dropped = 0

    for (_domain, _style), indices in segment_indices.items():
        accepted_embs: list[np.ndarray] = []
        for idx in indices:
            row = rows[idx]
            emb = embeddings[idx]

            # Seeds всегда сохраняем
            if row["source"] == "seed":
                kept_mask[idx] = True
                accepted_embs.append(emb)
                continue

            # Первая вариация каждого style для seed — гарантируем
            seed_style_key = (row.get("seed_id", ""), row.get("style", ""))
            if seed_style_key not in seen_seed_style:
                seen_seed_style.add(seed_style_key)
                kept_mask[idx] = True
                accepted_embs.append(emb)
                continue

            # Cosine dedup внутри сегмента
            is_dup = False
            for acc_emb in accepted_embs:
                sim = float(np.dot(emb, acc_emb))
                if sim >= threshold:
                    is_dup = True
                    break
            if is_dup:
                total_dropped += 1
            else:
                kept_mask[idx] = True
                accepted_embs.append(emb)

    result = [r for r, keep in zip(rows, kept_mask) if keep]
    logger.info(
        "Cosine dedup: %d → %d (отброшено %d, порог %.2f)",
        len(rows), len(result), total_dropped, threshold,
    )
    return result


# ---------------------------------------------------------------------------
# Seed audit: drop / relabel / flags
# ---------------------------------------------------------------------------

def load_seed_audit() -> dict[str, dict[str, str]]:
    """Загрузка seed_audit.csv в lookup dict {seed_id: row}."""
    if not SEED_AUDIT_FILE.exists():
        logger.warning("seed_audit.csv не найден, audit не применяется")
        return {}
    df = pd.read_csv(SEED_AUDIT_FILE, dtype=str).fillna("")
    return {row["seed_id"]: row.to_dict() for _, row in df.iterrows()}


def apply_audit(
    seeds: list[dict[str, Any]],
    audit: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Применение решений аудита к seeds.

    - status=drop → исключаем seed
    - status=relabel → применяем new_domain/new_subtype
    - faq_category → пробрасываем в seed для последующего экспорта

    Returns:
        Отфильтрованный список seeds.
    """
    if not audit:
        return seeds

    result = []
    dropped = 0
    relabeled = 0
    for seed in seeds:
        sid = seed["id"]
        info = audit.get(sid)
        if not info:
            result.append(seed)
            continue

        status = info.get("status", "keep")
        if status == "drop":
            logger.info("Audit drop: %s '%s'", sid, seed["text"][:60])
            dropped += 1
            continue

        if status == "relabel":
            new_domain = info.get("new_domain", "")
            new_subtype = info.get("new_subtype", "")
            if new_domain:
                seed["route_domain"] = new_domain
            if new_subtype:
                seed["subtype"] = new_subtype
            relabeled += 1
            logger.debug("Audit relabel: %s → %s/%s", sid, new_domain, new_subtype)

        # Пробрасываем faq_category
        faq_cat = info.get("faq_category", "")
        seed["faq_category"] = faq_cat
        result.append(seed)

    logger.info(
        "Audit applied: %d seeds → %d (dropped=%d, relabeled=%d)",
        len(seeds), len(result), dropped, relabeled,
    )
    return result


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_seeds(path: Path | None = None) -> list[dict[str, Any]]:
    """Загрузка seed-фраз из YAML."""
    p = path or SEEDS_FILE
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    seeds = data.get("seeds", [])
    logger.info("Загружено %d seeds из %s", len(seeds), p.name)
    return seeds


def load_variation_prompt() -> str:
    """Загрузка system prompt для генерации вариаций."""
    return _VARIATION_PROMPT_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[str, list[dict]]:
    """Загрузка checkpoint: {seed_id: [variations]}."""
    if _CHECKPOINT_FILE.exists():
        with open(_CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict[str, list[dict]]) -> None:
    """Сохранение checkpoint."""
    _CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Парсинг LLM-ответа (адаптировано из D4 _parse_variations)
# ---------------------------------------------------------------------------

def parse_variations(raw: str) -> list[dict[str, str]]:
    """Парсинг JSON-ответа LLM с вариациями.

    Обрабатывает:
    - Голый JSON массив [...]
    - Обёртку {"key": [...]}
    - Markdown code fence ```json ... ```
    - Фильтр не-русских символов
    """
    content = raw.strip()
    # Удаление markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)
    try:
        data = json.loads(content)
        # Если обёртка {"variations": [...]} или {"data": [...]}
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    data = val
                    break
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, dict) and "query" in item:
                    query = item["query"].strip()
                    if not query:
                        continue
                    if not _ALLOWED_QUERY_RE.match(query):
                        logger.debug("Отброшена вариация (не-кириллица): %s", query[:60])
                        continue
                    if _UKRAINIAN_CHARS_RE.search(query):
                        logger.debug("Отброшена вариация (укр. символы): %s", query[:60])
                        continue
                    results.append({
                        "query": query,
                        "style": item.get("style", "unknown"),
                    })
            return results
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Не удалось распарсить вариации: %s", raw[:200])
    return []


# ---------------------------------------------------------------------------
# Генерация вариаций для одного seed
# ---------------------------------------------------------------------------

def build_user_message(seed: dict[str, Any]) -> str:
    """Формирование user message для LLM."""
    return (
        f"Оригинальный запрос: \"{seed['text']}\"\n"
        f"Домен: {seed['route_domain']}\n"
        f"Подтип: {seed['subtype']}"
    )


def generate_for_seed(
    client: OpenRouterClient,
    system_prompt: str,
    seed: dict[str, Any],
) -> dict[str, Any]:
    """Генерация вариаций для одного seed через Grok.

    Returns:
        Dict с ключами: rows (list[dict]), meta (dict с raw/stats/rejects).
    """
    user_msg = build_user_message(seed)
    meta: dict[str, Any] = {
        "raw_response": "",
        "usage": {},
        "parsed_count": 0,
        "accepted_count": 0,
        "rejected_domain_check": 0,
        "reject_reasons": [],
    }
    try:
        raw_text, usage = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            model=VARIATION_MODEL,
            temperature=TEMPERATURE_VARIATION,
            max_tokens=MAX_TOKENS_VARIATION,
        )
    except Exception as exc:
        logger.warning("LLM ошибка для %s: %s", seed["id"], exc)
        meta["error"] = str(exc)
        return {"rows": [], "meta": meta}

    meta["raw_response"] = raw_text
    meta["usage"] = usage

    variations = parse_variations(raw_text)
    meta["parsed_count"] = len(variations)

    rows: list[dict[str, Any]] = []
    expected_domain = seed["route_domain"]
    for var in variations:
        if not _check_domain_consistency(var["query"], expected_domain, seed["id"]):
            meta["rejected_domain_check"] += 1
            meta["reject_reasons"].append(
                {"text": var["query"][:80], "reason": "domain_marker"}
            )
            continue
        row = {
            "id": f"var_{seed['id']}_{uuid.uuid4().hex[:6]}",
            "text": var["query"],
            "route_domain": seed["route_domain"],
            "subtype": seed["subtype"],
            "explicit_booking": seed.get("explicit_booking", False),
            "urgency": seed.get("urgency", "normal"),
            "is_offtopic": seed.get("is_offtopic", False),
            "specialization_hint": seed.get("specialization_hint", ""),
            "feedback_flag": seed.get("feedback_flag", False),
            "faq_category": seed.get("faq_category", ""),
            "style": var.get("style", "unknown"),
            "source": "grok_variation",
            "seed_id": seed["id"],
        }
        rows.append(row)

    meta["accepted_count"] = len(rows)
    return {"rows": rows, "meta": meta}


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------

def stratified_sample(
    seeds: list[dict[str, Any]],
    n: int,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Стратифицированный отбор N seeds равномерно по route_domain."""
    rng = random.Random(random_state)
    by_domain: dict[str, list] = defaultdict(list)
    for s in seeds:
        by_domain[s["route_domain"]].append(s)

    per_domain = max(1, n // len(by_domain))
    result: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        k = min(per_domain, len(pool))
        result.extend(rng.sample(pool, k))

    # Добрать остаток если n > per_domain * num_domains
    remaining = n - len(result)
    if remaining > 0:
        used = {s["id"] for s in result}
        extras = [s for s in seeds if s["id"] not in used]
        result.extend(rng.sample(extras, min(remaining, len(extras))))

    logger.info(
        "Stratified sample: %d seeds (%s)",
        len(result),
        {d: sum(1 for s in result if s['route_domain'] == d) for d in sorted(by_domain)},
    )
    return result


def run_generation(
    limit: int | None = None,
    sample: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Полный pipeline генерации датасета.

    Args:
        limit: первые N seeds (для тестирования)
        sample: стратифицированный отбор N seeds из всех доменов
        resume: продолжить с checkpoint
    """
    seeds = load_seeds()
    audit = load_seed_audit()
    seeds = apply_audit(seeds, audit)
    system_prompt = load_variation_prompt()
    client = OpenRouterClient()
    checkpoint = load_checkpoint() if resume else {}

    if sample:
        seeds = stratified_sample(seeds, sample)
    elif limit:
        seeds = seeds[:limit]
        logger.info("Ограничение: первые %d seeds", limit)

    # Множество seed_id текущего прогона (для фильтрации checkpoint)
    current_seed_ids = {s["id"] for s in seeds}

    all_rows: list[dict[str, Any]] = []
    # Включаем сами seed-фразы как отдельные samples
    for seed in seeds:
        seed_row = {
            "id": seed["id"],
            "text": seed["text"],
            "route_domain": seed["route_domain"],
            "subtype": seed["subtype"],
            "explicit_booking": seed.get("explicit_booking", False),
            "urgency": seed.get("urgency", "normal"),
            "is_offtopic": seed.get("is_offtopic", False),
            "specialization_hint": seed.get("specialization_hint", ""),
            "feedback_flag": seed.get("feedback_flag", False),
            "faq_category": seed.get("faq_category", ""),
            "source": "seed",
            "seed_id": seed["id"],
        }
        all_rows.append(seed_row)

    # Восстановление из checkpoint (только seeds текущего прогона)
    restored = 0
    if checkpoint:
        for seed_id, entry in checkpoint.items():
            if seed_id not in current_seed_ids:
                continue
            rows = entry["rows"] if isinstance(entry, dict) else entry
            all_rows.extend(rows)
            restored += 1
        if restored:
            logger.info("Восстановлено из checkpoint: %d/%d seeds", restored, len(checkpoint))
        skipped_ckpt = len(checkpoint) - restored
        if skipped_ckpt > 0:
            logger.info("Пропущено из checkpoint (не в текущем прогоне): %d seeds", skipped_ckpt)

    total = len(seeds)
    skipped = 0
    generated = 0
    start = time.perf_counter()

    for i, seed in enumerate(seeds, 1):
        sid = seed["id"]
        if sid in checkpoint:
            skipped += 1
            continue

        if seed.get("subtype") in _SKIP_LLM_SUBTYPES:
            logger.info("Skip LLM для %s (subtype=%s)", sid, seed["subtype"])
            continue

        result = generate_for_seed(client, system_prompt, seed)
        vars_list = result["rows"]
        n_vars = len(vars_list)
        if n_vars < MIN_VARIATIONS_WARN:
            logger.warning(
                "Low yield: %s вернул %d/%d вариаций (parsed=%d, rejected=%d)",
                sid, n_vars, VARIATIONS_PER_SEED,
                result["meta"]["parsed_count"],
                result["meta"]["rejected_domain_check"],
            )
        checkpoint[sid] = {"rows": vars_list, "meta": result["meta"]}
        all_rows.extend(vars_list)
        generated += 1

        # Сохраняем checkpoint после каждого seed
        save_checkpoint(checkpoint)

        if i % 10 == 0 or i == total:
            elapsed = time.perf_counter() - start
            stats = client.get_stats()
            print(
                f"  [{i}/{total}] +{n_vars} вариаций | "
                f"total: {len(all_rows)} | "
                f"tokens: {stats['total_tokens']:,} | "
                f"{elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    stats = client.get_stats()
    print(
        f"\nГенерация завершена: {len(all_rows)} сэмплов "
        f"({generated} новых seeds, {skipped} из checkpoint) "
        f"за {elapsed:.0f}s",
    )
    print(f"Токены: in={stats['total_tokens_in']:,}, out={stats['total_tokens_out']:,}")

    # Cosine дедупликация внутри каждого домена
    all_rows = deduplicate_by_similarity(all_rows)

    df = pd.DataFrame(all_rows)
    # Приводим к нужным колонкам
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CSV_COLUMNS]

    # Сохранение
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Сохранено: {_OUTPUT_FILE} ({len(df)} строк)")

    return df


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------

_DEFAULT_CONCURRENCY = 8


async def generate_for_seed_async(
    client: OpenRouterClient,
    system_prompt: str,
    seed: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Async версия generate_for_seed с bounded concurrency."""
    async with semaphore:
        user_msg = build_user_message(seed)
        meta: dict[str, Any] = {
            "raw_response": "",
            "usage": {},
            "parsed_count": 0,
            "accepted_count": 0,
            "rejected_domain_check": 0,
            "reject_reasons": [],
        }
        try:
            raw_text, usage = await client.async_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                model=VARIATION_MODEL,
                temperature=TEMPERATURE_VARIATION,
                max_tokens=MAX_TOKENS_VARIATION,
            )
        except Exception as exc:
            logger.warning("LLM ошибка для %s: %s", seed["id"], exc)
            meta["error"] = str(exc)
            return {"seed_id": seed["id"], "rows": [], "meta": meta}

        meta["raw_response"] = raw_text
        meta["usage"] = usage

        variations = parse_variations(raw_text)
        meta["parsed_count"] = len(variations)

        rows: list[dict[str, Any]] = []
        expected_domain = seed["route_domain"]
        for var in variations:
            if not _check_domain_consistency(var["query"], expected_domain, seed["id"]):
                meta["rejected_domain_check"] += 1
                meta["reject_reasons"].append(
                    {"text": var["query"][:80], "reason": "domain_marker"}
                )
                continue
            row = {
                "id": f"var_{seed['id']}_{uuid.uuid4().hex[:6]}",
                "text": var["query"],
                "route_domain": seed["route_domain"],
                "subtype": seed["subtype"],
                "explicit_booking": seed.get("explicit_booking", False),
                "urgency": seed.get("urgency", "normal"),
                "is_offtopic": seed.get("is_offtopic", False),
                "specialization_hint": seed.get("specialization_hint", ""),
                "feedback_flag": seed.get("feedback_flag", False),
                "faq_category": seed.get("faq_category", ""),
                "style": var.get("style", "unknown"),
                "source": "grok_variation",
                "seed_id": seed["id"],
            }
            rows.append(row)

        meta["accepted_count"] = len(rows)
        return {"seed_id": seed["id"], "rows": rows, "meta": meta}


async def run_generation_async(
    limit: int | None = None,
    sample: int | None = None,
    resume: bool = True,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> pd.DataFrame:
    """Async pipeline генерации с bounded concurrency.

    Args:
        limit: первые N seeds
        sample: стратифицированный отбор N seeds
        resume: продолжить с checkpoint
        concurrency: макс. параллельных LLM вызовов
    """
    seeds = load_seeds()
    audit = load_seed_audit()
    seeds = apply_audit(seeds, audit)
    system_prompt = load_variation_prompt()
    client = OpenRouterClient()
    checkpoint = load_checkpoint() if resume else {}

    if sample:
        seeds = stratified_sample(seeds, sample)
    elif limit:
        seeds = seeds[:limit]
        logger.info("Ограничение: первые %d seeds", limit)

    current_seed_ids = {s["id"] for s in seeds}

    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_row = {
            "id": seed["id"],
            "text": seed["text"],
            "route_domain": seed["route_domain"],
            "subtype": seed["subtype"],
            "explicit_booking": seed.get("explicit_booking", False),
            "urgency": seed.get("urgency", "normal"),
            "is_offtopic": seed.get("is_offtopic", False),
            "specialization_hint": seed.get("specialization_hint", ""),
            "feedback_flag": seed.get("feedback_flag", False),
            "faq_category": seed.get("faq_category", ""),
            "source": "seed",
            "seed_id": seed["id"],
        }
        all_rows.append(seed_row)

    # Восстановление из checkpoint (только seeds текущего прогона)
    restored = 0
    if checkpoint:
        for seed_id, entry in checkpoint.items():
            if seed_id not in current_seed_ids:
                continue
            rows = entry["rows"] if isinstance(entry, dict) else entry
            all_rows.extend(rows)
            restored += 1
        if restored:
            logger.info("Восстановлено из checkpoint: %d/%d seeds", restored, len(checkpoint))

    # Фильтруем seeds для генерации
    to_generate = [
        s for s in seeds
        if s["id"] not in checkpoint
        and s.get("subtype") not in _SKIP_LLM_SUBTYPES
    ]
    skipped_llm = sum(
        1 for s in seeds
        if s.get("subtype") in _SKIP_LLM_SUBTYPES and s["id"] not in checkpoint
    )
    if skipped_llm:
        logger.info("Skip LLM (greeting/farewell/unclear): %d seeds", skipped_llm)

    logger.info(
        "Async generation: %d seeds to generate (concurrency=%d), %d from checkpoint",
        len(to_generate), concurrency, restored,
    )

    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    # Запускаем все задачи параллельно
    tasks = [
        generate_for_seed_async(client, system_prompt, seed, semaphore)
        for seed in to_generate
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Обработка результатов
    generated = 0
    errors = 0
    for res in results:
        if isinstance(res, Exception):
            logger.error("Task exception: %s", res)
            errors += 1
            continue
        sid = res["seed_id"]
        vars_list = res["rows"]
        n_vars = len(vars_list)
        if n_vars < MIN_VARIATIONS_WARN:
            logger.warning(
                "Low yield: %s вернул %d/%d вариаций (parsed=%d, rejected=%d)",
                sid, n_vars, VARIATIONS_PER_SEED,
                res["meta"]["parsed_count"],
                res["meta"]["rejected_domain_check"],
            )
        checkpoint[sid] = {"rows": vars_list, "meta": res["meta"]}
        all_rows.extend(vars_list)
        generated += 1

    # Сохраняем checkpoint один раз после всех задач
    save_checkpoint(checkpoint)
    await client.async_close()

    elapsed = time.perf_counter() - start
    stats = client.get_stats()
    print(
        f"\nAsync генерация завершена: {len(all_rows)} сэмплов "
        f"({generated} новых, {restored} из checkpoint, {errors} ошибок) "
        f"за {elapsed:.0f}s",
    )
    print(f"Токены: in={stats['total_tokens_in']:,}, out={stats['total_tokens_out']:,}")

    # Cosine дедупликация
    all_rows = deduplicate_by_similarity(all_rows)

    df = pd.DataFrame(all_rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CSV_COLUMNS]

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Сохранено: {_OUTPUT_FILE} ({len(df)} строк)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="D1 v6 dataset generation")
    parser.add_argument("--limit", type=int, default=None, help="Первые N seeds")
    parser.add_argument("--sample", type=int, default=None, help="Стратифицированный отбор N seeds")
    parser.add_argument("--no-resume", action="store_true", help="Начать заново (без checkpoint)")
    parser.add_argument("--sync", action="store_true", help="Sync режим (без параллельности)")
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY, help="Параллельных запросов")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.sync:
        run_generation(limit=args.limit, sample=args.sample, resume=not args.no_resume)
    else:
        asyncio.run(
            run_generation_async(
                limit=args.limit,
                sample=args.sample,
                resume=not args.no_resume,
                concurrency=args.concurrency,
            )
        )


if __name__ == "__main__":
    main()
