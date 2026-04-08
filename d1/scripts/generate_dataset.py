"""Генерация датасета D1 v6 из seed-фраз через Grok LLM вариации.

Процесс:
1. Загрузка seeds из d1/data/d1_v6_seeds.yaml
2. Для каждого seed → Grok генерирует 15 вариаций (5 hurried + 5 anxious + 5 normal)
3. Фильтрация: только кириллица, удаление пустых
4. Дедупликация: cosine similarity < 0.85 внутри домена
5. Сохранение в d1/data/d1_v6_full.csv

Resume: после каждого seed результат записывается в checkpoint JSON.
Можно прервать и продолжить — уже обработанные seeds пропускаются.

Запуск:
    cd study && python -m d1.scripts.generate_dataset [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Добавляем корень study в sys.path для импорта utils
_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import (
    DATA_DIR,
    PROMPTS_DIR,
    SEEDS_FILE,
    VARIATION_MODEL,
    VARIATIONS_PER_SEED,
    TEMPERATURE_VARIATION,
    MAX_TOKENS_VARIATION,
    CSV_COLUMNS,
)
from utils.openrouter_client import OpenRouterClient
from utils.taxonomy_v6 import validate_domain, validate_subtype

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
) -> list[dict[str, Any]]:
    """Генерация вариаций для одного seed через Grok.

    Returns:
        Список dict с полями для CSV (text, route_domain, subtype, ...)
    """
    user_msg = build_user_message(seed)
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
        return []

    variations = parse_variations(raw_text)

    results = []
    for var in variations:
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
            "source": "grok_variation",
            "seed_id": seed["id"],
        }
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------

def run_generation(
    limit: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Полный pipeline генерации датасета.

    Args:
        limit: ограничить количество seeds (для тестирования)
        resume: продолжить с checkpoint
    """
    seeds = load_seeds()
    system_prompt = load_variation_prompt()
    client = OpenRouterClient()
    checkpoint = load_checkpoint() if resume else {}

    if limit:
        seeds = seeds[:limit]
        logger.info("Ограничение: первые %d seeds", limit)

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
            "source": "seed",
            "seed_id": seed["id"],
        }
        all_rows.append(seed_row)

    # Восстановление из checkpoint
    if checkpoint:
        for seed_id, vars_list in checkpoint.items():
            all_rows.extend(vars_list)
        logger.info("Восстановлено из checkpoint: %d seeds", len(checkpoint))

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

        vars_list = generate_for_seed(client, system_prompt, seed)
        checkpoint[sid] = vars_list
        all_rows.extend(vars_list)
        generated += 1

        # Сохраняем checkpoint после каждого seed
        save_checkpoint(checkpoint)

        if i % 10 == 0 or i == total:
            elapsed = time.perf_counter() - start
            stats = client.get_stats()
            print(
                f"  [{i}/{total}] +{len(vars_list)} вариаций | "
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="D1 v6 dataset generation")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить seeds (для теста)")
    parser.add_argument("--no-resume", action="store_true", help="Начать заново (без checkpoint)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_generation(limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
