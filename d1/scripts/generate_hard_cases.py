"""Генерация candidate pool hard cases через grok-4.1-fast (Task 1 из roadmap v3).

Процесс (в соответствии с планом §1):

1. **LLM-генерация** candidate pool через OpenRouter (`grok-4.1-fast`) —
   N кандидатов на каждый из 7 сценариев (~100-140 шт. суммарно).
2. **Anti-contamination**: BGE-M3 semantic overlap:
   - vs существующих hard_cases.yaml → auto_reject если max_sim >= 0.92
   - vs train seeds (d1_v6_seeds.yaml) → аналогично
   - внутри pool → drop near-duplicates sim >= 0.95
3. **Сохранение** (intermediate):
   - `d1/data/hard_cases_candidates.yaml` — все пропущенные через anti-contamination
   - `d1/data/hard_cases_similarity.csv` — лог сравнений для трассировки
   - `d1/data/hard_cases_audit.csv` — schema для ручного audit (пустые audit_decision)

Ручной audit и включение в gold происходит ПОСЛЕ этого скрипта (см. план).

Запуск:
    cd study && python -m d1.scripts.generate_hard_cases [--concurrency N] [--dry-scenario ID]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import pandas as pd
import yaml

# MPS (Apple GPU) backend иногда даёт RuntimeWarning invalid/overflow в matmul
# при L2-нормализованных эмбеддингах — результаты корректны (sim ∈ [0, 1]).
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in matmul",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered in matmul",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in matmul",
)

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.baselines.b2_embedding import B2EmbeddingClassifier
from d1.config import (
    DATA_DIR,
    HARD_CASES_FILE,
    LEAKAGE_COSINE_THRESHOLD,
    PROMPTS_DIR,
    SEEDS_FILE,
    VARIATION_MODEL,
)
from d1.scripts.generate_dataset import (
    _ALLOWED_QUERY_RE,
    _UKRAINIAN_CHARS_RE,
    _check_domain_consistency,
)
from utils.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_PROMPT_FILE = PROMPTS_DIR / "d1_hard_case_candidates.md"
_CANDIDATES_FILE = DATA_DIR / "hard_cases_candidates.yaml"
_SIMILARITY_FILE = DATA_DIR / "hard_cases_similarity.csv"
_AUDIT_FILE = DATA_DIR / "hard_cases_audit.csv"

# Пороги anti-contamination (по плану §1 Этап B.1)
_LEAKAGE_REJECT_THRESHOLD = LEAKAGE_COSINE_THRESHOLD       # 0.92 vs existing
_INTRA_POOL_DUP_THRESHOLD = 0.95                           # внутри candidate pool

# Генерация
_TEMPERATURE = 0.9
_MAX_TOKENS = 2200            # хватает на 20 кандидатов с полями
_CONCURRENCY_DEFAULT = 4

# ---------------------------------------------------------------------------
# Сценарии (7 штук по плану §1, сценарии 10-16)
# ---------------------------------------------------------------------------

_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "anamnesis_faq_confusion",
        "count": 20,
        "route_domain": "anamnesis",
        # НЕ применяем _check_domain_consistency: сценарий по природе содержит
        # faq-маркеры ("это нормально", "почему у меня") → ожидаемое поведение.
        "check_domain_consistency": False,
        "description": (
            "Личная жалоба, которую человек формулирует как общий вопрос. "
            "Граница procedure_info (faq) vs личный symptom (anamnesis). "
            "ВСЕ кейсы — anamnesis (личный опыт), не faq."
        ),
        "annotation_rule": (
            "subtype ∈ {symptom, complaint}, urgency=normal обычно, urgent если упомянута сильная боль/кровь"
        ),
        "seed_examples": [
            "После удаления зуба третий день болит — это нормально?",
            "Почему у меня кровит десна когда чищу зубы?",
            "У меня щека опухла после пломбы, это пройдёт само?",
            "Зуб ноет на холодное, это кариес наверное?",
        ],
    },
    {
        "id": "post_treatment_complications",
        "count": 15,
        "route_domain": "anamnesis",
        "check_domain_consistency": True,
        "description": (
            "Осложнения после стоматологического лечения. Обычно urgent "
            "(температура, отёк, сильная боль)."
        ),
        "annotation_rule": "subtype ∈ {symptom, complaint}, urgency=urgent",
        "seed_examples": [
            "После пломбирования канала щека раздулась и температура 38",
            "Через 2 дня после удаления опухла десна и гной",
            "После имплантации болит сильнее чем в первый день",
        ],
    },
    {
        "id": "mixed_pain_price_booking",
        "count": 15,
        "route_domain": "anamnesis",
        # НЕ применяем check: по природе содержит price/booking маркеры при
        # anamnesis-разметке (domain_priority решает в prod-логике).
        "check_domain_consistency": False,
        "description": (
            "Боль/симптом + вопрос про цену или запись в одном сообщении. "
            "domain_priority → anamnesis (симптом побеждает)."
        ),
        "annotation_rule": (
            "subtype ∈ {symptom, complaint, treatment_interest}, explicit_booking=true если есть запись, urgency=normal"
        ),
        "seed_examples": [
            "Болит зуб, сколько стоит лечение и когда можно прийти?",
            "Выпала пломба, сколько поставить новую?",
            "Хочу чистку, но десна кровоточит",
        ],
    },
    {
        "id": "vague_short_urgent",
        "count": 15,
        "route_domain": "anamnesis",
        # Короткие фразы часто не содержат regex-маркеров "бол\w+ зуб" →
        # ложные срабатывания. Отключаем для этого сценария.
        "check_domain_consistency": False,
        "description": (
            "Одно-двусловные экстренные фразы. Краткость не снимает urgency. "
            "Модель должна распознать emergency даже без контекста."
        ),
        "annotation_rule": "subtype=symptom, urgency=urgent, длина <= 4 слова",
        "seed_examples": ["Кровь", "Скорая", "Плохо ребёнку", "Срочно помогите"],
    },
    {
        "id": "pediatric_trauma",
        "count": 12,
        "route_domain": "anamnesis",
        "check_domain_consistency": True,
        "description": (
            "Детские травмы зубов / челюсти (падения, удары). ВСЕГДА urgent, "
            "потому что ребёнок."
        ),
        "annotation_rule": "subtype ∈ {complaint, symptom}, urgency=urgent, упоминание ребёнка/возраста",
        "seed_examples": [
            "Ребёнок 3 года ударился зубами об качели, кровь из десны",
            "Сын упал с велосипеда, передний зуб шатается",
            "Дочке 5 лет разбила губу, зуб выбит полностью",
        ],
    },
    {
        "id": "allergy_bleeding_swelling",
        "count": 15,
        "route_domain": "anamnesis",
        "check_domain_consistency": True,
        "description": (
            "Аллергические реакции на препараты, кровотечения, сильные отёки — "
            "все urgent."
        ),
        "annotation_rule": "subtype=symptom, urgency=urgent",
        "seed_examples": [
            "Аллергия на анестезию, губа распухла, не проходит два дня",
            "Кровь из лунки после удаления не останавливается 3 часа",
            "После лечения отёк дошёл до глаза",
        ],
    },
    {
        "id": "switch_additional",
        "count": 15,
        "route_domain": "mixed",   # зависит от active_domain → new_domain
        # Для switch route_domain определяется LLM (не фиксирован сценарием),
        # поэтому check не применяем — он использовал бы неверный expected.
        "check_domain_consistency": False,
        "description": (
            "Дополнительные switch-кейсы (переход между доменами в диалоге). "
            "Каждый кандидат имеет active_domain (предыдущий контекст) и text "
            "(новое сообщение, попадающее в другой домен). route_domain — это "
            "НОВЫЙ домен, в который переключается пользователь."
        ),
        "annotation_rule": (
            "route_domain определяется текстом (symptom → anamnesis, price/info → faq, "
            "запись → booking); active_domain ≠ route_domain"
        ),
        "seed_examples": [
            "Стоп, у меня ещё зуб болит",           # booking → anamnesis
            "А какая цена у импланта?",              # anamnesis → faq
            "Ладно, запишите на пятницу",            # faq → booking
        ],
    },
]


# ---------------------------------------------------------------------------
# Промпт
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _build_prompt(scenario: dict[str, Any], template: str) -> str:
    """Подстановка параметров сценария в шаблон промпта."""
    seeds_block = "\n".join(f"- {s!r}" for s in scenario["seed_examples"])

    if scenario["id"] == "switch_additional":
        active_block = (
            "## Для switch-сценария:\n"
            "Каждый кандидат обязан содержать `active_domain` — контекст предыдущей "
            "реплики ассистента (anamnesis | faq | booking), отличный от `route_domain` "
            "(определяемого текстом кандидата)."
        )
    else:
        active_block = ""

    return (
        template
        .replace("{scenario_id}", scenario["id"])
        .replace("{scenario_route_domain}", scenario["route_domain"])
        .replace("{scenario_description}", scenario["description"])
        .replace("{annotation_rule}", scenario["annotation_rule"])
        .replace("{scenario_seed_examples_block}", seeds_block)
        .replace("{scenario_active_domain_block}", active_block)
        .replace("{n_candidates}", str(scenario["count"]))
    )


# ---------------------------------------------------------------------------
# Парсинг LLM ответа
# ---------------------------------------------------------------------------

_VALID_DOMAINS = {"anamnesis", "faq", "booking", "unsupported"}
_VALID_URGENCY = {"normal", "urgent"}


def _parse_candidates(raw: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Парсинг JSON списка кандидатов из ответа LLM + фильтры.

    Применяемые фильтры (отбрасываем кандидата):
    - пустой или не-кириллический текст
    - украинские символы
    - urgency/subtype вне допустимых значений
    - switch без `active_domain` или active_domain == route_domain
    """
    content = raw.strip()
    if content.startswith("```"):
        # Срезаем markdown fence
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Не удалось распарсить JSON для %s: %s", scenario["id"], raw[:200])
        return []

    if isinstance(data, dict):
        # Иногда LLM оборачивает в {"candidates": [...]}
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        logger.warning("Ожидался JSON массив для %s", scenario["id"])
        return []

    is_switch = scenario["id"] == "switch_additional"
    apply_domain_check = scenario.get("check_domain_consistency", False)
    accepted: list[dict[str, Any]] = []

    for raw_item in data:
        if not isinstance(raw_item, dict):
            continue
        text = (raw_item.get("text") or "").strip()
        if not text or not _ALLOWED_QUERY_RE.match(text):
            continue
        if _UKRAINIAN_CHARS_RE.search(text):
            continue

        urgency = (raw_item.get("urgency") or "normal").strip()
        if urgency not in _VALID_URGENCY:
            continue

        subtype = (raw_item.get("subtype") or "").strip()
        if not subtype:
            continue

        item: dict[str, Any] = {
            "text": text,
            "subtype": subtype,
            "urgency": urgency,
        }

        if is_switch:
            active_domain = (raw_item.get("active_domain") or "").strip()
            route_domain = (raw_item.get("route_domain") or "").strip()
            if active_domain not in _VALID_DOMAINS or route_domain not in _VALID_DOMAINS:
                logger.debug(
                    "switch skip (invalid domains): active=%r, route=%r, text=%r",
                    active_domain, route_domain, text[:60],
                )
                continue
            if active_domain == route_domain:
                logger.debug(
                    "switch skip (active==route=%s): text=%r", active_domain, text[:60],
                )
                continue
            item["active_domain"] = active_domain
            item["route_domain"] = route_domain
        else:
            item["route_domain"] = scenario["route_domain"]

        # SSoT domain consistency check (reuse из generate_dataset.py).
        # Применяется только для сценариев с фиксированным expected_domain,
        # без ожидаемых маркеров чужих доменов.
        if apply_domain_check and not _check_domain_consistency(
            text, item["route_domain"], scenario["id"],
        ):
            logger.debug(
                "domain leak skip [%s]: %r (expected=%s)",
                scenario["id"], text[:60], item["route_domain"],
            )
            continue

        accepted.append(item)

    return accepted


# ---------------------------------------------------------------------------
# Async LLM генерация
# ---------------------------------------------------------------------------

async def _generate_scenario(
    client: OpenRouterClient,
    template: str,
    scenario: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Генерация кандидатов для одного сценария."""
    prompt = _build_prompt(scenario, template)
    async with semaphore:
        try:
            raw, _usage = await client.async_chat(
                messages=[{"role": "user", "content": prompt}],
                model=VARIATION_MODEL,
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 — сохраняем flow
            logger.error("LLM ошибка для сценария %s: %s", scenario["id"], exc)
            return []

    parsed = _parse_candidates(raw, scenario)
    logger.info(
        "%s: получено %d/%d валидных кандидатов",
        scenario["id"], len(parsed), scenario["count"],
    )
    return [
        {**item, "scenario": scenario["id"]}
        for item in parsed
    ]


async def _run_generation(
    scenarios: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Параллельная генерация всех сценариев."""
    template = _load_prompt_template()
    client = OpenRouterClient()
    sem = asyncio.Semaphore(concurrency)

    tasks = [_generate_scenario(client, template, sc, sem) for sc in scenarios]
    results = await asyncio.gather(*tasks)
    await client.async_close()

    stats = client.get_stats()
    logger.info(
        "Генерация завершена: calls=%d, tokens_in=%d, tokens_out=%d",
        stats["total_calls"], stats["total_tokens_in"], stats["total_tokens_out"],
    )

    all_candidates: list[dict[str, Any]] = []
    for cands in results:
        all_candidates.extend(cands)
    return all_candidates


# ---------------------------------------------------------------------------
# Anti-contamination: BGE-M3 semantic overlap
# ---------------------------------------------------------------------------

def _load_existing_hard_texts() -> list[dict[str, str]]:
    """Все тексты из текущего hard_cases.yaml (для leakage check)."""
    with open(HARD_CASES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        {"id": row["id"], "text": row["text"]}
        for row in data.get("hard_cases", [])
    ]


def _load_train_seed_texts() -> list[dict[str, str]]:
    """Все seed-тексты (для leakage check vs train distribution)."""
    with open(SEEDS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        {"id": row["id"], "text": row["text"]}
        for row in data.get("seeds", [])
    ]


def _encode_texts(encoder: Any, texts: list[str]) -> np.ndarray:
    """Embed текстов через BGE-M3 + L2-нормализация (cosine == dot)."""
    embs = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _max_sim_and_idx(
    query_emb: np.ndarray, target_embs: np.ndarray,
) -> tuple[float, int]:
    """Максимальный cosine + индекс ближайшего target."""
    sims = target_embs @ query_emb
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


def _anti_contamination(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Применить anti-contamination check → (accepted_candidates, similarity_df).

    accepted: auto_reject=False и intra-pool near-dup не помечен.
    similarity_df: полный лог для трассировки (включая отброшенных).
    """
    if not candidates:
        return [], pd.DataFrame()

    # Используем B2EmbeddingClassifier только как провайдер encoder'а — не обучаем.
    encoder_holder = B2EmbeddingClassifier()
    encoder = encoder_holder._get_encoder()

    hard_rows = _load_existing_hard_texts()
    seed_rows = _load_train_seed_texts()

    logger.info(
        "Anti-contamination: %d кандидатов vs %d hard_cases + %d train seeds",
        len(candidates), len(hard_rows), len(seed_rows),
    )

    # Один encode batch для всего: candidates | hard | seed
    cand_texts = [c["text"] for c in candidates]
    hard_texts = [r["text"] for r in hard_rows]
    seed_texts = [r["text"] for r in seed_rows]

    cand_embs = _encode_texts(encoder, cand_texts)
    hard_embs = _encode_texts(encoder, hard_texts) if hard_texts else np.zeros((0, cand_embs.shape[1]))
    seed_embs = _encode_texts(encoder, seed_texts) if seed_texts else np.zeros((0, cand_embs.shape[1]))

    similarity_rows: list[dict[str, Any]] = []
    accepted_flags = [True] * len(candidates)

    # (1) vs hard + seed
    for i, cand in enumerate(candidates):
        max_hard_sim, hard_idx = (0.0, -1)
        if hard_embs.shape[0] > 0:
            max_hard_sim, hard_idx = _max_sim_and_idx(cand_embs[i], hard_embs)
        max_seed_sim, seed_idx = (0.0, -1)
        if seed_embs.shape[0] > 0:
            max_seed_sim, seed_idx = _max_sim_and_idx(cand_embs[i], seed_embs)

        auto_reject = (
            max_hard_sim >= _LEAKAGE_REJECT_THRESHOLD
            or max_seed_sim >= _LEAKAGE_REJECT_THRESHOLD
        )
        if auto_reject:
            accepted_flags[i] = False

        similarity_rows.append({
            "candidate_idx": i,
            "scenario": cand["scenario"],
            "text": cand["text"],
            "max_sim_to_hard": round(max_hard_sim, 4),
            "nearest_hard_id": hard_rows[hard_idx]["id"] if hard_idx >= 0 else "",
            "max_sim_to_seed": round(max_seed_sim, 4),
            "nearest_seed_id": seed_rows[seed_idx]["id"] if seed_idx >= 0 else "",
            "auto_reject_reason": (
                "sim>=0.92_to_hard" if max_hard_sim >= _LEAKAGE_REJECT_THRESHOLD
                else "sim>=0.92_to_seed" if max_seed_sim >= _LEAKAGE_REJECT_THRESHOLD
                else ""
            ),
            "intra_pool_duplicate_of": "",
        })

    # (2) Intra-pool: среди тех, кто ещё accepted, отсев near-duplicates
    # (sim >= 0.95). Оставляем первого вхождения.
    for i in range(len(candidates)):
        if not accepted_flags[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not accepted_flags[j]:
                continue
            sim = float(cand_embs[i] @ cand_embs[j])
            if sim >= _INTRA_POOL_DUP_THRESHOLD:
                accepted_flags[j] = False
                similarity_rows[j]["intra_pool_duplicate_of"] = candidates[i]["text"][:60]
                similarity_rows[j]["auto_reject_reason"] = (
                    similarity_rows[j]["auto_reject_reason"] or "intra_pool_dup>=0.95"
                )

    accepted = [c for c, ok in zip(candidates, accepted_flags) if ok]
    logger.info(
        "Anti-contamination accept: %d / %d (reject %d)",
        len(accepted), len(candidates), len(candidates) - len(accepted),
    )

    return accepted, pd.DataFrame(similarity_rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _assign_candidate_ids(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Присвоить candidate_id = cand_001, cand_002, ... (стабильный порядок)."""
    result = []
    for i, cand in enumerate(candidates, 1):
        result.append({"candidate_id": f"cand_{i:03d}", **cand})
    return result


def _save_candidates_yaml(candidates: list[dict[str, Any]]) -> None:
    """Сохранение pool в YAML (flow=false для читаемости, UTF-8)."""
    payload = {
        "generated_by": VARIATION_MODEL,
        "prompt_file": str(_PROMPT_FILE.relative_to(_STUDY_ROOT)),
        "scenarios": [s["id"] for s in _SCENARIOS],
        "anti_contamination_thresholds": {
            "leakage_reject": _LEAKAGE_REJECT_THRESHOLD,
            "intra_pool_dup": _INTRA_POOL_DUP_THRESHOLD,
        },
        "candidates": candidates,
    }
    with open(_CANDIDATES_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)
    logger.info("Candidates → %s (%d шт.)", _CANDIDATES_FILE, len(candidates))


def _save_similarity_csv(df: pd.DataFrame) -> None:
    df.to_csv(_SIMILARITY_FILE, index=False, encoding="utf-8")
    logger.info("Similarity report → %s (%d строк)", _SIMILARITY_FILE, len(df))


def _save_audit_template(candidates: list[dict[str, Any]], sim_df: pd.DataFrame) -> None:
    """Создание/обновление audit schema для ручной работы.

    Колонки: candidate_id, scenario, text, proposed_*, similarity metrics,
    audit_decision (ПУСТОЕ — заполняет человек), final_text, notes.
    """
    sim_lookup = (
        sim_df.set_index("candidate_idx")[
            ["max_sim_to_hard", "nearest_hard_id", "max_sim_to_seed",
             "nearest_seed_id", "auto_reject_reason", "intra_pool_duplicate_of"]
        ].to_dict("index")
        if not sim_df.empty else {}
    )

    rows: list[dict[str, Any]] = []
    # Кандидаты без auto_reject → в audit. Отклонённые auto_reject НЕ попадают
    # в audit (они уже отфильтрованы), но лог сохранён в similarity.csv.
    for i, cand in enumerate(candidates):
        sim_info = sim_lookup.get(i, {})
        rows.append({
            "candidate_id": cand["candidate_id"],
            "scenario": cand["scenario"],
            "text": cand["text"],
            "proposed_route_domain": cand["route_domain"],
            "proposed_subtype": cand["subtype"],
            "proposed_urgency": cand["urgency"],
            "proposed_active_domain": cand.get("active_domain", ""),
            "max_sim_to_hard": sim_info.get("max_sim_to_hard", ""),
            "nearest_hard_id": sim_info.get("nearest_hard_id", ""),
            "max_sim_to_seed": sim_info.get("max_sim_to_seed", ""),
            # audit-поля (пустые — заполняет человек)
            "audit_decision": "",        # accept | reject | rewrite
            "source_after_audit": "",    # grok_audited | grok_rewritten | (пусто, если rejected)
            "final_text": "",            # заполняется при rewrite
            "final_route_domain": "",
            "final_subtype": "",
            "final_urgency": "",
            "notes": "",
        })

    pd.DataFrame(rows).to_csv(_AUDIT_FILE, index=False, encoding="utf-8")
    logger.info("Audit template → %s (%d строк, ждёт ручной audit)", _AUDIT_FILE, len(rows))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация hard case candidates")
    parser.add_argument(
        "--concurrency", type=int, default=_CONCURRENCY_DEFAULT,
        help="Параллельных LLM вызовов",
    )
    parser.add_argument(
        "--dry-scenario", type=str, default=None,
        help="Только один сценарий (id) — для sanity check",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scenarios = _SCENARIOS
    if args.dry_scenario:
        scenarios = [s for s in _SCENARIOS if s["id"] == args.dry_scenario]
        if not scenarios:
            raise SystemExit(
                f"Сценарий '{args.dry_scenario}' не найден. "
                f"Доступно: {[s['id'] for s in _SCENARIOS]}",
            )

    # 1. LLM генерация
    raw_candidates = asyncio.run(_run_generation(scenarios, args.concurrency))
    logger.info("Сырых кандидатов после парсинга: %d", len(raw_candidates))
    if not raw_candidates:
        logger.error("Ни одного валидного кандидата. Прерываем.")
        raise SystemExit(1)

    # 2. Anti-contamination
    accepted, similarity_df = _anti_contamination(raw_candidates)

    # 3. Присвоение ID (после отсева)
    accepted = _assign_candidate_ids(accepted)

    # 4. Persistence
    _save_candidates_yaml(accepted)
    _save_similarity_csv(similarity_df)
    _save_audit_template(accepted, similarity_df)

    print(
        f"\n✓ Готово: {len(accepted)} кандидатов после anti-contamination "
        f"(из {len(raw_candidates)} сгенерированных).\n"
        f"  Candidates YAML: {_CANDIDATES_FILE}\n"
        f"  Similarity CSV:  {_SIMILARITY_FILE}\n"
        f"  Audit CSV:       {_AUDIT_FILE}\n"
        f"Следующий шаг: ручной audit по плану §1 Этап B (accept/reject/rewrite).",
    )


if __name__ == "__main__":
    main()
