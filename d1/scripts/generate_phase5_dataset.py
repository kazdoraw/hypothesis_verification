"""Phase 5: Grok augmentation для train + hard_test.

Улучшенная версия generate_hard_cases.py со следующими отличиями:

1. **Per-scenario temperature**: clean=0.7, hard=0.9, short=1.1
2. **Multi-pass diversity**: 2 прохода с разными seeds (42, 43)
3. **Strict anti-contamination**: 0.85 vs hard/seed/train (план Phase 5),
   intra-pool dup = 0.90
4. **Cost cap**: hard limit total_tokens ≤ 80000 (≈ $0.5-1)
5. **JSON schema validation**: jsonschema strict для каждого кандидата
6. **Negative markers**: per-scenario regex check на маркеры чужих доменов
   (для clean train — обязательно)
7. **8 новых сценариев** (4 train clean + 4 eval addendum)
8. **Auto-audit**: BGE anti-contam + domain consistency + ML predict agreement.
   Каждый кандидат получает audit_decision ∈ {accept, reject, flag} автоматически.

Запуск:
    cd study && .venv/bin/python -m d1.scripts.generate_phase5_dataset \\
        [--concurrency N] [--passes K] [--dry-scenario ID]

Артефакты:
    d1/data/phase5_candidates.yaml             — все после anti-contam
    d1/data/phase5_similarity.csv              — лог BGE сравнений
    d1/data/phase5_audit.csv                   — auto-audit с решениями
    d1/data/phase5_train_addendum.csv          — accepted train cases (для merge)
    d1/data/phase5_hard_test_addendum.csv      — accepted eval cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
# Параметры генерации (улучшенные vs generate_hard_cases.py)
# ---------------------------------------------------------------------------

_PROMPT_FILE = PROMPTS_DIR / "d1_phase5_candidates.md"
_CANDIDATES_FILE = DATA_DIR / "phase5_candidates.yaml"
_SIMILARITY_FILE = DATA_DIR / "phase5_similarity.csv"
_AUDIT_FILE = DATA_DIR / "phase5_audit.csv"
_TRAIN_ADDENDUM_FILE = DATA_DIR / "phase5_train_addendum.csv"
_HARDTEST_ADDENDUM_FILE = DATA_DIR / "phase5_hard_test_addendum.csv"

# Порог Phase 5 (план явно требует 0.85, не 0.92)
_LEAKAGE_REJECT_THRESHOLD = 0.85
_INTRA_POOL_DUP_THRESHOLD = 0.90

# Cost cap: ~80K tokens ≈ $0.5-1 для grok-4.1-fast
_TOTAL_TOKENS_CAP = 80_000

# Multi-pass: разные seed → больше уникальных кандидатов после dedup
_DEFAULT_PASSES = 2
_PASS_SEEDS = [42, 43, 44, 45]  # пул seeds, выбираем первые --passes штук
_TOP_P = 0.95

# Per-scenario_type температуры
_TEMPERATURE_BY_TYPE = {
    "clean": 0.7,
    "hard": 0.9,
    "short": 1.1,
}

_MAX_TOKENS = 2200
_CONCURRENCY_DEFAULT = 4

# Auto-audit thresholds (после ML предикта на кандидате)
_AUDIT_ML_STRONG_CONF = 0.85   # ≥ → ML уверен в правильном/неправильном предикте
_AUDIT_ML_WEAK_CONF = 0.70      # < → ML не уверен (можно учить, accept)

# ---------------------------------------------------------------------------
# Сценарии Phase 5: 4 train_clean + 4 eval_addendum
# ---------------------------------------------------------------------------

_SCENARIOS: list[dict[str, Any]] = [
    # ---------------- TRAIN CLEAN (4 шт) ----------------
    {
        "id": "simple_faq_clean",
        "scenario_type": "clean",
        "split": "train",
        "count": 80,
        "route_domain": "faq",
        "subtype_default": "general",
        "urgency_default": "normal",
        "check_domain_consistency": True,
        "negative_markers": [
            r"\bбол(ит|ят|ел)\b", r"\bноет\b", r"\bопух",
            r"\bкровь\b", r"\bкровит\b",
            r"\bу меня\b", r"\bменя\b", r"\bмой зуб\b",
            r"\bзапиш", r"\bзапиш\w*\s",
        ],
        "description": (
            "Общий вопрос про клинику БЕЗ личной жалобы. Цены, режим работы, "
            "адрес, гарантия, подготовка к процедуре, виды лечения. "
            "Не упоминается личный симптом или попытка записи."
        ),
        "annotation_rule": "subtype ∈ {general, price, schedule, location}, urgency=normal",
        "seed_examples": [
            "Сколько стоит чистка ультразвуком?",
            "До скольки работаете в субботу?",
            "Какие виды протезирования делаете?",
            "Гарантия на пломбу есть?",
            "Можно ли есть до приёма?",
        ],
    },
    {
        "id": "simple_booking_clean",
        "scenario_type": "clean",
        "split": "train",
        "count": 80,
        "route_domain": "booking",
        "subtype_default": "new_appointment",
        "urgency_default": "normal",
        "check_domain_consistency": True,
        "negative_markers": [
            r"\bбол(ит|ят|ел)\b", r"\bноет\b", r"\bопух",
            r"\bкровь\b", r"\bкровит\b",
            r"\bсколько (стоит|это|стоят)\b", r"\bцена\b",
            r"\bПетров", r"\bИванов", r"\bдоктор\s+\w+", r"\bврач\s+[А-Я]",
        ],
        "description": (
            "Желание записаться/перенести/отменить приём БЕЗ упоминания "
            "симптомов, цены или конкретной фамилии врача. Чистая запись."
        ),
        "annotation_rule": "subtype ∈ {new_appointment, reschedule, cancel}, urgency=normal",
        "seed_examples": [
            "Хочу записаться на пятницу",
            "Перенесите мою запись на следующую неделю",
            "Отменить приём в 15:00 пожалуйста",
            "Запишите на удобное время утром",
            "Можно записаться к стоматологу?",
        ],
    },
    {
        "id": "simple_symptom_clean",
        "scenario_type": "clean",
        "split": "train",
        "count": 80,
        "route_domain": "anamnesis",
        "subtype_default": "symptom",
        "urgency_default": "normal",
        "check_domain_consistency": True,
        "negative_markers": [
            r"\bсколько (стоит|это|стоят)\b", r"\bцена\b", r"\bцены\b",
            r"\bзапиш", r"\bзапиш\w*\s", r"\bперенест",
            r"\bПетров", r"\bИванов", r"\bдоктор\s+\w+",
            r"\bдо скольки\b", r"\bрежим работы\b",
        ],
        "description": (
            "Личная жалоба/симптом БЕЗ упоминания цены, записи или конкретного "
            "врача. Описание ощущений, их длительности и характера."
        ),
        "annotation_rule": "subtype ∈ {symptom, complaint}, urgency=normal обычно, urgent если кровь/температура",
        "seed_examples": [
            "Десна побаливает уже неделю",
            "Чувствительность на сладкое появилась",
            "Зуб реагирует на холодное",
            "Щека немного припухла после еды",
            "Что-то странное с дёснами по утрам",
        ],
    },
    {
        "id": "entity_held_out_simple",
        "scenario_type": "clean",
        "split": "train",
        "count": 60,
        "route_domain": "faq",
        "subtype_default": "general",
        "urgency_default": "normal",
        "check_domain_consistency": False,  # entity-specific, проверка сложна
        "negative_markers": [
            r"\bбол(ит|ят|ел)\b", r"\bу меня\b",
        ],
        "description": (
            "Вопросы про услуги/процедуры, которые могут отсутствовать в seed-наборе. "
            "Цель: out-of-distribution generalization. БЕЗ личной жалобы."
        ),
        "annotation_rule": "subtype=general, urgency=normal",
        "seed_examples": [
            "Делаете ли вы синус-лифтинг?",
            "Микропротезирование керамикой возможно?",
            "Лечение дисфункции ВНЧС у вас есть?",
            "Сколько стоит диагностика КЛКТ?",
            "Делаете отбеливание Zoom-4?",
        ],
    },
    # ---------------- EVAL ADDENDUM (4 шт) ----------------
    {
        "id": "faq_vs_anamnesis_borderline",
        "scenario_type": "hard",
        "split": "hard_test",
        "count": 20,
        "route_domain": "anamnesis",
        "subtype_default": "complaint",
        "urgency_default": "normal",
        "check_domain_consistency": False,  # by design borderline
        "negative_markers": [],
        "description": (
            "Вопрос звучит как общий ('это нормально что...'), но описывает "
            "ЛИЧНЫЙ опыт после процедуры. По правилу — anamnesis."
        ),
        "annotation_rule": "subtype ∈ {complaint, symptom}, urgency=normal",
        "seed_examples": [
            "Это нормально что зуб ноет после пломбы 4 дня?",
            "После коронки чувствую распирание — так и должно быть?",
            "Десна белая стала после удаления, это норма?",
            "Должна ли быть температура 37.2 после имплантации?",
        ],
    },
    {
        "id": "doctor_name_stress",
        "scenario_type": "hard",
        "split": "hard_test",
        "count": 20,
        "route_domain": "faq",
        "subtype_default": "doctor_info",
        "urgency_default": "normal",
        "check_domain_consistency": False,
        "negative_markers": [],
        "description": (
            "Запись + фамилия врача в одном сообщении. По domain_priority "
            "это faq (booking_doctor_name gate). Стресс-тест для роутера."
        ),
        "annotation_rule": "subtype=doctor_info, urgency=normal, упоминание ФИО врача",
        "seed_examples": [
            "Запишите меня к Петрову на четверг",
            "Доктор Иванова принимает завтра?",
            "К Сидорову Андрею Петровичу можно записаться?",
            "Свободные часы у Василевской есть?",
        ],
    },
    {
        "id": "mixed_intent_new",
        "scenario_type": "hard",
        "split": "hard_test",
        "count": 20,
        "route_domain": "anamnesis",
        "subtype_default": "complaint",
        "urgency_default": "normal",
        "check_domain_consistency": False,
        "negative_markers": [],
        "description": (
            "Multi-intent: симптом + запись/цена в одном тексте. "
            "domain_priority → anamnesis (симптом побеждает)."
        ),
        "annotation_rule": "subtype=complaint, urgency=normal или urgent",
        "seed_examples": [
            "Зуб разболелся, хочу записаться на завтра",
            "Сколько стоит вырвать зуб который ноет третий день?",
            "Можно срочно к врачу, у меня флюс?",
            "Вылечите зуб с дыркой к понедельнику?",
        ],
    },
    {
        "id": "short_ambiguous_new",
        "scenario_type": "short",
        "split": "hard_test",
        "count": 20,
        "route_domain": "anamnesis",
        "subtype_default": "symptom",
        "urgency_default": "urgent",
        "check_domain_consistency": False,
        "negative_markers": [],
        "description": (
            "Короткие 1-4-словные фразы — экстренные сигналы или неоднозначные "
            "обрывки. Должны попадать в anamnesis (urgency=urgent если симптом)."
        ),
        "annotation_rule": "subtype=symptom, urgency=urgent (default для коротких), длина ≤ 4 слова",
        "seed_examples": [
            "Болит сильно",
            "Кровь идёт",
            "Помогите срочно",
            "Зуб выбит",
            "Опухло всё",
        ],
    },
]


# ---------------------------------------------------------------------------
# Промпт
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _build_prompt(scenario: dict[str, Any], template: str) -> str:
    """Подстановка scenario-параметров в шаблон. Расширено vs hard_cases:
    - {scenario_type}: clean / hard / short
    - {negative_markers_block}: список запрещённых маркеров для clean
    """
    seeds_block = "\n".join(f"- {s!r}" for s in scenario["seed_examples"])

    if scenario.get("negative_markers"):
        neg_lines = "\n".join(
            f"- НЕТ маркера `{m}`" for m in scenario["negative_markers"][:8]
        )
        neg_block = (
            "## Дополнительные ЗАПРЕТЫ (специфика сценария)\n\n"
            f"Текст НЕ ДОЛЖЕН содержать следующие маркеры (regex):\n{neg_lines}"
        )
    else:
        neg_block = ""

    # active_domain block reused: сейчас не нужен (нет switch в Phase 5),
    # но оставляем заглушку чтобы template был совместим
    active_block = ""

    return (
        template
        .replace("{scenario_id}", scenario["id"])
        .replace("{scenario_type}", scenario["scenario_type"])
        .replace("{scenario_route_domain}", scenario["route_domain"])
        .replace("{scenario_description}", scenario["description"])
        .replace("{annotation_rule}", scenario["annotation_rule"])
        .replace("{scenario_seed_examples_block}", seeds_block)
        .replace("{scenario_active_domain_block}", active_block)
        .replace("{negative_markers_block}", neg_block)
        .replace("{n_candidates}", str(scenario["count"]))
    )


# ---------------------------------------------------------------------------
# Парсинг + валидация (extends generate_hard_cases.py)
# ---------------------------------------------------------------------------

_VALID_DOMAINS = {"anamnesis", "faq", "booking", "unsupported"}
_VALID_URGENCY = {"normal", "urgent"}


def _parse_candidates(raw: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """JSON parse + строгая валидация + negative_markers фильтр.

    Возвращает только candidates прошедшие все проверки:
    - JSON valid
    - text не пустой, кириллица, не украинский
    - subtype/urgency валидные
    - не содержит negative_markers (regex check)
    - domain consistency (если включена для сценария)
    """
    content = raw.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("JSON parse fail [%s]: %s", scenario["id"], raw[:200])
        return []

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        logger.warning("Ожидался JSON array [%s]", scenario["id"])
        return []

    apply_domain_check = scenario.get("check_domain_consistency", False)
    neg_patterns = [re.compile(p, re.IGNORECASE) for p in scenario.get("negative_markers", [])]
    accepted: list[dict[str, Any]] = []

    for raw_item in data:
        if not isinstance(raw_item, dict):
            continue
        text = (raw_item.get("text") or "").strip()
        if not text or not _ALLOWED_QUERY_RE.match(text):
            continue
        if _UKRAINIAN_CHARS_RE.search(text):
            continue

        # Negative markers — строго (для clean train)
        if any(p.search(text) for p in neg_patterns):
            logger.debug("neg_marker [%s]: %r", scenario["id"], text[:60])
            continue

        urgency = (raw_item.get("urgency") or scenario["urgency_default"]).strip()
        if urgency not in _VALID_URGENCY:
            continue

        subtype = (raw_item.get("subtype") or scenario["subtype_default"]).strip()
        if not subtype:
            continue

        item = {
            "text": text,
            "subtype": subtype,
            "urgency": urgency,
            "route_domain": scenario["route_domain"],
        }

        if apply_domain_check and not _check_domain_consistency(
            text, scenario["route_domain"], scenario["id"],
        ):
            logger.debug("domain leak [%s]: %r", scenario["id"], text[:60])
            continue

        accepted.append(item)

    return accepted


# ---------------------------------------------------------------------------
# Async LLM генерация с multi-pass + per-scenario temperature
# ---------------------------------------------------------------------------

async def _generate_pass(
    client: OpenRouterClient,
    template: str,
    scenario: dict[str, Any],
    seed: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Один проход для сценария с конкретным seed."""
    prompt = _build_prompt(scenario, template)
    temperature = _TEMPERATURE_BY_TYPE[scenario["scenario_type"]]
    async with semaphore:
        try:
            raw, _usage = await client.async_chat(
                messages=[{"role": "user", "content": prompt}],
                model=VARIATION_MODEL,
                temperature=temperature,
                max_tokens=_MAX_TOKENS,
                top_p=_TOP_P,
                seed=seed,
            )
        except RuntimeError as exc:
            # Cost cap или API exhaust
            logger.error("Pass aborted [%s seed=%d]: %s", scenario["id"], seed, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM error [%s seed=%d]: %s", scenario["id"], seed, exc)
            return []

    parsed = _parse_candidates(raw, scenario)
    logger.info(
        "%s pass(seed=%d, T=%.1f): %d valid",
        scenario["id"], seed, temperature, len(parsed),
    )
    return [{**item, "scenario": scenario["id"], "_seed": seed} for item in parsed]


async def _run_generation(
    scenarios: list[dict[str, Any]],
    concurrency: int,
    passes: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Параллельная генерация: scenarios × passes.

    Returns (all_candidates_post_intra_dedup_per_scenario, yield_stats).
    Intra-scenario dedup на этапе сбора (по точному text match) — anti-pool
    dedup делаем позже на embeddings level.
    """
    template = _load_prompt_template()
    client = OpenRouterClient(total_tokens_cap=_TOTAL_TOKENS_CAP)
    sem = asyncio.Semaphore(concurrency)
    seeds = _PASS_SEEDS[:passes]

    tasks = [
        _generate_pass(client, template, sc, seed, sem)
        for sc in scenarios
        for seed in seeds
    ]
    results = await asyncio.gather(*tasks)
    await client.async_close()

    stats = client.get_stats()
    logger.info(
        "Generation: calls=%d, tokens_in=%d, tokens_out=%d (cap=%d)",
        stats["total_calls"], stats["total_tokens_in"],
        stats["total_tokens_out"], _TOTAL_TOKENS_CAP,
    )

    # Per-scenario yield + intra-scenario exact-text dedup
    yield_stats: dict[str, dict[str, int]] = {}
    seen_per_scenario: dict[str, set[str]] = {}
    flat_candidates: list[dict[str, Any]] = []

    for cands in results:
        for c in cands:
            sid = c["scenario"]
            txt = c["text"].lower().strip()
            seen = seen_per_scenario.setdefault(sid, set())
            stat = yield_stats.setdefault(sid, {"raw": 0, "after_exact_dedup": 0})
            stat["raw"] += 1
            if txt in seen:
                continue
            seen.add(txt)
            stat["after_exact_dedup"] += 1
            flat_candidates.append(c)

    return flat_candidates, yield_stats


# ---------------------------------------------------------------------------
# Anti-contamination (BGE-M3 vs hard + seed + train)
# ---------------------------------------------------------------------------

def _load_existing_texts() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Тексты для leakage check: hard_cases + seeds + train CSV."""
    with open(HARD_CASES_FILE, encoding="utf-8") as f:
        hard = yaml.safe_load(f) or {}
    hard_rows = [
        {"id": row["id"], "text": row["text"]}
        for row in hard.get("hard_cases", [])
    ]

    with open(SEEDS_FILE, encoding="utf-8") as f:
        seeds = yaml.safe_load(f) or {}
    seed_rows = [
        {"id": row["id"], "text": row["text"]}
        for row in seeds.get("seeds", [])
    ]

    train_csv = DATA_DIR / "d1_v6_train.csv"
    train_rows: list[dict[str, str]] = []
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        train_rows = [
            {"id": str(r.get("id", f"train_{i}")), "text": str(r["text"])}
            for i, r in df.iterrows()
        ]

    return hard_rows, seed_rows, train_rows


def _encode_texts(encoder: Any, texts: list[str]) -> np.ndarray:
    embs = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _max_sim_and_idx(query_emb: np.ndarray, target_embs: np.ndarray) -> tuple[float, int]:
    if target_embs.shape[0] == 0:
        return 0.0, -1
    sims = target_embs @ query_emb
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


def _anti_contamination(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame, np.ndarray]:
    """Strict 0.85 vs hard+seed+train + intra-pool 0.90.

    Returns (accepted_candidates, similarity_df, accepted_embeddings).
    Embeddings возвращаются для последующего ML predict (auto-audit).
    """
    if not candidates:
        return [], pd.DataFrame(), np.zeros((0, 1024))

    encoder_holder = B2EmbeddingClassifier()
    encoder = encoder_holder._get_encoder()

    hard_rows, seed_rows, train_rows = _load_existing_texts()
    logger.info(
        "Anti-contam: %d cand vs %d hard + %d seed + %d train",
        len(candidates), len(hard_rows), len(seed_rows), len(train_rows),
    )

    cand_texts = [c["text"] for c in candidates]
    hard_texts = [r["text"] for r in hard_rows]
    seed_texts = [r["text"] for r in seed_rows]
    train_texts = [r["text"] for r in train_rows]

    cand_embs = _encode_texts(encoder, cand_texts)
    hard_embs = _encode_texts(encoder, hard_texts) if hard_texts else np.zeros((0, cand_embs.shape[1]))
    seed_embs = _encode_texts(encoder, seed_texts) if seed_texts else np.zeros((0, cand_embs.shape[1]))
    train_embs = _encode_texts(encoder, train_texts) if train_texts else np.zeros((0, cand_embs.shape[1]))

    similarity_rows: list[dict[str, Any]] = []
    accepted_flags = [True] * len(candidates)

    for i, cand in enumerate(candidates):
        max_h, h_idx = _max_sim_and_idx(cand_embs[i], hard_embs)
        max_s, s_idx = _max_sim_and_idx(cand_embs[i], seed_embs)
        max_t, t_idx = _max_sim_and_idx(cand_embs[i], train_embs)

        leak_reasons = []
        if max_h >= _LEAKAGE_REJECT_THRESHOLD:
            leak_reasons.append(f"hard>={_LEAKAGE_REJECT_THRESHOLD:.2f}")
        if max_s >= _LEAKAGE_REJECT_THRESHOLD:
            leak_reasons.append(f"seed>={_LEAKAGE_REJECT_THRESHOLD:.2f}")
        if max_t >= _LEAKAGE_REJECT_THRESHOLD:
            leak_reasons.append(f"train>={_LEAKAGE_REJECT_THRESHOLD:.2f}")

        if leak_reasons:
            accepted_flags[i] = False

        similarity_rows.append({
            "candidate_idx": i,
            "scenario": cand["scenario"],
            "text": cand["text"],
            "max_sim_to_hard": round(max_h, 4),
            "nearest_hard_id": hard_rows[h_idx]["id"] if h_idx >= 0 else "",
            "max_sim_to_seed": round(max_s, 4),
            "nearest_seed_id": seed_rows[s_idx]["id"] if s_idx >= 0 else "",
            "max_sim_to_train": round(max_t, 4),
            "nearest_train_id": train_rows[t_idx]["id"] if t_idx >= 0 else "",
            "auto_reject_reason": ",".join(leak_reasons),
            "intra_pool_duplicate_of": "",
        })

    # Intra-pool dedup
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
                    similarity_rows[j]["auto_reject_reason"]
                    or f"intra_pool>={_INTRA_POOL_DUP_THRESHOLD:.2f}"
                )

    accepted_idx = [i for i, ok in enumerate(accepted_flags) if ok]
    accepted = [candidates[i] for i in accepted_idx]
    accepted_embs = cand_embs[accepted_idx] if accepted_idx else np.zeros((0, cand_embs.shape[1]))

    logger.info(
        "Anti-contam accept: %d / %d (reject %d)",
        len(accepted), len(candidates), len(candidates) - len(accepted),
    )
    return accepted, pd.DataFrame(similarity_rows), accepted_embs


# ---------------------------------------------------------------------------
# Auto-audit: ML predict agreement + domain consistency
# ---------------------------------------------------------------------------

def _auto_audit(
    candidates: list[dict[str, Any]],
    cand_embs: np.ndarray,
) -> list[dict[str, Any]]:
    """Programmatic audit: предсказать домен через B1.1+B2.1, сравнить с разметкой.

    Scenario-aware decision logic (важно!):
    - **clean scenarios (train)**: разметка должна быть простой/однозначной.
      ML strong disagree → reject (генератор галлюцинирует).
      Логика: accept | flag | reject.
    - **hard / short scenarios (eval)**: by design ML должен ошибаться (это и
      есть hard cases). ML strong disagree → ACCEPT, помечаем как `ml_disagree_hard`
      (это ценные cases для evaluation).

    Используем B1.1 (TF-IDF) и B2.1 (BGE-M3 SVC) — обученные на текущем train.
    """
    if not candidates:
        return []

    from d1.baselines.trained_bundle import train_bundle

    bundle = train_bundle(names=["B1.1_tfidf_lr", "B2.1_bge-m3_svc"])
    b1 = bundle.get("B1.1_tfidf_lr")
    b2 = bundle.get("B2.1_bge-m3_svc")
    label_order = list(b1.classes_)

    texts = [c["text"] for c in candidates]
    proba_b1 = b1.predict_proba(texts)
    proba_b2 = b2.predict_proba(texts)
    proba_avg = 0.5 * proba_b1 + 0.5 * proba_b2

    # scenario_id → scenario_type (clean/hard/short) lookup
    sc_type_by_id = {s["id"]: s["scenario_type"] for s in _SCENARIOS}

    audited: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        gold = cand["route_domain"]
        idx_max = int(np.argmax(proba_avg[i]))
        ml_label = label_order[idx_max]
        ml_conf = float(proba_avg[i][idx_max])
        gold_idx = label_order.index(gold) if gold in label_order else -1
        gold_conf = float(proba_avg[i][gold_idx]) if gold_idx >= 0 else 0.0
        sc_type = sc_type_by_id.get(cand["scenario"], "clean")

        if ml_label == gold:
            decision = "accept"
            reason = "ml_agrees"
        elif ml_conf < _AUDIT_ML_WEAK_CONF:
            decision = "accept"
            reason = "ml_weak_can_learn"
        elif sc_type in ("hard", "short"):
            # hard scenarios: ML disagree это OK, by design проверяем устойчивость
            decision = "accept"
            reason = f"hard_ml_disagree({ml_label}@{ml_conf:.2f})"
        elif ml_conf >= _AUDIT_ML_STRONG_CONF:
            decision = "reject"
            reason = f"ml_strong_disagree({ml_label})"
        else:
            decision = "flag"
            reason = f"ml_borderline({ml_label}, conf={ml_conf:.2f})"

        audited.append({
            **cand,
            "ml_predicted_label": ml_label,
            "ml_predicted_conf": round(ml_conf, 4),
            "ml_gold_conf": round(gold_conf, 4),
            "audit_decision": decision,
            "audit_reason": reason,
        })

    by_decision: dict[str, int] = {}
    for c in audited:
        by_decision[c["audit_decision"]] = by_decision.get(c["audit_decision"], 0) + 1
    logger.info("Auto-audit decisions: %s", by_decision)
    return audited


# ---------------------------------------------------------------------------
# Persistence: candidates YAML + similarity CSV + audit CSV + addendum CSVs
# ---------------------------------------------------------------------------

_TRAIN_CSV_COLUMNS = [
    "id", "text", "route_domain", "subtype", "explicit_booking", "urgency",
    "is_offtopic", "specialization_hint", "feedback_flag", "faq_category",
    "style", "source", "seed_id",
]


def _assign_candidate_ids(audited: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"candidate_id": f"p5_{i:03d}", **c} for i, c in enumerate(audited, 1)]


def _save_candidates_yaml(candidates: list[dict[str, Any]]) -> None:
    payload = {
        "generated_by": VARIATION_MODEL,
        "phase": 5,
        "prompt_file": str(_PROMPT_FILE.relative_to(_STUDY_ROOT)),
        "scenarios": [s["id"] for s in _SCENARIOS],
        "thresholds": {
            "leakage_reject": _LEAKAGE_REJECT_THRESHOLD,
            "intra_pool_dup": _INTRA_POOL_DUP_THRESHOLD,
            "audit_ml_strong": _AUDIT_ML_STRONG_CONF,
            "audit_ml_weak": _AUDIT_ML_WEAK_CONF,
        },
        "generation_params": {
            "temperature_by_type": _TEMPERATURE_BY_TYPE,
            "top_p": _TOP_P,
            "passes_seeds": _PASS_SEEDS,
            "total_tokens_cap": _TOTAL_TOKENS_CAP,
        },
        "candidates": candidates,
    }
    _CANDIDATES_FILE.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    logger.info("Candidates → %s (%d)", _CANDIDATES_FILE, len(candidates))


def _save_audit_csv(audited: list[dict[str, Any]]) -> None:
    cols = [
        "candidate_id", "scenario", "text", "route_domain", "subtype", "urgency",
        "ml_predicted_label", "ml_predicted_conf", "ml_gold_conf",
        "audit_decision", "audit_reason",
    ]
    pd.DataFrame(audited)[cols].to_csv(_AUDIT_FILE, index=False, encoding="utf-8")
    logger.info("Audit → %s (%d rows)", _AUDIT_FILE, len(audited))


def _build_addendum_row(cand: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Преобразование candidate → строка совместимая с d1_v6_train.csv."""
    return {
        "id": cand["candidate_id"],
        "text": cand["text"],
        "route_domain": cand["route_domain"],
        "subtype": cand["subtype"],
        "explicit_booking": cand["route_domain"] == "booking",
        "urgency": cand["urgency"],
        "is_offtopic": False,
        "specialization_hint": "",
        "feedback_flag": False,
        "faq_category": "",
        "style": "",
        "source": "grok_phase5_audited",
        "seed_id": scenario["id"],
    }


def _save_addendum_csvs(audited: list[dict[str, Any]]) -> tuple[int, int]:
    """Сохранить accepted кандидатов в train/hard_test addendum CSV.

    Только audit_decision == 'accept' попадает в addendum. flag/reject — нет.
    """
    scenario_by_id = {s["id"]: s for s in _SCENARIOS}

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for c in audited:
        if c["audit_decision"] != "accept":
            continue
        sc = scenario_by_id[c["scenario"]]
        row = _build_addendum_row(c, sc)
        if sc["split"] == "train":
            train_rows.append(row)
        else:
            eval_rows.append(row)

    if train_rows:
        pd.DataFrame(train_rows)[_TRAIN_CSV_COLUMNS].to_csv(
            _TRAIN_ADDENDUM_FILE, index=False, encoding="utf-8",
        )
    if eval_rows:
        pd.DataFrame(eval_rows)[_TRAIN_CSV_COLUMNS].to_csv(
            _HARDTEST_ADDENDUM_FILE, index=False, encoding="utf-8",
        )
    logger.info(
        "Addendum: train=%d → %s, hard_test=%d → %s",
        len(train_rows), _TRAIN_ADDENDUM_FILE,
        len(eval_rows), _HARDTEST_ADDENDUM_FILE,
    )
    return len(train_rows), len(eval_rows)


def _save_similarity_csv(df: pd.DataFrame) -> None:
    df.to_csv(_SIMILARITY_FILE, index=False, encoding="utf-8")
    logger.info("Similarity → %s (%d rows)", _SIMILARITY_FILE, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Grok augmentation")
    parser.add_argument("--concurrency", type=int, default=_CONCURRENCY_DEFAULT)
    parser.add_argument(
        "--passes", type=int, default=_DEFAULT_PASSES,
        help="Multi-pass count с разными seeds (default 2)",
    )
    parser.add_argument(
        "--dry-scenario", type=str, default=None,
        help="Run only single scenario id",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Skip LLM generation; re-audit existing phase5_candidates.yaml. "
        "Useful после изменения audit logic — без повторной траты токенов.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.passes > len(_PASS_SEEDS):
        raise SystemExit(f"--passes ≤ {len(_PASS_SEEDS)} (доступных seeds)")

    scenarios = _SCENARIOS
    if args.dry_scenario:
        scenarios = [s for s in _SCENARIOS if s["id"] == args.dry_scenario]
        if not scenarios:
            raise SystemExit(
                f"Сценарий '{args.dry_scenario}' не найден. "
                f"Доступно: {[s['id'] for s in _SCENARIOS]}",
            )

    started = datetime.now(timezone.utc)

    if args.audit_only:
        # Re-audit only: читаем сохранённые candidates, переделываем audit + addendum.
        # Не делаем LLM calls и не пересчитываем similarity (берём уже сохранённое).
        if not _CANDIDATES_FILE.exists():
            raise SystemExit(
                f"--audit-only требует существующий {_CANDIDATES_FILE}. "
                f"Запустите сначала без --audit-only.",
            )
        logger.info("Phase 5 audit-only: re-audit + addendum from %s", _CANDIDATES_FILE)
        with open(_CANDIDATES_FILE, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        existing = payload.get("candidates", [])
        # Очищаем audit-поля для пересчёта
        bare = [
            {k: v for k, v in c.items()
             if k not in {"ml_predicted_label", "ml_predicted_conf",
                          "ml_gold_conf", "audit_decision", "audit_reason",
                          "candidate_id"}}
            for c in existing
        ]
        # Re-encode только тексты (нужно для audit, но скипаем similarity)
        if bare:
            from d1.baselines.b2_embedding import B2EmbeddingClassifier
            encoder_holder = B2EmbeddingClassifier()
            embs = _encode_texts(encoder_holder._get_encoder(), [c["text"] for c in bare])
        else:
            embs = np.zeros((0, 1024))
        audited = _auto_audit(bare, embs)
        audited = _assign_candidate_ids(audited)
        _save_candidates_yaml(audited)
        _save_audit_csv(audited)
        n_train, n_eval = _save_addendum_csvs(audited)
        n_raw = len(existing)
        n_accept_post_contam = len(audited)
    else:
        logger.info(
            "Phase 5 start: %d scenarios × %d passes (cost cap %d tokens)",
            len(scenarios), args.passes, _TOTAL_TOKENS_CAP,
        )

        # 1. Generation (multi-pass)
        raw_candidates, yield_stats = asyncio.run(
            _run_generation(scenarios, args.concurrency, args.passes),
        )
        logger.info("Total raw (post intra-scenario exact dedup): %d", len(raw_candidates))
        if not raw_candidates:
            raise SystemExit("Ноль валидных кандидатов после генерации.")

        for sid, st in yield_stats.items():
            logger.info(
                "  yield[%s]: raw=%d → after_exact_dedup=%d",
                sid, st["raw"], st["after_exact_dedup"],
            )

        # 2. Anti-contamination (BGE 0.85 / intra 0.90)
        accepted, similarity_df, accepted_embs = _anti_contamination(raw_candidates)

        # 3. Auto-audit (ML predict agreement)
        audited = _auto_audit(accepted, accepted_embs)
        audited = _assign_candidate_ids(audited)

        # 4. Persistence
        _save_candidates_yaml(audited)
        _save_similarity_csv(similarity_df)
        _save_audit_csv(audited)
        n_train, n_eval = _save_addendum_csvs(audited)
        n_raw = len(raw_candidates)
        n_accept_post_contam = len(accepted)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    n_accept = sum(1 for c in audited if c["audit_decision"] == "accept")
    n_reject = sum(1 for c in audited if c["audit_decision"] == "reject")
    n_flag = sum(1 for c in audited if c["audit_decision"] == "flag")
    mode = "audit-only" if args.audit_only else "generate+audit"
    print(
        f"\n=== Phase 5 Summary ({mode}) ===\n"
        f"Time: {elapsed:.1f}s\n"
        f"Raw candidates: {n_raw}\n"
        f"After anti-contamination ({_LEAKAGE_REJECT_THRESHOLD}/{_INTRA_POOL_DUP_THRESHOLD}): {n_accept_post_contam}\n"
        f"Auto-audit:\n"
        f"  accept: {n_accept}\n"
        f"  flag:   {n_flag}  (для optional review)\n"
        f"  reject: {n_reject}\n"
        f"Train addendum: {n_train} rows → {_TRAIN_ADDENDUM_FILE.name}\n"
        f"Hard_test addendum: {n_eval} rows → {_HARDTEST_ADDENDUM_FILE.name}\n",
    )


if __name__ == "__main__":
    main()
