"""Построение gold_map для retrieval-метрик.

Gold map определяет эталонные chunk_ids для каждого eval sample.
Используется для вычисления Hit@k, Recall@k, MRR в retrieval_metrics.py.

Стратегии автозаполнения по категориям:
- factoid: SUBTYPE_TO_CHUNK_IDS (детерминированный SSoT маппинг)
- complaint_routing: expected_specialization → doctor chunks
- doctor_lookup: стемминг фамилии / keyword → specialization → doctors
- pricing: _PRICE_KEYWORDS (keyword → price chunk_ids)
- out_of_scope: [] (корректно по определению)
- reasoning: из EvalSample.gold_chunk_ids (ручная разметка)
"""

from __future__ import annotations

import re
from typing import Any

from d4.models import EvalSample


# ---------------------------------------------------------------------------
# SSoT: factoid subtype → эталонные chunk_ids
# Источник маппинга: taxonomy.yaml:expected_fields → chunker.py:CLINIC_SECTION_MAP
# ---------------------------------------------------------------------------

SUBTYPE_TO_CHUNK_IDS: dict[str, list[str]] = {
    "address_location": ["clinic_location"],
    "working_hours": ["clinic_working_hours"],
    "phone_contact": ["clinic_contacts"],
    "sedation_options": ["clinic_patient_comfort_and_safety"],
    "financing": ["clinic_financing"],
    "children_accepted": ["clinic_children_info"],
    "licenses_documents": ["clinic_licenses", "clinic_documents"],
}

# Ключевые слова в запросе → формальная специализация
# (для doctor_lookup.by_specialization_list)
_SPEC_KEYWORDS: dict[str, str] = {
    "ортодонт": "Ортодонтия",
    "хирург": "Хирургическая стоматология",
    "терапевт": "Терапевтическая стоматология",
    "ортопед": "Ортопедическая стоматология",
    "детск": "Детская стоматология",
}

_SURNAME_STEM_LEN = 4

# Reasoning seed-запросы → специализации, чьи doctor chunks нужны для ответа.
# Для LLM-вариаций используется keyword-fallback через _SPEC_KEYWORDS.
# Ценовые ключевые слова → price chunk_ids (SSoT).
# Определяется экспертом 1 раз при добавлении прайса.
# Для split-категорий (>max_chunk_tokens) перечисляются все sub-chunks.
_PRICE_KEYWORDS: dict[str, list[str]] = {
    # Имплантация
    "имплант": ["price_implantation"],
    # Ортодонтия
    "брекет": ["price_orthodontics_p1", "price_orthodontics_p2"],
    "ортодонт": ["price_orthodontics_p1", "price_orthodontics_p2"],
    "прикус": ["price_orthodontics_p1", "price_orthodontics_p2"],
    "выравнив": ["price_orthodontics_p1", "price_orthodontics_p2"],
    # Протезирование
    "протез": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "винир": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "коронк": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "металлокерамик": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "циркон": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "керамик": ["price_prosthetics_p1", "price_prosthetics_p2"],
    "мост": ["price_prosthetics_p1", "price_prosthetics_p2"],
    # Хирургия
    "удален": ["price_adult_surgery_removals"],
    "зуб мудрост": ["price_adult_surgery_removals"],
    # Терапия
    "гигиен": ["price_adult_therapy"],
    "кариес": ["price_adult_therapy"],
    "пульпит": ["price_adult_therapy"],
    "лечен": ["price_adult_therapy"],
    "пломб": ["price_adult_therapy"],
    "отбелив": ["price_adult_therapy"],
    "чистк": ["price_adult_therapy"],
    # Консультации
    "консультац": ["price_primary_appointment_consultations"],
    "осмотр": ["price_primary_appointment_consultations"],
    "прием": ["price_primary_appointment_consultations"],
    # Диагностика
    "рентген": ["price_diagnostics_2d_3d_viziograf"],
    "снимок": ["price_diagnostics_2d_3d_viziograf"],
    "кт": ["price_diagnostics_2d_3d_viziograf"],
    # Детская стоматология
    "детск": ["price_children_therapy"],
    "ребенк": ["price_children_therapy"],
    "ребёнк": ["price_children_therapy"],
    # Седация
    "седаци": ["price_sedation_removal_in_sleep_children"],
    "наркоз": ["price_sedation_removal_in_sleep_children"],
}


_REASONING_GOLD_SPECS: dict[str, list[str]] = {
    "Кто из хирургов самый опытный?": ["Хирургическая стоматология"],
    "Какой хирург опытнее?": ["Хирургическая стоматология"],
    "Есть ли детский стоматолог с опытом >10 лет?": ["Детская стоматология"],
    "Если болит при жевании, к кому идти?": ["Терапевтическая стоматология"],
    "К какому врачу если ребёнку нужны брекеты?": ["Ортодонтия"],
    "Чем терапевт отличается от ортопеда?": [
        "Терапевтическая стоматология",
        "Ортопедическая стоматология",
    ],
}


# ---------------------------------------------------------------------------
# Резолверы: категория → chunk_ids
# ---------------------------------------------------------------------------


def _resolve_doctors_by_spec(
    specialization: str,
    doctors: list[dict[str, Any]],
) -> list[str]:
    """Все doctor chunk_ids с данной специализацией."""
    target = specialization.strip().lower()
    return [
        f"doctor_{d['id']}"
        for d in doctors
        if d.get("specialization", "").strip().lower() == target
    ]


def _resolve_doctor_by_surname(
    query: str,
    doctors: list[dict[str, Any]],
) -> list[str]:
    """Поиск врача по фамилии в тексте запроса.

    Стемминг первых N символов фамилии — устойчив к русским падежам.
    Пример: «Ермакова» → stem «Ерма» → match doctor_1 (Ермаков).
    """
    query_words = [w.lower() for w in re.findall(r"[А-ЯЁа-яё]+", query)]
    matches: list[str] = []

    for doc in doctors:
        surname = doc.get("full_name", "").split()[0]
        if len(surname) < _SURNAME_STEM_LEN:
            continue
        stem = surname[:_SURNAME_STEM_LEN].lower()
        if any(w.startswith(stem) and len(w) >= _SURNAME_STEM_LEN for w in query_words):
            matches.append(f"doctor_{doc['id']}")

    return matches


def _resolve_spec_from_query(
    query: str,
    doctors: list[dict[str, Any]],
) -> list[str]:
    """Определение специализации из ключевых слов запроса.

    Для doctor_lookup.by_specialization_list: «ортодонт» → Ортодонтия → все врачи.
    """
    query_lower = query.lower()
    for keyword, spec in _SPEC_KEYWORDS.items():
        if keyword in query_lower:
            return _resolve_doctors_by_spec(spec, doctors)
    return []


def _resolve_reasoning_gold(
    sample: EvalSample,
    doctors: list[dict[str, Any]],
) -> list[str]:
    """Gold chunks для reasoning: статический маппинг + keyword-fallback.

    Приоритет:
    1. sample.gold_chunk_ids (ручная разметка, если заполнена)
    2. _REASONING_GOLD_SPECS (точное совпадение seed-запроса)
    3. _SPEC_KEYWORDS (keyword-fallback для LLM-вариаций)
    """
    if sample.gold_chunk_ids:
        return list(sample.gold_chunk_ids)

    specs = _REASONING_GOLD_SPECS.get(sample.query)
    if specs:
        result: list[str] = []
        for spec in specs:
            result.extend(_resolve_doctors_by_spec(spec, doctors))
        return result

    # Keyword-fallback для LLM-вариаций reasoning запросов
    query_lower = sample.query.lower()
    result = []
    for keyword, spec in _SPEC_KEYWORDS.items():
        if keyword in query_lower:
            result.extend(_resolve_doctors_by_spec(spec, doctors))
    return result


def _resolve_pricing_gold(query: str) -> list[str]:
    """Gold chunks для pricing: keyword match → price chunk_ids."""
    query_lower = query.lower()
    chunk_ids: list[str] = []
    for keyword, ids in _PRICE_KEYWORDS.items():
        if keyword in query_lower:
            for cid in ids:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
    return chunk_ids


# ---------------------------------------------------------------------------
# Главная утилита
# ---------------------------------------------------------------------------


def build_gold_map(
    samples: list[EvalSample],
    doctors: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Построение gold_map: {sample_id: [gold_chunk_ids]}.

    Args:
        samples: eval set (из eval_set_raw.yaml или eval_set.yaml)
        doctors: список врачей из doctors.yaml

    Returns:
        {sample_id: [gold_chunk_ids]}
    """
    gold_map: dict[str, list[str]] = {}

    for sample in samples:
        chunk_ids: list[str] = []

        if sample.category == "factoid":
            chunk_ids = list(SUBTYPE_TO_CHUNK_IDS.get(sample.subtype, []))

        elif sample.category == "complaint_routing":
            if sample.expected_specialization:
                chunk_ids = _resolve_doctors_by_spec(
                    sample.expected_specialization, doctors,
                )

        elif sample.category == "doctor_lookup":
            if sample.subtype == "by_specialization_list":
                chunk_ids = _resolve_spec_from_query(sample.query, doctors)
            else:
                chunk_ids = _resolve_doctor_by_surname(sample.query, doctors)

        elif sample.category == "pricing":
            chunk_ids = _resolve_pricing_gold(sample.query)

        elif sample.category == "out_of_scope":
            chunk_ids = []

        elif sample.category == "reasoning":
            chunk_ids = _resolve_reasoning_gold(sample, doctors)

        gold_map[sample.sample_id] = chunk_ids

    return gold_map


# ---------------------------------------------------------------------------
# Диагностика покрытия
# ---------------------------------------------------------------------------


def print_gold_map_report(
    gold_map: dict[str, list[str]],
    samples: list[EvalSample],
    valid_chunk_ids: set[str] | None = None,
) -> None:
    """Отчёт о покрытии gold_map: сколько заполнено, есть ли ошибки.

    Args:
        gold_map: построенный gold_map
        samples: eval set (для категорий и answerable)
        valid_chunk_ids: множество chunk_ids из chunks.json (для валидации)
    """
    sample_map = {s.sample_id: s for s in samples}
    by_category: dict[str, list[str]] = {}
    empty_answerable: list[str] = []
    invalid_chunks: list[tuple[str, str]] = []

    for sid, chunks in gold_map.items():
        sample = sample_map.get(sid)
        if not sample:
            continue
        by_category.setdefault(sample.category, []).append(sid)

        if sample.answerable and not chunks and sample.category != "reasoning":
            empty_answerable.append(sid)

        if valid_chunk_ids:
            for cid in chunks:
                if cid not in valid_chunk_ids:
                    invalid_chunks.append((sid, cid))

    total = len(gold_map)
    filled = sum(1 for v in gold_map.values() if v)
    print(f"Gold map: {filled}/{total} сэмплов с непустыми gold_chunk_ids")

    for cat in sorted(by_category):
        sids = by_category[cat]
        cat_filled = sum(1 for sid in sids if gold_map.get(sid))
        print(f"  {cat}: {cat_filled}/{len(sids)}")

    if empty_answerable:
        print(f"\n⚠ Answerable сэмплы БЕЗ gold chunks: {empty_answerable}")

    if invalid_chunks:
        print(f"\n⚠ Невалидные chunk_ids:")
        for sid, cid in invalid_chunks:
            print(f"  {sid} → {cid}")

    if not empty_answerable and not invalid_chunks:
        print("\n✓ Все проверки пройдены")
