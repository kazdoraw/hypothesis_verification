"""Генерация eval set запросов по таксономии.

Комбинированный подход: LLM-генерация + шаблон для ручного дополнения.
- generate_sample_templates() — seed-шаблоны из examples таксономии
- generate_llm_variations() — LLM-генерация вариаций (casual/typo/indirect)
- deduplicate_queries() — дедупликация по cosine similarity < 0.85
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import yaml

from d4.config import GenerationConfig
from d4.models import Difficulty, EvalSample, KBChunk

logger = logging.getLogger(__name__)

# Путь к prompt для генерации вариаций
_VARIATION_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "query_variation.md"

_SURNAME_STEM_LEN = 4

# ---------------------------------------------------------------------------
# Резолвер врача по фамилии (перенесён из gold_map.py, Фаза 3.1)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Загрузка таксономии
# ---------------------------------------------------------------------------


def load_taxonomy(taxonomy_path: str | Path) -> dict[str, Any]:
    """Загрузка таксономии из YAML."""
    with open(taxonomy_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


# ---------------------------------------------------------------------------
# Подсчёт целевого количества запросов
# ---------------------------------------------------------------------------


def count_target_queries(taxonomy: dict[str, Any]) -> dict[str, int]:
    """Подсчёт целевого количества запросов по категориям из таксономии.

    Returns:
        словарь {category.subtype: n_queries}
    """
    categories = taxonomy.get("taxonomy", {})
    targets: dict[str, int] = {}
    total = 0

    for cat_name, cat_data in categories.items():
        subtypes = cat_data.get("subtypes", {})
        for sub_name, sub_data in subtypes.items():
            n = sub_data.get("n_queries", 5)
            key = f"{cat_name}.{sub_name}"
            targets[key] = n
            total += n

    return targets


# ---------------------------------------------------------------------------
# Генерация шаблонов (без LLM — структура для дальнейшего заполнения)
# ---------------------------------------------------------------------------


def generate_sample_templates(
    taxonomy: dict[str, Any],
    seed: int = 42,
) -> list[EvalSample]:
    """Генерация шаблонов EvalSample из таксономии.

    Создаёт по одному шаблону на каждый example из таксономии.
    Остальные генерируются через LLM (в notebook 01).
    """
    random.seed(seed)
    categories = taxonomy.get("taxonomy", {})
    samples: list[EvalSample] = []
    counter = 0

    for cat_name, cat_data in categories.items():
        answerable = cat_data.get("answerable", True)
        subtypes = cat_data.get("subtypes", {})

        for sub_name, sub_data in subtypes.items():
            examples = sub_data.get("examples", [])
            expected_spec = sub_data.get("expected_specialization")

            for example in examples:
                counter += 1
                sample = EvalSample(
                    sample_id=f"seed_{counter:04d}",
                    query=example,
                    category=cat_name,
                    subtype=sub_name,
                    answerable=answerable,
                    expected_answer="",
                    expected_specialization=expected_spec,
                    difficulty=Difficulty.EASY if answerable else Difficulty.MEDIUM,
                    notes="seed example",
                )
                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Сериализация / десериализация eval set
# ---------------------------------------------------------------------------


def save_eval_set(samples: list[EvalSample], output_path: str | Path) -> None:
    """Сохранение eval set в YAML."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [sample.model_dump(mode="json") for sample in samples]
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_eval_set(eval_set_path: str | Path) -> list[EvalSample]:
    """Загрузка eval set из YAML (backward-compatible).

    Автоматически мигрирует:
    - gold_chunk_ids: list[str] → list[list[str]]
    - gold_facts: list[str] → list[GoldFact] (fact_type="text")
    """
    from d4.models import GoldFact

    with open(eval_set_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    samples = []
    for item in data:
        raw_gold = item.get("gold_chunk_ids", [])
        if raw_gold and isinstance(raw_gold[0], str):
            item["gold_chunk_ids"] = [raw_gold]

        raw_facts = item.get("gold_facts", [])
        if raw_facts and isinstance(raw_facts[0], str):
            item["gold_facts"] = [
                {"fact_type": "text", "canonical_value": f}
                for f in raw_facts
            ]

        samples.append(EvalSample(**item))
    return samples


# ---------------------------------------------------------------------------
# Дедупликация (cosine similarity)
# ---------------------------------------------------------------------------


def deduplicate_queries(
    samples: list[EvalSample],
    model_name: str | None = None,
    threshold: float | None = None,
    *,
    config: GenerationConfig | None = None,
) -> list[EvalSample]:
    """Удаление дубликатов по cosine similarity между запросами.

    Args:
        samples: список запросов для дедупликации
        model_name: имя embedding модели (приоритет над config)
        threshold: порог cosine similarity (приоритет над config)
        config: GenerationConfig — versioned параметры из generation_config.yaml

    Приоритет: явные аргументы > config > hardcoded defaults.
    Требует sentence-transformers. Если не установлен — возвращает без изменений.
    """
    effective_model = model_name
    effective_threshold = threshold

    if config is not None:
        if effective_model is None:
            effective_model = config.dedup.model
        if effective_threshold is None:
            effective_threshold = config.dedup.threshold

    if effective_model is None:
        raise ValueError(
            "model_name обязателен: передайте явно или через config (generation_config.yaml)"
        )
    if effective_threshold is None:
        effective_threshold = 0.85

    if len(samples) <= 1:
        return samples

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("WARN: sentence-transformers не установлен, дедупликация пропущена")
        return samples

    model = SentenceTransformer(effective_model)
    queries = [s.query for s in samples]
    embeddings = model.encode(queries, normalize_embeddings=True)

    sim_matrix = np.dot(embeddings, embeddings.T)

    keep_indices: list[int] = []
    for i in range(len(samples)):
        is_duplicate = False
        for j in keep_indices:
            if sim_matrix[i][j] >= effective_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    removed = len(samples) - len(keep_indices)
    if removed > 0:
        print(f"Дедупликация: удалено {removed} дубликатов "
              f"(model={effective_model}, threshold={effective_threshold})")

    return [samples[i] for i in keep_indices]


# ---------------------------------------------------------------------------
# Генерация уникальных ID
# ---------------------------------------------------------------------------


def assign_unique_ids(samples: list[EvalSample]) -> list[EvalSample]:
    """Присвоение уникальных sample_id."""
    for i, sample in enumerate(samples):
        sample.sample_id = f"q_{i + 1:04d}"
    return samples


def print_eval_set_summary(samples: list[EvalSample]) -> None:
    """Вывод сводки по eval set."""
    from collections import Counter

    cats = Counter(s.category for s in samples)
    answerable = sum(1 for s in samples if s.answerable)

    print(f"Eval set: {len(samples)} запросов")
    print(f"  answerable: {answerable}, unanswerable: {len(samples) - answerable}")
    print("  по категориям:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")


# ---------------------------------------------------------------------------
# LLM-генерация вариаций запросов
# ---------------------------------------------------------------------------


def _load_variation_prompt() -> str:
    """Загрузка system prompt для генерации вариаций."""
    return _VARIATION_PROMPT_PATH.read_text(encoding="utf-8")


def _build_variation_user_message(sample: EvalSample) -> str:
    """Формирование user message для LLM: seed-запрос + метаданные."""
    return (
        f"Оригинальный запрос: \"{sample.query}\"\n"
        f"Категория: {sample.category}\n"
        f"Подтип: {sample.subtype}\n"
        f"Отвечаемый: {'да' if sample.answerable else 'нет'}"
    )


_ALLOWED_QUERY_RE = re.compile(
    r"^[\u0400-\u04FF\s\d\.\,\!\?\-\:\;\(\)\"\'«»\…\—\–₽%\+\@\#\&\*\/]+$",
)


def _parse_variations(raw: str) -> list[dict[str, str]]:
    """Парсинг JSON-ответа LLM с вариациями.

    Обрабатывает как голый массив [...], так и обёртку {"key": [...]}.
    Фильтрует вариации с не-русскими символами (артефакты LLM).

    Returns:
        список {query, style} или пустой список при ошибке парсинга.
    """
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)
    try:
        data = json.loads(content)
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
                    if not _ALLOWED_QUERY_RE.match(query):
                        logger.warning("Отброшена вариация с не-русскими символами: %s", query[:80])
                        continue
                    results.append({"query": query, "style": item.get("style", "unknown")})
            return results
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Не удалось распарсить вариации: %s", raw[:200])
    return []


def _generate_variations_for_sample(
    client: OpenAI,
    model: str,
    system_prompt: str,
    sample: EvalSample,
    temperature: float,
    max_tokens: int,
) -> list[EvalSample]:
    """Генерация вариаций для одного seed-запроса через LLM.

    Возвращает список новых EvalSample, наследующих метаданные от seed.
    """
    user_message = _build_variation_user_message(sample)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
        variations = _parse_variations(raw)
    except Exception as exc:
        logger.warning("LLM ошибка для %s: %s", sample.sample_id, exc)
        return []

    family_id = sample.seed_family_id or sample.sample_id

    results: list[EvalSample] = []
    for var in variations:
        new_sample = EvalSample(
            sample_id="",  # будет назначен позже через assign_unique_ids()
            query=var["query"],
            category=sample.category,
            subtype=sample.subtype,
            answerable=sample.answerable,
            expected_answer="",  # заполняется в Фазе 1.2
            expected_doctor=sample.expected_doctor,
            expected_specialization=sample.expected_specialization,
            expected_branch=sample.expected_branch,
            expected_service=sample.expected_service,
            difficulty=Difficulty.MEDIUM,
            notes=f"llm_variation of {sample.sample_id}; style: {var['style']}",
            seed_family_id=family_id,
        )
        results.append(new_sample)
    return results


def generate_llm_variations(
    seed_samples: list[EvalSample],
    api_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_workers: int = 8,
    timeout_sec: float = 90.0,
) -> list[EvalSample]:
    """Параллельная LLM-генерация вариаций для всех seed-запросов.

    Для каждого seed-запроса генерирует 3 вариации (casual, typo, indirect).

    Args:
        seed_samples: исходные seed-запросы из таксономии
        api_url: URL OpenRouter API
        api_key: API ключ
        model: модель LLM для генерации
        temperature: температура (0.7 для разнообразия)
        max_tokens: лимит токенов на ответ
        max_workers: параллельных потоков
        timeout_sec: таймаут на один запрос

    Returns:
        список новых EvalSample (без ID — назначаются позже)
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=httpx.Timeout(timeout_sec, connect=10.0),
    )
    system_prompt = _load_variation_prompt()
    all_variations: list[EvalSample] = []
    total = len(seed_samples)
    done = 0
    start = time.perf_counter()

    print(f"  генерация вариаций: {total} seed-запросов, {max_workers} потоков")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_variations_for_sample,
                client, model, system_prompt, sample, temperature, max_tokens,
            ): sample
            for sample in seed_samples
        }
        for future in as_completed(futures):
            try:
                variations = future.result()
                all_variations.extend(variations)
            except Exception as exc:
                sample = futures[future]
                logger.warning("Ошибка генерации для %s: %s", sample.sample_id, exc)
            done += 1
            if done % 10 == 0:
                elapsed = time.perf_counter() - start
                print(f"  вариации: {done}/{total} ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - start
    print(f"  вариации готово: {len(all_variations)} из {total} seeds за {elapsed:.1f}s")
    return all_variations


# ---------------------------------------------------------------------------
# Заполнение метаданных: expected_doctor (Фаза 1.3)
# ---------------------------------------------------------------------------


def populate_annotations(
    samples: list[EvalSample],
    doctors: list[dict[str, Any]],
) -> list[EvalSample]:
    """Заполнение expected_doctor и expected_answer (только out_of_scope).

    expected_answer для answerable категорий генерируется отдельно
    через generate_reference_answers() (reference LLM), что устраняет
    циркулярность: эталоны не захардкожены автором эксперимента.

    Стратегия:
    - doctor_info/by_surname: определяем expected_doctor детерминированно из KB
    - out_of_scope: фиксированный текст отказа (не зависит от KB фактов)
    - остальные категории: expected_answer остаётся пустым → заполняется LLM

    Args:
        samples: список EvalSample (in-place модификация)
        doctors: список врачей из doctors.yaml

    Returns:
        тот же список samples с заполненными метаданными
    """
    doc_by_id = {f"doctor_{d['id']}": d for d in doctors}

    for sample in samples:
        if sample.category == "doctor_info" and sample.subtype == "by_surname":
            chunk_ids = _resolve_doctor_by_surname(sample.query, doctors)
            if chunk_ids and chunk_ids[0] in doc_by_id:
                doc = doc_by_id[chunk_ids[0]]
                sample.expected_doctor = doc["full_name"]

        elif sample.category == "out_of_scope":
            sample.expected_answer = (
                "К сожалению, я не могу ответить на этот вопрос. "
                "Пожалуйста, позвоните: +7 (842) 231-45-55."
            )

    filled = sum(1 for s in samples if s.expected_answer)
    doctors_filled = sum(1 for s in samples if s.expected_doctor)
    print(f"  метаданные: expected_answer (out_of_scope) у {filled}/{len(samples)}")
    print(f"  метаданные: expected_doctor у {doctors_filled}/{len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Генерация expected_answer через reference LLM (устранение циркулярности)
# ---------------------------------------------------------------------------

_REFERENCE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reference_answer.md"


def _load_reference_prompt() -> str:
    """Загрузка system prompt для генерации эталонных ответов."""
    return _REFERENCE_PROMPT_PATH.read_text(encoding="utf-8")


def _build_reference_user_message(
    sample: EvalSample,
    kb_text: str,
    chunks: list[KBChunk] | None = None,
) -> str:
    """Формирование user message для reference LLM.

    Если chunks переданы — собирает контекст ТОЛЬКО из gold chunks
    (устранение Bias 3: reference answers != S1 full context).
    Иначе — используется kb_text целиком (legacy-режим).
    """
    if chunks and sample.gold_chunk_ids:
        chunk_map = {c.id: c for c in chunks}
        gold_ids: set[str] = set()
        for alt in sample.gold_chunk_ids:
            gold_ids.update(alt)
        gold_chunks = [chunk_map[cid] for cid in gold_ids if cid in chunk_map]
        context = "\n\n".join(
            f"[{c.id}] {c.title}\n{c.content}" for c in gold_chunks
        )
    else:
        context = kb_text

    return (
        f"## База знаний (релевантные фрагменты)\n\n{context}\n\n"
        f"---\n\n"
        f"## Запрос пациента\n\n{sample.query}\n\n"
        f"## Метаданные\n"
        f"Категория: {sample.category}\n"
        f"Подтип: {sample.subtype}"
    )


def _parse_reference_response(raw: str) -> dict[str, Any]:
    """Парсинг JSON-ответа reference LLM."""
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)
    try:
        data = json.loads(content)
        return data
    except (json.JSONDecodeError, ValueError):
        logger.warning("Не удалось распарсить reference answer: %s", raw[:200])
        return {"answer": "", "doctor": None}


def _generate_reference_for_sample(
    client: OpenAI,
    model: str,
    system_prompt: str,
    sample: EvalSample,
    kb_text: str,
    temperature: float,
    max_tokens: int,
    chunks: list[KBChunk] | None = None,
) -> str:
    """Генерация эталонного ответа для одного сэмпла через reference LLM.

    Args:
        chunks: если переданы, reference answer строится по gold chunks,
                а не по полной KB (устранение Bias 3)

    Returns:
        текст эталонного ответа (строка)
    """
    user_message = _build_reference_user_message(sample, kb_text, chunks)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = _parse_reference_response(raw)
        return data.get("answer", "")
    except Exception as exc:
        logger.warning(
            "Reference LLM ошибка для %s: %s", sample.sample_id, exc,
        )
        return ""


def generate_reference_answers(
    samples: list[EvalSample],
    kb_text: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_workers: int = 8,
    timeout_sec: float = 90.0,
    chunks: list[KBChunk] | None = None,
) -> list[EvalSample]:
    """Генерация expected_answer для всех answerable сэмплов через reference LLM.

    Заменяет захардкоженные _FACTOID_ANSWERS / _REASONING_ANSWERS.
    Использует отдельную LLM (judge config) для устранения циркулярности:
    эталонные ответы генерируются независимой моделью, а не автором.

    Сэмплы с answerable=False (out_of_scope) пропускаются —
    их expected_answer задаётся в populate_annotations().

    Args:
        samples: список EvalSample (in-place модификация expected_answer)
        kb_text: полный текст KB (fallback если chunks не передан)
        api_url: URL OpenRouter API
        api_key: API ключ
        model: модель reference LLM (напр. openai/gpt-5.4-mini)
        temperature: температура (0.0 для детерминизма)
        max_tokens: лимит токенов на ответ
        max_workers: параллельных потоков
        timeout_sec: таймаут на один запрос
        chunks: KB chunks для сборки gold-only контекста (устранение Bias 3)

    Returns:
        тот же список samples с заполненными expected_answer
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=httpx.Timeout(timeout_sec, connect=10.0),
    )
    system_prompt = _load_reference_prompt()

    answerable_samples = [s for s in samples if s.answerable]
    total = len(answerable_samples)
    if not total:
        print("  reference answers: нет answerable сэмплов")
        return samples

    done = 0
    filled = 0
    start = time.perf_counter()
    mode = "gold-context" if chunks else "full-KB"
    print(f"  reference answers ({mode}): {total} сэмплов, {max_workers} потоков")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_reference_for_sample,
                client, model, system_prompt, sample,
                kb_text, temperature, max_tokens, chunks,
            ): sample
            for sample in answerable_samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                answer = future.result()
                if answer:
                    sample.expected_answer = answer
                    filled += 1
            except Exception as exc:
                logger.warning(
                    "Ошибка reference answer для %s: %s",
                    sample.sample_id, exc,
                )
            done += 1
            if done % 10 == 0:
                elapsed = time.perf_counter() - start
                print(f"  reference answers: {done}/{total} ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - start
    print(f"  reference answers готово: {filled}/{total} за {elapsed:.1f}s")
    return samples


# ---------------------------------------------------------------------------
# Запись gold_chunk_ids из gold_map обратно в EvalSample (Фаза 1.4)
# ---------------------------------------------------------------------------


def apply_gold_chunk_ids(
    samples: list[EvalSample],
    gold_map: dict[str, list[list[str]]],
) -> list[EvalSample]:
    """Записывает gold_chunk_ids из gold_map в каждый EvalSample.

    Args:
        samples: список EvalSample (in-place модификация)
        gold_map: {sample_id: [[alt1], [alt2], ...]} multi-gold

    Returns:
        тот же список samples с заполненными gold_chunk_ids
    """
    applied = 0
    for sample in samples:
        ids = gold_map.get(sample.sample_id, [])
        if ids:
            sample.gold_chunk_ids = ids
            applied += 1
    print(f"  gold_chunk_ids: заполнено у {applied}/{len(samples)} сэмплов")
    return samples


# ---------------------------------------------------------------------------
# Dev/test split (Bias 7: held-out test set)
# ---------------------------------------------------------------------------


def _infer_seed_family(sample: EvalSample, *, group_field: str = "seed_family_id") -> str:
    """Определяет group ID: явное поле (по group_field) → из notes → sample_id.

    Поддерживает два формата notes:
    - v1: "llm_variation of seed_0003; style: typo" → "seed_0003"
    - v2: "llm_variation of q_0003"                 → "q_0003"
    """
    value = getattr(sample, group_field, "")
    if value:
        return value
    m = re.search(r"llm_variation of ((?:seed_|q_)\w+)", sample.notes)
    if m:
        return m.group(1)
    return sample.sample_id


def split_eval_set(
    samples: list[EvalSample],
    test_ratio: float | None = None,
    seed: int | None = None,
    *,
    config: GenerationConfig | None = None,
) -> tuple[list[EvalSample], list[EvalSample]]:
    """Group-stratified split: seed + вариации = одна группа, баланс по category.

    Все вариации одного seed попадают целиком в dev ИЛИ в test (не разделяются),
    предотвращая data leakage. Баланс по category обеспечивается вручную
    через группировку по (family, category).

    Из config.split используются ВСЕ поля:
    - method: только "group_stratified" (ValueError при другом)
    - group_field: поле EvalSample для определения группы ("seed_family_id")
    - test_ratio, seed: числовые параметры split

    Args:
        samples: полный eval set
        test_ratio: доля test set (приоритет над config, default 0.3)
        seed: random seed (приоритет над config, default 42)
        config: GenerationConfig — versioned параметры из generation_config.yaml

    Returns:
        (dev_samples, test_samples)
    """
    import random
    from collections import Counter, defaultdict

    effective_ratio = test_ratio
    effective_seed = seed
    group_field = "seed_family_id"

    if config is not None:
        if config.split.method != "group_stratified":
            raise ValueError(
                f"split.method='{config.split.method}' не поддерживается. "
                f"Единственный реализованный метод: 'group_stratified'."
            )
        group_field = config.split.group_field
        if effective_ratio is None:
            effective_ratio = config.split.test_ratio
        if effective_seed is None:
            effective_seed = config.split.seed

    if effective_ratio is None:
        effective_ratio = 0.3
    if effective_seed is None:
        effective_seed = 42

    rng = random.Random(effective_seed)

    families: dict[str, list[EvalSample]] = defaultdict(list)
    for s in samples:
        fid = _infer_seed_family(s, group_field=group_field)
        families[fid].append(s)

    family_cat: dict[str, str] = {}
    for fid, members in families.items():
        cats = Counter(m.category for m in members)
        family_cat[fid] = cats.most_common(1)[0][0]

    cat_families: dict[str, list[str]] = defaultdict(list)
    for fid, cat in family_cat.items():
        cat_families[cat].append(fid)

    test_families: set[str] = set()
    for cat, fids in cat_families.items():
        rng.shuffle(fids)
        n_test = max(1, round(len(fids) * effective_ratio))
        if n_test >= len(fids):
            n_test = max(0, len(fids) - 1)
        test_families.update(fids[:n_test])

    dev_samples = [s for s in samples if _infer_seed_family(s, group_field=group_field) not in test_families]
    test_samples = [s for s in samples if _infer_seed_family(s, group_field=group_field) in test_families]

    if not test_samples:
        logger.warning("Eval set слишком мал для group split, всё в dev")
        return list(samples), []

    dev_cats = Counter(s.category for s in dev_samples)
    test_cats = Counter(s.category for s in test_samples)
    n_families = len(families)
    n_test_families = len(test_families)

    print(f"Group-stratified split: dev={len(dev_samples)}, test={len(test_samples)} "
          f"(ratio={len(test_samples)/len(samples):.1%})")
    print(f"  groups: {n_families} families, {n_test_families} in test")
    print(f"  dev categories:  {dict(sorted(dev_cats.items()))}")
    print(f"  test categories: {dict(sorted(test_cats.items()))}")

    return dev_samples, test_samples


# ---------------------------------------------------------------------------
# Валидация intent вариаций (Bias 5)
# ---------------------------------------------------------------------------


def validate_variation_intent(
    seed: EvalSample,
    variation: EvalSample,
    api_url: str,
    api_key: str,
    model: str,
    timeout_sec: float = 30.0,
) -> bool:
    """Проверяет сохранение intent между seed и вариацией через LLM.

    Вариации с изменённым intent отбрасываются, не переразмечиваются.

    Args:
        seed: оригинальный seed-запрос
        variation: LLM-сгенерированная вариация
        api_url: URL OpenRouter API
        api_key: API ключ
        model: lightweight judge model
        timeout_sec: таймаут

    Returns:
        True если intent сохранён, False если изменился
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=httpx.Timeout(timeout_sec, connect=10.0),
    )

    system_msg = (
        "Ты — классификатор интентов для стоматологической клиники. "
        "Тебе дадут два запроса пациента. Определи, одинаковый ли у них intent "
        "(какую информацию хочет получить пациент). "
        "Ответь ТОЛЬКО одним словом: same или shifted."
    )
    user_msg = (
        f"Запрос A: \"{seed.query}\"\n"
        f"Запрос B: \"{variation.query}\"\n"
        f"Категория A: {seed.category}/{seed.subtype}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        return "same" in answer
    except Exception as exc:
        logger.warning(
            "Intent validation ошибка для %s→%s: %s",
            seed.sample_id, variation.query[:40], exc,
        )
        return True  # при ошибке API сохраняем вариацию (conservative)
