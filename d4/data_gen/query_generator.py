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
from openai import OpenAI

from d4.models import Difficulty, EvalSample

logger = logging.getLogger(__name__)

# Путь к prompt для генерации вариаций
_VARIATION_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "query_variation.md"

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
            expected_fields = sub_data.get("expected_fields", [])

            for example in examples:
                counter += 1
                sample = EvalSample(
                    sample_id=f"seed_{counter:04d}",
                    query=example,
                    category=cat_name,
                    subtype=sub_name,
                    answerable=answerable,
                    expected_answer="",  # заполняется экспертом
                    expected_specialization=expected_spec,
                    difficulty=Difficulty.EASY if answerable else Difficulty.MEDIUM,
                    notes=f"seed example; fields: {expected_fields}" if expected_fields else "seed example",
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
    """Загрузка eval set из YAML."""
    with open(eval_set_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [EvalSample(**item) for item in data]


# ---------------------------------------------------------------------------
# Дедупликация (cosine similarity)
# ---------------------------------------------------------------------------


def deduplicate_queries(
    samples: list[EvalSample],
    model_name: str,
    threshold: float = 0.85,
) -> list[EvalSample]:
    """Удаление дубликатов по cosine similarity между запросами.

    Args:
        samples: список запросов для дедупликации
        model_name: имя embedding модели (из experiment.yaml → embedding.model)
        threshold: порог cosine similarity для дубликата

    Требует sentence-transformers. Если не установлен — возвращает без изменений.
    """
    if len(samples) <= 1:
        return samples

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("WARN: sentence-transformers не установлен, дедупликация пропущена")
        return samples

    model = SentenceTransformer(model_name)
    queries = [s.query for s in samples]
    embeddings = model.encode(queries, normalize_embeddings=True)

    # Cosine similarity матрица (нормализованные → dot product)
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Жадный отбор: оставляем sample, если нет уже отобранного с similarity >= threshold
    keep_indices: list[int] = []
    for i in range(len(samples)):
        is_duplicate = False
        for j in keep_indices:
            if sim_matrix[i][j] >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    removed = len(samples) - len(keep_indices)
    if removed > 0:
        print(f"Дедупликация: удалено {removed} дубликатов (threshold={threshold})")

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
    - doctor_lookup: определяем expected_doctor детерминированно из KB
    - out_of_scope: фиксированный текст отказа (не зависит от KB фактов)
    - остальные категории: expected_answer остаётся пустым → заполняется LLM

    Args:
        samples: список EvalSample (in-place модификация)
        doctors: список врачей из doctors.yaml

    Returns:
        тот же список samples с заполненными метаданными
    """
    from d4.evaluation.gold_map import (
        _resolve_doctor_by_surname,
    )

    doc_by_id = {f"doctor_{d['id']}": d for d in doctors}

    for sample in samples:
        if sample.category == "doctor_lookup":
            if sample.subtype != "by_specialization_list":
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
) -> str:
    """Формирование user message для reference LLM."""
    return (
        f"## База знаний клиники\n\n{kb_text}\n\n"
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
) -> str:
    """Генерация эталонного ответа для одного сэмпла через reference LLM.

    Returns:
        текст эталонного ответа (строка)
    """
    user_message = _build_reference_user_message(sample, kb_text)
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
) -> list[EvalSample]:
    """Генерация expected_answer для всех answerable сэмплов через reference LLM.

    Заменяет захардкоженные _FACTOID_ANSWERS / _REASONING_ANSWERS.
    Использует отдельную LLM (judge config) для устранения циркулярности:
    эталонные ответы генерируются независимой моделью, а не автором.

    Сэмплы с answerable=False (out_of_scope) пропускаются —
    их expected_answer задаётся в populate_annotations().

    Args:
        samples: список EvalSample (in-place модификация expected_answer)
        kb_text: полный текст KB (все chunks, сериализованные)
        api_url: URL OpenRouter API
        api_key: API ключ
        model: модель reference LLM (напр. openai/gpt-5.4-mini)
        temperature: температура (0.0 для детерминизма)
        max_tokens: лимит токенов на ответ
        max_workers: параллельных потоков
        timeout_sec: таймаут на один запрос

    Returns:
        тот же список samples с заполненными expected_answer
    """
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
    print(f"  reference answers: {total} сэмплов, {max_workers} потоков")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_reference_for_sample,
                client, model, system_prompt, sample,
                kb_text, temperature, max_tokens,
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
    gold_map: dict[str, list[str]],
) -> list[EvalSample]:
    """Записывает gold_chunk_ids из gold_map в каждый EvalSample.

    Args:
        samples: список EvalSample (in-place модификация)
        gold_map: {sample_id: [gold_chunk_ids]} из build_gold_map()

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
