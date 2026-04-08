"""Метрики качества извлечения анамнеза.

1. Completeness — доля заполненных полей.
2. Semantic similarity — embedding сравнение extracted vs reference.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from d2.config import EMBEDDING_MODEL, SEMANTIC_THRESHOLD
from d2.models import CaseResult, SchemaRun
from d2.cases import CASES


@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    """Ленивая загрузка модели embeddings (singleton)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity двух текстов через sentence-transformers."""
    encoder = _get_encoder()
    embeddings = encoder.encode([text_a, text_b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


def compute_completeness(run: SchemaRun, reference: dict[str, str]) -> float:
    """Доля reference-полей, покрытых extracted данными.

    Для S1/S2: прямое совпадение ключей.
    Для S3: best-match по значениям (ключи могут быть на русском).
    """
    if not reference:
        return 1.0 if run.extracted else 0.0

    extracted_values = _flatten_extracted(run.extracted)
    covered = 0
    for ref_key, ref_value in reference.items():
        if not ref_value:
            covered += 1
            continue
        # Прямое совпадение по ключу
        if ref_key in run.extracted and run.extracted[ref_key].strip():
            covered += 1
            continue
        # Best-match: есть ли хоть одно извлечённое значение, семантически близкое к reference?
        if extracted_values:
            best = max(cosine_similarity(v, ref_value) for v in extracted_values)
            if best >= 0.55:  # порог "покрыто"
                covered += 1
    return covered / len(reference)


def compute_similarity(run: SchemaRun, reference: dict[str, str]) -> dict:
    """Embedding similarity extracted vs reference.

    Для S1/S2 — прямое сравнение по ключам.
    Для S3 (fields dict) — best-match: каждый reference ищет ближайшее значение.

    Возвращает:
        {"per_field": {"symptoms": 0.85, ...}, "mean": 0.82}
    """
    # Собираем все извлечённые значения (плоский список)
    extracted_values = _flatten_extracted(run.extracted)

    per_field: dict[str, float] = {}
    for field_name, ref_value in reference.items():
        if not ref_value:
            per_field[field_name] = 0.0
            continue

        # Прямое совпадение по ключу
        direct = run.extracted.get(field_name, "")
        if direct:
            per_field[field_name] = cosine_similarity(direct, ref_value)
            continue

        # Best-match среди всех значений (для S3 с русскими ключами)
        if extracted_values:
            best = max(cosine_similarity(v, ref_value) for v in extracted_values)
            per_field[field_name] = best
        else:
            per_field[field_name] = 0.0

    mean = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"per_field": per_field, "mean": round(mean, 3)}


def _flatten_extracted(extracted: dict[str, str]) -> list[str]:
    """Плоский список всех значений, включая вложенные dict (S3.fields)."""
    import ast
    values: list[str] = []
    for v in extracted.values():
        if not v:
            continue
        # S3 может хранить Python dict-строку — пробуем распарсить
        if v.strip().startswith("{"):
            try:
                inner = ast.literal_eval(v)
                if isinstance(inner, dict):
                    values.extend(str(iv) for iv in inner.values() if iv)
                    continue
            except (ValueError, SyntaxError):
                pass
        values.append(v)
    return values


def compute_case_metrics(result: CaseResult) -> dict:
    """Вычислить все метрики для одного кейса.

    Возвращает dict: {schema_name: {completeness, similarity, turns, tokens}}.
    """
    # Находим reference_fields из cases.py по case_id
    reference = {}
    for case in CASES:
        if case.case_id == result.case_id:
            reference = case.reference_fields
            break

    metrics: dict[str, dict] = {}
    for schema_name, run in result.runs.items():
        sim = compute_similarity(run, reference) if reference else {"per_field": {}, "mean": 0.0}
        total_tokens = run.tokens_doctor + run.tokens_patient

        metrics[schema_name] = {
            "completeness": round(compute_completeness(run, reference), 3),
            "similarity_mean": sim["mean"],
            "similarity_per_field": sim["per_field"],
            "above_threshold": sum(1 for v in sim["per_field"].values() if v >= SEMANTIC_THRESHOLD),
            "turns": run.turns,
            "tokens": total_tokens,
            "duration_s": run.duration_s,
        }

    return metrics
