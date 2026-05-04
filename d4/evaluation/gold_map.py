"""Построение gold_map для retrieval-метрик.

Gold map определяет эталонные chunk_ids для каждого eval sample.
Используется для вычисления Hit@k, Recall@k, MRR в retrieval_metrics.py.

После рефакторинга (Фаза 3.1): gold_chunk_ids хранятся непосредственно
в EvalSample (multi-gold формат list[list[str]]). Hardcoded таблицы
SUBTYPE_TO_CHUNK_IDS, _PRICE_KEYWORDS, _REASONING_GOLD_SPECS удалены
для устранения циркулярности (Bias 1, 2, 8).
"""

from __future__ import annotations

from d4.models import EvalSample


def build_gold_map(
    samples: list[EvalSample],
) -> dict[str, list[list[str]]]:
    """Извлекает gold_chunk_ids из EvalSample (заполненных при annotation).

    Args:
        samples: eval set с размеченными gold_chunk_ids

    Returns:
        {sample_id: [[alt1_chunk_ids], [alt2_chunk_ids], ...]}
    """
    return {s.sample_id: s.gold_chunk_ids for s in samples if s.gold_chunk_ids}


def print_gold_map_report(
    gold_map: dict[str, list[list[str]]],
    samples: list[EvalSample],
    valid_chunk_ids: set[str] | None = None,
) -> None:
    """Отчёт о покрытии gold_map: сколько заполнено, есть ли ошибки.

    Args:
        gold_map: построенный gold_map (multi-gold формат)
        samples: eval set (для категорий и answerable)
        valid_chunk_ids: множество chunk_ids из chunks.json (для валидации)
    """
    sample_map = {s.sample_id: s for s in samples}
    by_category: dict[str, list[str]] = {}
    empty_answerable: list[str] = []
    invalid_chunks: list[tuple[str, str]] = []

    for sid, alternatives in gold_map.items():
        sample = sample_map.get(sid)
        if not sample:
            continue
        by_category.setdefault(sample.category, []).append(sid)

        if sample.answerable and not alternatives:
            empty_answerable.append(sid)

        if valid_chunk_ids:
            for alt in alternatives:
                for cid in alt:
                    if cid not in valid_chunk_ids:
                        invalid_chunks.append((sid, cid))

    total = len(samples)
    filled = len(gold_map)
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
