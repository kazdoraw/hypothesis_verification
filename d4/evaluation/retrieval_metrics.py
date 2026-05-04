"""Retrieval-метрики для S2-S4/S4r (обязательный блок).

Без них нельзя отделить провал retrieval от провала generation.
- Hit@k: gold chunk в top-k (any alternative)
- Recall@k: доля gold chunks в top-k (best alternative)
- MRR (Mean Reciprocal Rank, best alternative)
- gold_chunk_in_context (bool, best alternative)
- pre_rerank_hit / pre_rerank_recall (только S4r, для rerank lift)

Multi-gold: gold_chunk_ids = list[list[str]] — несколько допустимых
путей к правильному ответу. Метрики считаются по ЛУЧШЕЙ альтернативе.
"""

from __future__ import annotations

from d4.models import EvalSample, RetrievalScore, StrategyResult


def _best_hit_recall(
    gold_chunk_ids: list[list[str]],
    candidate_set: set[str],
) -> tuple[bool, float]:
    """Возвращает (best_hit, best_recall) по multi-gold альтернативам."""
    best_hit = False
    best_recall = 0.0
    for alt_gold in gold_chunk_ids:
        gold_set = set(alt_gold)
        if not gold_set:
            continue
        hit = bool(gold_set & candidate_set)
        recall = len(gold_set & candidate_set) / len(gold_set)
        best_hit = best_hit or hit
        best_recall = max(best_recall, recall)
    return best_hit, best_recall


def compute_retrieval_score(
    result: StrategyResult,
    gold_chunk_ids: list[list[str]],
) -> RetrievalScore:
    """Вычисление retrieval-метрик для одного результата (multi-gold).

    Для S4r дополнительно вычисляются pre_rerank_hit и pre_rerank_recall
    на основе RetrievalResult.pre_rerank_chunk_ids (top-N до reranking).
    """
    retrieved = result.retrieval.chunk_ids
    retrieved_set = set(retrieved)

    if not gold_chunk_ids:
        return RetrievalScore(
            sample_id=result.sample_id,
            strategy_id=result.strategy_id,
            gold_chunk_ids=gold_chunk_ids,
            retrieved_chunk_ids=retrieved,
        )

    best_hit = False
    best_recall = 0.0
    best_rr = 0.0
    best_in_context = False

    for alt_gold in gold_chunk_ids:
        gold_set = set(alt_gold)
        if not gold_set:
            continue

        hit = bool(gold_set & retrieved_set)
        recall = len(gold_set & retrieved_set) / len(gold_set)
        rr = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in gold_set:
                rr = 1.0 / rank
                break
        in_context = gold_set.issubset(retrieved_set)

        best_hit = best_hit or hit
        best_recall = max(best_recall, recall)
        best_rr = max(best_rr, rr)
        best_in_context = best_in_context or in_context

    pre_hit = False
    pre_recall = 0.0
    pre_rerank_ids = result.retrieval.pre_rerank_chunk_ids
    if pre_rerank_ids:
        pre_hit, pre_recall = _best_hit_recall(
            gold_chunk_ids, set(pre_rerank_ids),
        )

    return RetrievalScore(
        sample_id=result.sample_id,
        strategy_id=result.strategy_id,
        gold_chunk_ids=gold_chunk_ids,
        retrieved_chunk_ids=retrieved,
        hit_at_k=best_hit,
        recall_at_k=best_recall,
        reciprocal_rank=best_rr,
        gold_chunk_in_context=best_in_context,
        pre_rerank_hit=pre_hit,
        pre_rerank_recall=pre_recall,
    )


def compute_batch_retrieval(
    results: list[StrategyResult],
    gold_map: dict[str, list[list[str]]],
    strategy_ids: list[str] | None = None,
) -> list[RetrievalScore]:
    """Batch вычисление retrieval-метрик (multi-gold).

    Args:
        results: все StrategyResult
        gold_map: {sample_id: [[alt1], [alt2], ...]}
        strategy_ids: фильтр стратегий (None = S2, S3, S4, S4r)

    Returns:
        список RetrievalScore
    """
    allowed = set(strategy_ids) if strategy_ids else {"S2", "S3", "S4", "S4r", "S5"}
    scores: list[RetrievalScore] = []
    skipped_no_gold = 0

    for result in results:
        if result.strategy_id.value not in allowed:
            continue
        gold_ids = gold_map.get(result.sample_id, [])
        if not gold_ids:
            skipped_no_gold += 1
            continue
        score = compute_retrieval_score(result, gold_ids)
        scores.append(score)

    if skipped_no_gold > 0:
        import logging
        logging.getLogger(__name__).info(
            "Retrieval: пропущено %d сэмплов без gold_chunk_ids", skipped_no_gold,
        )

    return scores


def aggregate_retrieval_metrics(
    scores: list[RetrievalScore],
) -> dict[str, dict[str, float]]:
    """Агрегация retrieval-метрик по стратегиям.

    Для S4r дополнительно: pre_rerank_hit_rate, pre_rerank_recall, rerank_hit_lift.

    Returns:
        {strategy_id: {hit_rate, mean_recall, mrr, gold_in_context_rate, ...}}
    """
    from collections import defaultdict

    by_strategy: dict[str, list[RetrievalScore]] = defaultdict(list)
    for s in scores:
        by_strategy[s.strategy_id.value].append(s)

    result: dict[str, dict[str, float]] = {}
    for sid, strategy_scores in by_strategy.items():
        n = len(strategy_scores)
        if n == 0:
            continue

        hit_rate = sum(1 for s in strategy_scores if s.hit_at_k) / n
        mean_recall = sum(s.recall_at_k for s in strategy_scores) / n

        row: dict[str, float] = {
            "hit_rate": hit_rate,
            "mean_recall": mean_recall,
            "mrr": sum(s.reciprocal_rank for s in strategy_scores) / n,
            "gold_in_context_rate": sum(
                1 for s in strategy_scores if s.gold_chunk_in_context
            ) / n,
            "n_samples": float(n),
        }

        pre_rerank_entries = [s for s in strategy_scores if s.pre_rerank_hit or s.pre_rerank_recall > 0]
        has_pre_rerank = any(s.pre_rerank_hit or s.pre_rerank_recall > 0 for s in strategy_scores)
        if has_pre_rerank or sid == "S4r":
            pre_hit_rate = sum(1 for s in strategy_scores if s.pre_rerank_hit) / n
            pre_recall = sum(s.pre_rerank_recall for s in strategy_scores) / n
            row["pre_rerank_hit_rate"] = pre_hit_rate
            row["pre_rerank_recall"] = pre_recall
            row["rerank_hit_lift"] = hit_rate - pre_hit_rate
            row["rerank_recall_lift"] = mean_recall - pre_recall

        result[sid] = row

    return result


def aggregate_retrieval_by_category(
    scores: list[RetrievalScore],
    samples: list[EvalSample],
) -> dict[str, dict[str, dict[str, float]]]:
    """Разбивка retrieval hit@k по category для каждой стратегии.

    Returns:
        {strategy_id: {category: {hit_rate, mean_recall, n}}}
    """
    from collections import defaultdict

    sample_cat = {s.sample_id: s.category for s in samples}

    grouped: dict[str, dict[str, list[RetrievalScore]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for s in scores:
        cat = sample_cat.get(s.sample_id, "unknown")
        grouped[s.strategy_id.value][cat].append(s)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for sid, cats in grouped.items():
        result[sid] = {}
        for cat, cat_scores in cats.items():
            n = len(cat_scores)
            result[sid][cat] = {
                "hit_rate": sum(1 for s in cat_scores if s.hit_at_k) / n,
                "mean_recall": sum(s.recall_at_k for s in cat_scores) / n,
                "n": float(n),
            }

    return result
