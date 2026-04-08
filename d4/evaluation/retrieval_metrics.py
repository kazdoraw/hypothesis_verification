"""Retrieval-метрики для S2-S4 (обязательный блок).

Без них нельзя отделить провал retrieval от провала generation.
- Hit@k: gold chunk в top-k
- Recall@k: доля gold chunks в top-k
- MRR (Mean Reciprocal Rank)
- gold_chunk_in_context (bool)
"""

from __future__ import annotations

from d4.models import RetrievalScore, StrategyID, StrategyResult


def compute_retrieval_score(
    result: StrategyResult,
    gold_chunk_ids: list[str],
) -> RetrievalScore:
    """Вычисление retrieval-метрик для одного результата.

    Args:
        result: результат стратегии (содержит retrieval.chunk_ids)
        gold_chunk_ids: эталонные chunk_ids (из экспертной разметки)

    Returns:
        RetrievalScore
    """
    retrieved = result.retrieval.chunk_ids
    gold_set = set(gold_chunk_ids)

    if not gold_chunk_ids:
        return RetrievalScore(
            sample_id=result.sample_id,
            strategy_id=result.strategy_id,
            gold_chunk_ids=gold_chunk_ids,
            retrieved_chunk_ids=retrieved,
        )

    # Hit@k: хотя бы один gold chunk в retrieved
    hit_at_k = bool(gold_set & set(retrieved))

    # Recall@k: доля gold chunks найденных в retrieved
    found = len(gold_set & set(retrieved))
    recall_at_k = found / len(gold_set) if gold_set else 0.0

    # MRR: reciprocal rank первого gold chunk в retrieved
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            reciprocal_rank = 1.0 / rank
            break

    # gold_chunk_in_context: все gold chunks присутствуют
    gold_chunk_in_context = gold_set.issubset(set(retrieved))

    return RetrievalScore(
        sample_id=result.sample_id,
        strategy_id=result.strategy_id,
        gold_chunk_ids=gold_chunk_ids,
        retrieved_chunk_ids=retrieved,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        gold_chunk_in_context=gold_chunk_in_context,
    )


def compute_batch_retrieval(
    results: list[StrategyResult],
    gold_map: dict[str, list[str]],
    strategy_ids: list[str] | None = None,
) -> list[RetrievalScore]:
    """Batch вычисление retrieval-метрик.

    Args:
        results: все StrategyResult
        gold_map: {sample_id: [gold_chunk_ids]} — экспертная разметка
        strategy_ids: фильтр стратегий (None = S2, S3, S4)

    Returns:
        список RetrievalScore
    """
    allowed = set(strategy_ids) if strategy_ids else {"S2", "S3", "S4"}
    scores: list[RetrievalScore] = []
    skipped_no_gold = 0

    for result in results:
        if result.strategy_id.value not in allowed:
            continue
        gold_ids = gold_map.get(result.sample_id, [])
        # Пропускаем сэмплы без gold_chunk_ids (out_of_scope и т.д.)
        # чтобы не занижать метрики нулями
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

    Returns:
        {strategy_id: {hit_rate, mean_recall, mrr, gold_in_context_rate}}
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
        result[sid] = {
            "hit_rate": sum(1 for s in strategy_scores if s.hit_at_k) / n,
            "mean_recall": sum(s.recall_at_k for s in strategy_scores) / n,
            "mrr": sum(s.reciprocal_rank for s in strategy_scores) / n,
            "gold_in_context_rate": sum(1 for s in strategy_scores if s.gold_chunk_in_context) / n,
            "n_samples": float(n),
        }

    return result
