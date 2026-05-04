"""Агрегатные метрики по стратегиям.

Собирает deterministic scores, judge scores, retrieval scores
и системные метрики в единую summary таблицу.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from d4.models import (
    DeterministicScore,
    EvalSample,
    JudgeScore,
    StrategyResult,
)


def _safe_mean(values: list[float]) -> float:
    """Среднее или 0.0 для пустого списка."""
    return float(np.mean(values)) if values else 0.0


def _safe_rate(bools: list[bool]) -> float:
    """Доля True в списке."""
    return sum(bools) / len(bools) if bools else 0.0


def aggregate_deterministic(
    scores: list[DeterministicScore],
) -> dict[str, dict[str, float]]:
    """Агрегация deterministic scores по стратегиям.

    Returns:
        {strategy_id: {answerability_accuracy, doctor_match_rate, ...}}
    """
    by_strategy: dict[str, list[DeterministicScore]] = defaultdict(list)
    for s in scores:
        by_strategy[s.strategy_id.value].append(s)

    result: dict[str, dict[str, float]] = {}
    for sid, group in by_strategy.items():
        n = len(group)
        ucr_values = [s.unsupported_claim_rate for s in group if s.total_claims > 0]
        fmr_values = [s.fact_match_rate for s in group if s.fact_match_rate is not None]

        result[sid] = {
            # --- Primary metrics ---
            "n_samples": float(n),
            "answerability_accuracy": _safe_rate([s.answerability_correct for s in group]),
            "fact_match_rate": _safe_mean(fmr_values),
            "doctor_match_rate": _safe_rate([s.doctor_match for s in group]),
            "specialization_match_rate": _safe_rate([s.specialization_match for s in group]),
            "branch_match_rate": _safe_rate([s.branch_match for s in group]),
            "service_match_rate": _safe_rate([s.service_match for s in group]),
            # --- Secondary metrics (не использовать в decision memo) ---
            "unsupported_claim_rate_secondary": _safe_mean(ucr_values),
            "total_claims": float(sum(s.total_claims for s in group)),
            "unsupported_claims": float(sum(s.unsupported_claims for s in group)),
        }

    return result


def aggregate_fact_match_by_category(
    scores: list[DeterministicScore],
    samples: list[EvalSample],
) -> dict[str, dict[str, float]]:
    """Разбивка fact_match_rate по category для каждой стратегии.

    Returns:
        {strategy_id: {category: mean_fact_match_rate}}
    """
    sample_cat = {s.sample_id: s.category for s in samples}

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        cat = sample_cat.get(s.sample_id, "unknown")
        if s.fact_match_rate is not None:
            grouped[s.strategy_id.value][cat].append(s.fact_match_rate)

    result: dict[str, dict[str, float]] = {}
    for sid, cats in grouped.items():
        result[sid] = {cat: _safe_mean(vals) for cat, vals in cats.items()}
    return result


def aggregate_judge(
    scores: list[JudgeScore],
) -> dict[str, dict[str, float]]:
    """Агрегация judge scores по стратегиям.

    Returns:
        {strategy_id: {mean_factual_accuracy, mean_completeness, hallucination_rate}}
    """
    by_strategy: dict[str, list[JudgeScore]] = defaultdict(list)
    for s in scores:
        by_strategy[s.strategy_id.value].append(s)

    result: dict[str, dict[str, float]] = {}
    for sid, group in by_strategy.items():
        n = len(group)
        result[sid] = {
            "n_samples": float(n),
            "mean_factual_accuracy": _safe_mean([float(s.factual_accuracy) for s in group]),
            "mean_completeness": _safe_mean([float(s.completeness) for s in group]),
            "hallucination_rate": _safe_rate([s.hallucination for s in group]),
        }

    return result


def aggregate_system_metrics(
    results: list[StrategyResult],
) -> dict[str, dict[str, float]]:
    """Агрегация системных метрик по стратегиям.

    Returns:
        {strategy_id: {latency_p50, latency_p95, mean_tokens_prompt, ...}}
    """
    by_strategy: dict[str, list[StrategyResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_id.value].append(r)

    result: dict[str, dict[str, float]] = {}
    for sid, group in by_strategy.items():
        latencies = [r.latency_ms for r in group if r.error is None]
        prompts = [float(r.tokens_prompt) for r in group if r.error is None]
        completions = [float(r.tokens_completion) for r in group if r.error is None]
        contexts = [float(r.context_length) for r in group]
        errors = [r for r in group if r.error is not None]

        result[sid] = {
            "n_samples": float(len(group)),
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "mean_tokens_prompt": _safe_mean(prompts),
            "mean_tokens_completion": _safe_mean(completions),
            "mean_context_length": _safe_mean(contexts),
            "fail_rate": len(errors) / len(group) if group else 0.0,
        }

    return result


def aggregate_route_metrics(
    results: list[StrategyResult],
    det_scores: list[DeterministicScore] | None = None,
) -> dict[str, dict[str, float]]:
    """S5 route accounting: share_direct/fallback, quality/cost/latency by route.

    Только для стратегий с непустым route_taken (S5, B0).

    Returns:
        {strategy_id: {share_direct, share_fallback, fmr_direct, fmr_fallback, ...}}
    """
    routed = [r for r in results if r.route_taken]
    if not routed:
        return {}

    det_map: dict[tuple[str, str], DeterministicScore] = {}
    if det_scores:
        for d in det_scores:
            det_map[(d.sample_id, d.strategy_id.value)] = d

    by_strategy: dict[str, list[StrategyResult]] = defaultdict(list)
    for r in routed:
        by_strategy[r.strategy_id.value].append(r)

    result: dict[str, dict[str, float]] = {}
    for sid, group in by_strategy.items():
        n = len(group)
        direct = [r for r in group if r.route_taken == "direct"]
        fallback = [r for r in group if r.route_taken == "fallback"]

        row: dict[str, float] = {
            "n_samples": float(n),
            "share_direct": len(direct) / n if n else 0.0,
            "share_fallback": len(fallback) / n if n else 0.0,
            "mean_latency_direct": _safe_mean([r.latency_ms for r in direct]),
            "mean_latency_fallback": _safe_mean([r.latency_ms for r in fallback]),
            "mean_tokens_direct": _safe_mean([float(r.tokens_prompt + r.tokens_completion) for r in direct]),
            "mean_tokens_fallback": _safe_mean([float(r.tokens_prompt + r.tokens_completion) for r in fallback]),
        }

        if det_scores:
            fmr_direct = [
                det_map[(r.sample_id, sid)].fact_match_rate
                for r in direct
                if (r.sample_id, sid) in det_map
                and det_map[(r.sample_id, sid)].fact_match_rate is not None
            ]
            fmr_fallback = [
                det_map[(r.sample_id, sid)].fact_match_rate
                for r in fallback
                if (r.sample_id, sid) in det_map
                and det_map[(r.sample_id, sid)].fact_match_rate is not None
            ]
            row["fmr_direct"] = _safe_mean(fmr_direct)
            row["fmr_fallback"] = _safe_mean(fmr_fallback)

        result[sid] = row

    return result


def build_summary_table(
    deterministic: dict[str, dict[str, float]],
    judge: dict[str, dict[str, float]],
    system: dict[str, dict[str, float]],
    retrieval: dict[str, dict[str, float]] | None = None,
    route: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Объединение всех метрик в единую summary таблицу.

    Returns:
        {strategy_id: {все метрики}}
    """
    all_sids = set(deterministic) | set(judge) | set(system)
    summary: dict[str, dict[str, Any]] = {}

    for sid in sorted(all_sids):
        row: dict[str, Any] = {"strategy_id": sid}
        row.update(deterministic.get(sid, {}))
        row.update({f"judge_{k}": v for k, v in judge.get(sid, {}).items()})
        row.update({f"sys_{k}": v for k, v in system.get(sid, {}).items()})
        if retrieval and sid in retrieval:
            row.update({f"ret_{k}": v for k, v in retrieval[sid].items()})
        if route and sid in route:
            row.update({f"route_{k}": v for k, v in route[sid].items()})
        summary[sid] = row

    return summary
