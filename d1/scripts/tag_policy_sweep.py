"""Phase 2: tag-policy threshold sweep + verification (selection ТОЛЬКО на val).

Цель — найти min(threshold) per tag ∈ {simple_faq, simple_booking, simple_symptom},
который удовлетворяет stopping rule на val:
    accepted_acc(val, tag)        ≥ 0.95
    AND accepted_recall_anam(val) ≥ 0.98

Затем верифицировать выбранные thresholds на test/hard_test/blind_test/extended_eval
БЕЗ участия этих splits в selection (anti-overfit guard).

Артефакты:
    d1/results/tag_policy_pareto_candidates.csv — grid × tag × val metrics.
    d1/results/tag_policy_sweep_results.csv     — verification per eval_set.

Регрессия (sanity): пересчёт current DEFAULT_TAG_POLICIES на pre-policy trace
должен дать те же coverage / accepted_acc что и `simple_router_results.csv`.

Запуск:
    cd study && .venv/bin/python -m d1.scripts.tag_policy_sweep
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from d1.baselines.simple_router import DEFAULT_TAG_POLICIES, TagPolicy
from d1.config import RESULTS_DIR
from d1.scripts.cross_tune_sanity import (
    ACC_THRESHOLD,
    ALLOWED_LABELS_FOR_TUNING,
    DEFAULT_DENSE,
    DEFAULT_SPARSE,
    GRID,
    RECALL_ANAM_THRESHOLD,
    TAGS_TO_TUNE,
    _PrePolicyTrace,
    _build_hybrid_router,
    _metrics_for_indices,
    collect_pre_policy_trace,
)

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SETS: tuple[str, ...] = (
    "val", "test", "hard_test", "blind_test", "extended_eval",
)
SELECTION_SPLIT = "val"


# ---------------------------------------------------------------------------
# Step 1: pre-policy traces для всех splits (один bundle, разные splits).
# ---------------------------------------------------------------------------

def collect_traces_for_eval_sets(
    eval_sets: tuple[str, ...] = DEFAULT_EVAL_SETS,
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
) -> dict[str, _PrePolicyTrace]:
    """Загрузить bundle один раз → собрать pre-policy traces для всех splits."""
    hybrid, gate = _build_hybrid_router(sparse_name, dense_name)
    traces: dict[str, _PrePolicyTrace] = {}
    for eval_set in eval_sets:
        logger.info("Pre-policy trace: %s", eval_set)
        traces[eval_set] = collect_pre_policy_trace(
            eval_set=eval_set,
            sparse_name=sparse_name,
            dense_name=dense_name,
            hybrid=hybrid,
            gate=gate,
        )
    return traces


# ---------------------------------------------------------------------------
# Step 2: pareto candidates на val (grid × tag × val metrics).
# ---------------------------------------------------------------------------

def build_pareto_candidates(
    trace: _PrePolicyTrace,
    grid: tuple[float, ...] = GRID,
    tags: tuple[str, ...] = TAGS_TO_TUNE,
) -> pd.DataFrame:
    """Для каждого (tag, threshold) посчитать val metrics.

    При тюнинге одного tag остальные tag-policy зафиксированы на
    DEFAULT_TAG_POLICIES (1D sweep — independent per tag).
    """
    n = len(trace)
    indices = np.arange(n)
    rows: list[dict[str, object]] = []
    for tag in tags:
        allowed = ALLOWED_LABELS_FOR_TUNING[tag]
        for thr in sorted(grid):
            candidate = dict(DEFAULT_TAG_POLICIES)
            candidate[tag] = TagPolicy(min_confidence=thr, allowed_labels=allowed)
            recall_anam, coverage, per_tag = _metrics_for_indices(
                trace, indices, candidate,
            )
            acc_tag = per_tag.get(tag, float("nan"))
            passes = (
                not np.isnan(acc_tag)
                and acc_tag >= ACC_THRESHOLD
                and recall_anam >= RECALL_ANAM_THRESHOLD
            )
            rows.append({
                "tag": tag,
                "threshold": thr,
                "val_accepted_acc_tag": round(acc_tag, 4) if not np.isnan(acc_tag) else None,
                "val_accepted_recall_anam": round(recall_anam, 4),
                "val_coverage_global": round(coverage, 4),
                "passes_stopping_rule": passes,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: select min threshold per tag по pareto df.
# ---------------------------------------------------------------------------

def select_thresholds(pareto_df: pd.DataFrame) -> dict[str, float | None]:
    """Min(threshold) на котором passes_stopping_rule=True. None — если ни одного."""
    selected: dict[str, float | None] = {}
    for tag, group in pareto_df.groupby("tag"):
        passing = group[group["passes_stopping_rule"]]
        if passing.empty:
            selected[tag] = None
        else:
            selected[tag] = float(passing["threshold"].min())
    return selected


# ---------------------------------------------------------------------------
# Step 4: verification — selected thresholds применяем к каждому eval_set.
# ---------------------------------------------------------------------------

def build_verification_table(
    traces: dict[str, _PrePolicyTrace],
    selected: dict[str, float | None],
) -> pd.DataFrame:
    """Применить selected thresholds к каждому eval_set, собрать metrics.

    Также включает baseline (current DEFAULT_TAG_POLICIES) для сравнения —
    позволяет видеть Δ coverage / Δ accepted_acc.
    """
    rows: list[dict[str, object]] = []

    selected_policies = dict(DEFAULT_TAG_POLICIES)
    for tag, thr in selected.items():
        if thr is None:
            continue
        selected_policies[tag] = TagPolicy(
            min_confidence=thr,
            allowed_labels=ALLOWED_LABELS_FOR_TUNING[tag],
        )

    for eval_set, trace in traces.items():
        n = len(trace)
        indices = np.arange(n)

        recall_a, cov_a, per_tag_a = _metrics_for_indices(
            trace, indices, dict(DEFAULT_TAG_POLICIES),
        )
        recall_b, cov_b, per_tag_b = _metrics_for_indices(
            trace, indices, selected_policies,
        )
        rows.append({
            "eval_set": eval_set,
            "n": n,
            "current_coverage": round(cov_a, 4),
            "current_recall_anam": round(recall_a, 4),
            "current_acc_simple_faq": _safe_round(per_tag_a.get("simple_faq")),
            "current_acc_simple_booking": _safe_round(per_tag_a.get("simple_booking")),
            "current_acc_simple_symptom": _safe_round(per_tag_a.get("simple_symptom")),
            "selected_coverage": round(cov_b, 4),
            "selected_recall_anam": round(recall_b, 4),
            "selected_acc_simple_faq": _safe_round(per_tag_b.get("simple_faq")),
            "selected_acc_simple_booking": _safe_round(per_tag_b.get("simple_booking")),
            "selected_acc_simple_symptom": _safe_round(per_tag_b.get("simple_symptom")),
            "delta_coverage": round(cov_b - cov_a, 4),
        })
    return pd.DataFrame(rows)


def _safe_round(x: float | None, n: int = 4) -> float | None:
    """Round; NaN → None для CSV (пусто вместо NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(x, n)


# ---------------------------------------------------------------------------
# Step 5: regression check — current DEFAULT_TAG_POLICIES на val ↔ simple_router_results.
# ---------------------------------------------------------------------------

def regression_check_against_simple_router(
    traces: dict[str, _PrePolicyTrace],
    simple_router_results_path: Path | None = None,
    tolerance: float = 0.005,
) -> pd.DataFrame:
    """Verify pre-policy trace + DEFAULT_TAG_POLICIES ≈ simple_router_results.csv.

    Это sanity gate: если расхождение > tolerance, значит pre-policy trace
    не репрезентирует SimpleRouter (например, забыли gate-defer и т.п.).
    """
    path = simple_router_results_path or (RESULTS_DIR / "simple_router_results.csv")
    if not path.exists():
        logger.warning("simple_router_results.csv не найден: %s", path)
        return pd.DataFrame()

    expected = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, exp_row in expected.iterrows():
        eval_set = exp_row["eval_set"]
        if eval_set not in traces:
            continue
        trace = traces[eval_set]
        recall_actual, coverage_actual, _ = _metrics_for_indices(
            trace, np.arange(len(trace)), dict(DEFAULT_TAG_POLICIES),
        )
        # Внимание: pre-policy trace не учитывает complexity-defer (gate.decide
        # для defer-тэгов даёт action=defer, но мы используем hybrid.route_batch
        # на ВСЕХ текстах). Поэтому coverage из trace будет ВЫШЕ чем фактический.
        # Регрессионная проверка делается через accepted_recall_anam — он
        # инвариантен к gate-defer (anam с complexity → пропадает из accepted
        # И в trace и в SimpleRouter).
        rows.append({
            "eval_set": eval_set,
            "expected_coverage": exp_row["coverage"],
            "trace_coverage_no_gate": round(coverage_actual, 4),
            "expected_recall_anam": exp_row["accepted_recall_anam"],
            "trace_recall_anam": round(recall_actual, 4),
            "recall_anam_match": (
                abs(recall_actual - exp_row["accepted_recall_anam"]) <= tolerance
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_tag_policy_sweep(
    eval_sets: tuple[str, ...] = DEFAULT_EVAL_SETS,
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Phase 2 main entrypoint.

    Returns:
        dict с ключами selected_thresholds, pareto_df, verification_df,
        regression_df.
    """
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = collect_traces_for_eval_sets(eval_sets=eval_sets)
    pareto_df = build_pareto_candidates(traces[SELECTION_SPLIT])
    selected = select_thresholds(pareto_df)
    verification_df = build_verification_table(traces, selected)
    regression_df = regression_check_against_simple_router(traces)

    pareto_path = out_dir / "tag_policy_pareto_candidates.csv"
    pareto_df.to_csv(pareto_path, index=False)
    logger.info("Saved: %s", pareto_path)

    verif_path = out_dir / "tag_policy_sweep_results.csv"
    verification_df.to_csv(verif_path, index=False)
    logger.info("Saved: %s", verif_path)

    return {
        "selected_thresholds": selected,
        "pareto_df": pareto_df,
        "verification_df": verification_df,
        "regression_df": regression_df,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run_tag_policy_sweep()

    print("\n" + "=" * 88)
    print("  TAG-POLICY SWEEP — selected thresholds (selection on val ONLY)")
    print("=" * 88)
    for tag, thr in out["selected_thresholds"].items():
        current = DEFAULT_TAG_POLICIES.get(tag)
        current_thr = current.min_confidence if current else "(none)"
        print(f"  {tag:18s}: current={current_thr}  →  selected={thr}")

    print("\n" + "=" * 88)
    print("  PARETO CANDIDATES (val) — passes stopping rule")
    print("=" * 88)
    pareto = out["pareto_df"]
    print(pareto.to_string(index=False))

    print("\n" + "=" * 88)
    print("  VERIFICATION (current DEFAULT vs SELECTED)")
    print("=" * 88)
    print(out["verification_df"].to_string(index=False))

    print("\n" + "=" * 88)
    print("  REGRESSION SANITY (DEFAULT applied on trace ↔ simple_router_results)")
    print("=" * 88)
    if out["regression_df"].empty:
        print("  simple_router_results.csv не найден — skip")
    else:
        print(out["regression_df"].to_string(index=False))


if __name__ == "__main__":
    main()
