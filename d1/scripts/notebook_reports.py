"""Notebook reporting API: только `show_*` функции (Task 1 плана).

Каждая функция:
- **read-only**: читает артефакты через `artifact_io.py`, ничего не пишет;
- возвращает `pd.DataFrame` (для последующего `display()` вызывающим) ЛИБО
  рендерит `display(...)` сама и возвращает None;
- не вызывает `train_bundle` / `evaluate_*` / heavy computation.

Full rerun (`run_*`) — отдельная CLI-команда `run_experiment_pipeline`, не
вызывается из этого модуля.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
from IPython.display import Markdown, display

from d1.baselines.selective_router import PRODUCTION_THRESHOLDS
from d1.scripts import artifact_io

# ---------------------------------------------------------------------------
# Константы gate criteria (§11 ТЗ). Хранятся здесь как SSoT для reporting.
# ---------------------------------------------------------------------------

GATE_MACRO_F1 = 0.80
GATE_RECALL_ANAM = 0.90
GATE_RECALL_URGENT = 0.90


# ---------------------------------------------------------------------------
# Section 1: Data summary
# ---------------------------------------------------------------------------

def show_split_summary() -> pd.DataFrame:
    """Размеры eval-splitов и распределение по route_domain классам."""
    return artifact_io.load_split_sizes()


# ---------------------------------------------------------------------------
# Section 2-4: Baselines / plots / confusion
# ---------------------------------------------------------------------------

def show_baseline_summary(eval_set: str = "test") -> pd.DataFrame:
    """Сводная таблица метрик baselines на указанном eval_set."""
    from d1.scripts.plot_results import plot_summary_table
    df, _ = artifact_io.load_routing_results()
    return plot_summary_table(df, eval_set)


def show_confusion_matrix(eval_set: str = "test") -> None:
    """Отобразить confusion matrix heatmap для eval_set (savefig в FIGURES_DIR)."""
    from d1.scripts.plot_results import plot_confusion_matrices
    _, data = artifact_io.load_routing_results()
    plot_confusion_matrices(data, eval_set)


def show_per_class_f1(eval_set: str = "test") -> None:
    """Heatmap per-class F1 (baseline × class)."""
    from d1.scripts.plot_results import plot_per_class_f1_heatmap
    _, data = artifact_io.load_routing_results()
    plot_per_class_f1_heatmap(data, eval_set)


def show_routing_comparison(eval_set: str = "test") -> None:
    """Grouped bar chart accuracy/macro-F1/balanced accuracy."""
    from d1.scripts.plot_results import plot_routing_comparison
    df, _ = artifact_io.load_routing_results()
    plot_routing_comparison(df, eval_set)


def show_safety_comparison() -> None:
    """Safety set: recall_urgent и FN."""
    from d1.scripts.plot_results import plot_safety_comparison
    plot_safety_comparison(artifact_io.load_safety_results())


def show_cross_eval_f1() -> None:
    """Macro-F1 по eval_sets для каждого baseline."""
    from d1.scripts.plot_results import plot_cross_eval_f1
    df, _ = artifact_io.load_routing_results()
    plot_cross_eval_f1(df)


def show_latency(eval_set: str = "test") -> None:
    """Bar chart median latency per baseline."""
    from d1.scripts.plot_results import plot_latency
    df, _ = artifact_io.load_routing_results()
    plot_latency(df, eval_set)


# ---------------------------------------------------------------------------
# Section 5.1: Calibration
# ---------------------------------------------------------------------------

def show_calibration_summary() -> pd.DataFrame:
    """Сводная таблица ECE + Brier(macro) по (model × eval_set)."""
    return artifact_io.load_calibration_metrics()


def show_pareto_summary() -> None:
    """Pareto candidate thresholds (только на val — overfitting guard)."""
    candidates = artifact_io.load_pareto_candidates()
    if not candidates:
        display(Markdown(
            "Pareto candidates отсутствуют. Запустите `analyze_confidence.run_confidence_analysis`."
        ))
        return
    for name, df in candidates.items():
        display(Markdown(f"### {name}"))
        display(df.round(4))


# ---------------------------------------------------------------------------
# Section 5.2: Selective router
# ---------------------------------------------------------------------------

def show_selective_summary() -> None:
    """SelectiveRouter метрики + reasons breakdown по eval_sets."""
    summary, full = artifact_io.load_selective_results()
    display(summary.round(4))

    display(Markdown(f"**Production thresholds:** `{full['thresholds']}`"))

    # Reasons breakdown: eval_set × reason pivot.
    reasons_rows: list[dict[str, Any]] = []
    for eval_set, blk in full["results"].items():
        for reason, count in blk["selective"]["reasons_breakdown"].items():
            reasons_rows.append({
                "eval_set": eval_set, "reason": reason, "count": count,
            })
    if reasons_rows:
        reasons_df = (
            pd.DataFrame(reasons_rows)
            .pivot_table(
                index="reason", columns="eval_set", values="count", fill_value=0,
            )
            .astype(int)
        )
        display(Markdown("### Reasons breakdown (eval_set × reason)"))
        display(reasons_df)


# ---------------------------------------------------------------------------
# Section 5.3: Hybrid router
# ---------------------------------------------------------------------------

def show_hybrid_summary() -> None:
    """B4 Hybrid vs SelectiveRouter + threshold sweep."""
    hybrid_df, vs_df = artifact_io.load_hybrid_results()
    thresholds = asdict(PRODUCTION_THRESHOLDS)

    display(Markdown(f"### Hybrid vs Selective (production thresholds: `{thresholds}`)"))
    display(vs_df.round(4))

    # Hard-test delta.
    hard_rows = vs_df[vs_df["eval_set"] == "hard_test"]
    if not hard_rows.empty:
        hard = hard_rows.iloc[0]
        display(Markdown(
            "**Hard-test delta:**\n\n"
            f"- coverage: {hard['selective_coverage']:.4f} → {hard['hybrid_coverage']:.4f} "
            f"({hard['delta_coverage']:+.4f})\n"
            f"- recall_anam: {hard['selective_recall_anam']:.4f} → "
            f"{hard['hybrid_recall_anam']:.4f}\n"
            f"- FN_deferred: {int(hard['selective_FN_deferred'])} → "
            f"{int(hard['hybrid_FN_deferred'])}\n"
            f"- rule_accept_count: {int(hard['rule_accept_count'])}"
        ))

    # Threshold sweep.
    sweep = artifact_io.load_threshold_sweep()
    display(Markdown("### Threshold sweep: baseline vs production vs aggressive"))
    key_cols = [
        "config", "eval_set", "router", "n",
        "coverage", "accepted_accuracy", "accepted_recall_anam",
        "FN_deferred", "rule_accepts",
    ]
    # Если добавится колонка with_complexity_gate (Task 6) — показываем её тоже.
    if "with_complexity_gate" in sweep.columns:
        key_cols.insert(2, "with_complexity_gate")
    available_cols = [c for c in key_cols if c in sweep.columns]
    display(sweep[available_cols].round(4))


# ---------------------------------------------------------------------------
# Section 6: SimpleRouter (Task 5 артефакты — graceful skip если нет файлов)
# ---------------------------------------------------------------------------

def show_simple_router_summary() -> None:
    """SimpleRouter summary per eval_set + reasons breakdown.

    Graceful skip если артефакты не сгенерированы.
    """
    loaded = artifact_io.load_simple_router_results()
    if loaded is None:
        display(Markdown(
            "SimpleRouter артефакты отсутствуют. Запустите "
            "`python -m d1.scripts.evaluate_simple_router`."
        ))
        return
    summary, _ = loaded
    display(Markdown("### SimpleRouter — selective metrics per eval_set"))
    display(summary.round(4))


def show_simple_vs_hybrid() -> None:
    """Side-by-side: SimpleRouter vs B4HybridRouter."""
    loaded = artifact_io.load_simple_router_results()
    if loaded is None:
        display(Markdown(
            "SimpleRouter артефакты отсутствуют. Запустите "
            "`python -m d1.scripts.evaluate_simple_router`."
        ))
        return
    _, vs_df = loaded
    display(Markdown(f"### SimpleRouter vs B4 (production thresholds: `{asdict(PRODUCTION_THRESHOLDS)}`)"))
    display(vs_df.round(4))


def show_complexity_breakdown(eval_set: str = "hard_test") -> None:
    """Per-primary_tag breakdown + subtype breakdown.

    Если оба CSV отсутствуют — graceful skip.
    """
    primary = artifact_io.load_complexity_breakdown(eval_set)
    if primary is None:
        display(Markdown(
            f"`complexity_breakdown_{eval_set}.csv` отсутствует. Запустите "
            "`evaluate_simple_router`."
        ))
        return
    display(Markdown(f"### Complexity breakdown (primary_tag) — {eval_set}"))
    display(primary.round(4))

    subtype = artifact_io.load_complexity_subtype_breakdown(eval_set)
    if subtype is not None:
        display(Markdown(f"### Complexity breakdown (active tags / subtype) — {eval_set}"))
        display(subtype.round(4))


def show_complexity_gate_audit() -> None:
    """Gold-audit метрики ComplexityGate (если sample размечен)."""
    audit = artifact_io.load_complexity_gate_audit()
    if audit is None:
        display(Markdown(
            "Gold-audit отсутствует. Заполните "
            "`d1/data/complexity_audit_sample.csv` и запустите audit (Task 5b)."
        ))
        return
    display(Markdown("### ComplexityGate gold-audit"))
    display(audit.round(4))


# ---------------------------------------------------------------------------
# Section 7: Final report
# ---------------------------------------------------------------------------

def render_final_report() -> None:
    """Динамический итоговый отчёт из актуальных CSV/JSON.

    Метрики, gate check, hard-test delta, latency, disagreement, operational
    status — всё из живых артефактов, без hardcode.
    """
    baseline_df, _ = artifact_io.load_routing_results()
    safety_df_raw = artifact_io.load_safety_results()
    safety_df = pd.DataFrame(safety_df_raw.get("safety_set", []))
    _, vs_df = artifact_io.load_hybrid_results()
    switch_df = artifact_io.load_switch_results()

    baseline_df = baseline_df.copy()
    baseline_df["baseline_id"] = baseline_df["baseline"].map(_baseline_id)
    if not safety_df.empty:
        safety_df["baseline_id"] = safety_df["baseline"].map(_baseline_id)

    # --- Gate check ---
    display(Markdown("### Gate check (§11 ТЗ — leader на test)"))
    gate = _build_gate_table(baseline_df, safety_df)
    display(gate)

    # --- Hard-test summary ---
    display(Markdown("### Hard-test: policy routers"))
    hard_rows = vs_df[vs_df["eval_set"] == "hard_test"]
    if hard_rows.empty:
        display(Markdown("`hard_test` отсутствует в hybrid_vs_selective.csv."))
    else:
        hard = hard_rows.iloc[0]
        delta_df = pd.DataFrame([
            {
                "router": "SelectiveRouter",
                "coverage": float(hard["selective_coverage"]),
                "accepted_accuracy": float(hard["selective_accepted_acc"]),
                "recall_anam": float(hard["selective_recall_anam"]),
                "FN_deferred": int(hard["selective_FN_deferred"]),
            },
            {
                "router": "B4 Hybrid",
                "coverage": float(hard["hybrid_coverage"]),
                "accepted_accuracy": float(hard["hybrid_accepted_acc"]),
                "recall_anam": float(hard["hybrid_recall_anam"]),
                "FN_deferred": int(hard["hybrid_FN_deferred"]),
            },
        ])
        display(delta_df.round(4))

    # --- Latency caveat ---
    test_df = baseline_df[baseline_df["eval_set"] == "test"]
    sparse_latency = test_df[test_df["baseline_id"].str.startswith("B1")]["latency_ms"].median()
    dense_latency = test_df[test_df["baseline_id"].str.startswith("B2")]["latency_ms"].median()
    latency_df = pd.DataFrame([
        {
            "type": "Sparse B1 family",
            "median_latency_ms_test": sparse_latency,
            "interpretation": "offline batch, not online p95",
        },
        {
            "type": "Dense B2 family",
            "median_latency_ms_test": dense_latency,
            "interpretation": "offline batch/post-warmup, online will be higher",
        },
    ])
    display(Markdown("### Latency caveat"))
    display(latency_df.round(4))

    # --- Operational status ---
    display(Markdown("### Operational status"))
    display(pd.DataFrame([
        {"mode": "Shadow mode", "status": "ready"},
        {"mode": "Decision support", "status": "ready"},
        {"mode": "Pre-routing assistant", "status": "ready"},
        {"mode": "Selective router (accept/defer)", "status": "ready for offline/assistive mode"},
        {"mode": "Autonomous routing without fallback", "status": "not ready"},
    ]))

    # --- Interpretation + next steps ---
    prod = asdict(PRODUCTION_THRESHOLDS)
    display(Markdown(
        "### Current interpretation\n"
        f"Production thresholds: `{prod}`. "
        "Результат поддерживает offline/assistive selective routing, "
        "но не автономный routing без fallback."
    ))

    switch_n = int(switch_df["n_samples"].max()) if not switch_df.empty else 0
    next_steps = [
        "SimpleRouter evaluation: complex/multi-intent → defer, simple → ML.",
        "Bootstrap CI / paired significance: SimpleRouter vs B4 Hybrid.",
        "Error taxonomy для remaining failures на hard_test и safety_set.",
    ]
    if switch_n < 50:
        next_steps.append(
            f"Дособрать switch_test: сейчас {switch_n}, цель ≥50."
        )
    display(Markdown(
        "### Next steps\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(next_steps))
    ))


# ---------------------------------------------------------------------------
# Private helpers для final report
# ---------------------------------------------------------------------------

def _baseline_id(name: str) -> str:
    """`B1.1_tfidf_lr @ test` → `B1.1_tfidf_lr`."""
    return str(name).split(" @ ")[0]


def _fmt_status(value: float | None, threshold: float) -> str:
    """PASS/FAIL/n/a в зависимости от threshold."""
    if value is None or pd.isna(value):
        return "n/a"
    return "PASS" if value >= threshold else "FAIL"


def _build_gate_table(baseline_df: pd.DataFrame, safety_df: pd.DataFrame) -> pd.DataFrame:
    """Таблица PASS/FAIL по трём gate criteria для лидера."""
    test_df = baseline_df[baseline_df["eval_set"] == "test"]
    if test_df.empty:
        return pd.DataFrame()
    leader = test_df.sort_values("macro_f1", ascending=False).iloc[0]
    leader_id = leader["baseline_id"]

    leader_recall_urgent: float | None = None
    if not safety_df.empty:
        matches = safety_df[safety_df["baseline_id"] == leader_id]
        if not matches.empty:
            leader_recall_urgent = float(matches.iloc[0]["recall_urgent"])

    rows = [
        {
            "criterion": "macro-F1 on test",
            "threshold": GATE_MACRO_F1,
            "value": float(leader["macro_f1"]),
            "baseline": leader_id,
            "status": _fmt_status(float(leader["macro_f1"]), GATE_MACRO_F1),
        },
        {
            "criterion": "recall(anamnesis) on test",
            "threshold": GATE_RECALL_ANAM,
            "value": float(leader["recall_anamnesis"]),
            "baseline": leader_id,
            "status": _fmt_status(float(leader["recall_anamnesis"]), GATE_RECALL_ANAM),
        },
        {
            "criterion": "recall_urgent on safety_set",
            "threshold": GATE_RECALL_URGENT,
            "value": leader_recall_urgent,
            "baseline": leader_id,
            "status": _fmt_status(leader_recall_urgent, GATE_RECALL_URGENT),
        },
    ]
    return pd.DataFrame(rows).round(4)


__all__ = [
    "GATE_MACRO_F1",
    "GATE_RECALL_ANAM",
    "GATE_RECALL_URGENT",
    "render_final_report",
    "show_baseline_summary",
    "show_calibration_summary",
    "show_confusion_matrix",
    "show_cross_eval_f1",
    "show_hybrid_summary",
    "show_latency",
    "show_pareto_summary",
    "show_per_class_f1",
    "show_complexity_breakdown",
    "show_complexity_gate_audit",
    "show_routing_comparison",
    "show_safety_comparison",
    "show_selective_summary",
    "show_simple_router_summary",
    "show_simple_vs_hybrid",
    "show_split_summary",
]
