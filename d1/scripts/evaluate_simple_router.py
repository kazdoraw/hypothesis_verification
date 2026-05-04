"""Evaluate SimpleRouter vs B4HybridRouter (Task 5 плана).

SimpleRouter = `ComplexityGate → B4HybridRouter` cascade. Оцениваем coverage,
accepted accuracy/recall, defer split (complexity vs hybrid), descriptive
complexity breakdown по primary_tag и subtype.

Артефакты (см. plan Task 5):
- `simple_router_results.csv` — summary per eval_set.
- `simple_router_results.json` — full report.
- `simple_router_decisions_<eval_set>.csv` — per-sample trace + primary_tag/active_tags.
- `complexity_breakdown_<eval_set>.csv` — per-primary_tag aggregation.
- `complexity_subtype_breakdown_<eval_set>.csv` — per-subtype-tag aggregation.
- `simple_vs_hybrid.csv` — side-by-side.

Запуск:
    cd study && python -m d1.scripts.evaluate_simple_router
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from d1.baselines.b4_hybrid import B4HybridRouter
from d1.baselines.complexity_gate import ComplexityDecision, ComplexityGate
from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    RouteDecision,
    SelectiveRouter,
    SelectiveThresholds,
    compute_accepted_only_report,
    compute_selective_report,
)
from d1.baselines.simple_router import SimpleRouter
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SETS = [
    "val", "test", "hard_test",
    "blind_test", "entity_held_out", "extended_eval",
]
DEFAULT_SPARSE = "B1.1_tfidf_lr"
DEFAULT_DENSE = "B2.1_bge-m3_svc"
RULES_NAME = "B0_rules"


# ---------------------------------------------------------------------------
# Pure helpers (тестируются изолированно от train_bundle / I/O)
# ---------------------------------------------------------------------------

def _is_complexity_defer(decision: RouteDecision) -> bool:
    """Differentiate complexity-defer от ML-defer по reason prefix."""
    return decision.action == "defer" and decision.reason.startswith("complexity:")


def _is_tag_policy_defer(decision: RouteDecision) -> bool:
    """Tag-aware post-ML guard defer (`tag_policy:<primary_tag>`)."""
    return decision.action == "defer" and decision.reason.startswith("tag_policy:")


def _decisions_to_trace_rows(
    texts: list[str],
    y_true: list[str],
    decisions: list[RouteDecision],
    gate_decisions: list[ComplexityDecision],
) -> list[dict[str, Any]]:
    """Per-sample trace для simple_router_decisions_<eval>.csv.

    Расширяет hybrid trace формат двумя колонками:
    - `primary_tag` (str): результат ComplexityGate.
    - `active_tags` (str, semicolon-joined): все True tags из gate.
    """
    if not (len(texts) == len(y_true) == len(decisions) == len(gate_decisions)):
        raise ValueError(
            "_decisions_to_trace_rows: длины texts/y_true/decisions/gate_decisions "
            "должны совпадать",
        )
    rows: list[dict[str, Any]] = []
    for text, gold, d, gd in zip(texts, y_true, decisions, gate_decisions):
        active = sorted(name for name, on in gd.tags.items() if on)
        rows.append({
            "text_preview": text[:80],
            "gold": gold,
            "predicted": d.label,
            "action": d.action,
            "reason": d.reason,
            "is_rule_accept": d.reason.startswith("rule:"),
            "is_complexity_defer": _is_complexity_defer(d),
            "is_tag_policy_defer": _is_tag_policy_defer(d),
            "confidence": round(d.confidence, 4),
            "margin": round(d.margin, 4),
            "correct": d.action == "accept" and d.label == gold,
            "primary_tag": gd.primary_tag,
            "active_tags": ";".join(active),
        })
    return rows


def _compute_summary_row(
    eval_set: str,
    router_name: str,
    n: int,
    y_true: list[str],
    decisions: list[RouteDecision],
) -> dict[str, Any]:
    """Selective-style summary row + complexity_defer_rate / hybrid_defer_rate.

    Чистая функция: не пишет файлы, не читает датасеты.
    """
    if n != len(y_true) or n != len(decisions):
        raise ValueError(
            f"_compute_summary_row: n={n} != len(y_true)={len(y_true)} "
            f"!= len(decisions)={len(decisions)}",
        )
    n_accept = sum(1 for d in decisions if d.action == "accept")
    n_defer = n - n_accept
    n_complexity = sum(1 for d in decisions if _is_complexity_defer(d))
    n_tag_policy = sum(1 for d in decisions if _is_tag_policy_defer(d))
    n_hybrid = n_defer - n_complexity - n_tag_policy

    correct_accept = sum(
        1 for d, g in zip(decisions, y_true)
        if d.action == "accept" and d.label == g
    )
    accepted_acc = correct_accept / n_accept if n_accept else 0.0

    # accepted recall на anamnesis (selective-style).
    n_anam_total_accepted = sum(
        1 for d, g in zip(decisions, y_true)
        if d.action == "accept" and g == "anamnesis"
    )
    n_anam_correct_accepted = sum(
        1 for d, g in zip(decisions, y_true)
        if d.action == "accept" and g == "anamnesis" and d.label == "anamnesis"
    )
    accepted_recall_anam = (
        n_anam_correct_accepted / n_anam_total_accepted
        if n_anam_total_accepted else 0.0
    )

    # FN_deferred — gold=anamnesis, ушедшие в defer (любого типа).
    fn_deferred = sum(
        1 for d, g in zip(decisions, y_true)
        if d.action == "defer" and g == "anamnesis"
    )

    return {
        "eval_set": eval_set,
        "router": router_name,
        "n": n,
        "coverage": round(n_accept / n if n else 0.0, 4),
        "accepted_accuracy": round(accepted_acc, 4),
        "accepted_recall_anam": round(accepted_recall_anam, 4),
        "defer_rate": round(n_defer / n if n else 0.0, 4),
        "complexity_defer_rate": round(n_complexity / n if n else 0.0, 4),
        "tag_policy_defer_rate": round(n_tag_policy / n if n else 0.0, 4),
        "hybrid_defer_rate": round(n_hybrid / n if n else 0.0, 4),
        "false_negative_deferred": fn_deferred,
    }


def _build_complexity_breakdown(
    y_true: list[str],
    decisions: list[RouteDecision],
    gate_decisions: list[ComplexityDecision],
) -> pd.DataFrame:
    """Per-primary_tag aggregation для complexity_breakdown_<eval>.csv.

    Колонки: `primary_tag`, `n`, `n_accept`, `n_defer_complexity`,
    `n_defer_tag_policy`, `n_defer_ml`, `accepted_accuracy`,
    `accepted_recall_anamnesis`.
    """
    # Группировка по primary_tag.
    rows: list[dict[str, Any]] = []
    primary_tags = sorted({gd.primary_tag for gd in gate_decisions})
    for tag in primary_tags:
        idx = [i for i, gd in enumerate(gate_decisions) if gd.primary_tag == tag]
        rows.append(_aggregate_breakdown_row("primary_tag", tag, idx, y_true, decisions))
    return pd.DataFrame(rows)


def _build_subtype_breakdown(
    y_true: list[str],
    decisions: list[RouteDecision],
    gate_decisions: list[ComplexityDecision],
) -> pd.DataFrame:
    """Per-active-tag aggregation (independent of primary_tag).

    Учитывает все tags=True в gate_decisions.tags, что позволяет видеть
    subtype-распределение внутри `mixed_intent` bucket'а.
    """
    rows: list[dict[str, Any]] = []
    # Собираем все tag-имена (фиксированный набор из ComplexityGate).
    all_tags: set[str] = set()
    for gd in gate_decisions:
        all_tags.update(gd.tags.keys())
    for tag in sorted(all_tags):
        idx = [i for i, gd in enumerate(gate_decisions) if gd.tags.get(tag)]
        if not idx:
            continue
        rows.append(_aggregate_breakdown_row("tag", tag, idx, y_true, decisions))
    return pd.DataFrame(rows)


def _aggregate_breakdown_row(
    label_col: str,
    label_value: str,
    indices: list[int],
    y_true: list[str],
    decisions: list[RouteDecision],
) -> dict[str, Any]:
    """Общий aggregate для primary_tag и subtype breakdown."""
    n = len(indices)
    n_accept = sum(1 for i in indices if decisions[i].action == "accept")
    n_def_cx = sum(1 for i in indices if _is_complexity_defer(decisions[i]))
    n_def_tag = sum(1 for i in indices if _is_tag_policy_defer(decisions[i]))
    n_def_ml = n - n_accept - n_def_cx - n_def_tag

    correct_accept = sum(
        1 for i in indices
        if decisions[i].action == "accept" and decisions[i].label == y_true[i]
    )
    accepted_acc = correct_accept / n_accept if n_accept else 0.0

    n_anam_acc_total = sum(
        1 for i in indices
        if decisions[i].action == "accept" and y_true[i] == "anamnesis"
    )
    n_anam_acc_correct = sum(
        1 for i in indices
        if decisions[i].action == "accept"
        and y_true[i] == "anamnesis"
        and decisions[i].label == "anamnesis"
    )
    accepted_recall_anam = (
        n_anam_acc_correct / n_anam_acc_total if n_anam_acc_total else 0.0
    )

    return {
        label_col: label_value,
        "n": n,
        "n_accept": n_accept,
        "n_defer_complexity": n_def_cx,
        "n_defer_tag_policy": n_def_tag,
        "n_defer_ml": n_def_ml,
        "accepted_accuracy": round(accepted_acc, 4),
        "accepted_recall_anamnesis": round(accepted_recall_anam, 4),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Phase 1: accepted-error slice (post-hoc, без перезапуска SimpleRouter).
# ---------------------------------------------------------------------------

ACCEPTED_ERROR_COLUMNS: tuple[str, ...] = (
    "text_preview", "gold", "predicted",
    "primary_tag", "confidence", "margin", "reason",
)


def build_accepted_error_slice(
    eval_set: str,
    decisions_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Filter accepted-errors из simple_router_decisions_<eval>.csv.

    Accepted-error = (action == "accept") AND (correct == False). Не дублирует
    `build_complexity_audit_sample` (тот про defer-разметку); тут — фокус на
    ошибках уже принятых ML моделью, чтобы локализовать причины снижения
    accepted_acc для Phase 2/3.

    Args:
        eval_set: имя split (test, blind_test, hard_test, extended_eval, ...).
        decisions_path: опциональный путь к decisions CSV. По умолчанию —
            `RESULTS_DIR / f"simple_router_decisions_{eval_set}.csv"`.
        out_dir: куда писать `accepted_errors_<eval>.csv`. По умолчанию RESULTS_DIR.

    Returns:
        DataFrame с колонками `ACCEPTED_ERROR_COLUMNS`.
    """
    decisions_path = decisions_path or (
        RESULTS_DIR / f"simple_router_decisions_{eval_set}.csv"
    )
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"decisions CSV не найден: {decisions_path}. "
            "Сначала прогоните evaluate_simple_router.",
        )
    df = pd.read_csv(decisions_path)
    # `correct` пишется bool через pandas; поддержим оба варианта чтения.
    correct_mask = df["correct"].astype(str).str.lower().isin({"true", "1"})
    errors = df[(df["action"] == "accept") & ~correct_mask].copy()
    cols = list(ACCEPTED_ERROR_COLUMNS)
    errors = errors.reindex(columns=cols)

    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"accepted_errors_{eval_set}.csv"
    errors.to_csv(out_path, index=False)
    logger.info("Saved: %s (n=%d)", out_path, len(errors))
    return errors


def _build_routers(
    thresholds: SelectiveThresholds,
    sparse_name: str,
    dense_name: str,
) -> tuple[SelectiveRouter, B4HybridRouter, SimpleRouter]:
    """Построить три роутера на одних fitted моделях."""
    logger.info(
        "Loading TrainedBundle: rules=%s, sparse=%s, dense=%s",
        RULES_NAME, sparse_name, dense_name,
    )
    bundle = train_bundle(
        names=[RULES_NAME, sparse_name, dense_name],
        use_cache=True,
    )
    selective = SelectiveRouter(
        sparse_model=bundle.get(sparse_name),
        dense_model=bundle.get(dense_name),
        thresholds=thresholds,
    )
    hybrid = B4HybridRouter(bundle=bundle, selective=selective)
    simple = SimpleRouter(hybrid=hybrid, complexity_gate=ComplexityGate())
    return selective, hybrid, simple


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_simple_router(
    eval_sets: list[str] | None = None,
    thresholds: SelectiveThresholds | None = None,
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
    out_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Прогон SimpleRouter + B4HybridRouter side-by-side.

    Returns:
        dict eval_set → {"simple": ..., "hybrid": ...} с selective metrics.
    """
    eval_sets = eval_sets or DEFAULT_EVAL_SETS
    thresholds = thresholds or PRODUCTION_THRESHOLDS
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _, hybrid, simple = _build_routers(thresholds, sparse_name, dense_name)
    simple_name = "SimpleRouter[ComplexityGate->B4]"
    hybrid_name = "B4HybridRouter"

    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    gate = ComplexityGate()  # для recovery gate_decisions для tracing/breakdown

    for eval_set in eval_sets:
        df = _load_split(eval_set)
        texts = df["text"].tolist()
        y_true = df["route_domain"].tolist()
        n = len(texts)

        gate_decisions = gate.decide_batch(texts)

        # --- SimpleRouter ---
        simple_decisions = simple.route_batch(texts)
        simple_summary = _compute_summary_row(
            eval_set, simple_name, n, y_true, simple_decisions,
        )
        simple_selective_report = compute_selective_report(
            y_true=y_true, decisions=simple_decisions,
            router_name=simple_name, thresholds=thresholds,
        )
        simple_accepted_report = compute_accepted_only_report(
            y_true=y_true, decisions=simple_decisions,
            baseline_name=f"{simple_name}_accepted_only",
        )

        # --- B4 baseline ---
        hybrid_decisions = hybrid.route_batch(texts)
        hybrid_summary = _compute_summary_row(
            eval_set, hybrid_name, n, y_true, hybrid_decisions,
        )
        hybrid_selective_report = compute_selective_report(
            y_true=y_true, decisions=hybrid_decisions,
            router_name=hybrid_name, thresholds=thresholds,
        )

        results[eval_set] = {
            "simple": {
                "selective": simple_selective_report.summary_dict(),
                "accepted_only_metrics": simple_accepted_report.summary_dict(),
            },
            "hybrid": {
                "selective": hybrid_selective_report.summary_dict(),
            },
        }

        # --- Per-sample trace (SimpleRouter) ---
        trace_path = out_dir / f"simple_router_decisions_{eval_set}.csv"
        pd.DataFrame(_decisions_to_trace_rows(
            texts, y_true, simple_decisions, gate_decisions,
        )).to_csv(trace_path, index=False)
        logger.info("Saved: %s", trace_path)

        # --- Complexity breakdown (primary_tag) ---
        breakdown_path = out_dir / f"complexity_breakdown_{eval_set}.csv"
        _build_complexity_breakdown(y_true, simple_decisions, gate_decisions).to_csv(
            breakdown_path, index=False,
        )
        logger.info("Saved: %s", breakdown_path)

        # --- Subtype breakdown (active tags) ---
        subtype_path = out_dir / f"complexity_subtype_breakdown_{eval_set}.csv"
        _build_subtype_breakdown(y_true, simple_decisions, gate_decisions).to_csv(
            subtype_path, index=False,
        )
        logger.info("Saved: %s", subtype_path)

        # Summary CSV row.
        summary_rows.append(simple_summary)

        # Side-by-side comparison.
        compare_rows.append({
            "eval_set": eval_set,
            "n": n,
            "simple_coverage": simple_summary["coverage"],
            "hybrid_coverage": hybrid_summary["coverage"],
            "delta_coverage": round(
                simple_summary["coverage"] - hybrid_summary["coverage"], 4,
            ),
            "simple_accepted_acc": simple_summary["accepted_accuracy"],
            "hybrid_accepted_acc": hybrid_summary["accepted_accuracy"],
            "delta_accepted_acc": round(
                simple_summary["accepted_accuracy"]
                - hybrid_summary["accepted_accuracy"], 4,
            ),
            "simple_recall_anam": simple_summary["accepted_recall_anam"],
            "hybrid_recall_anam": hybrid_summary["accepted_recall_anam"],
            "simple_FN_deferred": simple_summary["false_negative_deferred"],
            "hybrid_FN_deferred": hybrid_summary["false_negative_deferred"],
            "complexity_defer_rate": simple_summary["complexity_defer_rate"],
            "tag_policy_defer_rate": simple_summary["tag_policy_defer_rate"],
        })

    # --- Save aggregates ---
    summary_path = out_dir / "simple_router_results.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info("Saved: %s", summary_path)

    json_path = out_dir / "simple_router_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "simple_name": simple_name,
            "hybrid_name": hybrid_name,
            "thresholds": asdict(thresholds),
            "eval_sets": eval_sets,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    compare_path = out_dir / "simple_vs_hybrid.csv"
    pd.DataFrame(compare_rows).to_csv(compare_path, index=False)
    logger.info("Saved: %s", compare_path)

    # --- Accepted-error slices (per eval_set) ---
    # Генерируется автоматически: иначе файлы accepted_errors_*.csv остаются
    # с устаревшими данными от предыдущих прогонов.
    for eval_set in eval_sets:
        try:
            build_accepted_error_slice(eval_set, out_dir=out_dir)
        except FileNotFoundError as exc:
            logger.warning(
                "accepted_errors для %s не сгенерирован: %s", eval_set, exc,
            )

    print("\n" + "=" * 110)
    print(f"  SIMPLE vs HYBRID  thresholds={asdict(thresholds)}")
    print("=" * 110)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1 v6 SimpleRouter (ComplexityGate→B4) evaluation",
    )
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument(
        "--sparse", default=DEFAULT_SPARSE,
        help="имя sparse baseline (см. BASELINE_CONFIGS)",
    )
    parser.add_argument(
        "--dense", default=DEFAULT_DENSE,
        help="имя dense baseline (см. BASELINE_CONFIGS)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="директория для артефактов (default: d1/results). "
             "Используйте для cascade comparison, чтобы не перезаписывать.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    evaluate_simple_router(
        eval_sets=args.eval_sets,
        sparse_name=args.sparse,
        dense_name=args.dense,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
