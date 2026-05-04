"""Evaluate B4HybridRouter vs SelectiveRouter (Task 5 roadmap).

Orchestration layer для сравнения двух policy:
- **SelectiveRouter** (Task 4) — baseline: ML-only cascade.
- **B4HybridRouter** (Task 5) — rules-first cascade + ML fallback.

Для обоих прогоняем одинаковые eval sets с одинаковыми thresholds, чтобы
выделить чистый эффект rules-first на coverage и safety.

Артефакты (отдельные от closed-set, по требованию плана §5):
- `hybrid_results.json` — per-eval_set SelectiveReport + closed-set report для B4.
- `hybrid_results.csv` — плоская сводная таблица.
- `hybrid_vs_selective.csv` — side-by-side сравнение (coverage, accuracy, safety).
- `hybrid_decisions_<eval_set>.csv` — per-sample trace.

Запуск:
    cd study && python -m d1.scripts.evaluate_hybrid
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
from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    RouteDecision,
    SelectiveRouter,
    SelectiveThresholds,
    compute_accepted_only_report,
    compute_selective_report,
)
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


def _load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _decisions_to_csv_rows(
    texts: list[str],
    y_true: list[str],
    decisions: list[RouteDecision],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text, gold, d in zip(texts, y_true, decisions):
        rows.append({
            "text_preview": text[:80],
            "gold": gold,
            "predicted": d.label,
            "action": d.action,
            "reason": d.reason,
            "is_rule_accept": d.reason.startswith("rule:"),
            "confidence": round(d.confidence, 4),
            "margin": round(d.margin, 4),
            "correct": d.action == "accept" and d.label == gold,
        })
    return rows


def _build_routers(
    thresholds: SelectiveThresholds,
    sparse_name: str,
    dense_name: str,
) -> tuple[SelectiveRouter, B4HybridRouter]:
    """Построить оба роутера на одних и тех же fitted моделях."""
    logger.info("=== Loading TrainedBundle: rules=%s, sparse=%s, dense=%s ===",
                RULES_NAME, sparse_name, dense_name)
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
    return selective, hybrid


def evaluate_hybrid(
    eval_sets: list[str] | None = None,
    thresholds: SelectiveThresholds | None = None,
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
    out_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Прогон B4HybridRouter + SelectiveRouter бок о бок.

    Returns:
        dict eval_set → {"hybrid": {selective, accepted_only_metrics},
                         "selective": {selective, accepted_only_metrics}}
    """
    eval_sets = eval_sets or DEFAULT_EVAL_SETS
    thresholds = thresholds or PRODUCTION_THRESHOLDS
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    selective, hybrid = _build_routers(thresholds, sparse_name, dense_name)
    hybrid_name = f"B4_hybrid[{sparse_name}+{dense_name}]"
    selective_name = f"{sparse_name}+{dense_name}_selective"

    results: dict[str, dict[str, Any]] = {}
    hybrid_csv_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    for eval_set in eval_sets:
        df = _load_split(eval_set)
        texts = df["text"].tolist()
        y_true = df["route_domain"].tolist()

        # --- B4 hybrid ---
        h_decisions = hybrid.route_batch(texts)
        h_selective = compute_selective_report(
            y_true=y_true, decisions=h_decisions,
            router_name=hybrid_name, thresholds=thresholds,
        )
        h_accepted = compute_accepted_only_report(
            y_true=y_true, decisions=h_decisions,
            baseline_name=f"{hybrid_name}_accepted_only",
        )

        # --- Selective baseline (same run для честного сравнения) ---
        s_decisions = selective.route_batch(texts)
        s_selective = compute_selective_report(
            y_true=y_true, decisions=s_decisions,
            router_name=selective_name, thresholds=thresholds,
        )
        s_accepted = compute_accepted_only_report(
            y_true=y_true, decisions=s_decisions,
            baseline_name=f"{selective_name}_accepted_only",
        )

        results[eval_set] = {
            "hybrid": {
                "selective": h_selective.summary_dict(),
                "accepted_only_metrics": h_accepted.summary_dict(),
            },
            "selective": {
                "selective": s_selective.summary_dict(),
                "accepted_only_metrics": s_accepted.summary_dict(),
            },
        }

        # Per-sample trace — только для hybrid (selective trace уже сохранён
        # в evaluate_selective.py; здесь важна именно cascade-трассировка).
        trace_path = out_dir / f"hybrid_decisions_{eval_set}.csv"
        pd.DataFrame(_decisions_to_csv_rows(texts, y_true, h_decisions)).to_csv(
            trace_path, index=False,
        )
        logger.info("Saved: %s", trace_path)

        # Flat row for hybrid summary CSV
        hybrid_csv_rows.append({
            "eval_set": eval_set,
            "router": hybrid_name,
            "n": h_selective.n_samples,
            "coverage": round(h_selective.coverage, 4),
            "accepted_accuracy": round(h_selective.accepted_accuracy, 4),
            "accepted_recall_anam": round(h_selective.accepted_recall_anamnesis, 4),
            "defer_rate": round(h_selective.defer_rate, 4),
            "false_negative_deferred": h_selective.false_negative_deferred,
            "rule_accept_count": sum(
                1 for d in h_decisions if d.reason.startswith("rule:")
            ),
            "accepted_subset_accuracy": round(h_accepted.accuracy, 4),
            "accepted_subset_macro_f1": round(h_accepted.macro_f1, 4),
        })

        # Side-by-side comparison
        compare_rows.append({
            "eval_set": eval_set,
            "n": h_selective.n_samples,
            "hybrid_coverage": round(h_selective.coverage, 4),
            "selective_coverage": round(s_selective.coverage, 4),
            "delta_coverage": round(h_selective.coverage - s_selective.coverage, 4),
            "hybrid_accepted_acc": round(h_selective.accepted_accuracy, 4),
            "selective_accepted_acc": round(s_selective.accepted_accuracy, 4),
            "delta_accepted_acc": round(
                h_selective.accepted_accuracy - s_selective.accepted_accuracy, 4,
            ),
            "hybrid_recall_anam": round(h_selective.accepted_recall_anamnesis, 4),
            "selective_recall_anam": round(s_selective.accepted_recall_anamnesis, 4),
            "hybrid_FN_deferred": h_selective.false_negative_deferred,
            "selective_FN_deferred": s_selective.false_negative_deferred,
            "rule_accept_count": sum(
                1 for d in h_decisions if d.reason.startswith("rule:")
            ),
        })

    # --- Save summaries ---
    json_path = out_dir / "hybrid_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "hybrid_name": hybrid_name,
            "selective_name": selective_name,
            "thresholds": asdict(thresholds),
            "eval_sets": eval_sets,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    csv_path = out_dir / "hybrid_results.csv"
    pd.DataFrame(hybrid_csv_rows).to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    compare_path = out_dir / "hybrid_vs_selective.csv"
    pd.DataFrame(compare_rows).to_csv(compare_path, index=False)
    logger.info("Saved: %s", compare_path)

    # --- Print comparison ---
    print("\n" + "=" * 110)
    print(f"  HYBRID vs SELECTIVE  thresholds={asdict(thresholds)}")
    print("=" * 110)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1 v6 B4HybridRouter vs SelectiveRouter evaluation",
    )
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--sparse", default=DEFAULT_SPARSE)
    parser.add_argument("--dense", default=DEFAULT_DENSE)
    parser.add_argument(
        "--anamnesis-threshold",
        type=float,
        default=PRODUCTION_THRESHOLDS.anamnesis_threshold,
    )
    parser.add_argument(
        "--faq-anamnesis-margin",
        type=float,
        default=PRODUCTION_THRESHOLDS.faq_anamnesis_margin,
    )
    parser.add_argument(
        "--general-threshold",
        type=float,
        default=PRODUCTION_THRESHOLDS.general_threshold,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    thresholds = SelectiveThresholds(
        anamnesis_threshold=args.anamnesis_threshold,
        faq_anamnesis_margin=args.faq_anamnesis_margin,
        general_threshold=args.general_threshold,
    )
    evaluate_hybrid(
        eval_sets=args.eval_sets,
        thresholds=thresholds,
        sparse_name=args.sparse,
        dense_name=args.dense,
    )


if __name__ == "__main__":
    main()
