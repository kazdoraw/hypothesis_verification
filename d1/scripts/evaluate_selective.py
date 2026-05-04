"""Evaluate SelectiveRouter на всех routing eval sets (Task 4 roadmap).

Orchestration layer:
- загружает fitted B1.1 + B2.1 через TrainedBundle (use_cache=True);
- строит SelectiveRouter с thresholds (дефолт или override через CLI);
- прогоняет через val/test/hard_test/blind_test/entity_held_out/extended_eval;
- для `safety_set` отдельно считает safety_on_accepted;
- сохраняет:
    * `selective_results.json` — все SelectiveReport + closed-set метрики;
    * `selective_results.csv` — плоская сводная таблица для сравнения;
    * `selective_decisions_<eval_set>.csv` — per-sample decision trace.

Запуск:
    cd study && python -m d1.scripts.evaluate_selective
    cd study && python -m d1.scripts.evaluate_selective \\
        --anamnesis-threshold 0.48 --general-threshold 0.63
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    RouteDecision,
    SelectiveReport,
    SelectiveRouter,
    SelectiveThresholds,
    compute_accepted_only_report,
    compute_selective_report,
)
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

# Eval sets для selective eval — только routing-сеты (safety отдельно).
DEFAULT_EVAL_SETS = [
    "val", "test", "hard_test",
    "blind_test", "entity_held_out", "extended_eval",
]

# Дефолтные имена моделей — SSoT Task 4 (B1.1 sparse + B2.1 dense).
DEFAULT_SPARSE = "B1.1_tfidf_lr"
DEFAULT_DENSE = "B2.1_bge-m3_svc"


# ---------------------------------------------------------------------------
# Data loading (reuse логики run_baselines.py, но лёгкий импорт без side effects)
# ---------------------------------------------------------------------------

def _load_split(name: str) -> pd.DataFrame:
    """Загрузка split CSV по имени."""
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Decision trace persistence
# ---------------------------------------------------------------------------

def _decisions_to_csv_rows(
    texts: list[str], y_true: list[str], decisions: list[RouteDecision],
) -> list[dict[str, Any]]:
    """Per-sample trace для аудита (коротко, без полного текста во избежание PII-утечек)."""
    rows: list[dict[str, Any]] = []
    for text, gold, d in zip(texts, y_true, decisions):
        rows.append({
            "text_preview": text[:80],
            "gold": gold,
            "predicted": d.label,
            "action": d.action,
            "reason": d.reason,
            "confidence": round(d.confidence, 4),
            "margin": round(d.margin, 4),
            "sparse_dense_agree": d.sparse_dense_agree,
            "dense_label": d.dense_label,
            "correct": d.action == "accept" and d.label == gold,
        })
    return rows


# ---------------------------------------------------------------------------
# Orchestration: evaluate_selective
# ---------------------------------------------------------------------------

def evaluate_selective(
    eval_sets: list[str] | None = None,
    thresholds: SelectiveThresholds | None = None,
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
    out_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Прогнать SelectiveRouter на всех eval sets, сохранить артефакты.

    Returns:
        dict eval_set → {selective: SelectiveReport.summary_dict,
                         accepted_only_metrics: RoutingReport.summary_dict}
    """
    eval_sets = eval_sets or DEFAULT_EVAL_SETS
    thresholds = thresholds or PRODUCTION_THRESHOLDS
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Bundle + router (SSoT) ---
    logger.info("=== Loading TrainedBundle: sparse=%s, dense=%s ===",
                sparse_name, dense_name)
    bundle = train_bundle(names=[sparse_name, dense_name], use_cache=True)
    router = SelectiveRouter(
        sparse_model=bundle.get(sparse_name),
        dense_model=bundle.get(dense_name),
        thresholds=thresholds,
    )
    router_name = f"{sparse_name}+{dense_name}_selective"

    # --- Run eval ---
    results: dict[str, dict[str, Any]] = {}
    all_csv_rows: list[dict[str, Any]] = []

    for eval_set in eval_sets:
        df = _load_split(eval_set)
        texts = df["text"].tolist()
        y_true = df["route_domain"].tolist()

        decisions = router.route_batch(texts)

        selective_rep = compute_selective_report(
            y_true=y_true, decisions=decisions,
            router_name=router_name, thresholds=thresholds,
        )
        accepted_rep = compute_accepted_only_report(
            y_true=y_true, decisions=decisions,
            baseline_name=f"{router_name}_accepted_only",
        )

        results[eval_set] = {
            "selective": selective_rep.summary_dict(),
            "accepted_only_metrics": accepted_rep.summary_dict(),
        }

        # Per-sample trace CSV
        trace_path = out_dir / f"selective_decisions_{eval_set}.csv"
        pd.DataFrame(_decisions_to_csv_rows(texts, y_true, decisions)).to_csv(
            trace_path, index=False,
        )
        logger.info("Saved: %s", trace_path)

        # Flat row for summary CSV
        summary_row = {
            "eval_set": eval_set,
            "router": router_name,
            "n": selective_rep.n_samples,
            "coverage": round(selective_rep.coverage, 4),
            "accepted_accuracy": round(selective_rep.accepted_accuracy, 4),
            "accepted_recall_anam": round(selective_rep.accepted_recall_anamnesis, 4),
            "defer_rate": round(selective_rep.defer_rate, 4),
            "false_negative_deferred": selective_rep.false_negative_deferred,
            "accepted_subset_accuracy": round(accepted_rep.accuracy, 4),
            "accepted_subset_macro_f1": round(accepted_rep.macro_f1, 4),
        }
        all_csv_rows.append(summary_row)

    # --- Save summaries ---
    json_path = out_dir / "selective_results.json"
    json_data: dict[str, Any] = {
        "router_name": router_name,
        "thresholds": asdict(thresholds),
        "eval_sets": eval_sets,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    csv_path = out_dir / "selective_results.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    # --- Print summary ---
    print("\n" + "=" * 88)
    print(f"  SELECTIVE ROUTING SUMMARY  router={router_name}")
    print(f"  thresholds={asdict(thresholds)}")
    print("=" * 88)
    df = pd.DataFrame(all_csv_rows)
    print(df.to_string(index=False))
    print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="D1 v6 SelectiveRouter evaluation")
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
    evaluate_selective(
        eval_sets=args.eval_sets,
        thresholds=thresholds,
        sparse_name=args.sparse,
        dense_name=args.dense,
    )


if __name__ == "__main__":
    main()
