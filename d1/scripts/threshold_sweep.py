"""Threshold sweep для SelectiveRouter + B4HybridRouter (Task 5 retune + Task 6).

Прогоняет несколько threshold configs параллельно и собирает сводку
для выбора production thresholds. Task 6: параметр with_complexity_gate —
при True оборачивает SelectiveRouter/B4HybridRouter в SimpleRouter (ComplexityGate cascade).

Каждый config оценивается на val/test/hard_test (без overfitting на blind).
Результат:
- `threshold_sweep_results.csv` — все (config × eval_set × router × with_complexity_gate) точки.
- `threshold_sweep_summary.csv` — pivot по (eval_set, router, with_complexity_gate) × config.

Запуск:
    cd study && python -m d1.scripts.threshold_sweep                    # оба варианта
    cd study && python -m d1.scripts.threshold_sweep --no-complexity-gate # без gate
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from d1.baselines.b4_hybrid import B4HybridRouter
from d1.baselines.complexity_gate import ComplexityGate
from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    SelectiveRouter,
    SelectiveThresholds,
    compute_accepted_only_report,
    compute_selective_report,
)
from d1.baselines.simple_router import SimpleRouter
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)


# Конфиги для сравнения. Обоснование в комментариях.
# SSoT: `pareto_candidates_b1_1_tf-idf_lr_val.csv` (sparse) для anamnesis_threshold
# и general_threshold (оба применяются к sparse top1 proba в SelectiveRouter).
THRESHOLD_CONFIGS: list[tuple[str, SelectiveThresholds]] = [
    # Baseline (стартовые дефолты Task 4 до production retune).
    ("baseline_0.55_0.70", SelectiveThresholds(
        anamnesis_threshold=0.55, faq_anamnesis_margin=0.15, general_threshold=0.70,
    )),
    # Production config: SSoT после retune, см. selective_router.PRODUCTION_THRESHOLDS.
    ("moderate_0.48_0.63", PRODUCTION_THRESHOLDS),
    # Aggressive coverage: sparse pareto @ t=0.43 для anamnesis (cov 83%, acc 90%).
    # Оставляем faq_anamnesis_margin строгим: safety guard против faq-override.
    ("aggressive_0.43_0.55", SelectiveThresholds(
        anamnesis_threshold=0.43, faq_anamnesis_margin=0.15, general_threshold=0.55,
    )),
]

EVAL_SETS = ["val", "test", "hard_test"]
SPARSE_NAME = "B1.1_tfidf_lr"
DENSE_NAME = "B2.1_bge-m3_svc"
RULES_NAME = "B0_rules"


def _load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    return pd.read_csv(path, dtype=str).fillna("")


def sweep(with_complexity_gate: bool | None = None) -> pd.DataFrame:
    """Прогон (config × eval_set × router × with_complexity_gate).

    Args:
        with_complexity_gate: если None — прогоняются оба варианта (False+True);
            если True/False — только указанный. При True base router оборачивается
            в SimpleRouter (только hybrid wrapping имеет смысл: SimpleRouter внутри
            использует B4HybridRouter; selective прогоняется без изменений).

    Returns:
        DataFrame с колонкой `with_complexity_gate: bool`.
    """
    gate_variants: list[bool]
    if with_complexity_gate is None:
        gate_variants = [False, True]
    else:
        gate_variants = [with_complexity_gate]

    # SSoT: один bundle — все routers одинаковые fitted модели.
    bundle = train_bundle(
        names=[RULES_NAME, SPARSE_NAME, DENSE_NAME],
        use_cache=True,
    )

    rows: list[dict] = []
    for config_name, thresholds in THRESHOLD_CONFIGS:
        logger.info("=== Config: %s ===", config_name)
        selective = SelectiveRouter(
            sparse_model=bundle.get(SPARSE_NAME),
            dense_model=bundle.get(DENSE_NAME),
            thresholds=thresholds,
        )
        hybrid = B4HybridRouter(bundle=bundle, selective=selective)

        for eval_set in EVAL_SETS:
            df = _load_split(eval_set)
            texts = df["text"].tolist()
            y_true = df["route_domain"].tolist()

            for gate_on in gate_variants:
                # SimpleRouter оборачивает только hybrid (он внутри вызывает selective).
                # Для selective gate без hybrid не имеет продуктового смысла,
                # но для баланса pivot-таблицы пропускаем selective+gate.
                router_pairs: list[tuple[str, object]] = []
                if gate_on:
                    simple = SimpleRouter(
                        hybrid=hybrid, complexity_gate=ComplexityGate(),
                    )
                    router_pairs.append(("hybrid", simple))
                else:
                    router_pairs.append(("selective", selective))
                    router_pairs.append(("hybrid", hybrid))

                for router_label, router in router_pairs:
                    decisions = router.route_batch(texts)
                    sel = compute_selective_report(
                        y_true=y_true, decisions=decisions,
                        router_name=f"{router_label}_{config_name}_gate{gate_on}",
                        thresholds=thresholds,
                    )
                    acc = compute_accepted_only_report(
                        y_true=y_true, decisions=decisions,
                    )
                    rows.append({
                        "config": config_name,
                        **asdict(thresholds),
                        "eval_set": eval_set,
                        "router": router_label,
                        "with_complexity_gate": gate_on,
                        "n": sel.n_samples,
                        "coverage": round(sel.coverage, 4),
                        "accepted_accuracy": round(sel.accepted_accuracy, 4),
                        "accepted_recall_anam": round(sel.accepted_recall_anamnesis, 4),
                        "defer_rate": round(sel.defer_rate, 4),
                        "FN_deferred": sel.false_negative_deferred,
                        "accepted_subset_macro_f1": round(acc.macro_f1, 4),
                        "rule_accepts": sum(
                            1 for d in decisions if d.reason.startswith("rule:")
                        ),
                    })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 v6 threshold sweep")
    parser.add_argument(
        "--no-complexity-gate",
        dest="complexity_gate",
        action="store_false",
        default=None,
        help="Отключить gate-вариант (default: прогон обоих)",
    )
    parser.add_argument(
        "--only-complexity-gate",
        dest="complexity_gate",
        action="store_true",
        help="Только gate-вариант",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = sweep(with_complexity_gate=args.complexity_gate)

    full_path = out_dir / "threshold_sweep_results.csv"
    df.to_csv(full_path, index=False)
    logger.info("Saved: %s", full_path)

    # Pivot по (eval_set, router, with_complexity_gate) × config
    metric_cols = ["coverage", "accepted_accuracy", "accepted_recall_anam", "FN_deferred"]
    pivot = df.pivot_table(
        index=["eval_set", "router", "with_complexity_gate"],
        columns="config",
        values=metric_cols,
        aggfunc="first",
    )
    summary_path = out_dir / "threshold_sweep_summary.csv"
    pivot.to_csv(summary_path)
    logger.info("Saved: %s", summary_path)

    # Print key comparison для быстрого eyeballing
    print("\n" + "=" * 110)
    print("  THRESHOLD SWEEP RESULTS")
    print("=" * 110)
    key_cols = [
        "config", "eval_set", "router", "with_complexity_gate", "n",
        "coverage", "accepted_accuracy", "accepted_recall_anam",
        "FN_deferred", "rule_accepts",
    ]
    print(df[key_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
