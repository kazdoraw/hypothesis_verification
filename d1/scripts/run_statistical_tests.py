"""Orchestration для bootstrap CI / paired significance (Task 8).

Pure статистика живёт в `d1.baselines.statistical_tests`. Этот скрипт:
- загружает eval CSV;
- получает predictions всех 5 baseline через `TrainedBundle`;
- сохраняет `bootstrap_ci.csv` и `paired_tests.csv`.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from d1.baselines.statistical_tests import (
    family_bootstrap_ci_report,
    paired_family_bootstrap,
)
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SETS = ["test", "hard_test", "safety_set"]

PRED_COLS: dict[str, str] = {
    "B0_rules": "pred_b0_rules",
    "B1.1_tfidf_lr": "pred_b1_1",
    "B1.3_fasttext": "pred_b1_3",
    "B2.1_bge-m3_svc": "pred_b2_1",
    "B2.5_e5-small_svc": "pred_b2_5",
}

# Маппинг eval_set → подмножество метрик. На safety_set оставляем только
# recall_urgent: macro_f1 на one-class set — artefact (≈0.24 у всех моделей,
# т.к. F1 для отсутствующих классов = 0).
# На остальных eval-сетах используем только macro_f1: достаточно для
# сравнения моделей, recall_urgent на test/hard_test малоинформативен
# (test=17 urgent, hard_test=44 — CI пересекает 0). Per-class recall
# (включая anamnesis_recall) живёт в baseline_results.json для диагностики
# слабых классов; в статистические тесты не выносим.
_METRIC_FILTER: dict[str, set[str]] = {
    "safety_set": {"recall_urgent"},
}
_DEFAULT_METRIC_KEYS: set[str] = {"macro_f1"}


def _allowed_metrics(eval_set: str) -> dict[str, Any]:
    """Подмножество METRICS, релевантное для данного eval_set."""
    from d1.baselines.statistical_tests import METRICS
    keys = _METRIC_FILTER.get(eval_set, _DEFAULT_METRIC_KEYS)
    return {k: METRICS[k] for k in keys if k in METRICS}


def run_statistical_tests(
    eval_sets: list[str] | None = None,
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сохранить bootstrap CI + paired tests CSV."""
    eval_sets = eval_sets or DEFAULT_EVAL_SETS
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_names = list(PRED_COLS.keys())
    bundle = train_bundle(names=baseline_names, use_cache=True)
    ci_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for eval_set in eval_sets:
        df = _prediction_frame(eval_set, bundle)
        available = {
            baseline: pred_col
            for baseline, pred_col in PRED_COLS.items()
            if pred_col in df.columns
        }

        metrics_for_eval = _allowed_metrics(eval_set)
        for baseline, pred_col in available.items():
            for metric_name, metric_fn in metrics_for_eval.items():
                if metric_name == "recall_urgent" and not _has_urgent(df):
                    continue
                ci_rows.append(family_bootstrap_ci_report(
                    df=df,
                    baseline=baseline,
                    eval_set=eval_set,
                    metric_name=metric_name,
                    pred_col=pred_col,
                    metric_fn=metric_fn,
                    n_bootstrap=n_bootstrap,
                    method="BCa",
                    rng_seed=rng_seed,
                ))

        for a, b in _paired_plan(available):
            col_a = available[a]
            col_b = available[b]
            for metric_name, metric_fn in metrics_for_eval.items():
                if metric_name == "recall_urgent" and not _has_urgent(df):
                    continue
                result = paired_family_bootstrap(
                    df=df,
                    pred_col_a=col_a,
                    pred_col_b=col_b,
                    metric_fn=metric_fn,
                    n_bootstrap=n_bootstrap,
                    rng_seed=rng_seed,
                )
                row = {
                    "baseline_a": a,
                    "baseline_b": b,
                    "eval_set": eval_set,
                    "metric": metric_name,
                    **result,
                }
                # CI 95% не пересекает 0 → разница статистически значима.
                row["significant"] = bool(
                    row["delta_ci_low"] > 0 or row["delta_ci_high"] < 0
                )
                paired_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    paired_df = pd.DataFrame(paired_rows)

    ci_path = out_dir / "bootstrap_ci.csv"
    paired_path = out_dir / "paired_tests.csv"
    ci_df.to_csv(ci_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    logger.info("Saved: %s", ci_path)
    logger.info("Saved: %s", paired_path)

    print("\n=== BOOTSTRAP CI ===")
    print(ci_df.to_string(index=False))
    print("\n=== PAIRED TESTS ===")
    print(paired_df.to_string(index=False))
    return ci_df, paired_df


def _prediction_frame(eval_set: str, bundle) -> pd.DataFrame:
    """Собрать true/pred frame для одного eval set."""
    split = _load_split(eval_set)
    texts = split["text"].tolist()
    df = pd.DataFrame({
        "id": split.get("id", pd.Series([f"row_{i}" for i in range(len(split))])),
        "text": split["text"],
        "true_label": split["route_domain"],
        "urgency": split.get("urgency", pd.Series([""] * len(split))),
        "seed_id": split.get("seed_id", pd.Series([""] * len(split))),
    })
    for name, col in PRED_COLS.items():
        df[col] = bundle.get(name).predict(texts)
    return df


def _load_split(eval_set: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{eval_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _has_urgent(df: pd.DataFrame) -> bool:
    return bool(df["urgency"].isin(["urgent", "emergency", "high"]).any())


def _paired_plan(available: dict[str, str]) -> list[tuple[str, str]]:
    """Все пары baseline'ов с доступными колонками (лексикографический порядок)."""
    names = sorted(available.keys())
    return [(a, b) for a, b in itertools.combinations(names, 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 bootstrap CI / paired tests")
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_statistical_tests(
        eval_sets=args.eval_sets,
        n_bootstrap=args.n_bootstrap,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_statistical_tests"]
