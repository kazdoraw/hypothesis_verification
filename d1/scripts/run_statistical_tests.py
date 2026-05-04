"""Orchestration для bootstrap CI / paired significance (Task 8).

Pure статистика живёт в `d1.baselines.statistical_tests`. Этот скрипт:
- загружает eval CSV;
- получает predictions B1.1/B2.1 через `TrainedBundle`;
- добавляет selective/hybrid decisions из trace CSV, если они есть;
- сохраняет `bootstrap_ci.csv` и `paired_tests.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from d1.baselines.statistical_tests import (
    METRICS,
    family_bootstrap_ci_report,
    paired_family_bootstrap,
)
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SETS = ["test", "hard_test", "safety_set"]
SPARSE_NAME = "B1.1_tfidf_lr"
DENSE_NAME = "B2.1_bge-m3_svc"

PRED_COLS = {
    SPARSE_NAME: "pred_b1_1",
    DENSE_NAME: "pred_b2_1",
    "SelectiveRouter": "pred_selective",
    "B4_hybrid": "pred_b4_hybrid",
}


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

    bundle = train_bundle(names=[SPARSE_NAME, DENSE_NAME], use_cache=True)
    ci_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for eval_set in eval_sets:
        df = _prediction_frame(eval_set, bundle)
        available = {
            baseline: pred_col
            for baseline, pred_col in PRED_COLS.items()
            if pred_col in df.columns
        }

        for baseline, pred_col in available.items():
            for metric_name, metric_fn in METRICS.items():
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
            for metric_name, metric_fn in METRICS.items():
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
                paired_rows.append({
                    "baseline_a": a,
                    "baseline_b": b,
                    "eval_set": eval_set,
                    "metric": metric_name,
                    **result,
                })

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
    df[PRED_COLS[SPARSE_NAME]] = bundle.get(SPARSE_NAME).predict(texts)
    df[PRED_COLS[DENSE_NAME]] = bundle.get(DENSE_NAME).predict(texts)

    _attach_decision_trace(df, eval_set, "selective", PRED_COLS["SelectiveRouter"])
    _attach_decision_trace(df, eval_set, "hybrid", PRED_COLS["B4_hybrid"])
    return df


def _attach_decision_trace(
    df: pd.DataFrame,
    eval_set: str,
    prefix: str,
    pred_col: str,
) -> None:
    """Добавить selective/hybrid prediction trace, если CSV существует.

    `defer` кодируется как `__defer__`: это не label, а abstain outcome.
    Так full-outcome recall/F1 честно штрафует deferred clinical cases, не
    маппя их в `anamnesis`.
    """
    path = RESULTS_DIR / f"{prefix}_decisions_{eval_set}.csv"
    if not path.exists():
        return
    trace = pd.read_csv(path, dtype=str).fillna("")
    if len(trace) != len(df):
        logger.warning("Skip %s: len(trace)=%d != len(df)=%d", path, len(trace), len(df))
        return
    df[pred_col] = [
        pred if action == "accept" else "__defer__"
        for pred, action in zip(trace["predicted"], trace["action"])
    ]


def _load_split(eval_set: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{eval_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _has_urgent(df: pd.DataFrame) -> bool:
    return bool(df["urgency"].isin(["urgent", "emergency", "high"]).any())


def _paired_plan(available: dict[str, str]) -> list[tuple[str, str]]:
    """Пары из roadmap, только если обе колонки доступны.

    `p_value_one_sided` проверяет направление A > B, поэтому B4 ставим слева
    в сравнении с SelectiveRouter: это ровно гипотеза о приросте hybrid-policy.
    """
    candidates = [
        (SPARSE_NAME, DENSE_NAME),
        (SPARSE_NAME, "B4_hybrid"),
        (DENSE_NAME, "B4_hybrid"),
        ("B4_hybrid", "SelectiveRouter"),
    ]
    return [(a, b) for a, b in candidates if a in available and b in available]


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
