"""Learning curves для D1 v6 (Task 9 roadmap).

Эксперимент отвечает на вопрос: достаточно ли текущего train объёма, или
качество на hard/safety срезах ещё растёт от добавления новых seed families.

Архитектурный контракт:
- обучение только через `train_bundle`;
- subsampling по `seed_id` families, не по строкам;
- sub-sampled модели не пишутся в persistent `results/models` — используется
  временный CSV и временный cache_dir внутри `TemporaryDirectory`.
"""

from __future__ import annotations

import argparse
import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from d1.baselines.statistical_tests import METRICS
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_BASELINES = ["B1.1_tfidf_lr", "B2.1_bge-m3_svc"]
DEFAULT_EVAL_SETS = ["test", "hard_test", "safety_set"]
DEFAULT_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_RANDOM_SEEDS = [11, 23, 37, 53, 71]


def _sample_train_by_families(
    train_df: pd.DataFrame,
    fraction: float,
    rng_seed: int,
) -> pd.DataFrame:
    """Subsample train по seed families.

    Все строки выбранного `seed_id` попадают в sample целиком. Это сохраняет
    зависимость вариаций одного seed и не даёт leakage-like row sampling.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if "seed_id" not in train_df.columns:
        raise ValueError("train_df должен содержать seed_id")

    non_empty = train_df["seed_id"].astype(str).str.strip().ne("")
    if not bool(non_empty.all()):
        raise ValueError("learning curves требуют непустые seed_id в train")

    unique_seed_ids = sorted(train_df["seed_id"].astype(str).unique().tolist())
    n_take = max(1, int(round(len(unique_seed_ids) * fraction)))
    if fraction >= 1.0:
        selected = set(unique_seed_ids)
    else:
        rng = np.random.default_rng(rng_seed)
        selected = set(rng.choice(unique_seed_ids, size=n_take, replace=False).tolist())
    return train_df[train_df["seed_id"].astype(str).isin(selected)].reset_index(drop=True)


def _load_split(eval_set: str) -> pd.DataFrame:
    """Загрузить фиксированный eval split."""
    path = DATA_DIR / f"{DATASET_PREFIX}_{eval_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _predict_frame(eval_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Frame true/pred для метрик из statistical_tests."""
    texts = eval_df["text"].tolist()
    return pd.DataFrame({
        "true_label": eval_df["route_domain"].tolist(),
        "urgency": eval_df.get("urgency", pd.Series([""] * len(eval_df))).tolist(),
        "pred": model.predict(texts),
    })


def _evaluate_model(eval_df: pd.DataFrame, model: Any) -> dict[str, float]:
    """Посчитать метрики learning-curve точки."""
    frame = _predict_frame(eval_df, model)
    values: dict[str, float] = {}
    for metric_name, metric_fn in METRICS.items():
        if metric_name == "recall_urgent" and not frame["urgency"].isin(["urgent", "emergency", "high"]).any():
            continue
        values[metric_name] = float(metric_fn(frame, "pred"))
    return values


def _with_group_stats(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавить mean/std по baseline/eval/metric/fraction."""
    keys = ["baseline", "eval_set", "metric", "fraction"]
    summary = (
        raw.groupby(keys, as_index=False)["value"]
        .agg(mean="mean", std="std", n_runs="count")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    enriched = raw.merge(summary, on=keys, how="left")
    enriched = enriched.rename(columns={"mean": "group_mean", "std": "group_std"})
    return enriched, summary


def plot_learning_curves(summary: pd.DataFrame) -> plt.Figure:
    """Построить learning curves: baseline × eval_set × metric."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eval_sets = [e for e in DEFAULT_EVAL_SETS if e in set(summary["eval_set"])]
    metrics = [m for m in ["macro_f1", "recall_anamnesis", "recall_urgent"] if m in set(summary["metric"])]
    n_rows = len(metrics)
    n_cols = len(eval_sets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * max(1, n_cols), 3.2 * max(1, n_rows)),
        squeeze=False,
        sharex=True,
    )

    for r, metric in enumerate(metrics):
        for c, eval_set in enumerate(eval_sets):
            ax = axes[r][c]
            subset = summary[
                (summary["metric"] == metric)
                & (summary["eval_set"] == eval_set)
            ]
            for baseline, part in subset.groupby("baseline"):
                part = part.sort_values("fraction")
                ax.errorbar(
                    part["fraction"],
                    part["mean"],
                    yerr=part["std"].fillna(0.0),
                    marker="o",
                    capsize=3,
                    label=baseline,
                )
            ax.set_title(f"{eval_set} — {metric}", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(True, alpha=0.3)
            if r == n_rows - 1:
                ax.set_xlabel("Train fraction")
            if c == 0:
                ax.set_ylabel("Score")
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Learning Curves — family-level train subsampling", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / "learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", path)
    return fig


def run_learning_curves(
    fractions: list[float] | None = None,
    random_seeds: list[int] | None = None,
    train_csv_path: Path | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запустить learning curve experiment.

    `n_jobs` оставлен как явный параметр будущего распараллеливания; текущая
    реализация намеренно последовательная, чтобы не перегружать encoder/BLAS.
    """
    if n_jobs != 1:
        logger.warning("n_jobs=%s игнорируется: реализация последовательная", n_jobs)

    fractions = fractions or DEFAULT_FRACTIONS
    random_seeds = random_seeds or DEFAULT_RANDOM_SEEDS
    train_path = train_csv_path or DATA_DIR / f"{DATASET_PREFIX}_train.csv"
    train_df = pd.read_csv(train_path, dtype=str).fillna("")
    eval_frames = {eval_set: _load_split(eval_set) for eval_set in DEFAULT_EVAL_SETS}

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        for seed in random_seeds:
            sample_df = _sample_train_by_families(train_df, fraction, seed)
            logger.info(
                "Learning curve: fraction=%.2f seed=%s rows=%d families=%d",
                fraction, seed, len(sample_df), sample_df["seed_id"].nunique(),
            )
            with tempfile.TemporaryDirectory(prefix="d1_learning_curve_") as tmp:
                tmp_dir = Path(tmp)
                sub_train_path = tmp_dir / "train.csv"
                sample_df.to_csv(sub_train_path, index=False, encoding="utf-8")
                # device_override="cpu": embedding модели не уезжают на MPS.
                # Иначе на 25 итерациях (5 fractions × 5 seeds) Apple MPS
                # allocator не освобождает GPU-память между итерациями
                # и падает с OOM (~26 GiB накопленных тензоров BGE-M3).
                bundle = train_bundle(
                    names=DEFAULT_BASELINES,
                    use_cache=False,
                    cache_dir=tmp_dir / "models",
                    train_csv_path=sub_train_path,
                    device_override="cpu",
                )
                for baseline in DEFAULT_BASELINES:
                    model = bundle.get(baseline)
                    for eval_set, eval_df in eval_frames.items():
                        values = _evaluate_model(eval_df, model)
                        for metric, value in values.items():
                            rows.append({
                                "baseline": baseline,
                                "eval_set": eval_set,
                                "metric": metric,
                                "fraction": fraction,
                                "seed": seed,
                                "value": round(value, 6),
                                "train_rows": len(sample_df),
                                "train_families": sample_df["seed_id"].nunique(),
                            })
                del bundle
            gc.collect()

    raw = pd.DataFrame(rows)
    raw, summary = _with_group_stats(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "learning_curves.csv"
    summary_path = RESULTS_DIR / "learning_curves_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Saved: %s", raw_path)
    logger.info("Saved: %s", summary_path)
    plot_learning_curves(summary)
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 learning curves")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_learning_curves(fractions=args.fractions, random_seeds=args.seeds)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_BASELINES",
    "DEFAULT_EVAL_SETS",
    "DEFAULT_FRACTIONS",
    "DEFAULT_RANDOM_SEEDS",
    "_sample_train_by_families",
    "plot_learning_curves",
    "run_learning_curves",
]
