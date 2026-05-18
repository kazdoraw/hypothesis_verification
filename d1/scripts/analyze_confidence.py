"""Calibration analysis для D1 baselines (closed-set scope).

Анализирует только то, что имеет смысл в чистой closed-set классификации:
1. Confidence distribution (correct vs incorrect predictions);
2. Reliability tables (per-bin observed accuracy vs avg confidence);
3. ECE + Brier OvR macro/per-class (machine-readable);
4. Reliability diagrams (OvR через sklearn CalibrationDisplay).

Запуск:
    cd study && python -m d1.scripts.analyze_confidence
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from sklearn.calibration import CalibrationDisplay

from d1.baselines.b1_tfidf import B1TfidfClassifier
from d1.baselines.b2_embedding import B2EmbeddingClassifier
from d1.baselines.calibration import (
    compute_brier_ovr,
    compute_ece,
    compute_reliability_table,
)
from d1.baselines.trained_bundle import train_bundle
from d1.config import RESULTS_DIR
from d1.scripts.run_baselines import extract_texts_labels, load_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Default eval sets для калибровочного отчёта. `val` исключён —
# используется только для tuning, не для финальной отчётности.
DEFAULT_EVAL_SETS = ["test", "hard_test"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionBundle:
    """Predictions + probabilities для одной модели на одном eval set."""
    name: str
    preds: list[str]
    proba: np.ndarray  # (n_samples, n_classes)
    classes: list[str]
    confidence: np.ndarray  # max(proba) per sample
    true_labels: list[str]

    @property
    def correct(self) -> np.ndarray:
        return np.array(self.preds) == np.array(self.true_labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_models() -> tuple[B1TfidfClassifier, B2EmbeddingClassifier]:
    """Загрузка B1.1 (sparse LR) и B2.1 (dense SVC) через TrainedBundle."""
    logger.info("Loading B1.1 и B2.1 через train_bundle (use_cache=True)")
    bundle = train_bundle(
        names=["B1.1_tfidf_lr", "B2.1_bge-m3_svc"], use_cache=True,
    )
    return bundle.get("B1.1_tfidf_lr"), bundle.get("B2.1_bge-m3_svc")


def _predict_bundle(
    model: B1TfidfClassifier | B2EmbeddingClassifier,
    name: str,
    texts: list[str],
    true_labels: list[str],
) -> PredictionBundle:
    """Predict + predict_proba → PredictionBundle."""
    preds = model.predict(texts)
    proba = model.predict_proba(texts)
    classes = model.classes_
    confidence = np.max(proba, axis=1)
    return PredictionBundle(
        name=name,
        preds=preds,
        proba=proba,
        classes=classes,
        confidence=confidence,
        true_labels=true_labels,
    )


# ---------------------------------------------------------------------------
# Confidence distribution (correct vs incorrect)
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    bundle: PredictionBundle,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Гистограмма confidence для correct/incorrect predictions.

    Полезна как иллюстрация: «насколько модель уверена в правильных vs
    ошибочных предсказаниях». Хорошо откалиброванная модель показывает
    очевидную сепарацию (правильные — выше confidence, ошибочные — ниже).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (mask, label, color) in zip(
        axes,
        [(bundle.correct, "Correct", "#2ecc71"), (~bundle.correct, "Incorrect", "#e74c3c")],
    ):
        conf = bundle.confidence[mask]
        ax.hist(conf, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(f"{bundle.name} — {label} (n={mask.sum()})")
        ax.set_xlabel("Confidence (max prob)")
        ax.set_ylabel("Count")
        if conf.size:
            ax.axvline(conf.mean(), color="black", linestyle="--", alpha=0.6, label=f"mean={conf.mean():.3f}")
            ax.legend()

    fig.suptitle(f"Confidence Distribution: {bundle.name} ({eval_set})", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"confidence_dist_{_slug(bundle.name)}_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# Calibration artifacts (machine-readable)
# ---------------------------------------------------------------------------

def save_calibration_artifacts(
    bundle: PredictionBundle,
    eval_set: str,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Вычислить и сохранить machine-readable calibration artifacts.

    Сохраняет 2 файла на (model × eval_set):
      - reliability_table_<model>_<eval_set>.csv  (per-bin observed accuracy)
      - calibration_metrics_<model>_<eval_set>.json  (ECE, Brier per-class + macro)

    `threshold_table_*.csv` намеренно не пишется — был relic'ом эпохи
    selective routing (см. docstring модуля).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(bundle.name)

    reliability = compute_reliability_table(
        confidence=bundle.confidence,
        correct=bundle.correct,
        n_bins=10,
    )
    rel_path = results_dir / f"reliability_table_{slug}_{eval_set}.csv"
    reliability.to_csv(rel_path, index=False)

    ece = compute_ece(bundle.confidence, bundle.correct, n_bins=10)
    brier = compute_brier_ovr(
        y_true=bundle.true_labels,
        proba=bundle.proba,
        classes=bundle.classes,
    )
    metrics = {
        "baseline": bundle.name,
        "eval_set": eval_set,
        "n_samples": len(bundle.true_labels),
        "ece_n_bins": 10,
        "ece": round(ece, 6),
        "brier_ovr": {
            "per_class": {k: round(v, 6) for k, v in brier["per_class"].items()},
            "macro": round(brier["macro"], 6),
        },
    }
    metrics_path = results_dir / f"calibration_metrics_{slug}_{eval_set}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(
        "Calibration [%s @ %s]: ECE=%.4f, Brier(macro)=%.4f",
        bundle.name, eval_set, ece, brier["macro"],
    )
    return {
        "reliability_table": rel_path,
        "metrics": metrics_path,
    }


def plot_reliability_ovr(
    bundle: PredictionBundle,
    eval_set: str = "test",
    n_bins: int = 10,
    save: bool = True,
) -> Figure:
    """Reliability diagram (OvR) через sklearn CalibrationDisplay.from_predictions.

    Для multiclass строим по одному subplot на класс: (y == cls) vs proba[:, i].
    """
    true_arr = np.asarray(bundle.true_labels)
    classes = bundle.classes
    n_classes = len(classes)

    n_cols = min(n_classes, 2)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, cls in enumerate(classes):
        ax = axes_flat[i]
        y_bin = (true_arr == cls).astype(int)
        if y_bin.sum() == 0:
            ax.text(0.5, 0.5, f"no samples\nclass={cls}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{cls} (n=0)")
            continue
        CalibrationDisplay.from_predictions(
            y_bin, bundle.proba[:, i], n_bins=n_bins, ax=ax, name=cls,
        )
        ax.set_title(f"{cls} (n_pos={int(y_bin.sum())})")

    for j in range(n_classes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Reliability (OvR) — {bundle.name} [{eval_set}]",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"reliability_{_slug(bundle.name)}_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_confidence_analysis(
    eval_sets: list[str] | None = None,
) -> dict[str, Any]:
    """Полный calibration analysis (closed-set scope).

    Делает: confidence distribution + reliability table + ECE/Brier metrics
    + reliability diagram. Для B1.1 (sparse) и B2.1 (dense) на каждом
    eval-сете в `eval_sets` (по умолчанию test + hard_test).

    Returns:
        Словарь {eval_set: {"sparse": PredictionBundle, "dense": PredictionBundle}}.
    """
    if eval_sets is None:
        eval_sets = list(DEFAULT_EVAL_SETS)

    sparse_model, dense_model = _train_models()
    results: dict[str, Any] = {}

    for eval_set in eval_sets:
        logger.info("=== Calibration Analysis: %s ===", eval_set)
        eval_df = load_split(eval_set)
        texts, labels, _ = extract_texts_labels(eval_df)

        sparse_bundle = _predict_bundle(sparse_model, "B1.1 TF-IDF+LR", texts, labels)
        dense_bundle = _predict_bundle(dense_model, "B2.1 BGE-M3+SVC", texts, labels)

        for bundle in (sparse_bundle, dense_bundle):
            plot_confidence_distribution(bundle, eval_set)
            save_calibration_artifacts(bundle, eval_set)
            plot_reliability_ovr(bundle, eval_set)

        results[eval_set] = {
            "sparse": sparse_bundle,
            "dense": dense_bundle,
        }

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _save_fig(fig: Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_confidence_analysis()
