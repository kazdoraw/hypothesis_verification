"""Confidence / disagreement analysis для D1 baselines.

Анализирует:
1. Распределение confidence (max predict_proba) по правильным/ошибочным predictions
2. Precision-Recall @ confidence threshold (reject policy)
3. Disagreement: где sparse и dense расходятся, кто чаще прав
4. Thresholded hybrid: accept/reject/fallback зоны

Запуск:
    cd study && python -m d1.scripts.analyze_confidence
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    compute_threshold_table,
    find_pareto_candidates,
)
from d1.baselines.eval_metrics import LABEL_ORDER
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR
from d1.scripts.run_baselines import extract_texts_labels, load_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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


@dataclass
class DisagreementResult:
    """Результат disagreement analysis."""
    total: int
    agree: int
    disagree: int
    disagree_pct: float
    sparse_right_when_disagree: int
    dense_right_when_disagree: int
    both_wrong_when_disagree: int
    disagree_samples: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_models() -> tuple[B1TfidfClassifier, B2EmbeddingClassifier]:
    """Загрузка B1.1 (sparse LR) и B2.1 (dense SVC) через TrainedBundle.

    SSoT обучения — `d1/baselines/trained_bundle.py` (Task 0). Если кэш валиден
    (params + dataset + code + env + schema не изменились), модели загружаются
    мгновенно; иначе — пере-обучение с сохранением в results/models.
    """
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
# 1. Confidence distribution
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    bundle: PredictionBundle,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Гистограмма confidence для correct/incorrect predictions."""
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
        ax.axvline(conf.mean(), color="black", linestyle="--", alpha=0.6, label=f"mean={conf.mean():.3f}")
        ax.legend()

    fig.suptitle(f"Confidence Distribution: {bundle.name} ({eval_set})", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"confidence_dist_{_slug(bundle.name)}_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# 2. Precision-Recall @ threshold (reject policy)
# ---------------------------------------------------------------------------

def plot_threshold_curve(
    bundle: PredictionBundle,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Coverage vs Accuracy при разных confidence thresholds."""
    thresholds = np.linspace(0.3, 0.99, 50)
    coverages = []
    accuracies = []
    recall_anam_list = []

    true_arr = np.array(bundle.true_labels)
    pred_arr = np.array(bundle.preds)

    for t in thresholds:
        mask = bundle.confidence >= t
        if mask.sum() == 0:
            coverages.append(0)
            accuracies.append(0)
            recall_anam_list.append(0)
            continue
        coverages.append(mask.sum() / len(mask))
        accuracies.append((pred_arr[mask] == true_arr[mask]).mean())

        # recall anamnesis: из ВСЕХ anamnesis — сколько покрыто И правильно
        anam_mask = true_arr == "anamnesis"
        accepted_anam = mask & anam_mask
        if anam_mask.sum() > 0:
            correct_anam = (pred_arr[accepted_anam] == "anamnesis").sum()
            recall_anam_list.append(correct_anam / anam_mask.sum())
        else:
            recall_anam_list.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_acc = "#3498db"
    color_cov = "#e67e22"
    color_rec = "#e74c3c"

    ax1.plot(thresholds, accuracies, color=color_acc, linewidth=2, label="Accuracy (accepted)")
    ax1.plot(thresholds, recall_anam_list, color=color_rec, linewidth=2, linestyle="--", label="Recall(anamnesis)")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverages, color=color_cov, linewidth=2, linestyle=":", label="Coverage")
    ax2.set_ylabel("Coverage", color=color_cov)
    ax2.set_ylim(0, 1.05)

    # Gate lines
    ax1.axhline(0.90, color="#95a5a6", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(0.31, 0.91, "gate: 0.90", fontsize=9, color="#95a5a6")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title(
        f"Threshold Trade-off: {bundle.name} ({eval_set})\n"
        f"accepted accuracy vs coverage vs overall recall(anamnesis)",
        fontsize=11,
    )
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"threshold_curve_{_slug(bundle.name)}_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# 3. Disagreement analysis
# ---------------------------------------------------------------------------

def analyze_disagreement(
    sparse: PredictionBundle,
    dense: PredictionBundle,
) -> DisagreementResult:
    """Анализ расхождений между sparse и dense моделями."""
    true_arr = np.array(sparse.true_labels)
    s_arr = np.array(sparse.preds)
    d_arr = np.array(dense.preds)

    agree_mask = s_arr == d_arr
    disagree_mask = ~agree_mask

    n_disagree = disagree_mask.sum()
    sparse_right = ((s_arr[disagree_mask] == true_arr[disagree_mask])).sum()
    dense_right = ((d_arr[disagree_mask] == true_arr[disagree_mask])).sum()
    both_wrong = n_disagree - sparse_right - dense_right

    # Disagreement samples
    rows = []
    for i in np.where(disagree_mask)[0]:
        rows.append({
            "idx": int(i),
            "true": true_arr[i],
            "sparse_pred": s_arr[i],
            "dense_pred": d_arr[i],
            "sparse_conf": float(sparse.confidence[i]),
            "dense_conf": float(dense.confidence[i]),
            "sparse_correct": s_arr[i] == true_arr[i],
            "dense_correct": d_arr[i] == true_arr[i],
        })

    return DisagreementResult(
        total=len(true_arr),
        agree=int(agree_mask.sum()),
        disagree=n_disagree,
        disagree_pct=n_disagree / len(true_arr) * 100,
        sparse_right_when_disagree=int(sparse_right),
        dense_right_when_disagree=int(dense_right),
        both_wrong_when_disagree=int(both_wrong),
        disagree_samples=pd.DataFrame(rows),
    )


def plot_disagreement(
    result: DisagreementResult,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Визуализация disagreement: pie + bar по классам."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie: agree vs disagree
    ax = axes[0]
    sizes = [result.agree, result.disagree]
    labels = [f"Agree\n({result.agree})", f"Disagree\n({result.disagree})"]
    colors = ["#2ecc71", "#e74c3c"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title(f"Agreement ({eval_set}, n={result.total})")

    # Bar: who's right when they disagree
    ax = axes[1]
    cats = ["Sparse right", "Dense right", "Both wrong"]
    vals = [result.sparse_right_when_disagree, result.dense_right_when_disagree, result.both_wrong_when_disagree]
    colors_bar = ["#3498db", "#9b59b6", "#95a5a6"]
    bars = ax.bar(cats, vals, color=colors_bar, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v), ha="center", fontsize=11)
    ax.set_title(f"Outcome When Models Disagree ({eval_set})")
    ax.set_ylabel("Count")

    fig.suptitle("Sparse (B1.1) vs Dense (B2.1) Disagreement", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"disagreement_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


def plot_disagreement_by_class(
    result: DisagreementResult,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Heatmap: disagreement breakdown по true class."""
    df = result.disagree_samples
    if df.empty:
        logger.warning("No disagreements to plot")
        return plt.figure()

    # Crosstab: true class → who's right
    df["winner"] = "both_wrong"
    df.loc[df["sparse_correct"], "winner"] = "sparse_right"
    df.loc[df["dense_correct"], "winner"] = "dense_right"

    ct = pd.crosstab(df["true"], df["winner"])
    for col in ["sparse_right", "dense_right", "both_wrong"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["sparse_right", "dense_right", "both_wrong"]]
    ct = ct.reindex([l for l in LABEL_ORDER if l in ct.index])

    fig, ax = plt.subplots(figsize=(8, 4))
    ct.plot(kind="barh", stacked=True, ax=ax, color=["#3498db", "#9b59b6", "#95a5a6"])
    ax.set_title(f"Disagreement by True Class ({eval_set})")
    ax.set_xlabel("Count")
    ax.legend(title="Winner", loc="lower right")
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"disagreement_by_class_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# 4. Confidence vs Disagreement scatter
# ---------------------------------------------------------------------------

def plot_confidence_scatter(
    sparse: PredictionBundle,
    dense: PredictionBundle,
    eval_set: str = "test",
    save: bool = True,
) -> Figure:
    """Scatter: sparse confidence vs dense confidence, colored by agreement."""
    s_arr = np.array(sparse.preds)
    d_arr = np.array(dense.preds)
    true_arr = np.array(sparse.true_labels)

    agree = s_arr == d_arr
    both_correct = agree & (s_arr == true_arr)
    agree_wrong = agree & (s_arr != true_arr)
    disagree = ~agree

    fig, ax = plt.subplots(figsize=(8, 8))

    for mask, label, color, marker, alpha in [
        (both_correct, "Agree & correct", "#2ecc71", "o", 0.3),
        (agree_wrong, "Agree & wrong", "#e67e22", "s", 0.6),
        (disagree, "Disagree", "#e74c3c", "^", 0.6),
    ]:
        if mask.sum() > 0:
            ax.scatter(
                sparse.confidence[mask],
                dense.confidence[mask],
                c=color, marker=marker, alpha=alpha, s=20, label=f"{label} ({mask.sum()})",
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel(f"Sparse Confidence ({sparse.name})")
    ax.set_ylabel(f"Dense Confidence ({dense.name})")
    ax.set_title(f"Confidence Scatter ({eval_set})")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.2, 1.02)
    fig.tight_layout()

    if save:
        path = _save_fig(fig, f"confidence_scatter_{eval_set}")
        logger.info("Saved: %s", path)
    return fig


# ---------------------------------------------------------------------------
# 5. Calibration artifacts (Task 3b/3c)
# ---------------------------------------------------------------------------

def save_calibration_artifacts(
    bundle: PredictionBundle,
    eval_set: str,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Вычислить и сохранить machine-readable calibration artifacts.

    Сохраняет 3 файла на (model × eval_set):
      - threshold_table_<model>_<eval_set>.csv  (coverage/acc/recall sweep)
      - reliability_table_<model>_<eval_set>.csv  (per-bin stats)
      - calibration_metrics_<model>_<eval_set>.json  (ECE, Brier per-class + macro)

    Args:
        bundle: PredictionBundle с proba, preds, true_labels.
        eval_set: имя eval-сета (val/test/hard_test/...).
        results_dir: корень артефактов (дефолт — RESULTS_DIR).

    Returns:
        Словарь {"threshold_table": Path, "reliability_table": Path,
                 "metrics": Path}.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(bundle.name)

    # Threshold sweep
    threshold_table = compute_threshold_table(
        proba=bundle.proba,
        preds=bundle.preds,
        true_labels=bundle.true_labels,
        classes=bundle.classes,
    )
    tt_path = results_dir / f"threshold_table_{slug}_{eval_set}.csv"
    threshold_table.to_csv(tt_path, index=False)

    # Reliability (на top-label confidence)
    reliability = compute_reliability_table(
        confidence=bundle.confidence,
        correct=bundle.correct,
        n_bins=10,
    )
    rel_path = results_dir / f"reliability_table_{slug}_{eval_set}.csv"
    reliability.to_csv(rel_path, index=False)

    # ECE + Brier
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
        "threshold_table": tt_path,
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
    Стандартный sklearn API — см. [docs](https://scikit-learn.org/stable/modules/
    generated/sklearn.calibration.CalibrationDisplay.html).
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
        # Проверка: для класса без положительных примеров reliability
        # плохо определена — рисуем заглушку, чтобы не падать.
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


def save_val_pareto_candidates(
    bundles_by_name: dict[str, PredictionBundle],
    top_k: int = 7,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Pareto candidate thresholds — ТОЛЬКО на val (защита от overfitting).

    Args:
        bundles_by_name: {model_name: val_bundle}.
        top_k: число кандидатов на baseline.
        results_dir: корень артефактов.

    Returns:
        {model_name: path_to_candidates_csv}.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, bundle in bundles_by_name.items():
        table = compute_threshold_table(
            proba=bundle.proba,
            preds=bundle.preds,
            true_labels=bundle.true_labels,
            classes=bundle.classes,
        )
        candidates = find_pareto_candidates(
            table,
            objectives=("accepted_accuracy", "overall_recall_anamnesis"),
            top_k=top_k,
        )
        slug = _slug(name)
        path = results_dir / f"pareto_candidates_{slug}_val.csv"
        candidates.to_csv(path, index=False)
        paths[name] = path
        logger.info(
            "Pareto candidates [%s @ val]: %d thresholds saved to %s",
            name, len(candidates), path.name,
        )
    return paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_confidence_analysis(
    eval_sets: list[str] | None = None,
) -> dict[str, Any]:
    """Полный confidence/disagreement analysis.

    Returns:
        Словарь с результатами для каждого eval set.
    """
    if eval_sets is None:
        # Task 3b: val добавлен для построения Pareto candidates.
        # test/hard_test используются только для отчётности (read-only oracle).
        eval_sets = ["val", "test", "hard_test"]

    sparse_model, dense_model = _train_models()
    results: dict[str, Any] = {}
    val_bundles_for_pareto: dict[str, PredictionBundle] = {}

    for eval_set in eval_sets:
        logger.info("=== Confidence Analysis: %s ===", eval_set)
        eval_df = load_split(eval_set)
        texts, labels, _ = extract_texts_labels(eval_df)

        sparse_bundle = _predict_bundle(sparse_model, "B1.1 TF-IDF+LR", texts, labels)
        dense_bundle = _predict_bundle(dense_model, "B2.1 BGE-M3+SVC", texts, labels)

        # 1. Confidence distributions
        plot_confidence_distribution(sparse_bundle, eval_set)
        plot_confidence_distribution(dense_bundle, eval_set)

        # 2. Threshold curves (visual)
        plot_threshold_curve(sparse_bundle, eval_set)
        plot_threshold_curve(dense_bundle, eval_set)

        # 3. Disagreement
        dis = analyze_disagreement(sparse_bundle, dense_bundle)
        plot_disagreement(dis, eval_set)
        plot_disagreement_by_class(dis, eval_set)

        # 4. Confidence scatter
        plot_confidence_scatter(sparse_bundle, dense_bundle, eval_set)

        # 5. Calibration artifacts (Task 3b/3c): machine-readable CSV + JSON
        for bundle in (sparse_bundle, dense_bundle):
            save_calibration_artifacts(bundle, eval_set)
            plot_reliability_ovr(bundle, eval_set)

        # Копим val bundles для Pareto (только на val — roadmap §3)
        if eval_set == "val":
            val_bundles_for_pareto[sparse_bundle.name] = sparse_bundle
            val_bundles_for_pareto[dense_bundle.name] = dense_bundle

        logger.info(
            "Disagreement %s: %d/%d (%.1f%%) — sparse_right=%d, dense_right=%d, both_wrong=%d",
            eval_set, dis.disagree, dis.total, dis.disagree_pct,
            dis.sparse_right_when_disagree, dis.dense_right_when_disagree,
            dis.both_wrong_when_disagree,
        )

        results[eval_set] = {
            "sparse": sparse_bundle,
            "dense": dense_bundle,
            "disagreement": dis,
        }

    # Pareto candidates — только на val. НЕ на test/hard_test (overfitting guard).
    if val_bundles_for_pareto:
        save_val_pareto_candidates(val_bundles_for_pareto, top_k=7)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "_").replace(".", "_")


def _save_fig(fig: Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Дефолт задан в run_confidence_analysis: [val, test, hard_test].
    # val нужен для Pareto candidates; test/hard_test — read-only отчётность.
    results = run_confidence_analysis()
    plt.close("all")
    print(f"\n✓ Confidence analysis complete. Figures saved in {FIGURES_DIR}")
