"""Визуализация результатов D1 baselines.

Читает JSON/CSV из d1/results/, строит:
- сводную таблицу метрик
- grouped bar chart (accuracy / macro-F1 / balanced accuracy)
- confusion matrix heatmaps (по каждому baseline на test)
- safety bar chart (recall_urgent, FN)
- per-class F1 heatmap (baselines × classes)
- latency bar chart

Запуск:
    cd study && python -m d1.scripts.plot_results
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.baselines.eval_metrics import LABEL_ORDER
from d1.config import RESULTS_DIR

logger = logging.getLogger(__name__)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Визуальные константы
_DPI = 150
_PALETTE = "colorblind"
_SHORT_NAMES = {
    "B0_rules": "B0\nRules",
    "B1_tfidf_svc": "B1\nTF-IDF+SVC",
    "B1.1_tfidf_lr": "B1.1\nTF-IDF+LR",
    "B1.2_tfidf_lr_tuned": "B1.2\nTF-IDF+LR tuned",
    "B1.3_fasttext": "B1.3\nfastText",
    "B2_bge-m3_linear": "B2\nBGE+LR",
    "B2.1_bge-m3_svc": "B2.1\nBGE+SVC",
    "B2.2_bge-m3_centroid": "B2.2\nBGE+Centr",
    "B2.3_bge-m3_linear_tuned": "B2.3\nBGE+LR tuned",
    "B2.4_e5-small_linear": "B2.4\nE5-small+LR",
    "B2.5_e5-small_svc": "B2.5\nE5-small+SVC",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_routing_results() -> tuple[pd.DataFrame, dict]:
    """Backward-compat re-export. SSoT: `d1.scripts.artifact_io`."""
    from d1.scripts.artifact_io import load_routing_results

    return load_routing_results()


def _load_safety_results() -> dict:
    """Backward-compat re-export. SSoT: `d1.scripts.artifact_io`."""
    from d1.scripts.artifact_io import load_safety_results

    return load_safety_results()


def _baseline_short(name: str) -> str:
    """baseline full name → short display name."""
    key = name.split(" @ ")[0]
    return _SHORT_NAMES.get(key, key)


def _save(fig: plt.Figure, name: str) -> Path:
    """Сохранение фигуры."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Сводная таблица метрик (test)
# ---------------------------------------------------------------------------

def plot_summary_table(df: pd.DataFrame, eval_set: str = "test") -> pd.DataFrame:
    """Формирование и вывод сводной таблицы метрик."""
    subset = df[df["eval_set"] == eval_set].copy()
    subset["baseline_short"] = subset["baseline"].apply(
        lambda x: _baseline_short(x).replace("\n", " ")
    )

    cols = [
        "baseline_short", "accuracy", "macro_f1", "balanced_accuracy",
        "recall_anamnesis", "latency_ms",
    ]
    table = subset[cols].rename(columns={
        "baseline_short": "Baseline",
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Bal.Acc",
        "recall_anamnesis": "Recall(anam)",
        "latency_ms": "Latency, ms",
    })
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Grouped bar chart — routing quality
# ---------------------------------------------------------------------------

def plot_routing_comparison(df: pd.DataFrame, eval_set: str = "test") -> plt.Figure:
    """Grouped bar chart: Accuracy / Macro-F1 / Balanced Accuracy."""
    subset = df[df["eval_set"] == eval_set].copy()
    subset["short"] = subset["baseline"].apply(_baseline_short)

    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Macro-F1", "Balanced Accuracy"]

    x = np.arange(len(subset))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i * width, subset[metric], width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(subset["short"], fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Routing Quality — {eval_set} set", fontsize=14)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5, label="Gate: F1≥0.80")
    plt.tight_layout()
    _save(fig, f"routing_comparison_{eval_set}")
    return fig


# ---------------------------------------------------------------------------
# 3. Confusion matrix heatmaps
# ---------------------------------------------------------------------------

def plot_confusion_matrices(data: dict, eval_set: str = "test") -> plt.Figure:
    """Heatmap confusion matrix для каждого baseline на eval_set."""
    entries = data.get(eval_set, [])
    n = len(entries)
    if n == 0:
        logger.warning("Нет данных для %s", eval_set)
        return plt.figure()

    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, entry in enumerate(entries):
        ax = axes_flat[idx]
        cm = np.array(entry["confusion"])
        short = _baseline_short(entry["baseline"])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
            ax=ax, cbar=False,
        )
        ax.set_title(short, fontsize=11)
        ax.set_ylabel("True" if idx % n_cols == 0 else "")
        ax.set_xlabel("Predicted")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {eval_set}", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, f"confusion_matrices_{eval_set}")
    return fig


# ---------------------------------------------------------------------------
# 4. Per-class F1 heatmap
# ---------------------------------------------------------------------------

def plot_per_class_f1_heatmap(data: dict, eval_set: str = "test") -> plt.Figure:
    """Heatmap: baselines × classes → F1."""
    entries = data.get(eval_set, [])
    rows = []
    for entry in entries:
        row = {"baseline": _baseline_short(entry["baseline"])}
        for cls in LABEL_ORDER:
            row[cls] = entry.get("per_class", {}).get(cls, {}).get("f1", 0.0)
        rows.append(row)

    heatmap_df = pd.DataFrame(rows).set_index("baseline")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0.2, vmax=1.0, ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Per-class F1 — {eval_set}", fontsize=14)
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, f"per_class_f1_{eval_set}")
    return fig


# ---------------------------------------------------------------------------
# 5. Safety bar chart
# ---------------------------------------------------------------------------

def plot_safety_comparison(safety_data: dict) -> plt.Figure:
    """Bar chart: recall_urgent и FN по baselines."""
    entries = safety_data.get("safety_set", [])
    names = [_baseline_short(e["baseline"]) for e in entries]
    recall_urg = [e["recall_urgent"] for e in entries]
    fn = [e["false_negative_urgent"] for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = sns.color_palette(_PALETTE, len(names))
    ax1.bar(names, recall_urg, color=colors)
    ax1.set_title("Recall (urgent/emergency)", fontsize=11)
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(y=0.90, color="red", linestyle="--", alpha=0.7, label="Gate ≥ 0.90")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    for i, v in enumerate(recall_urg):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(names, fn, color=colors)
    ax2.set_title("False Negative Urgent (абс.)", fontsize=11)
    ax2.set_ylim(0, max(fn) + 3)
    for i, v in enumerate(fn):
        ax2.text(i, v + 0.3, str(v), ha="center", fontsize=9, fontweight="bold")

    n_samples = entries[0].get("n_samples", "?") if entries else "?"
    fig.suptitle(f"Safety Set (clinical-only, n={n_samples})", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "safety_comparison")
    return fig


# ---------------------------------------------------------------------------
# 6. Latency chart
# ---------------------------------------------------------------------------

def plot_latency(df: pd.DataFrame, eval_set: str = "test") -> plt.Figure:
    """Bar chart: latency per baseline."""
    subset = df[df["eval_set"] == eval_set].copy()
    subset["short"] = subset["baseline"].apply(_baseline_short)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette(_PALETTE, len(subset))
    bars = ax.bar(subset["short"], subset["latency_ms"], color=colors)
    ax.set_title(f"Offline Batch Latency — {eval_set} (post-warmup, not online p95)", fontsize=12)
    ax.set_ylabel("ms / request")
    for bar, val in zip(bars, subset["latency_ms"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", fontsize=9,
        )
    plt.tight_layout()
    _save(fig, f"latency_{eval_set}")
    return fig


# ---------------------------------------------------------------------------
# 7. Cross-eval-set comparison (test vs val vs hard_test)
# ---------------------------------------------------------------------------

def plot_cross_eval_f1(df: pd.DataFrame) -> plt.Figure:
    """Macro-F1 по eval sets для каждого baseline."""
    df_copy = df.copy()
    df_copy["baseline_id"] = df_copy["baseline"].str.split(" @ ").str[0]
    df_copy["baseline_short"] = df_copy["baseline_id"].map(_SHORT_NAMES).fillna(df_copy["baseline_id"])

    pivot = df_copy.pivot_table(
        index="eval_set", columns="baseline_short", values="macro_f1",
    )
    eval_order = ["val", "test", "hard_test"]
    pivot = pivot.reindex([e for e in eval_order if e in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_title("Macro-F1 по eval sets", fontsize=14)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5)
    ax.legend(title="Baseline", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _save(fig, "cross_eval_f1")
    return fig


def plot_policy_comparison(
    left: tuple[str, pd.Series],
    right: tuple[str, pd.Series],
    metrics: list[tuple[str, str]],
    eval_set: str,
    output_name: str,
    ci_csv: pd.DataFrame | None = None,
    ci_metric_map: dict[str, tuple[str, str]] | None = None,
) -> plt.Figure:
    """Параметризованный bar chart: 2 политики × N метрик (Task 7 плана).

    Args:
        left, right: (display_name, pandas.Series с column → value).
        metrics: список (column_name, display_label).
        eval_set: имя eval set для заголовка.
        output_name: базовое имя файла без расширения.
        ci_csv: bootstrap CI DataFrame (опционально). При None — error bars выкл.
        ci_metric_map: column_name → (left_baseline, right_baseline) в ci_csv.
            Если колонка отсутствует в map — error bars равны 0.

    Returns:
        matplotlib Figure (сохранён в FIGURES_DIR).
    """
    left_name, left_series = left
    right_name, right_series = right
    metric_map = ci_metric_map or {}

    left_vals = [float(left_series[col]) for col, _ in metrics]
    right_vals = [float(right_series[col]) for col, _ in metrics]

    def _err_for(side_baseline_name: str | None, col: str) -> list[float]:
        if ci_csv is None or side_baseline_name is None:
            return [0.0, 0.0]
        return _ci_error(ci_csv, side_baseline_name, eval_set, col)

    left_errs = []
    right_errs = []
    for col, _ in metrics:
        left_baseline, right_baseline = metric_map.get(col, (None, None))
        left_errs.append(_err_for(left_baseline, col))
        right_errs.append(_err_for(right_baseline, col))

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(_PALETTE, 2)
    ax.bar(
        x - width / 2, left_vals, width,
        yerr=np.asarray(left_errs, dtype=float).T, capsize=4,
        label=left_name, color=colors[0],
    )
    ax.bar(
        x + width / 2, right_vals, width,
        yerr=np.asarray(right_errs, dtype=float).T, capsize=4,
        label=right_name, color=colors[1],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{left_name} vs {right_name} — {eval_set}")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    _save(fig, output_name)
    return fig


def plot_selective_comparison(eval_set: str = "hard_test") -> plt.Figure:
    """SelectiveRouter vs B4 Hybrid через plot_policy_comparison (Task 7 refactor).

    CI берётся из `bootstrap_ci.csv` для full-outcome `recall_anamnesis`.
    """
    compare_path = RESULTS_DIR / "hybrid_vs_selective.csv"
    ci_path = RESULTS_DIR / "bootstrap_ci.csv"
    if not compare_path.exists():
        raise FileNotFoundError(f"Не найден {compare_path}")
    if not ci_path.exists():
        raise FileNotFoundError(f"Не найден {ci_path}. Запустите run_statistical_tests.")

    compare = pd.read_csv(compare_path)
    ci = pd.read_csv(ci_path)
    row = compare[compare["eval_set"] == eval_set]
    if row.empty:
        raise ValueError(f"eval_set={eval_set!r} отсутствует в hybrid_vs_selective.csv")
    row = row.iloc[0]

    # Адаптируем к CI-friendly именам метрик (recall_anamnesis full-outcome из ci_csv).
    left_series = pd.Series({
        "coverage": row["selective_coverage"],
        "accepted_acc": row["selective_accepted_acc"],
        "recall_anamnesis": _ci_point(ci, "SelectiveRouter", eval_set, "recall_anamnesis"),
    })
    right_series = pd.Series({
        "coverage": row["hybrid_coverage"],
        "accepted_acc": row["hybrid_accepted_acc"],
        "recall_anamnesis": _ci_point(ci, "B4_hybrid", eval_set, "recall_anamnesis"),
    })
    return plot_policy_comparison(
        left=("SelectiveRouter", left_series),
        right=("B4 Hybrid", right_series),
        metrics=[
            ("coverage", "Coverage"),
            ("accepted_acc", "Accepted accuracy"),
            ("recall_anamnesis", "Recall(anam), full outcome"),
        ],
        eval_set=eval_set,
        output_name=f"selective_comparison_{eval_set}",
        ci_csv=ci,
        ci_metric_map={
            "recall_anamnesis": ("SelectiveRouter", "B4_hybrid"),
        },
    )


def plot_simple_vs_hybrid(eval_set: str = "hard_test") -> plt.Figure:
    """SimpleRouter vs B4 Hybrid (Task 7).

    Читает `simple_vs_hybrid.csv`, выводит coverage / accepted_acc / recall_anam
    без CI (bootstrap для SimpleRouter добавим в отдельной итерации).
    """
    csv_path = RESULTS_DIR / "simple_vs_hybrid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден {csv_path}. Запустите evaluate_simple_router.")
    df = pd.read_csv(csv_path)
    sub = df[df["eval_set"] == eval_set]
    if sub.empty:
        raise ValueError(f"eval_set={eval_set!r} отсутствует в {csv_path}")
    row = sub.iloc[0]

    left = pd.Series({
        "coverage": row["simple_coverage"],
        "accepted_acc": row["simple_accepted_acc"],
        "recall_anam": row["simple_recall_anam"],
    })
    right = pd.Series({
        "coverage": row["hybrid_coverage"],
        "accepted_acc": row["hybrid_accepted_acc"],
        "recall_anam": row["hybrid_recall_anam"],
    })
    return plot_policy_comparison(
        left=("SimpleRouter", left),
        right=("B4 Hybrid", right),
        metrics=[
            ("coverage", "Coverage"),
            ("accepted_acc", "Accepted accuracy"),
            ("recall_anam", "Accepted recall(anam)"),
        ],
        eval_set=eval_set,
        output_name=f"simple_vs_hybrid_{eval_set}",
    )


def plot_complexity_abstain_matrix(eval_set: str = "hard_test") -> plt.Figure:
    """Heatmap: complexity primary_tag × outcome (accept / defer_complexity / defer_ml).

    Источник: `simple_router_decisions_<eval>.csv`. Диагностический вид —
    показывает распределение outcome по тэгам.
    """
    decisions_path = RESULTS_DIR / f"simple_router_decisions_{eval_set}.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Не найден {decisions_path}")
    df = pd.read_csv(decisions_path)

    def _outcome(row: pd.Series) -> str:
        if row["action"] == "accept":
            return "accept"
        return (
            "defer_complexity"
            if str(row["reason"]).startswith("complexity:")
            else "defer_ml"
        )

    df = df.copy()
    df["outcome"] = df.apply(_outcome, axis=1)
    matrix = (
        df.groupby(["primary_tag", "outcome"])
        .size()
        .unstack(fill_value=0)
    )
    # Фиксируем порядок outcome колонок.
    cols_order = [c for c in ["accept", "defer_ml", "defer_complexity"] if c in matrix.columns]
    matrix = matrix.reindex(columns=cols_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(matrix))))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="viridis", cbar=True,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Complexity tag × outcome — {eval_set}")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("primary_tag")
    plt.tight_layout()
    _save(fig, f"complexity_abstain_matrix_{eval_set}")
    return fig


def _ci_point(ci: pd.DataFrame, baseline: str, eval_set: str, metric: str) -> float:
    row = ci[
        (ci["baseline"] == baseline)
        & (ci["eval_set"] == eval_set)
        & (ci["metric"] == metric)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["point"])


def _ci_error(ci: pd.DataFrame, baseline: str, eval_set: str, metric: str) -> list[float]:
    row = ci[
        (ci["baseline"] == baseline)
        & (ci["eval_set"] == eval_set)
        & (ci["metric"] == metric)
    ]
    if row.empty:
        return [0.0, 0.0]
    r = row.iloc[0]
    point = float(r["point"])
    return [
        max(0.0, point - float(r["ci_lower"])),
        max(0.0, float(r["ci_upper"]) - point),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_plots() -> list[Path]:
    """Построение всех графиков. Возвращает список сохранённых файлов."""
    sns.set_theme(style="whitegrid", font_scale=1.0, palette=_PALETTE)

    df, data = _load_routing_results()
    safety_data = _load_safety_results()

    saved: list[Path] = []

    # Summary table
    table = plot_summary_table(df, "test")
    table_path = FIGURES_DIR / "summary_table.csv"
    table.to_csv(table_path, index=False)
    saved.append(table_path)
    print("\n=== ROUTING SUMMARY (test) ===")
    print(table.to_string(index=False))

    # Routing comparison
    for eval_set in ["test", "val", "hard_test"]:
        if eval_set in df["eval_set"].unique():
            plot_routing_comparison(df, eval_set)
            saved.append(FIGURES_DIR / f"routing_comparison_{eval_set}.png")

    # Confusion matrices
    for eval_set in ["test", "hard_test"]:
        if eval_set in data:
            plot_confusion_matrices(data, eval_set)
            saved.append(FIGURES_DIR / f"confusion_matrices_{eval_set}.png")

    # Per-class F1 heatmap
    for eval_set in ["test", "val", "hard_test"]:
        if eval_set in data:
            plot_per_class_f1_heatmap(data, eval_set)
            saved.append(FIGURES_DIR / f"per_class_f1_{eval_set}.png")

    # Safety
    plot_safety_comparison(safety_data)
    saved.append(FIGURES_DIR / "safety_comparison.png")

    # Latency
    plot_latency(df, "test")
    saved.append(FIGURES_DIR / f"latency_test.png")

    # Cross-eval
    plot_cross_eval_f1(df)
    saved.append(FIGURES_DIR / "cross_eval_f1.png")

    # Selective-policy comparison: отдельный график, не closed-set benchmark.
    if (RESULTS_DIR / "hybrid_vs_selective.csv").exists() and (RESULTS_DIR / "bootstrap_ci.csv").exists():
        plot_selective_comparison("hard_test")
        saved.append(FIGURES_DIR / "selective_comparison_hard_test.png")

    # SimpleRouter (Task 7): graceful skip если артефакты отсутствуют.
    if (RESULTS_DIR / "simple_router_results.csv").exists():
        plot_simple_vs_hybrid("hard_test")
        saved.append(FIGURES_DIR / "simple_vs_hybrid_hard_test.png")
        if (RESULTS_DIR / "simple_router_decisions_hard_test.csv").exists():
            plot_complexity_abstain_matrix("hard_test")
            saved.append(FIGURES_DIR / "complexity_abstain_matrix_hard_test.png")
    else:
        logger.info("Skip simple-router plots: artifacts missing")

    plt.close("all")

    print(f"\n✓ Сохранено {len(saved)} артефактов в {FIGURES_DIR}")
    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_all_plots()


if __name__ == "__main__":
    main()
