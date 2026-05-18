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
    "B1.1_tfidf_lr": "B1.1\nTF-IDF+LR",
    "B1.3_fasttext": "B1.3\nfastText",
    "B2.1_bge-m3_svc": "B2.1\nBGE+SVC",
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

def _load_latency_lookup() -> dict[str, float]:
    """Подтянуть median per-text latency из latency_breakdown.csv.

    Источник истины для latency — `latency_breakdown.csv` (n=100, repeats=5).
    Inline replay-latency в baseline_results.csv удалён (P0-1/P3-1):
    давал значения, расходящиеся с правильным бенчмарком в 2–3 раза.
    """
    path = RESULTS_DIR / "latency_breakdown.csv"
    if not path.exists():
        return {}
    lat = pd.read_csv(path)
    if "baseline" not in lat.columns or "total_ms_per_text_median" not in lat.columns:
        return {}
    return dict(zip(lat["baseline"], lat["total_ms_per_text_median"]))


def plot_summary_table(df: pd.DataFrame, eval_set: str = "test") -> pd.DataFrame:
    """Формирование и вывод сводной таблицы метрик."""
    subset = df[df["eval_set"] == eval_set].copy()
    subset["baseline_id"] = subset["baseline"].str.split(" @ ").str[0]
    subset["baseline_short"] = subset["baseline_id"].apply(
        lambda x: _SHORT_NAMES.get(x, x).replace("\n", " ")
    )

    latency_lookup = _load_latency_lookup()
    subset["latency_ms_per_text"] = subset["baseline_id"].map(latency_lookup)

    cols = [
        "baseline_short", "accuracy", "macro_f1", "balanced_accuracy",
        "false_faq_for_anamnesis", "latency_ms_per_text",
    ]
    table = subset[cols].rename(columns={
        "baseline_short": "Baseline",
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Bal.Acc",
        "false_faq_for_anamnesis": "FAQ-for-anam",
        "latency_ms_per_text": "Latency, ms/text (n=100)",
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

def plot_safety_misroute_breakdown(safety_data: dict) -> plt.Figure:
    """Stacked bar chart: куда уходят FN urgent для каждой модели.

    Источник: `safety_results.json` → поле `misrouted_to` (dict[label → count]).
    Визуализация дополняет `safety_comparison.png` (там — recall_urgent + FN total),
    показывая class-level breakdown: B1.1 чаще отправляет urgent в faq, B2.1 —
    в booking, B0_rules — в unsupported, и так далее.
    """
    entries = safety_data.get("safety_set", [])
    names = [_baseline_short(e["baseline"]) for e in entries]
    classes = ["faq", "booking", "unsupported"]
    counts: dict[str, list[int]] = {c: [] for c in classes}
    for entry in entries:
        misrouted = entry.get("misrouted_to") or {}
        if isinstance(misrouted, str):
            import ast
            try:
                misrouted = ast.literal_eval(misrouted)
            except (ValueError, SyntaxError):
                misrouted = {}
        for c in classes:
            counts[c].append(int(misrouted.get(c, 0) or 0))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottoms = np.zeros(len(names))
    colors = sns.color_palette(_PALETTE, len(classes))
    for c, color in zip(classes, colors):
        ax.bar(names, counts[c], bottom=bottoms, label=f"FN → {c}", color=color)
        bottoms += np.array(counts[c])
    ax.set_title("FN urgent — куда ушли пропуски anamnesis (safety_set)", fontsize=12)
    ax.set_ylabel("Кол-во пропусков (out of 87)")
    ax.legend(loc="upper right")
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(i, total + 0.2, str(int(total)), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, "safety_misroute_breakdown")
    return fig


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

def plot_latency() -> plt.Figure:
    """Bar chart: median + p95 per-text latency на CPU.

    Источник: `latency_breakdown.csv` (n=100, repeats=5). Раньше график
    тащил latency из baseline_results.csv (inline replay после warmup),
    что давало числа, расходящиеся с правильным бенчмарком в 2–3 раза.
    """
    path = RESULTS_DIR / "latency_breakdown.csv"
    if not path.exists():
        logger.warning("latency_breakdown.csv не найден — график пропущен")
        return plt.figure()

    df = pd.read_csv(path).copy()
    df["short"] = df["baseline"].map(_SHORT_NAMES).fillna(df["baseline"])
    df = df.sort_values("total_ms_per_text_median").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    width = 0.4
    colors = sns.color_palette(_PALETTE, 2)
    ax.bar(x - width / 2, df["total_ms_per_text_median"], width, label="median", color=colors[0])
    ax.bar(x + width / 2, df["total_ms_p95"], width, label="p95", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"])
    ax.set_yscale("log")
    ax.set_ylabel("ms / text  (log)")
    ax.set_title("CPU latency — per-text  (n=100, repeats=5)")
    ax.legend()
    for i, (m, p) in enumerate(zip(df["total_ms_per_text_median"], df["total_ms_p95"])):
        ax.text(i - width / 2, m * 1.1, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, p * 1.1, f"{p:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "latency_per_text")
    return fig


# ---------------------------------------------------------------------------
# 7. Cross-eval-set comparison (test vs val vs hard_test)
# ---------------------------------------------------------------------------

def plot_cross_eval_f1(df: pd.DataFrame) -> plt.Figure:
    """Macro-F1 по всем релевантным eval-сетам для каждого baseline.

    Включает `test`, `hard_test`, `blind_test`, `entity_held_out`,
    `extended_eval` (если в `df`). На `extended_eval` все baseline дают
    macro_F1 ≈ 0.21 (singleton class faq) — это подписывается явно в
    подзаголовке как known limitation. `val` исключён намеренно: используется
    только для tuning, не для финальной отчётности. `safety_set` исключён:
    one-class → macro_F1 = artefact (см. P0-3).
    """
    df_copy = df.copy()
    df_copy["baseline_id"] = df_copy["baseline"].str.split(" @ ").str[0]
    df_copy["baseline_short"] = df_copy["baseline_id"].map(_SHORT_NAMES).fillna(df_copy["baseline_id"])

    eval_order = ["test", "hard_test", "blind_test", "entity_held_out", "extended_eval"]
    df_copy = df_copy[df_copy["eval_set"].isin(eval_order)]
    sizes = df_copy.groupby("eval_set")["n_samples"].first().to_dict() if "n_samples" in df_copy.columns else {}

    pivot = df_copy.pivot_table(
        index="eval_set", columns="baseline_short", values="macro_f1",
    )
    pivot = pivot.reindex([e for e in eval_order if e in pivot.index])

    pivot.index = [
        f"{e}\n(n={int(sizes.get(e, 0))})" if sizes.get(e) else e
        for e in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_title("Macro-F1 по eval-сетам", fontsize=14)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5)
    ax.legend(title="Baseline", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.text(
        0.5, -0.02,
        "extended_eval: все примеры — faq → macro_F1 ≈ 0.21 (singleton class artefact).  "
        "blind_test n=38 → CI шумный.",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout()
    _save(fig, "cross_eval_f1")
    return fig


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

    # Routing comparison (val исключён — используется только для tuning)
    for eval_set in ["test", "hard_test"]:
        if eval_set in df["eval_set"].unique():
            plot_routing_comparison(df, eval_set)
            saved.append(FIGURES_DIR / f"routing_comparison_{eval_set}.png")

    # Confusion matrices
    for eval_set in ["test", "hard_test"]:
        if eval_set in data:
            plot_confusion_matrices(data, eval_set)
            saved.append(FIGURES_DIR / f"confusion_matrices_{eval_set}.png")

    # Per-class F1 heatmap (val исключён)
    for eval_set in ["test", "hard_test"]:
        if eval_set in data:
            plot_per_class_f1_heatmap(data, eval_set)
            saved.append(FIGURES_DIR / f"per_class_f1_{eval_set}.png")

    # Safety
    plot_safety_comparison(safety_data)
    saved.append(FIGURES_DIR / "safety_comparison.png")
    plot_safety_misroute_breakdown(safety_data)
    saved.append(FIGURES_DIR / "safety_misroute_breakdown.png")

    # Latency (источник истины — latency_breakdown.csv)
    plot_latency()
    saved.append(FIGURES_DIR / "latency_per_text.png")

    # Cross-eval
    plot_cross_eval_f1(df)
    saved.append(FIGURES_DIR / "cross_eval_f1.png")

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
