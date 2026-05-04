"""Versioned визуализации для Stage 1 / Stage 2A экспериментов.

Все графики сохраняются в run_paths.figures_dir.
Каждая функция возвращает Path к сохранённому файлу.

Колонки таблиц используют разделитель «::» (mode::metric).

Группы функций:
* Stage 2A: ``plot_retrieval_comparison``, ``plot_quality_comparison``,
  ``plot_mode_deltas``, ``plot_rank_shift`` — фокус на сравнении режимов
  представления (C0/C1/C2) на одной стратегии.
* Stage 1: ``plot_stage1_retrieval_strategies``,
  ``plot_stage1_quality_strategies`` — фокус на сравнении стратегий
  поиска (B0/S1/S2/S3/S4) внутри одного режима.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from d4.analysis.artifacts import RunPaths
from d4.analysis.representation_compare import _SEP

sns.set_style("whitegrid")
_DPI = 150
_FIGSIZE = (10, 6)

# Палитра режимов: C0=серый baseline, C1=синий, C2=зелёный
_MODE_PALETTE = {"plain": "#9e9e9e", "contextual": "#1976d2", "llm_enriched": "#388e3c"}

# Палитра стратегий для Stage 1: B0=серый descriptive, S1=жёлтый upper-bound,
# S2=розовый lexical, S3=зелёный лидер, S4=голубой hybrid.
# Подобрана так, чтобы лидер (S3) и upper-bound (S1) выделялись из retrieval-веток.
_STRATEGY_PALETTE = {
    "B0": "#9e9e9e",
    "S1": "#fbc02d",
    "S2": "#ec407a",
    "S3": "#388e3c",
    "S4": "#039be5",
    "S4r": "#673ab7",
    "S5": "#5d4037",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _extract_modes(table: pd.DataFrame) -> list[str]:
    """Извлекает уникальные modes из column names с разделителем «::»."""
    modes: set[str] = set()
    for col in table.columns:
        if _SEP in col and not col.endswith(f"{_SEP}delta"):
            mode = col.split(_SEP)[0]
            modes.add(mode)
    return sorted(modes)


# ---------------------------------------------------------------------------
# 1. Retrieval comparison (grouped bar)
# ---------------------------------------------------------------------------

def plot_retrieval_comparison(
    retrieval_table: pd.DataFrame,
    run_paths: RunPaths,
    filename: str = "retrieval_comparison.png",
) -> Path:
    """Grouped bar chart: hit_rate + MRR по modes для каждой strategy."""
    target_metrics = ["hit_rate", "mrr"]
    modes = _extract_modes(retrieval_table)
    strategies = retrieval_table.index.tolist()

    fig, axes = plt.subplots(1, len(target_metrics), figsize=(6 * len(target_metrics), 5))
    if len(target_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, target_metrics):
        x = np.arange(len(strategies))
        width = 0.8 / max(len(modes), 1)
        for i, mode in enumerate(modes):
            col = f"{mode}{_SEP}{metric}"
            vals = retrieval_table[col].values if col in retrieval_table.columns else [0] * len(strategies)
            bars = ax.bar(x + i * width, vals, width, label=mode,
                          color=_MODE_PALETTE.get(mode, "#666"))
            for bar, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(strategies)
        ax.set_title("hit@5" if metric == "hit_rate" else metric.upper())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Retrieval Comparison: C0 / C1 / C2", fontsize=12)
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))


# ---------------------------------------------------------------------------
# 2. Quality comparison (grouped bar)
# ---------------------------------------------------------------------------

def plot_quality_comparison(
    det_table: pd.DataFrame,
    run_paths: RunPaths,
    filename: str = "quality_comparison.png",
) -> Path:
    """Grouped bar chart: avg_fmr + doctor_match_rate по modes."""
    target_metrics = ["avg_fmr", "doctor_match_rate"]
    modes = _extract_modes(det_table)
    strategies = det_table.index.tolist()

    fig, axes = plt.subplots(1, len(target_metrics), figsize=(6 * len(target_metrics), 5))
    if len(target_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, target_metrics):
        x = np.arange(len(strategies))
        width = 0.8 / max(len(modes), 1)
        for i, mode in enumerate(modes):
            col = f"{mode}{_SEP}{metric}"
            vals = det_table[col].values if col in det_table.columns else [0] * len(strategies)
            bars = ax.bar(x + i * width, vals, width, label=mode,
                          color=_MODE_PALETTE.get(mode, "#666"))
            for bar, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(strategies)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Quality Comparison: C0 / C1 / C2", fontsize=12)
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))


# ---------------------------------------------------------------------------
# 3. Mode deltas heatmap
# ---------------------------------------------------------------------------

def plot_mode_deltas(
    retrieval_table: pd.DataFrame,
    det_table: pd.DataFrame,
    run_paths: RunPaths,
    baseline: str = "plain",
    filename: str = "mode_deltas.png",
) -> Path:
    """Heatmap: delta vs baseline для ключевых метрик."""
    delta_suffix = f"{_SEP}delta"
    delta_cols_ret = [c for c in retrieval_table.columns
                      if c.endswith(delta_suffix) and not c.startswith(f"{baseline}{_SEP}")]
    delta_cols_det = [c for c in det_table.columns
                      if c.endswith(delta_suffix) and not c.startswith(f"{baseline}{_SEP}")]

    parts = []
    if delta_cols_ret:
        parts.append(retrieval_table[delta_cols_ret])
    if delta_cols_det:
        parts.append(det_table[delta_cols_det])

    if not parts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Нет delta-колонок", ha="center", va="center")
        return _save(fig, run_paths.figure(filename))

    combined = pd.concat(parts, axis=1)
    # Убираем "::delta" из названий для читаемости
    combined.columns = [c.replace(delta_suffix, "") for c in combined.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(combined.columns) * 1.2), max(4, len(combined) * 0.8)))
    sns.heatmap(
        combined.astype(float),
        annot=True,
        fmt=".3f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Delta vs {baseline} (strategy × metric)")
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))


# ---------------------------------------------------------------------------
# 4. Rank shift per sample
# ---------------------------------------------------------------------------

def plot_rank_shift(
    rank_df: pd.DataFrame,
    run_paths: RunPaths,
    filename: str = "rank_shift.png",
) -> Path:
    """Per-sample reciprocal rank comparison between modes (scatter/strip)."""
    if rank_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Нет rank данных", ha="center", va="center")
        return _save(fig, run_paths.figure(filename))

    long = rank_df.reset_index().melt(
        id_vars="sample_id", var_name="mode_strategy", value_name="rr",
    )
    # Column format: {mode}_{strategy}_rr (from loaders, underscore-separated)
    long[["mode", "strategy", "_"]] = long["mode_strategy"].str.extract(
        r"^(.+?)_(S\d+r?|B\d+)_(rr)$",
    )
    long = long.dropna(subset=["mode"])

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.stripplot(
        data=long, x="strategy", y="rr", hue="mode",
        dodge=True, jitter=0.15, alpha=0.6,
        palette=_MODE_PALETTE, ax=ax,
    )
    ax.set_title("Reciprocal Rank per Sample × Strategy")
    ax.set_ylabel("Reciprocal Rank (0 = miss)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))


# ---------------------------------------------------------------------------
# Stage 1 plots: сравнение СТРАТЕГИЙ внутри одного режима (без mode-deltas)
# ---------------------------------------------------------------------------

def _bar_strategies(
    ax: plt.Axes,
    strategies: list[str],
    values: list[float],
    title: str,
    ylim: tuple[float, float] = (0.0, 1.05),
    fmt: str = "{:.3f}",
) -> None:
    """Один subplot: bar по стратегиям с подписью значений и палитрой."""
    x = np.arange(len(strategies))
    colors = [_STRATEGY_PALETTE.get(s, "#666") for s in strategies]
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        if pd.notna(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.01,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(*ylim)


def plot_stage1_retrieval_strategies(
    retrieval_table: pd.DataFrame,
    run_paths: RunPaths,
    mode: str = "plain",
    filename: str = "stage1_retrieval_strategies.png",
) -> Path:
    """Stage 1: bar chart по стратегиям × retrieval-метрикам в одном режиме.

    Метрики: hit_rate, MRR, mean_recall, gold_in_context_rate.

    Args:
        retrieval_table: output ``build_retrieval_table()``. Должен содержать
            колонки ``{mode}::hit_rate``, ``{mode}::mrr``, и т. д.
        mode: режим представления, для которого строим график. Stage 1 по
            умолчанию использует ``plain``.
    """
    metrics = [
        ("hit_rate", "hit@5"),
        ("mrr", "MRR"),
        ("mean_recall", "Mean Recall@5"),
        ("gold_in_context_rate", "Gold in Context"),
    ]
    strategies = retrieval_table.index.tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        col = f"{mode}{_SEP}{metric_key}"
        if col not in retrieval_table.columns:
            ax.text(0.5, 0.5, f"Нет {col}", ha="center", va="center")
            ax.set_title(metric_label)
            continue
        values = retrieval_table[col].tolist()
        _bar_strategies(ax, strategies, values, metric_label)

    fig.suptitle(
        f"Stage 1: Retrieval Comparison Across Strategies "
        f"(mode = {mode})",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))


def plot_stage1_quality_strategies(
    det_table: pd.DataFrame,
    run_paths: RunPaths,
    mode: str = "plain",
    filename: str = "stage1_quality_strategies.png",
) -> Path:
    """Stage 1: bar chart по стратегиям × answer-quality метрикам в одном режиме.

    Метрики: avg_fmr, doctor_match_rate, answerability_rate.

    Args:
        det_table: output ``build_deterministic_table()``. Должен содержать
            колонки ``{mode}::avg_fmr``, ``{mode}::doctor_match_rate``,
            ``{mode}::answerability_rate``.
        mode: режим представления, для которого строим график.
    """
    metrics = [
        ("avg_fmr", "Fact Match Rate"),
        ("doctor_match_rate", "Doctor Match Rate"),
        ("answerability_rate", "Answerability Rate"),
    ]
    strategies = det_table.index.tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        col = f"{mode}{_SEP}{metric_key}"
        if col not in det_table.columns:
            ax.text(0.5, 0.5, f"Нет {col}", ha="center", va="center")
            ax.set_title(metric_label)
            continue
        values = det_table[col].tolist()
        _bar_strategies(ax, strategies, values, metric_label)

    fig.suptitle(
        f"Stage 1: Answer Quality Across Strategies "
        f"(mode = {mode})",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, run_paths.figure(filename))
