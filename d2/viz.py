"""Визуализация результатов эксперимента D2.

Графики:
1. Radar — 6 осей оценки по схемам.
2. Heatmap — data_sufficiency × routing_match по кейсам × схемам.
3. Grouped bar — 6 метрик + routing match % по схемам.
4. Efficiency: tokens vs turns (scatter).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2.config import DIALOGS_DIR, FIGURES_DIR, REPORTS_DIR
from d2.models import CaseResult

SCHEMA_COLORS = {"S1": "#4CAF50", "S2": "#2196F3", "S3": "#FF9800"}
SCHEMA_LIST = ["S1", "S2", "S3"]
SCORE_AXES = [
    "specialist_score", "service_score", "examination_score",
    "data_sufficiency", "accuracy", "dialogue_quality",
]
SCORE_LABELS = [
    "Специалист", "Услуга", "Обследование",
    "Достаточность", "Точность", "Диалог",
]


def _load_all_results() -> list[CaseResult]:
    """Загрузить все case_XX.json из results/dialogs/."""
    results = []
    for path in sorted(DIALOGS_DIR.glob("case_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(CaseResult.model_validate(data))
    return results


def _load_scores() -> dict | None:
    """Загрузить оценки качества."""
    path = REPORTS_DIR / "d2_judge_scores.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


# --- Графики ---


def plot_quality_radar(scores: dict, save_path: Path) -> None:
    """Radar chart: средние оценки по 6 осям × 3 схемы."""
    schema_means: dict[str, list[float]] = {}
    for schema in SCHEMA_LIST:
        vals = {a: [] for a in SCORE_AXES}
        for case_scores in scores.values():
            if schema in case_scores:
                for a in SCORE_AXES:
                    vals[a].append(case_scores[schema].get(a, 0))
        schema_means[schema] = [
            sum(vals[a]) / len(vals[a]) if vals[a] else 0 for a in SCORE_AXES
        ]

    n = len(SCORE_AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for schema in SCHEMA_LIST:
        values = schema_means[schema] + schema_means[schema][:1]
        ax.plot(angles, values, "o-", label=schema,
                color=SCHEMA_COLORS[schema], linewidth=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=SCHEMA_COLORS[schema])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SCORE_LABELS, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_title("Качество по 6 осям × схемы", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_heatmap(scores: dict, results: list[CaseResult], save_path: Path) -> None:
    """Heatmap: data_sufficiency по кейсам × схемам с метками routing_match."""
    case_ids = sorted(scores.keys())
    case_labels = []
    for cid in case_ids:
        for r in results:
            if r.case_id == cid:
                case_labels.append(f"C{cid}: {r.case_type}")
                break
        else:
            case_labels.append(f"C{cid}")

    data = np.zeros((len(case_ids), len(SCHEMA_LIST)))
    match_data = np.full((len(case_ids), len(SCHEMA_LIST)), False)
    for i, cid in enumerate(case_ids):
        for j, schema in enumerate(SCHEMA_LIST):
            if schema in scores[cid]:
                s = scores[cid][schema]
                data[i, j] = s.get("data_sufficiency", 0)
                match_data[i, j] = s.get("routing_match", False)

    fig, ax = plt.subplots(figsize=(6, max(5, len(case_ids) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=10, aspect="auto")

    ax.set_xticks(range(len(SCHEMA_LIST)))
    ax.set_xticklabels(SCHEMA_LIST, fontsize=12)
    ax.set_yticks(range(len(case_labels)))
    ax.set_yticklabels(case_labels, fontsize=10)

    for i in range(len(case_ids)):
        for j in range(len(SCHEMA_LIST)):
            rm = "✓" if match_data[i, j] else "✗"
            label = f"{data[i, j]:.0f} {rm}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if data[i, j] < 4 else "black")

    ax.set_title("Достаточность данных для маршрутизации (✓/✗ = route match)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Data Sufficiency (0-10)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_schema_comparison(scores: dict, save_path: Path) -> None:
    """Grouped bar: сравнение 6 метрик + routing match % по схемам."""
    schema_means: dict[str, dict[str, float]] = {}
    route_rates: dict[str, float] = {}
    for schema in SCHEMA_LIST:
        vals = {a: [] for a in SCORE_AXES}
        matches, total = 0, 0
        for case_scores in scores.values():
            if schema in case_scores:
                for a in SCORE_AXES:
                    vals[a].append(case_scores[schema].get(a, 0))
                if case_scores[schema].get("routing_match"):
                    matches += 1
                total += 1
        schema_means[schema] = {
            a: sum(vals[a]) / len(vals[a]) if vals[a] else 0
            for a in SCORE_AXES
        }
        route_rates[schema] = (matches / total * 10) if total else 0

    labels = SCORE_LABELS + ["Route Match\n(×10)"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, schema in enumerate(SCHEMA_LIST):
        values = (
            [schema_means[schema][a] for a in SCORE_AXES]
            + [route_rates[schema]]
        )
        bars = ax.bar(x + i * width, values, width, label=schema,
                      color=SCHEMA_COLORS[schema], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", fontsize=9)

    ax.set_ylabel("Score (0-10)")
    ax.set_title("Сравнение схем: 6 метрик качества + маршрутизация")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_efficiency(results: list[CaseResult], save_path: Path) -> None:
    """Scatter: tokens vs turns по схемам."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for schema_name in SCHEMA_LIST:
        turns_l, tokens_l = [], []
        for r in results:
            run = r.runs.get(schema_name)
            if run:
                turns_l.append(run.turns)
                tokens_l.append(run.tokens_doctor + run.tokens_patient)
        ax.scatter(turns_l, tokens_l, label=schema_name,
                   color=SCHEMA_COLORS.get(schema_name, "gray"),
                   s=70, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Turns (ходов диалога)", fontsize=11)
    ax.set_ylabel("Tokens (суммарно)", fontsize=11)
    ax.set_title("Эффективность: tokens vs turns по схемам")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_plots() -> list[Path]:
    """Генерировать все графики и сохранить в results/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_all_results()
    scores = _load_scores()

    if not results:
        print("  Нет результатов для визуализации.")
        return []

    paths = []

    if scores:
        p = FIGURES_DIR / "d2_quality_radar.png"
        plot_quality_radar(scores, p)
        paths.append(p)
        print(f"  → {p}")

        p = FIGURES_DIR / "d2_quality_heatmap.png"
        plot_quality_heatmap(scores, results, p)
        paths.append(p)
        print(f"  → {p}")

        p = FIGURES_DIR / "d2_schema_comparison.png"
        plot_schema_comparison(scores, p)
        paths.append(p)
        print(f"  → {p}")
    else:
        print("  ⚠️  Нет оценок качества. Запустите: python -m d2.run --judge")

    p = FIGURES_DIR / "d2_efficiency.png"
    plot_efficiency(results, p)
    paths.append(p)
    print(f"  → {p}")

    return paths
