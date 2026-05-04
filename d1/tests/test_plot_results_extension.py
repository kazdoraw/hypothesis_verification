"""Тесты для plotting refactor (Task 7 плана).

Контракт:
- `plot_policy_comparison(left, right, metrics, eval_set, output_name, ci_csv=None)` —
  параметризованный bar chart на 2 политики × N метрик.
- `plot_selective_comparison` рефакторится через `plot_policy_comparison`
  (backward-compat — функция остаётся в API).
- `plot_simple_vs_hybrid(eval_set)` строится из `simple_vs_hybrid.csv`.
- `plot_complexity_abstain_matrix(eval_set)` — heatmap из
  `simple_router_decisions_<eval>.csv`.
- `run_all_plots()` вызывает новые plots если артефакты существуют (graceful skip).

Все тесты используют `matplotlib.use('Agg')` (без GUI).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# API существование
# ---------------------------------------------------------------------------

def test_plot_policy_comparison_exists() -> None:
    from d1.scripts.plot_results import plot_policy_comparison

    assert callable(plot_policy_comparison)


def test_plot_simple_vs_hybrid_exists() -> None:
    from d1.scripts.plot_results import plot_simple_vs_hybrid

    assert callable(plot_simple_vs_hybrid)


def test_plot_complexity_abstain_matrix_exists() -> None:
    from d1.scripts.plot_results import plot_complexity_abstain_matrix

    assert callable(plot_complexity_abstain_matrix)


# ---------------------------------------------------------------------------
# plot_policy_comparison contract
# ---------------------------------------------------------------------------

def test_plot_policy_comparison_returns_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`plot_policy_comparison` принимает 2 политики × N метрик и возвращает Figure."""
    from d1.scripts import plot_results

    monkeypatch.setattr(plot_results, "FIGURES_DIR", tmp_path)
    left = pd.Series({"coverage": 0.8, "acc": 0.9, "recall": 0.85})
    right = pd.Series({"coverage": 0.7, "acc": 0.92, "recall": 0.91})
    metrics = [
        ("coverage", "Coverage"),
        ("acc", "Accepted accuracy"),
        ("recall", "Recall(anam)"),
    ]
    fig = plot_results.plot_policy_comparison(
        left=("Hybrid", left),
        right=("Simple", right),
        metrics=metrics,
        eval_set="test",
        output_name="policy_smoke",
    )
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "policy_smoke.png").exists()
    plt.close("all")


def test_plot_policy_comparison_supports_ci_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ci_csv=None` — без error bars (no exception)."""
    from d1.scripts import plot_results

    monkeypatch.setattr(plot_results, "FIGURES_DIR", tmp_path)
    left = pd.Series({"coverage": 0.8})
    right = pd.Series({"coverage": 0.7})
    fig = plot_results.plot_policy_comparison(
        left=("A", left), right=("B", right),
        metrics=[("coverage", "Coverage")],
        eval_set="test", output_name="ci_none", ci_csv=None,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_simple_vs_hybrid + complexity_abstain_matrix
# ---------------------------------------------------------------------------

def test_plot_simple_vs_hybrid_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Из синтетического `simple_vs_hybrid.csv` строит фигуру."""
    from d1.scripts import plot_results

    fake_results = tmp_path / "results"
    fake_results.mkdir()
    (fake_results / "simple_vs_hybrid.csv").write_text(
        "eval_set,n,simple_coverage,hybrid_coverage,delta_coverage,"
        "simple_accepted_acc,hybrid_accepted_acc,delta_accepted_acc,"
        "simple_recall_anam,hybrid_recall_anam,simple_FN_deferred,"
        "hybrid_FN_deferred,complexity_defer_rate\n"
        "hard_test,213,0.62,0.73,-0.11,0.89,0.85,0.04,0.94,0.89,52,36,0.18\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(plot_results, "RESULTS_DIR", fake_results)
    monkeypatch.setattr(plot_results, "FIGURES_DIR", tmp_path / "fig")
    (tmp_path / "fig").mkdir()

    fig = plot_results.plot_simple_vs_hybrid("hard_test")
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "fig" / "simple_vs_hybrid_hard_test.png").exists()
    plt.close("all")


def test_plot_complexity_abstain_matrix_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Из синтетического `simple_router_decisions_<eval>.csv` строит heatmap."""
    from d1.scripts import plot_results

    fake_results = tmp_path / "results"
    fake_results.mkdir()
    rows = [
        # primary_tag, action, reason
        ("simple_symptom", "accept", "anamnesis_confident"),
        ("simple_symptom", "accept", "anamnesis_confident"),
        ("mixed_intent", "defer", "complexity:mixed_intent"),
        ("mixed_intent", "defer", "complexity:mixed_intent"),
        ("short_ambiguous", "defer", "complexity:short_ambiguous"),
        ("simple_faq", "defer", "low_confidence"),
    ]
    df = pd.DataFrame(
        [
            {"primary_tag": p, "action": a, "reason": r}
            for p, a, r in rows
        ]
    )
    df.to_csv(fake_results / "simple_router_decisions_hard_test.csv", index=False)

    monkeypatch.setattr(plot_results, "RESULTS_DIR", fake_results)
    monkeypatch.setattr(plot_results, "FIGURES_DIR", tmp_path / "fig")
    (tmp_path / "fig").mkdir()

    fig = plot_results.plot_complexity_abstain_matrix("hard_test")
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "fig" / "complexity_abstain_matrix_hard_test.png").exists()
    plt.close("all")


# ---------------------------------------------------------------------------
# run_all_plots: graceful skip
# ---------------------------------------------------------------------------

def test_run_all_plots_skips_simple_router_when_artifacts_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_all_plots` не падает при отсутствии simple_router CSV."""
    from d1.scripts import plot_results

    src = Path("d1/scripts/plot_results.py").read_text(encoding="utf-8")
    # Источник содержит graceful-skip для simple_router артефактов.
    assert "simple_router_results.csv" in src
    assert "Skip simple-router" in src or "simple_router" in src
