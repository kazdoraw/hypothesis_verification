"""Тесты для d1.scripts.plot_results (closed-set scope)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_short_names_cover_five_baselines() -> None:
    from d1.scripts.plot_results import _SHORT_NAMES

    assert len(_SHORT_NAMES) == 5
    for k in (
        "B0_rules",
        "B1.1_tfidf_lr",
        "B1.3_fasttext",
        "B2.1_bge-m3_svc",
        "B2.5_e5-small_svc",
    ):
        assert k in _SHORT_NAMES


def test_run_all_plots_no_policy_hooks_in_source() -> None:
    src = Path("d1/scripts/plot_results.py").read_text(encoding="utf-8")
    assert "simple_router" not in src
    assert "hybrid_vs_selective" not in src


def test_plot_routing_comparison_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from d1.scripts import plot_results

    monkeypatch.setattr(plot_results, "FIGURES_DIR", tmp_path)
    df = pd.DataFrame({
        "eval_set": ["test"] * 3,
        "baseline": [
            "B0_rules @ test",
            "B1.1_tfidf_lr @ test",
            "B2.1_bge-m3_svc @ test",
        ],
        "accuracy": [0.8, 0.85, 0.9],
        "macro_f1": [0.75, 0.82, 0.88],
        "balanced_accuracy": [0.78, 0.83, 0.87],
    })
    fig = plot_results.plot_routing_comparison(df, "test")
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "routing_comparison_test.png").exists()
    plt.close("all")
