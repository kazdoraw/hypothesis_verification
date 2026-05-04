"""Тесты bootstrap CI / paired significance (Task 8 roadmap)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "seed_id": ["s1", "s1", "s2", "s3"],
        "true_label": ["anamnesis", "faq", "anamnesis", "booking"],
        "urgency": ["urgent", "normal", "urgent", "normal"],
        "pred_a": ["anamnesis", "faq", "faq", "booking"],
        "pred_b": ["faq", "faq", "faq", "booking"],
    })


def test_generate_family_indices_is_deterministic() -> None:
    from d1.baselines.statistical_tests import _generate_family_indices

    seed_ids = ["s1", "s1", "s2", "s3"]
    first = _generate_family_indices(seed_ids, n_bootstrap=5, rng_seed=42)
    second = _generate_family_indices(seed_ids, n_bootstrap=5, rng_seed=42)

    assert [x.tolist() for x in first] == [x.tolist() for x in second]
    assert len(first) == 5
    assert all(len(idx) >= 1 for idx in first)


def test_paired_family_bootstrap_uses_identical_rows_for_pair() -> None:
    from d1.baselines.statistical_tests import paired_family_bootstrap

    calls: list[tuple[str, tuple[int, ...]]] = []

    def metric(df: pd.DataFrame, pred_col: str) -> float:
        calls.append((pred_col, tuple(df.index.tolist())))
        return float((df["true_label"] == df[pred_col]).mean())

    paired_family_bootstrap(
        _toy_df(),
        pred_col_a="pred_a",
        pred_col_b="pred_b",
        metric_fn=metric,
        n_bootstrap=8,
        rng_seed=7,
    )

    # Первые 2 вызова — point estimates на полном df. Далее пары bootstrap.
    bootstrap_calls = calls[2:]
    assert len(bootstrap_calls) == 16
    for a_call, b_call in zip(bootstrap_calls[0::2], bootstrap_calls[1::2]):
        assert a_call[0] == "pred_a"
        assert b_call[0] == "pred_b"
        assert a_call[1] == b_call[1]


def test_paired_family_bootstrap_is_deterministic() -> None:
    from d1.baselines.statistical_tests import accuracy_metric, paired_family_bootstrap

    a = paired_family_bootstrap(
        _toy_df(), "pred_a", "pred_b", accuracy_metric,
        n_bootstrap=50, rng_seed=123,
    )
    b = paired_family_bootstrap(
        _toy_df(), "pred_a", "pred_b", accuracy_metric,
        n_bootstrap=50, rng_seed=123,
    )

    assert a["delta_mean"] == pytest.approx(b["delta_mean"])
    assert a["p_value_one_sided"] == pytest.approx(b["p_value_one_sided"])
    assert a["rng_seed"] == 123


def test_family_bootstrap_ci_marks_row_level_fallback() -> None:
    from d1.baselines.statistical_tests import (
        accuracy_metric,
        family_bootstrap_ci_report,
    )

    df = _toy_df().assign(seed_id=["", "", "", ""])
    row = family_bootstrap_ci_report(
        df,
        baseline="toy",
        eval_set="hard_test",
        metric_name="accuracy",
        pred_col="pred_a",
        metric_fn=accuracy_metric,
        n_bootstrap=30,
        rng_seed=1,
    )

    assert row["row_level_fallback"] is True
    assert row["method"] == "BCa"
    assert 0 <= row["ci_lower"] <= row["ci_upper"] <= 1


def test_metric_functions() -> None:
    from d1.baselines.statistical_tests import (
        macro_f1_metric,
        recall_anamnesis_metric,
        recall_urgent_metric,
    )

    df = _toy_df()

    assert recall_anamnesis_metric(df, "pred_a") == pytest.approx(0.5)
    assert recall_urgent_metric(df, "pred_a") == pytest.approx(0.5)
    assert 0 <= macro_f1_metric(df, "pred_a") <= 1


def test_run_statistical_tests_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from d1.scripts import run_statistical_tests as rst

    split = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "text": ["t1", "t2", "t3", "t4"],
        "route_domain": ["anamnesis", "faq", "anamnesis", "booking"],
        "urgency": ["urgent", "normal", "urgent", "normal"],
        "seed_id": ["s1", "s1", "s2", "s3"],
    })
    (tmp_path / "toy_test.csv").write_text(split.to_csv(index=False), encoding="utf-8")

    class _Model:
        def __init__(self, preds: list[str]) -> None:
            self.preds = preds

        def predict(self, texts: list[str]) -> list[str]:
            return self.preds

    class _Bundle:
        def get(self, name: str):
            return {
                "B1.1_tfidf_lr": _Model(["anamnesis", "faq", "faq", "booking"]),
                "B2.1_bge-m3_svc": _Model(["faq", "faq", "faq", "booking"]),
            }[name]

    monkeypatch.setattr(rst, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rst, "DATASET_PREFIX", "toy")
    monkeypatch.setattr(rst, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rst, "train_bundle", lambda names, use_cache=True: _Bundle())

    rst.run_statistical_tests(
        eval_sets=["test"],
        n_bootstrap=20,
        rng_seed=9,
    )

    ci = pd.read_csv(tmp_path / "bootstrap_ci.csv")
    paired = pd.read_csv(tmp_path / "paired_tests.csv")
    assert {"baseline", "eval_set", "metric", "point", "ci_lower", "ci_upper"}.issubset(ci.columns)
    assert {"baseline_a", "baseline_b", "eval_set", "metric", "rng_seed"}.issubset(paired.columns)
    assert set(paired["rng_seed"]) == {9}


def test_paired_plan_orients_b4_improvement() -> None:
    from d1.scripts.run_statistical_tests import _paired_plan

    available = {
        "B1.1_tfidf_lr": "pred_b1_1",
        "B2.1_bge-m3_svc": "pred_b2_1",
        "SelectiveRouter": "pred_selective",
        "B4_hybrid": "pred_b4_hybrid",
    }

    pairs = _paired_plan(available)

    assert ("B4_hybrid", "SelectiveRouter") in pairs
    assert ("SelectiveRouter", "B4_hybrid") not in pairs


def test_statistical_tests_has_no_direct_fit_calls() -> None:
    src = Path("d1/baselines/statistical_tests.py").read_text(encoding="utf-8")
    assert ".fit(" not in src
