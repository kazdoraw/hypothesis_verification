"""TDD-тесты learning curve experiment (Task 9 roadmap)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _train_df() -> pd.DataFrame:
    rows = []
    labels = ["anamnesis", "faq", "booking", "unsupported"]
    for i in range(20):
        label = labels[i % len(labels)]
        rows.append({
            "id": f"row_{i}",
            "text": f"{label} sample {i}",
            "route_domain": label,
            "subtype": "",
            "explicit_booking": "",
            "urgency": "urgent" if label == "anamnesis" and i % 8 == 0 else "normal",
            "is_offtopic": "",
            "specialization_hint": "",
            "feedback_flag": "",
            "faq_category": "",
            "style": "",
            "source": "seed",
            "seed_id": f"seed_{i:02d}",
        })
    return pd.DataFrame(rows)


def test_sample_train_by_families_uses_seed_ids() -> None:
    from d1.scripts.learning_curves import _sample_train_by_families

    train = _train_df()
    sample = _sample_train_by_families(train, fraction=0.25, rng_seed=7)

    assert 1 <= sample["seed_id"].nunique() <= train["seed_id"].nunique()
    assert len(sample) == sample["seed_id"].nunique()
    assert set(sample["seed_id"]).issubset(set(train["seed_id"]))


def test_default_random_seeds_contract() -> None:
    from d1.scripts.learning_curves import DEFAULT_RANDOM_SEEDS

    assert len(DEFAULT_RANDOM_SEEDS) == 5
    assert len(set(DEFAULT_RANDOM_SEEDS)) == 5


def test_run_learning_curves_writes_raw_summary_and_figure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from d1.scripts import learning_curves as lc

    train = _train_df()
    eval_df = train.iloc[:8].copy()
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    eval_path = tmp_path / "toy_eval.csv"
    eval_df.to_csv(eval_path, index=False)

    class _Model:
        def predict(self, texts: list[str]) -> list[str]:
            return ["anamnesis" if "anamnesis" in t else "faq" for t in texts]

    class _Bundle:
        def get(self, name: str):
            return _Model()

    def fake_train_bundle(
        names, use_cache, cache_dir, train_csv_path=None, device_override=None,
    ):
        assert use_cache is False
        assert train_csv_path is not None
        assert cache_dir is not None
        return _Bundle()

    monkeypatch.setattr(lc, "train_bundle", fake_train_bundle)
    monkeypatch.setattr(lc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lc, "DATASET_PREFIX", "toy")
    monkeypatch.setattr(lc, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(lc, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(lc, "DEFAULT_BASELINES", ["B1.1_tfidf_lr"])
    monkeypatch.setattr(lc, "DEFAULT_EVAL_SETS", ["eval"])

    raw, summary = lc.run_learning_curves(
        fractions=[0.5, 1.0],
        random_seeds=[1, 2],
        train_csv_path=train_path,
        n_jobs=1,
    )

    assert (tmp_path / "learning_curves.csv").exists()
    assert (tmp_path / "learning_curves_summary.csv").exists()
    assert (tmp_path / "figures" / "learning_curves.png").exists()
    assert {"baseline", "eval_set", "metric", "fraction", "seed", "value", "group_std"}.issubset(raw.columns)
    assert {"mean", "std"}.issubset(summary.columns)


def test_learning_curves_has_no_direct_fit_calls() -> None:
    src = Path("d1/scripts/learning_curves.py").read_text(encoding="utf-8")
    assert ".fit(" not in src
    assert "unique_seed_ids" in src
