"""Тесты для formal error taxonomy (Task 7 roadmap).

Проверяем:
- rule-based категории по true/pred/subtype/urgency;
- `both_wrong` для кейсов, где sparse и dense оба ошиблись (любые предсказания);
- `models_disagree_both_wrong` для подмножества, где sparse != dense;
- сохранение taxonomy CSV + summary CSV;
- audit sample создаётся как шаблон для ручной проверки категорий;
- модуль использует `train_bundle`, но сам не обучает модели.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


class _FakeModel:
    """Минимальная модель с заранее заданными предсказаниями."""

    def __init__(self, preds: list[str]) -> None:
        self.preds = preds
        self.classes_ = ["anamnesis", "booking", "faq", "unsupported"]

    def predict(self, texts: list[str]) -> list[str]:
        assert len(texts) == len(self.preds)
        return self.preds


class _FakeBundle:
    def __init__(self, sparse_preds: list[str], dense_preds: list[str]) -> None:
        self.models = {
            "B1.1_tfidf_lr": _FakeModel(sparse_preds),
            "B2.1_bge-m3_svc": _FakeModel(dense_preds),
        }

    def get(self, name: str):
        return self.models[name]


def test_classify_error_types_can_return_multiple_tags() -> None:
    from d1.scripts.error_taxonomy import classify_error_types

    row = {
        "text": "болит зуб сколько стоит",
        "route_domain": "anamnesis",
        "subtype": "mixed_intent_price",
        "urgency": "normal",
        "pred_sparse": "faq",
        "pred_dense": "booking",
    }

    tags = classify_error_types(row)

    assert "anamnesis_to_faq" in tags
    assert "mixed_intent_error" in tags
    assert "both_wrong" in tags
    assert "models_disagree_both_wrong" in tags


def test_classify_error_types_required_single_tags() -> None:
    from d1.scripts.error_taxonomy import classify_error_types

    assert "anamnesis_to_booking" in classify_error_types({
        "text": "болит зуб запишите",
        "route_domain": "anamnesis",
        "subtype": "symptom",
        "urgency": "normal",
        "pred_sparse": "booking",
        "pred_dense": "anamnesis",
    })
    assert "faq_to_anamnesis" in classify_error_types({
        "text": "сколько стоит",
        "route_domain": "faq",
        "subtype": "price",
        "urgency": "normal",
        "pred_sparse": "anamnesis",
        "pred_dense": "faq",
    })
    assert "vague_short_error" in classify_error_types({
        "text": "цена",
        "route_domain": "faq",
        "subtype": "price",
        "urgency": "normal",
        "pred_sparse": "unsupported",
        "pred_dense": "faq",
    })


def test_build_error_taxonomy_uses_sparse_and_dense_predictions() -> None:
    from d1.scripts.error_taxonomy import build_error_taxonomy

    df = pd.DataFrame([
        {
            "id": "a",
            "text": "болит зуб сколько стоит",
            "route_domain": "anamnesis",
            "subtype": "mixed_intent_price",
            "urgency": "normal",
            "source": "manual_hard",
            "seed_id": "",
        },
        {
            "id": "b",
            "text": "сколько стоит",
            "route_domain": "faq",
            "subtype": "price",
            "urgency": "normal",
            "source": "seed",
            "seed_id": "seed_1",
        },
    ])

    tax = build_error_taxonomy(
        df,
        eval_set="toy",
        sparse_model=_FakeModel(["faq", "faq"]),
        dense_model=_FakeModel(["booking", "faq"]),
    )

    assert set(tax["id"]) == {"a"}
    assert {"pred_sparse", "pred_dense", "error_type"}.issubset(tax.columns)
    assert "both_wrong" in set(tax["error_type"])
    assert "models_disagree_both_wrong" in set(tax["error_type"])


def test_summarize_taxonomy_counts_pct_and_examples() -> None:
    from d1.scripts.error_taxonomy import summarize_taxonomy

    tax = pd.DataFrame([
        {
            "id": "a",
            "eval_set": "hard_test",
            "text": "t1",
            "error_type": "anamnesis_to_faq",
        },
        {
            "id": "a",
            "eval_set": "hard_test",
            "text": "t1",
            "error_type": "both_wrong",
        },
        {
            "id": "b",
            "eval_set": "hard_test",
            "text": "t2",
            "error_type": "vague_short_error",
        },
    ])

    summary = summarize_taxonomy(tax)

    row = summary.loc[summary["error_type"] == "anamnesis_to_faq"].iloc[0]
    assert row["count"] == 1
    assert row["pct_of_errors"] == pytest.approx(0.5)
    assert row["example_text"] == "t1"


def test_run_error_taxonomy_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from d1.scripts import error_taxonomy as et

    df = pd.DataFrame([
        {
            "id": "a",
            "text": "болит зуб сколько стоит",
            "route_domain": "anamnesis",
            "subtype": "mixed_intent_price",
            "urgency": "normal",
            "source": "manual_hard",
            "seed_id": "",
        },
        {
            "id": "b",
            "text": "цена",
            "route_domain": "faq",
            "subtype": "price",
            "urgency": "normal",
            "source": "seed",
            "seed_id": "seed_1",
        },
    ])

    monkeypatch.setattr(et, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(et, "DATA_DIR", tmp_path)
    monkeypatch.setattr(et, "DATASET_PREFIX", "toy")
    (tmp_path / "toy_hard_test.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    monkeypatch.setattr(
        et,
        "train_bundle",
        lambda names, use_cache=True: _FakeBundle(["faq", "unsupported"], ["booking", "faq"]),
    )

    outputs = et.run_error_taxonomy(eval_sets=["hard_test"], audit_sample_size=2)

    assert "hard_test" in outputs
    assert (tmp_path / "error_taxonomy_hard_test.csv").exists()
    assert (tmp_path / "error_taxonomy_summary.csv").exists()
    audit = pd.read_csv(tmp_path / "error_taxonomy_audit_sample.csv")
    assert {"rule_category_correct", "manual_error_type"}.issubset(audit.columns)


def test_error_taxonomy_has_no_direct_fit_calls() -> None:
    src = Path("d1/scripts/error_taxonomy.py").read_text(encoding="utf-8")

    assert ".fit(" not in src
    assert "train_bundle(" in src
