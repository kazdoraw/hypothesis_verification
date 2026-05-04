"""Тесты для ручного inference sandbox (Task 6 roadmap).

Контракт:
- логика живёт в `d1.scripts.interactive_inference`, не в ноутбуке;
- inference использует уже обученный `TrainedBundle` через SSoT;
- LLM-вызовов нет, `defer` трактуется только как abstain;
- simple/complex diagnostics не меняет train set и не обучает отдельную модель.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from d1.baselines.b0_rules import RulePrediction
from d1.baselines.selective_router import RouteDecision


CLASSES = ["anamnesis", "booking", "faq", "unsupported"]


class _FakeProbaModel:
    """Минимальный fitted model: predict + predict_proba + classes_."""

    def __init__(self, label: str, proba: list[float]) -> None:
        self.label = label
        self.proba = np.asarray([proba], dtype=float)
        self.classes_ = list(CLASSES)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label for _ in texts]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.repeat(self.proba, repeats=len(texts), axis=0)


class _FakeRules:
    """B0-like rules stub с confidence trace."""

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_with_confidence(texts)[0].route_domain for _ in texts]

    def predict_with_confidence(self, texts: list[str]) -> list[RulePrediction]:
        return [
            RulePrediction("anamnesis", 0.7, "2_patterns")
            for _ in texts
        ]


class _FakeBundle:
    """TrainedBundle-like stub для изолированного inference."""

    def __init__(self) -> None:
        self.models = {
            "B0_rules": _FakeRules(),
            "B1.1_tfidf_lr": _FakeProbaModel("anamnesis", [0.72, 0.08, 0.15, 0.05]),
            "B1.2_tfidf_lr_tuned": _FakeProbaModel("anamnesis", [0.70, 0.10, 0.15, 0.05]),
            "B2.1_bge-m3_svc": _FakeProbaModel("anamnesis", [0.80, 0.05, 0.10, 0.05]),
        }

    def get(self, name: str):
        return self.models[name]


class _FakeRouter:
    """Selective/B4-like router stub."""

    def __init__(self, reason: str = "anamnesis_confident") -> None:
        self.reason = reason

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        return [
            RouteDecision(
                label="anamnesis",
                confidence=0.72,
                margin=0.57,
                action="accept",
                sparse_dense_agree=True,
                reason=self.reason,
                dense_label="anamnesis",
            )
            for _ in texts
        ]


def _fake_manual_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        bundle=_FakeBundle(),
        selective=_FakeRouter(),
        hybrid=_FakeRouter(reason="rule:anamnesis:hits=2"),
    )


def test_infer_text_all_returns_closed_set_selective_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один запрос возвращает полный trace без реального обучения."""
    from d1.scripts import interactive_inference as ii

    monkeypatch.setattr(ii, "build_manual_router_bundle", lambda use_cache=True: _fake_manual_bundle())

    result = ii.infer_text("болит зуб сколько стоит", mode="all")

    assert result.text == "болит зуб сколько стоит"
    assert result.mode == "all"
    assert set(result.closed_set) == {
        "B0_rules", "B1.1_tfidf_lr", "B1.2_tfidf_lr_tuned", "B2.1_bge-m3_svc",
    }
    assert result.closed_set["B1.1_tfidf_lr"]["label"] == "anamnesis"
    assert result.closed_set["B1.1_tfidf_lr"]["top2"][0]["label"] == "anamnesis"
    assert result.selective is not None
    assert result.selective["action"] == "accept"
    assert result.hybrid is not None
    assert result.hybrid["reason"] == "rule:anamnesis:hits=2"
    assert result.rule_trace is not None
    assert result.rule_trace["hit_count"] == 2
    assert result.correctness is None


def test_infer_text_with_gold_label_counts_only_accepted_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Если gold указан — считаем correctness для closed-set и accepted policy."""
    from d1.scripts import interactive_inference as ii

    monkeypatch.setattr(ii, "build_manual_router_bundle", lambda use_cache=True: _fake_manual_bundle())

    result = ii.infer_text("болит зуб", mode="all", gold_label="anamnesis")

    assert result.correctness is not None
    assert result.correctness["closed_set"]["B1.1_tfidf_lr"] is True
    assert result.correctness["selective"] is True
    assert result.correctness["hybrid"] is True


def test_infer_text_with_defer_does_not_score_policy_correctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """defer — abstain outcome; для policy correctness возвращается None."""
    from d1.scripts import interactive_inference as ii

    class _DeferRouter(_FakeRouter):
        def route_batch(self, texts: list[str]) -> list[RouteDecision]:
            return [
                RouteDecision(
                    label="faq",
                    confidence=0.51,
                    margin=0.03,
                    action="defer",
                    sparse_dense_agree=False,
                    reason="sparse_dense_disagree",
                    dense_label="anamnesis",
                )
                for _ in texts
            ]

    monkeypatch.setattr(
        ii,
        "build_manual_router_bundle",
        lambda use_cache=True: SimpleNamespace(
            bundle=_FakeBundle(),
            selective=_DeferRouter(),
            hybrid=_DeferRouter(),
        ),
    )

    result = ii.infer_text("сколько стоит если болит зуб", mode="all", gold_label="anamnesis")

    assert result.correctness is not None
    assert result.selective["action"] == "defer"
    assert result.correctness["selective"] is None
    assert result.correctness["hybrid"] is None


def test_infer_many_validates_gold_label_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """gold_labels должен совпадать по длине с texts."""
    from d1.scripts import interactive_inference as ii

    monkeypatch.setattr(ii, "build_manual_router_bundle", lambda use_cache=True: _fake_manual_bundle())

    with pytest.raises(ValueError, match="gold_labels"):
        ii.infer_many(["а", "б"], gold_labels=["anamnesis"])


@pytest.mark.parametrize(
    ("text", "expected_tag"),
    [
        ("болит зуб сколько стоит лечение", "symptom_price"),
        ("десна опухла запишите меня", "symptom_booking"),
        ("зуб болит есть хирург", "symptom_doctor"),
        ("цена", "short_ambiguous"),
    ],
)
def test_tag_complexity_detects_required_tags(text: str, expected_tag: str) -> None:
    from d1.scripts.interactive_inference import tag_complexity

    tags = tag_complexity(text)

    assert tags[expected_tag] is True


def test_complexity_summary_saves_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """complexity_summary агрегирует hybrid_decisions и сохраняет artifact."""
    from d1.scripts import interactive_inference as ii

    decisions = pd.DataFrame([
        {
            "text_preview": "болит зуб сколько стоит",
            "gold": "anamnesis",
            "predicted": "anamnesis",
            "action": "accept",
            "correct": True,
        },
        {
            "text_preview": "десна опухла запишите",
            "gold": "anamnesis",
            "predicted": "faq",
            "action": "defer",
            "correct": False,
        },
    ])
    src = tmp_path / "hybrid_decisions_hard_test.csv"
    decisions.to_csv(src, index=False)
    monkeypatch.setattr(ii, "RESULTS_DIR", tmp_path)

    summary = ii.complexity_summary(src)

    assert {"tag", "n", "accepted", "deferred", "accepted_accuracy"}.issubset(summary.columns)
    assert (tmp_path / "complexity_summary_hard_test.csv").exists()
    symptom_price = summary.loc[summary["tag"] == "symptom_price"].iloc[0]
    assert symptom_price["n"] == 1


def test_interactive_inference_has_no_fit_or_llm_calls() -> None:
    """Task 6 не обучает модели и не вызывает LLM."""
    path = Path("d1/scripts/interactive_inference.py")
    src = path.read_text(encoding="utf-8")

    assert ".fit(" not in src
    forbidden = ["OpenRouterClient", "chat.completions", "LLM fallback", "вызов LLM"]
    for needle in forbidden:
        assert needle not in src
