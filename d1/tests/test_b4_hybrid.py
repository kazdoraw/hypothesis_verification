"""Unit + integration тесты для B4HybridRouter (Task 5 roadmap).

Проверяемые контракты:
- `_parse_hit_count` — "N_patterns" → N, "no_match" → 0.
- `RULE_ACCEPT_POLICY` — whitelist с min_hit_count по доменам:
    * anamnesis: >=2 (safety-conservative)
    * booking: >=3 (legal-sensitive)
    * faq: None (NEVER rule-accept, всегда ML arbitration)
    * unsupported: None (NEVER rule-accept, ML fallback)
- B4HybridRouter НЕ создаёт baseline'ы — берёт из TrainedBundle.
- B4HybridRouter переиспользует SelectiveRouter для ML-fallback.
- faq/unsupported **никогда** не rule-accepted, даже при большом hit_count.
- RouteDecision.reason имеет формат "rule:{domain}:hits={N}" для rule-accept.
- Output — единый тип `RouteDecision` (унификация с Task 4).

Запуск:
    cd study && python -m pytest d1/tests/test_b4_hybrid.py -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from d1.baselines.b0_rules import RulePrediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSES = ["anamnesis", "booking", "faq", "unsupported"]


class _FakeClassifier:
    """Минимальный stub с predict_proba + classes_."""

    def __init__(self, proba_map: dict[str, list[float]]) -> None:
        self.proba_map = proba_map
        self.classes_ = list(CLASSES)
        self._is_fitted = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.proba_map[t] for t in texts], dtype=float)

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.classes_[int(i)] for i in np.argmax(proba, axis=1)]


class _FakeRulesClassifier:
    """Stub B0 — возвращает заранее заготовленные RulePrediction."""

    def __init__(self, rule_map: dict[str, RulePrediction]) -> None:
        self.rule_map = rule_map

    def predict_with_confidence(self, texts: list[str]) -> list[RulePrediction]:
        return [self.rule_map[t] for t in texts]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.rule_map[t].route_domain for t in texts]


class _FakeBundle:
    """Stub TrainedBundle для изолированных тестов."""

    def __init__(self, models: dict) -> None:
        self.models = models

    def get(self, name: str):
        if name not in self.models:
            raise KeyError(name)
        return self.models[name]


def _proba(anamnesis=0.0, booking=0.0, faq=0.0, unsupported=0.0) -> list[float]:
    """Строка в порядке CLASSES."""
    return [anamnesis, booking, faq, unsupported]


# ---------------------------------------------------------------------------
# _parse_hit_count contract
# ---------------------------------------------------------------------------

def test_parse_hit_count_basic() -> None:
    from d1.baselines.b4_hybrid import _parse_hit_count

    assert _parse_hit_count("2_patterns") == 2
    assert _parse_hit_count("5_patterns") == 5
    assert _parse_hit_count("no_match") == 0


def test_parse_hit_count_single_hit() -> None:
    from d1.baselines.b4_hybrid import _parse_hit_count

    # "1_patterns" → 1 (ниже порога anamnesis=2 → ML fallback)
    assert _parse_hit_count("1_patterns") == 1


# ---------------------------------------------------------------------------
# RULE_ACCEPT_POLICY contract
# ---------------------------------------------------------------------------

def test_rule_accept_policy_structure() -> None:
    from d1.baselines.b4_hybrid import RULE_ACCEPT_POLICY

    # Все 4 домена представлены
    assert set(RULE_ACCEPT_POLICY) == {"anamnesis", "booking", "faq", "unsupported"}

    # anamnesis: >=2
    assert RULE_ACCEPT_POLICY["anamnesis"]["min_hit_count"] == 2
    # booking: >=3
    assert RULE_ACCEPT_POLICY["booking"]["min_hit_count"] == 3
    # faq: NEVER
    assert RULE_ACCEPT_POLICY["faq"]["min_hit_count"] is None
    # unsupported: NEVER
    assert RULE_ACCEPT_POLICY["unsupported"]["min_hit_count"] is None


def test_rule_accept_policy_has_rationale() -> None:
    """Каждое правило должно иметь rationale для аудита."""
    from d1.baselines.b4_hybrid import RULE_ACCEPT_POLICY

    for domain, policy in RULE_ACCEPT_POLICY.items():
        assert "rationale" in policy, f"Missing rationale for {domain}"
        assert policy["rationale"], f"Empty rationale for {domain}"


# ---------------------------------------------------------------------------
# Policy: rule-first accept (anamnesis + booking)
# ---------------------------------------------------------------------------

def _make_hybrid(
    rule_map: dict[str, RulePrediction],
    sparse_proba: dict[str, list[float]],
    dense_proba: dict[str, list[float]],
):
    from d1.baselines.b4_hybrid import B4HybridRouter
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds

    bundle = _FakeBundle({
        "B0_rules": _FakeRulesClassifier(rule_map),
        "B1.1_tfidf_lr": _FakeClassifier(sparse_proba),
        "B2.1_bge-m3_svc": _FakeClassifier(dense_proba),
    })
    selective = SelectiveRouter(
        sparse_model=bundle.get("B1.1_tfidf_lr"),
        dense_model=bundle.get("B2.1_bge-m3_svc"),
        thresholds=SelectiveThresholds(),
    )
    return B4HybridRouter(bundle=bundle, selective=selective)


def test_rule_accept_anamnesis_with_2_hits() -> None:
    """anamnesis + hit_count=2 → rule-accept, ML не вызывается."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("anamnesis", 0.7, "2_patterns")},
        # Sparse/dense возвращают другой домен, чтобы проверить что rule-first
        # действительно перекрывает ML
        sparse_proba={text: _proba(booking=0.8)},
        dense_proba={text: _proba(booking=0.8)},
    )
    decisions = hybrid.route_batch([text])
    d = decisions[0]
    assert d.action == "accept"
    assert d.label == "anamnesis"
    assert d.reason == "rule:anamnesis:hits=2"


def test_rule_accept_booking_with_3_hits() -> None:
    """booking + hit_count=3 → rule-accept."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("booking", 0.8, "3_patterns")},
        sparse_proba={text: _proba(faq=0.6)},
        dense_proba={text: _proba(faq=0.6)},
    )
    d = hybrid.route_batch([text])[0]
    assert d.action == "accept"
    assert d.label == "booking"
    assert d.reason == "rule:booking:hits=3"


def test_rule_accept_booking_below_threshold_falls_to_ml() -> None:
    """booking + hit_count=2 (ниже порога 3) → ML fallback."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("booking", 0.7, "2_patterns")},
        # Sparse/dense дают чёткий booking → ML accept через general_confident
        sparse_proba={text: _proba(booking=0.85)},
        dense_proba={text: _proba(booking=0.85)},
    )
    d = hybrid.route_batch([text])[0]
    assert d.action == "accept"
    assert d.reason == "general_confident"  # ML-принятое, не rule


def test_rule_accept_anamnesis_below_threshold_falls_to_ml() -> None:
    """anamnesis + hit_count=1 (ниже порога 2) → ML fallback."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("anamnesis", 0.6, "1_patterns")},
        sparse_proba={text: _proba(anamnesis=0.7)},
        dense_proba={text: _proba(anamnesis=0.7)},
    )
    d = hybrid.route_batch([text])[0]
    assert d.action == "accept"
    assert d.reason == "anamnesis_confident"  # ML R1


# ---------------------------------------------------------------------------
# CRITICAL SAFETY: faq/unsupported NEVER rule-accepted
# ---------------------------------------------------------------------------

def test_faq_never_rule_accepted_even_with_many_hits() -> None:
    """faq с hit_count=10 → ВСЁ РАВНО ML fallback (safety-critical контракт)."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("faq", 0.95, "10_patterns")},
        sparse_proba={text: _proba(faq=0.85)},
        dense_proba={text: _proba(faq=0.85)},
    )
    d = hybrid.route_batch([text])[0]
    # reason начинается НЕ с "rule:"
    assert not d.reason.startswith("rule:"), (
        f"faq должен всегда идти через ML, но получен rule-accept: {d.reason}"
    )


def test_unsupported_never_rule_accepted_even_with_many_hits() -> None:
    """unsupported с hit_count=10 → ВСЁ РАВНО ML fallback."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("unsupported", 0.95, "10_patterns")},
        sparse_proba={text: _proba(unsupported=0.8)},
        dense_proba={text: _proba(unsupported=0.8)},
    )
    d = hybrid.route_batch([text])[0]
    assert not d.reason.startswith("rule:"), (
        f"unsupported должен всегда идти через ML, но получен rule-accept: {d.reason}"
    )


# ---------------------------------------------------------------------------
# Rules no_match → ML fallback
# ---------------------------------------------------------------------------

def test_rules_no_match_falls_to_ml() -> None:
    """rules=no_match → ML fallback (hit_count=0 ниже любого порога)."""
    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("unsupported", 0.3, "no_match")},
        sparse_proba={text: _proba(faq=0.4, booking=0.3, anamnesis=0.2, unsupported=0.1)},
        dense_proba={text: _proba(faq=0.4, booking=0.3, anamnesis=0.2, unsupported=0.1)},
    )
    d = hybrid.route_batch([text])[0]
    # reason — из SelectiveRouter (не rule)
    assert not d.reason.startswith("rule:")


# ---------------------------------------------------------------------------
# Output type unification
# ---------------------------------------------------------------------------

def test_output_is_route_decision() -> None:
    """Output — единый RouteDecision (унификация с Task 4)."""
    from d1.baselines.b4_hybrid import B4HybridRouter
    from d1.baselines.selective_router import RouteDecision

    text = "q"
    hybrid = _make_hybrid(
        rule_map={text: RulePrediction("anamnesis", 0.7, "2_patterns")},
        sparse_proba={text: _proba(anamnesis=0.6)},
        dense_proba={text: _proba(anamnesis=0.6)},
    )
    decisions = hybrid.route_batch([text])
    assert len(decisions) == 1
    assert isinstance(decisions[0], RouteDecision)


# ---------------------------------------------------------------------------
# Contract: не создаёт модели, принимает bundle
# ---------------------------------------------------------------------------

def test_router_takes_models_from_bundle() -> None:
    """B4HybridRouter НЕ создаёт B0RulesClassifier() — берёт из bundle."""
    from d1.baselines.b4_hybrid import B4HybridRouter
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds

    text = "q"
    bundle = _FakeBundle({
        "B0_rules": _FakeRulesClassifier(
            {text: RulePrediction("anamnesis", 0.7, "2_patterns")},
        ),
        "B1.1_tfidf_lr": _FakeClassifier({text: _proba(anamnesis=0.6)}),
        "B2.1_bge-m3_svc": _FakeClassifier({text: _proba(anamnesis=0.6)}),
    })
    selective = SelectiveRouter(
        sparse_model=bundle.get("B1.1_tfidf_lr"),
        dense_model=bundle.get("B2.1_bge-m3_svc"),
        thresholds=SelectiveThresholds(),
    )
    hybrid = B4HybridRouter(bundle=bundle, selective=selective)

    # rules — тот же самый объект из bundle, не новая B0RulesClassifier
    assert hybrid.rules is bundle.get("B0_rules")


def test_custom_rule_policy_override() -> None:
    """RULE_ACCEPT_POLICY можно переопределить через конструктор."""
    from d1.baselines.b4_hybrid import B4HybridRouter
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds

    text = "q"
    bundle = _FakeBundle({
        "B0_rules": _FakeRulesClassifier(
            {text: RulePrediction("anamnesis", 0.6, "1_patterns")},
        ),
        "B1.1_tfidf_lr": _FakeClassifier({text: _proba(booking=0.4)}),
        "B2.1_bge-m3_svc": _FakeClassifier({text: _proba(booking=0.4)}),
    })
    selective = SelectiveRouter(
        sparse_model=bundle.get("B1.1_tfidf_lr"),
        dense_model=bundle.get("B2.1_bge-m3_svc"),
        thresholds=SelectiveThresholds(),
    )
    # Ослабляем порог anamnesis до 1
    custom_policy = {
        "anamnesis": {"min_hit_count": 1, "rationale": "test_override"},
        "booking": {"min_hit_count": 3, "rationale": "test"},
        "faq": {"min_hit_count": None, "rationale": "test"},
        "unsupported": {"min_hit_count": None, "rationale": "test"},
    }
    hybrid = B4HybridRouter(bundle=bundle, selective=selective, rule_policy=custom_policy)
    d = hybrid.route_batch([text])[0]
    assert d.reason == "rule:anamnesis:hits=1"  # принято по ослабленному порогу


# ---------------------------------------------------------------------------
# Integration smoke — реальный TrainedBundle
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_b4_hybrid_with_trained_bundle() -> None:
    """B4HybridRouter работает на реальном TrainedBundle."""
    from d1.baselines.b4_hybrid import B4HybridRouter
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds
    from d1.baselines.trained_bundle import train_bundle

    bundle = train_bundle(
        names=["B0_rules", "B1.1_tfidf_lr", "B2.1_bge-m3_svc"],
        use_cache=True,
    )
    selective = SelectiveRouter(
        sparse_model=bundle.get("B1.1_tfidf_lr"),
        dense_model=bundle.get("B2.1_bge-m3_svc"),
        thresholds=SelectiveThresholds(),
    )
    hybrid = B4HybridRouter(bundle=bundle, selective=selective)
    decisions = hybrid.route_batch([
        "болит зуб уже неделю, ноет постоянно",
        "сколько стоит имплантация",
        "хочу записаться на завтра",
    ])
    assert len(decisions) == 3
    for d in decisions:
        assert d.action in ("accept", "defer")
        # reason — либо rule:*, либо selective reason
        assert d.reason.startswith("rule:") or d.reason in {
            "anamnesis_confident",
            "faq_anamnesis_borderline",
            "sparse_dense_disagree",
            "general_confident",
            "low_confidence",
        }
