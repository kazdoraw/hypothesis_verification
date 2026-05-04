"""Unit + integration тесты для SelectiveRouter (Task 4 roadmap).

Проверяемые контракты:
- SelectiveThresholds: frozen dataclass, дефолты совпадают с планом.
- RouteDecision: содержит label, confidence, margin, action, sparse_dense_agree, reason.
- Policy priority (строго сверху вниз):
    R1 anamnesis_confident
    R2 faq_anamnesis_borderline (defer safety-conservative)
    R3 sparse_dense_disagree (defer)
    R4 general_confident
    R5 low_confidence (fallback defer)
- SelectiveRouter НЕ обучает модели, принимает fitted.
- Два режима eval:
    * compute_closed_set_report — только accepted, defer исключены.
    * compute_selective_report — selective metrics + safety_on_accepted.
- Safety guard: `false_negative_deferred` — urgent/anamnesis → defer
  (явный подсчёт, но не превращает selective в закрытый routing).

Запуск:
    cd study && python -m pytest d1/tests/test_selective_router.py -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: фейковые fitted baseline'ы (без обучения → быстрые тесты)
# ---------------------------------------------------------------------------

CLASSES = ["anamnesis", "booking", "faq", "unsupported"]


class _FakeClassifier:
    """Минимальный stub с predict_proba + classes_ + predict.

    `proba_map`: dict[text -> list[float]] в порядке CLASSES.
    """

    def __init__(self, proba_map: dict[str, list[float]]) -> None:
        self.proba_map = proba_map
        self.classes_ = list(CLASSES)
        self._is_fitted = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.proba_map[t] for t in texts], dtype=float)

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.classes_[int(i)] for i in np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# SelectiveThresholds — контракт dataclass
# ---------------------------------------------------------------------------

def test_selective_thresholds_is_frozen_dataclass() -> None:
    from d1.baselines.selective_router import SelectiveThresholds

    assert dataclasses.is_dataclass(SelectiveThresholds)
    t = SelectiveThresholds()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.anamnesis_threshold = 0.9  # type: ignore[misc]


def test_selective_thresholds_defaults_match_plan() -> None:
    from d1.baselines.selective_router import SelectiveThresholds

    t = SelectiveThresholds()
    assert t.anamnesis_threshold == pytest.approx(0.55)
    assert t.faq_anamnesis_margin == pytest.approx(0.15)
    assert t.general_threshold == pytest.approx(0.70)


def test_production_thresholds_matches_retune() -> None:
    """PRODUCTION_THRESHOLDS — SSoT для retuned values (Task 5 post-eval).

    Изменение этой константы должно сопровождаться новым threshold_sweep
    и обновлением обоснования в docstring.
    """
    from d1.baselines.selective_router import PRODUCTION_THRESHOLDS, SelectiveThresholds

    assert isinstance(PRODUCTION_THRESHOLDS, SelectiveThresholds)
    assert PRODUCTION_THRESHOLDS.anamnesis_threshold == pytest.approx(0.48)
    assert PRODUCTION_THRESHOLDS.faq_anamnesis_margin == pytest.approx(0.15)
    assert PRODUCTION_THRESHOLDS.general_threshold == pytest.approx(0.63)


# ---------------------------------------------------------------------------
# RouteDecision — контракт dataclass
# ---------------------------------------------------------------------------

def test_route_decision_has_required_fields() -> None:
    from d1.baselines.selective_router import RouteDecision

    fields = {f.name for f in dataclasses.fields(RouteDecision)}
    required = {
        "label", "confidence", "margin",
        "action", "sparse_dense_agree", "reason",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


# ---------------------------------------------------------------------------
# Policy rules — 5 приоритетов
# ---------------------------------------------------------------------------

def _make_router(
    sparse_proba: dict[str, list[float]],
    dense_proba: dict[str, list[float]],
    thresholds=None,
):
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds

    sparse = _FakeClassifier(sparse_proba)
    dense = _FakeClassifier(dense_proba)
    return SelectiveRouter(
        sparse_model=sparse,
        dense_model=dense,
        thresholds=thresholds or SelectiveThresholds(),
    )


def _proba_row(anamnesis=0.0, booking=0.0, faq=0.0, unsupported=0.0) -> list[float]:
    """Строка в порядке CLASSES = [anamnesis, booking, faq, unsupported]."""
    return [anamnesis, booking, faq, unsupported]


def test_rule_1_anamnesis_confident_accept() -> None:
    """top1=anamnesis, conf >= 0.55 → accept/anamnesis_confident."""
    text = "q"
    router = _make_router(
        sparse_proba={text: _proba_row(anamnesis=0.6, faq=0.2, booking=0.1, unsupported=0.1)},
        dense_proba={text: _proba_row(anamnesis=0.7, faq=0.1, booking=0.1, unsupported=0.1)},
    )
    decisions = router.route_batch([text])
    d = decisions[0]
    assert d.label == "anamnesis"
    assert d.action == "accept"
    assert d.reason == "anamnesis_confident"


def test_rule_2_faq_anamnesis_borderline_defer() -> None:
    """top1=faq, anamnesis в top2, margin<0.15 → defer/faq_anamnesis_borderline."""
    text = "q"
    # faq=0.50, anamnesis=0.45, margin=0.05 < 0.15
    router = _make_router(
        sparse_proba={text: _proba_row(faq=0.50, anamnesis=0.45, booking=0.03, unsupported=0.02)},
        dense_proba={text: _proba_row(faq=0.50, anamnesis=0.45, booking=0.03, unsupported=0.02)},
    )
    d = router.route_batch([text])[0]
    assert d.label == "faq"
    assert d.action == "defer"
    assert d.reason == "faq_anamnesis_borderline"


def test_rule_3_sparse_dense_disagree_defer() -> None:
    """sparse != dense → defer/sparse_dense_disagree (если не сработали R1/R2)."""
    text = "q"
    # sparse top1=booking, dense top1=faq (разные) — не anamnesis, не R2.
    router = _make_router(
        sparse_proba={text: _proba_row(booking=0.6, faq=0.2, anamnesis=0.1, unsupported=0.1)},
        dense_proba={text: _proba_row(faq=0.6, booking=0.2, anamnesis=0.1, unsupported=0.1)},
    )
    d = router.route_batch([text])[0]
    assert d.action == "defer"
    assert d.reason == "sparse_dense_disagree"
    assert d.sparse_dense_agree is False


def test_rule_4_general_confident_accept() -> None:
    """sparse==dense, conf >= 0.70 → accept/general_confident."""
    text = "q"
    router = _make_router(
        sparse_proba={text: _proba_row(booking=0.8, faq=0.1, anamnesis=0.05, unsupported=0.05)},
        dense_proba={text: _proba_row(booking=0.85, faq=0.1, anamnesis=0.03, unsupported=0.02)},
    )
    d = router.route_batch([text])[0]
    assert d.label == "booking"
    assert d.action == "accept"
    assert d.reason == "general_confident"


def test_rule_5_low_confidence_defer_fallback() -> None:
    """sparse==dense, conf < 0.70, не R1/R2 → defer/low_confidence."""
    text = "q"
    # top1=booking conf=0.40, нет anamnesis R1, нет R2 (faq), sparse==dense
    router = _make_router(
        sparse_proba={text: _proba_row(booking=0.40, faq=0.30, anamnesis=0.15, unsupported=0.15)},
        dense_proba={text: _proba_row(booking=0.40, faq=0.30, anamnesis=0.15, unsupported=0.15)},
    )
    d = router.route_batch([text])[0]
    assert d.action == "defer"
    assert d.reason == "low_confidence"


def test_rule_priority_r1_before_r2() -> None:
    """R1 (anamnesis_confident) имеет приоритет над R2 (faq_anamnesis_borderline).

    Если top1=anamnesis и conf>=threshold → сразу accept, R2 не триггерится.
    """
    text = "q"
    router = _make_router(
        sparse_proba={text: _proba_row(anamnesis=0.6, faq=0.3, booking=0.05, unsupported=0.05)},
        dense_proba={text: _proba_row(anamnesis=0.6, faq=0.3, booking=0.05, unsupported=0.05)},
    )
    d = router.route_batch([text])[0]
    assert d.action == "accept"
    assert d.reason == "anamnesis_confident"


def test_rule_priority_r2_before_r3() -> None:
    """R2 (faq_anamnesis_borderline) имеет приоритет над R3 (disagreement).

    Даже если sparse=faq, dense=anamnesis (disagree), R2 должен выиграть
    т.к. это более специфичный safety-сигнал (faq с anamnesis в top2).
    """
    text = "q"
    router = _make_router(
        sparse_proba={text: _proba_row(faq=0.50, anamnesis=0.45, booking=0.03, unsupported=0.02)},
        dense_proba={text: _proba_row(anamnesis=0.6, faq=0.3, booking=0.05, unsupported=0.05)},
    )
    d = router.route_batch([text])[0]
    assert d.action == "defer"
    assert d.reason == "faq_anamnesis_borderline"


# ---------------------------------------------------------------------------
# Contract: SelectiveRouter НЕ обучает модели
# ---------------------------------------------------------------------------

def test_router_does_not_call_fit() -> None:
    """SelectiveRouter принимает уже fitted модели, не вызывает .fit()."""
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds

    class _TrackerClassifier(_FakeClassifier):
        def __init__(self, proba_map):
            super().__init__(proba_map)
            self.fit_calls = 0

        def fit(self, *_args, **_kwargs):
            self.fit_calls += 1

    text = "q"
    sparse = _TrackerClassifier({text: _proba_row(booking=0.8, faq=0.2)})
    dense = _TrackerClassifier({text: _proba_row(booking=0.8, faq=0.2)})
    router = SelectiveRouter(
        sparse_model=sparse, dense_model=dense,
        thresholds=SelectiveThresholds(),
    )
    router.route_batch([text])
    assert sparse.fit_calls == 0
    assert dense.fit_calls == 0


# ---------------------------------------------------------------------------
# Reports: closed-set + selective
# ---------------------------------------------------------------------------

def test_closed_set_report_excludes_defer() -> None:
    """compute_closed_set_report: defer-кейсы исключаются (НЕ маппятся в anamnesis)."""
    from d1.baselines.selective_router import RouteDecision, compute_closed_set_report

    decisions = [
        RouteDecision(label="anamnesis", confidence=0.8, margin=0.5,
                      action="accept", sparse_dense_agree=True,
                      reason="anamnesis_confident"),
        RouteDecision(label="booking", confidence=0.4, margin=0.1,
                      action="defer", sparse_dense_agree=True,
                      reason="low_confidence"),
        RouteDecision(label="faq", confidence=0.9, margin=0.7,
                      action="accept", sparse_dense_agree=True,
                      reason="general_confident"),
    ]
    y_true = ["anamnesis", "anamnesis", "faq"]
    # closed-set: только accepted (1-й и 3-й), defer исключён
    report = compute_closed_set_report(y_true, decisions)
    assert report.n_samples == 2  # accepted only
    assert report.accuracy == pytest.approx(1.0)


def test_selective_report_fields() -> None:
    """compute_selective_report возвращает все поля из плана."""
    from d1.baselines.selective_router import RouteDecision, compute_selective_report

    decisions = [
        RouteDecision(label="anamnesis", confidence=0.8, margin=0.5,
                      action="accept", sparse_dense_agree=True,
                      reason="anamnesis_confident"),
        RouteDecision(label="booking", confidence=0.4, margin=0.1,
                      action="defer", sparse_dense_agree=True,
                      reason="low_confidence"),
        RouteDecision(label="faq", confidence=0.9, margin=0.7,
                      action="accept", sparse_dense_agree=True,
                      reason="general_confident"),
    ]
    y_true = ["anamnesis", "anamnesis", "faq"]
    rep = compute_selective_report(y_true, decisions, router_name="test_router")

    # coverage = 2/3
    assert rep.coverage == pytest.approx(2 / 3)
    # accepted_accuracy = 2/2
    assert rep.accepted_accuracy == pytest.approx(1.0)
    # defer_rate = 1/3
    assert rep.defer_rate == pytest.approx(1 / 3)
    # accepted_recall_anamnesis: 1 anamnesis в accepted корректно классифицирован
    assert rep.accepted_recall_anamnesis == pytest.approx(1.0)
    # router_name передаётся
    assert rep.router_name == "test_router"


def test_selective_report_false_negative_deferred() -> None:
    """false_negative_deferred считает urgent/anamnesis → defer."""
    from d1.baselines.selective_router import RouteDecision, compute_selective_report

    decisions = [
        # urgent anamnesis → defer — это FN для safety
        RouteDecision(label="faq", confidence=0.4, margin=0.05,
                      action="defer", sparse_dense_agree=True,
                      reason="faq_anamnesis_borderline"),
        # обычный booking → accept
        RouteDecision(label="booking", confidence=0.8, margin=0.5,
                      action="accept", sparse_dense_agree=True,
                      reason="general_confident"),
    ]
    y_true = ["anamnesis", "booking"]
    rep = compute_selective_report(y_true, decisions, router_name="test")
    assert rep.false_negative_deferred == 1


# ---------------------------------------------------------------------------
# Integration smoke: проверка что SelectiveRouter работает на TrainedBundle
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_selective_router_with_trained_bundle() -> None:
    """SelectiveRouter интегрируется с train_bundle() (B1.1 + B2.1)."""
    from d1.baselines.selective_router import SelectiveRouter, SelectiveThresholds
    from d1.baselines.trained_bundle import train_bundle

    bundle = train_bundle(names=["B1.1_tfidf_lr", "B2.1_bge-m3_svc"], use_cache=True)
    router = SelectiveRouter(
        sparse_model=bundle.get("B1.1_tfidf_lr"),
        dense_model=bundle.get("B2.1_bge-m3_svc"),
        thresholds=SelectiveThresholds(),
    )
    decisions = router.route_batch(["сколько стоит имплант", "болит зуб мудрости"])
    assert len(decisions) == 2
    for d in decisions:
        assert d.action in ("accept", "defer")
        assert d.reason in {
            "anamnesis_confident",
            "faq_anamnesis_borderline",
            "sparse_dense_disagree",
            "general_confident",
            "low_confidence",
        }
