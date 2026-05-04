"""Smoke tests: baselines обучаются и предсказывают без падений.

Запуск:
    cd study && python -m pytest d1/tests/test_baselines_smoke.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from d1.baselines.b0_rules import B0RulesClassifier
from d1.baselines.b1_tfidf import B1TfidfClassifier
from d1.baselines.b2_embedding import B2EmbeddingClassifier
from d1.baselines.eval_metrics import (
    SafetyReport,
    compute_all_metrics,
    compute_safety_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELS = ["anamnesis", "faq", "booking", "unsupported"]

_TOY_TEXTS = [
    "болит зуб уже неделю",
    "опухла щека после удаления",
    "температура и кровоточат десны",
    "ноет нижняя челюсть",
    "сколько стоит чистка зубов",
    "какие врачи работают по выходным",
    "где находится ваша клиника",
    "есть ли рассрочка на лечение",
    "хочу записаться на осмотр",
    "запишите меня к ортодонту на завтра",
    "можно перенести запись на следующую неделю",
    "нужен приём к терапевту",
    "привет",
    "спасибо",
    "ок",
    "расскажите про блокчейн",
]

_TOY_LABELS = [
    "anamnesis", "anamnesis", "anamnesis", "anamnesis",
    "faq", "faq", "faq", "faq",
    "booking", "booking", "booking", "booking",
    "unsupported", "unsupported", "unsupported", "unsupported",
]


@pytest.fixture(scope="module")
def toy_data() -> tuple[list[str], list[str]]:
    return _TOY_TEXTS, _TOY_LABELS


# ---------------------------------------------------------------------------
# B0: rules
# ---------------------------------------------------------------------------

class TestB0:
    def test_predict_returns_valid_labels(self, toy_data: tuple) -> None:
        texts, _ = toy_data
        b0 = B0RulesClassifier()
        preds = b0.predict(texts)
        assert len(preds) == len(texts)
        for p in preds:
            assert isinstance(p, str)


# ---------------------------------------------------------------------------
# B1: TF-IDF
# ---------------------------------------------------------------------------

class TestB1:
    def test_fit_predict(self, toy_data: tuple) -> None:
        texts, labels = toy_data
        b1 = B1TfidfClassifier()
        b1.fit(texts, labels)
        preds = b1.predict(texts)
        assert len(preds) == len(texts)
        for p in preds:
            assert p in LABELS

    def test_head_type_logistic(self, toy_data: tuple) -> None:
        texts, labels = toy_data
        b1 = B1TfidfClassifier(head_type="logistic")
        b1.fit(texts, labels)
        preds = b1.predict(texts)
        assert len(preds) == len(texts)
        proba = b1.predict_proba(texts)
        assert proba.shape == (len(texts), len(LABELS))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# B2: Embedding (основной — ранее ломался на multiclass)
# ---------------------------------------------------------------------------

class TestB2:
    @pytest.fixture(scope="class")
    def fitted_b2(self, toy_data: tuple) -> B2EmbeddingClassifier:
        texts, labels = toy_data
        b2 = B2EmbeddingClassifier()
        b2.fit(texts, labels)
        return b2

    def test_fit_does_not_crash(self, fitted_b2: B2EmbeddingClassifier) -> None:
        assert fitted_b2._is_fitted

    def test_predict_valid_labels(
        self,
        fitted_b2: B2EmbeddingClassifier,
        toy_data: tuple,
    ) -> None:
        texts, _ = toy_data
        preds = fitted_b2.predict(texts)
        assert len(preds) == len(texts)
        for p in preds:
            assert p in LABELS

    def test_predict_proba_shape(
        self,
        fitted_b2: B2EmbeddingClassifier,
        toy_data: tuple,
    ) -> None:
        texts, _ = toy_data
        proba = fitted_b2.predict_proba(texts)
        assert proba.shape == (len(texts), len(LABELS))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_head_type_svc(self, toy_data: tuple) -> None:
        texts, labels = toy_data
        b2 = B2EmbeddingClassifier(head_type="svc")
        b2.fit(texts, labels)
        preds = b2.predict(texts)
        assert len(preds) == len(texts)
        for p in preds:
            assert p in LABELS
        proba = b2.predict_proba(texts)
        assert proba.shape == (len(texts), len(LABELS))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_head_type_centroid_no_proba(self, toy_data: tuple) -> None:
        texts, labels = toy_data
        b2 = B2EmbeddingClassifier(head_type="centroid")
        b2.fit(texts, labels)
        preds = b2.predict(texts)
        assert len(preds) == len(texts)
        with pytest.raises(AttributeError):
            b2.predict_proba(texts)

    def test_determinism(self, toy_data: tuple) -> None:
        texts, labels = toy_data
        results = []
        for _ in range(2):
            b2 = B2EmbeddingClassifier()
            b2.fit(texts, labels)
            results.append(b2.predict(texts))
        assert results[0] == results[1], "B2 должен быть детерминистичен"


# ---------------------------------------------------------------------------
# Metrics: compute_safety_report serialization
# ---------------------------------------------------------------------------

class TestSafetyReport:
    def test_misrouted_to_uses_native_str(self) -> None:
        y_true = ["anamnesis"] * 5
        y_pred = ["anamnesis", "anamnesis", "faq", "booking", "anamnesis"]
        urgency = ["urgent"] * 5

        sr = compute_safety_report(
            y_true=y_true,
            y_pred=y_pred,
            urgency=urgency,
            baseline_name="test",
        )
        for key in sr.misrouted_to:
            assert type(key) is str, f"Ожидается str, получен {type(key)}"

    def test_summary_dict_json_serializable(self) -> None:
        y_true = ["anamnesis"] * 3
        y_pred = ["anamnesis", "faq", "booking"]
        urgency = ["urgent"] * 3

        sr = compute_safety_report(
            y_true=y_true,
            y_pred=y_pred,
            urgency=urgency,
            baseline_name="test",
        )
        # Не должен бросать TypeError при сериализации
        json.dumps(sr.summary_dict(), ensure_ascii=False)
