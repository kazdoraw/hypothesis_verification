"""B1: TF-IDF + LinearSVC baseline (§9.1 ТЗ).

Честный классический baseline для domain routing.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Дефолтное число fold-ов для CalibratedClassifierCV. На реальном train
# (~2K на 4 класса) cv=5 устойчивее, чем cv=3; smoke-тесты с микро-train
# фоллбекают через `_adaptive_cv` до minимально-валидного значения.
_DEFAULT_CALIBRATION_CV = 5


def _adaptive_cv(labels: list[str], requested: int = _DEFAULT_CALIBRATION_CV) -> int:
    """Возвращает min(requested, count_of_smallest_class).

    sklearn-CalibratedClassifierCV требует ``cv ≤ min_samples_per_class``.
    Без этого fallback тесты с micro-train (< requested примеров на класс)
    падают с ValueError.
    """
    if not labels:
        return 2
    min_count = min(Counter(labels).values())
    return max(2, min(requested, min_count))


# Параметры по умолчанию
_DEFAULT_TFIDF_PARAMS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50_000,
}

# NOTE: class_weight='balanced' убрано из дефолтов после аудита 2026-05-11.
# Train сильно дисбалансен (faq:anam:booking:unsupported ≈ 3.1:2.8:2.2:1.0),
# balanced разгоняет prior для минорных классов и снижает recall_anamnesis
# на anamnesis-heavy eval-сетах (safety_set, hard_test). Включается по
# желанию через head_params={"class_weight": "balanced"}.
_DEFAULT_SVC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 5000,
}

_DEFAULT_LR_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 2000,
    "solver": "lbfgs",
}


class B1TfidfClassifier:
    """TF-IDF + linear head classifier.

    char_wb n-grams (2,5) — работает хорошо для русского текста
    с опечатками и разговорным стилем.

    Args:
        head_type: "svc" (LinearSVC + CalibratedClassifierCV) или
                   "logistic" (LogisticRegression, нативный predict_proba)
    """

    def __init__(
        self,
        head_type: Literal["svc", "logistic"] = "svc",
        tfidf_params: dict[str, Any] | None = None,
        head_params: dict[str, Any] | None = None,
        calibrate: bool = True,
    ):
        self.head_type = head_type
        self.calibrate = calibrate
        self.tfidf_params = tfidf_params or {}
        self.head_params = head_params or {}
        self._tfidf_kw = {**_DEFAULT_TFIDF_PARAMS, **self.tfidf_params}
        self.pipeline: Pipeline | None = None
        self._is_fitted = False
        self.train_time_ms: float = 0.0

    def _build_head(self, labels: list[str]):
        """Строит классификатор с адаптивным cv для CalibratedClassifierCV."""
        if self.head_type == "logistic":
            lr_kw = {**_DEFAULT_LR_PARAMS, **self.head_params}
            return LogisticRegression(**lr_kw)
        svc_kw = {**_DEFAULT_SVC_PARAMS, **self.head_params}
        base_svc = LinearSVC(**svc_kw)
        if not self.calibrate:
            return base_svc
        return CalibratedClassifierCV(
            base_svc, cv=_adaptive_cv(labels), method="sigmoid",
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Обучение на train данных."""
        clf = self._build_head(labels)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self._tfidf_kw)),
            ("clf", clf),
        ])
        t0 = time.perf_counter()
        self.pipeline.fit(texts, labels)
        self.train_time_ms = (time.perf_counter() - t0) * 1000
        self._is_fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        """Batch prediction → route_domain labels."""
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability estimates (если калибровка включена).

        Returns:
            np.ndarray shape (n_samples, n_classes)
        """
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.named_steps["clf"].predict_proba(
                self.pipeline.named_steps["tfidf"].transform(texts)
            )
        raise AttributeError("predict_proba недоступен без калибровки")

    @property
    def classes_(self) -> list[str]:
        """Порядок классов."""
        return self.pipeline.classes_.tolist()

    def get_params_summary(self) -> dict[str, Any]:
        """Сводка параметров для отчёта."""
        tfidf = self.pipeline.named_steps["tfidf"]
        return {
            "head_type": self.head_type,
            "analyzer": tfidf.analyzer,
            "ngram_range": tfidf.ngram_range,
            "max_features": tfidf.max_features,
            "n_features": len(tfidf.vocabulary_) if self._is_fitted else 0,
            "train_time_ms": round(self.train_time_ms, 1),
        }
