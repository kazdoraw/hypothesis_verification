"""B1: TF-IDF + LinearSVC baseline (§9.1 ТЗ).

Честный классический baseline для domain routing.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Параметры по умолчанию
_DEFAULT_TFIDF_PARAMS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50_000,
}

_DEFAULT_SVC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 5000,
    "class_weight": "balanced",
}

_DEFAULT_LR_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
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
        tfidf_kw = {**_DEFAULT_TFIDF_PARAMS, **(tfidf_params or {})}

        if head_type == "logistic":
            lr_kw = {**_DEFAULT_LR_PARAMS, **(head_params or {})}
            clf = LogisticRegression(**lr_kw)
        else:
            svc_kw = {**_DEFAULT_SVC_PARAMS, **(head_params or {})}
            base_svc = LinearSVC(**svc_kw)
            clf = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid") if calibrate else base_svc

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kw)),
            ("clf", clf),
        ])
        self._is_fitted = False
        self.train_time_ms: float = 0.0

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Обучение на train данных."""
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
