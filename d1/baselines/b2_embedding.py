"""B2: Strong embedding baseline (§9.1 ТЗ).

Encoder (sentence-transformers) + linear head.
По ТЗ: не rubert-tiny2, а минимум один из e5/bge-m3.

Два режима:
- linear_head: LogisticRegression на эмбеддингах (основной)
- nearest_centroid: prototype classifier (дополнительный)
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC


_DEFAULT_CALIBRATION_CV = 5


def _adaptive_cv(labels: list[str], requested: int = _DEFAULT_CALIBRATION_CV) -> int:
    """min(requested, count_smallest_class). Защищает smoke-тесты с micro-train."""
    if not labels:
        return 2
    min_count = min(Counter(labels).values())
    return max(2, min(requested, min_count))


# Модели-кандидаты 
EMBEDDING_MODELS = {
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
}

DEFAULT_MODEL = "BAAI/bge-m3"

# NOTE: class_weight='balanced' убрано из дефолтов после аудита 2026-05-11.
# При сильном дисбалансе train balanced снижает recall_anamnesis на
# anamnesis-heavy eval-сетах. Включается через head_params при необходимости.
_LOGISTIC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 2000,
    "solver": "lbfgs",
}

_SVC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 5000,
}


class B2EmbeddingClassifier:
    """Embedding + linear head classifier.

    Args:
        model_name: HuggingFace model name (sentence-transformers compatible)
        head_type: "linear" (LogisticRegression), "centroid" (NearestCentroid)
                   или "svc" (LinearSVC + CalibratedClassifierCV)
        head_params: kwargs мержатся с дефолтными `_LOGISTIC_PARAMS` /
            `_SVC_PARAMS`. Для centroid игнорируется (NearestCentroid не имеет
            гиперпараметров уровня C/max_iter). Используется в B2.3 (tuned).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        head_type: Literal["linear", "centroid", "svc"] = "linear",
        head_params: dict[str, Any] | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.head_type = head_type
        self.head_params = head_params or {}
        # Устройство для SentenceTransformer. None → auto (MPS/CUDA/CPU).
        # Для production-aligned latency benchmark используем явный 'cpu'.
        self.device = device
        self._encoder = None
        self._head = None  # fitted classifier (LogisticRegression or NearestCentroid)
        self._is_fitted = False
        self.train_time_ms: float = 0.0
        self.encode_time_ms: float = 0.0

    def _get_encoder(self):
        """Lazy-load sentence-transformers модели (offline из кэша).

        При указанном `self.device` загружает энкодер сразу на нужное устройство;
        иначе sentence-transformers сам выбирает доступный (MPS/CUDA/CPU).
        """
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            from d1.config import resolve_model_path

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device

            self._encoder = SentenceTransformer(
                resolve_model_path(self.model_name),
                **kwargs,
            )
        return self._encoder

    def set_device(self, device: str) -> None:
        """Принудительно перенести encoder на указанное устройство.

        Используется в latency benchmark для production-aligned CPU-замера:
        даже если модель была обучена на MPS/CUDA, инференс замеряем на CPU
        (production deployment работает на CPU).

        Args:
            device: 'cpu', 'cuda', 'mps' и др. — любая строка, понятная torch.
        """
        self.device = device
        if self._encoder is not None:
            self._encoder.to(device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode текстов.

        Для e5 моделей добавляем prefix "query: " по рекомендации авторов.
        sentence-transformers возвращает уже L2-нормированный float32 при
        ``normalize_embeddings=True`` — повторная нормализация после
        ``astype(float64)`` не нужна (она ничего не меняет, только тратит
        время).
        """
        encoder = self._get_encoder()
        is_e5 = "e5" in self.model_name.lower()
        if is_e5:
            texts = [f"query: {t}" for t in texts]
        t0 = time.perf_counter()
        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        self.encode_time_ms = (time.perf_counter() - t0) * 1000

        return np.asarray(embeddings, dtype=np.float64)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Обучение: encode train → fit head."""
        X = self._encode(texts)

        t0 = time.perf_counter()
        if self.head_type == "linear":
            params = {**_LOGISTIC_PARAMS, **self.head_params}
            self._head = LogisticRegression(**params)
        elif self.head_type == "svc":
            params = {**_SVC_PARAMS, **self.head_params}
            self._head = CalibratedClassifierCV(
                LinearSVC(**params), cv=_adaptive_cv(labels), method="sigmoid",
            )
        else:
            self._head = NearestCentroid()
        # Apple Accelerate BLAS выдаёт ложные RuntimeWarning (matmul
        # overflow/divide-by-zero) на macOS — результат детерминистичен.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*(matmul|overflow|divide by zero).*",
                category=RuntimeWarning,
            )
            self._head.fit(X, labels)
        head_time = (time.perf_counter() - t0) * 1000

        self.train_time_ms = self.encode_time_ms + head_time
        self._is_fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        """Batch prediction."""
        X = self._encode(texts)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*(matmul|overflow|divide by zero).*",
                category=RuntimeWarning,
            )
            return self._head.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability estimates (для linear и svc head).

        Returns:
            np.ndarray shape (n_samples, n_classes)
        """
        if self.head_type not in {"linear", "svc"}:
            raise AttributeError(
                "predict_proba доступен только для linear и svc head"
            )
        X = self._encode(texts)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*(matmul|overflow|divide by zero).*",
                category=RuntimeWarning,
            )
            return self._head.predict_proba(X)

    @property
    def classes_(self) -> list[str]:
        """Порядок классов."""
        return self._head.classes_.tolist()

    def get_params_summary(self) -> dict[str, Any]:
        """Сводка параметров."""
        dim = 0
        if self._encoder is not None:
            dim = self._encoder.get_sentence_embedding_dimension()
        return {
            "model": self.model_name,
            "head_type": self.head_type,
            "embedding_dim": dim,
            "train_time_ms": round(self.train_time_ms, 1),
        }
