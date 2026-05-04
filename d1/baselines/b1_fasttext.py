"""B1.3: fastText baseline (lightweight production candidate).

API contract совпадает с B1/B2: fit/predict/predict_proba/classes_/get_params_summary.

Особенности:
- **Lazy import**: `import fasttext` внутри методов. ImportError → RuntimeError
  с подсказкой `pip install fasttext-wheel`.
- **Tmpfile management**: train format `__label__<cls> <text>` пишется в
  `tempfile.NamedTemporaryFile`, удаляется после `train_supervised`.
- **Seed**: fasttext не имеет seed kwarg, но `thread=1` даёт детерминированность.
- **Serialization**: explicit `save(path_dir)` / `load(path_dir)` (joblib не
  пишет C++ object). В `trained_bundle._save_to_cache` ветка для B1FastText.

Запуск (smoke):
    python -m d1.baselines.b1_fasttext
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Простой нормализатор: fasttext чувствителен к спец-символам и переводам строк.
_NORM_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Нормализация для fasttext: lower + collapse whitespace."""
    return _NORM_RE.sub(" ", text.strip().lower())


def _import_fasttext():
    """Lazy import с понятной ошибкой если пакет отсутствует."""
    try:
        import fasttext
        return fasttext
    except ImportError as exc:
        raise RuntimeError(
            "fasttext не установлен: pip install fasttext-wheel"
        ) from exc


class B1FastTextClassifier:
    """fastText supervised baseline (linear bag-of-ngrams + softmax).

    Args:
        dim: размерность word embeddings (default 100).
        epoch: число эпох обучения (default 25).
        lr: learning rate (default 0.5 — стандарт fasttext).
        wordNgrams: max ngram order (default 2).
        minCount: минимальная частота токена (default 1 для маленького train).
        thread: число потоков (1 — детерминированность).
    """

    def __init__(
        self,
        dim: int = 100,
        epoch: int = 25,
        lr: float = 0.5,
        wordNgrams: int = 2,
        minCount: int = 1,
        thread: int = 1,
    ) -> None:
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.minCount = minCount
        self.thread = thread
        self._model = None  # fasttext _FastText (после fit)
        self._classes: list[str] = []
        self._is_fitted = False
        self.train_size: int = 0
        self.train_time_ms: float = 0.0

    # -- training ---------------------------------------------------------

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Train в tempfile (`__label__<cls> <text>`)."""
        import time

        if len(texts) != len(labels):
            raise ValueError("len(texts) != len(labels)")
        fasttext = _import_fasttext()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False,
        ) as f:
            tmp_path = Path(f.name)
            for text, label in zip(texts, labels):
                f.write(f"__label__{label} {_normalize_text(text)}\n")

        try:
            t0 = time.perf_counter()
            self._model = fasttext.train_supervised(
                input=str(tmp_path),
                dim=self.dim,
                epoch=self.epoch,
                lr=self.lr,
                wordNgrams=self.wordNgrams,
                minCount=self.minCount,
                thread=self.thread,
                verbose=0,
            )
            self.train_time_ms = (time.perf_counter() - t0) * 1000
        finally:
            tmp_path.unlink(missing_ok=True)

        # classes_: order via fasttext model.labels (отсортируем для
        # детерминированного порядка между запусками).
        raw_labels = [lbl.replace("__label__", "") for lbl in self._model.labels]
        self._classes = sorted(set(raw_labels))
        self._is_fitted = True
        self.train_size = len(texts)

    # -- inference --------------------------------------------------------

    def predict(self, texts: list[str]) -> list[str]:
        """Top-1 label."""
        self._ensure_fitted()
        normalized = [_normalize_text(t) for t in texts]
        labels, _probs = self._model.predict(normalized, k=1)
        return [lbl[0].replace("__label__", "") for lbl in labels]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability per класс в порядке `classes_`.

        fasttext возвращает top-k только для requested labels, поэтому
        запрашиваем top-K где K=len(classes_) и raster'им в массив.
        """
        self._ensure_fitted()
        normalized = [_normalize_text(t) for t in texts]
        k = len(self._classes)
        labels_batch, probs_batch = self._model.predict(normalized, k=k)
        class_to_idx = {cls: i for i, cls in enumerate(self._classes)}
        out = np.zeros((len(texts), k), dtype=float)
        for i, (labels, probs) in enumerate(zip(labels_batch, probs_batch)):
            for lbl, prob in zip(labels, probs):
                cls = lbl.replace("__label__", "")
                if cls in class_to_idx:
                    out[i, class_to_idx[cls]] = prob
        return out

    @property
    def classes_(self) -> list[str]:
        self._ensure_fitted()
        return list(self._classes)

    def get_params_summary(self) -> dict[str, Any]:
        return {
            "model": "fasttext",
            "dim": self.dim,
            "epoch": self.epoch,
            "lr": self.lr,
            "wordNgrams": self.wordNgrams,
            "minCount": self.minCount,
            "thread": self.thread,
            "train_time_ms": round(self.train_time_ms, 1),
            "train_size": self.train_size,
        }

    # -- serialization (joblib не работает с C++ моделью) -----------------

    _MODEL_FILENAME = "fasttext.bin"
    _META_FILENAME = "fasttext_meta.json"

    def save(self, path_dir: Path) -> None:
        """Сохранить модель в директорию (бинарный + метаданные)."""
        self._ensure_fitted()
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path_dir / self._MODEL_FILENAME))
        meta = {
            "classes_": self._classes,
            "params": {
                "dim": self.dim,
                "epoch": self.epoch,
                "lr": self.lr,
                "wordNgrams": self.wordNgrams,
                "minCount": self.minCount,
                "thread": self.thread,
            },
            "train_size": self.train_size,
            "train_time_ms": self.train_time_ms,
        }
        (path_dir / self._META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path_dir: Path) -> "B1FastTextClassifier":
        """Восстановить fitted classifier из директории."""
        path_dir = Path(path_dir)
        meta_path = path_dir / cls._META_FILENAME
        model_path = path_dir / cls._MODEL_FILENAME
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"fasttext bundle отсутствует в {path_dir}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        params = meta["params"]
        instance = cls(**params)
        fasttext = _import_fasttext()
        instance._model = fasttext.load_model(str(model_path))
        instance._classes = list(meta["classes_"])
        instance._is_fitted = True
        instance.train_size = int(meta.get("train_size", 0))
        instance.train_time_ms = float(meta.get("train_time_ms", 0.0))
        return instance

    # -- internals --------------------------------------------------------

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("B1FastTextClassifier не обучен (вызовите .fit)")


__all__ = ["B1FastTextClassifier"]
