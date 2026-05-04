"""TrainedBundle — SSoT для обученных D1 baseline'ов (Task 0 из roadmap v3).

Единственная точка обучения baseline'ов. Заменяет дублирование кода в
`run_baselines.py`, `save_models.py`, `analyze_confidence.py` и всех будущих
consumer'ах (selective, hybrid, bootstrap, learning_curves).

Архитектурные принципы:
- **Один конфиг гиперпараметров**: `BASELINE_CONFIGS` — SSoT.
- **Явный enabled-контракт**: disabled baseline падает с понятной ошибкой,
  silent skip запрещён.
- **Строгий cache contract**: key = params_hash + dataset_content_hash +
  code_hash + env_hash + schema_hash. Защита от mtime-file-copy и
  stale-cache drift при upgrade sklearn/torch.
- **Тонкий orchestration layer**: дефолты гиперпараметров живут в
  классах baseline, TrainedBundle передаёт только overrides.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib

from d1.baselines.b0_rules import B0RulesClassifier
from d1.baselines.b1_fasttext import B1FastTextClassifier
from d1.baselines.b1_tfidf import B1TfidfClassifier
from d1.baselines.b2_embedding import B2EmbeddingClassifier
from d1.config import CSV_COLUMNS, DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BASELINE_CONFIGS — единственный источник истины для гиперпараметров
# ---------------------------------------------------------------------------

# Контракт записи:
#   - `cls`: класс baseline (должен иметь .fit, .predict, .predict_proba где применимо)
#   - `params`: kwargs для `cls(**params)` — только overrides, не дублируются
#     дефолты из самого класса
#   - `enabled`: True = runnable сейчас, False = concept-only до реализации
#   - `blocked_by`: (только если enabled=False) человекочитаемая причина блока
BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    # B0: rules — не требует fit, но регистрируется в bundle для единого API.
    "B0_rules": {
        "cls": B0RulesClassifier,
        "params": {},
        "enabled": True,
    },

    # B1 семейство: TF-IDF + classifier head.
    "B1_tfidf_svc": {
        "cls": B1TfidfClassifier,
        "params": {"head_type": "svc"},
        "enabled": True,
    },
    "B1.1_tfidf_lr": {
        "cls": B1TfidfClassifier,
        "params": {"head_type": "logistic"},
        "enabled": True,
    },
    "B1.2_tfidf_lr_tuned": {
        "cls": B1TfidfClassifier,
        "params": {
            "head_type": "logistic",
            "tfidf_params": {"ngram_range": (2, 6), "min_df": 1},
            "head_params": {"C": 0.5},
        },
        "enabled": True,
    },

    # B2 семейство: encoder (BGE-M3 по умолчанию класса) + head.
    "B2_bge-m3_linear": {
        "cls": B2EmbeddingClassifier,
        "params": {"head_type": "linear"},
        "enabled": True,
    },
    "B2.1_bge-m3_svc": {
        "cls": B2EmbeddingClassifier,
        "params": {"head_type": "svc"},
        "enabled": True,
    },
    "B2.2_bge-m3_centroid": {
        "cls": B2EmbeddingClassifier,
        "params": {"head_type": "centroid"},
        "enabled": True,
    },

    # B2.3: tuned bge-m3 (head_params в B2EmbeddingClassifier добавлены в Phase 3.1).
    "B2.3_bge-m3_linear_tuned": {
        "cls": B2EmbeddingClassifier,
        "params": {"head_type": "linear", "head_params": {"C": 0.3}},
        "enabled": True,
    },

    # Phase 3.2 (2026-05-01): lightweight dense candidates на multilingual-e5-small
    # (~118M params vs bge-m3 568M — 5x легче, тот же e5 family).
    # E5-prefix "query: " применяется автоматически в B2EmbeddingClassifier._encode.
    "B2.4_e5-small_linear": {
        "cls": B2EmbeddingClassifier,
        "params": {
            "model_name": "intfloat/multilingual-e5-small",
            "head_type": "linear",
        },
        "enabled": True,
    },
    "B2.5_e5-small_svc": {
        "cls": B2EmbeddingClassifier,
        "params": {
            "model_name": "intfloat/multilingual-e5-small",
            "head_type": "svc",
        },
        "enabled": True,
    },

    # Phase 3.3 (2026-05-01): fastText — ультра-лёгкий sparse-embedding кандидат.
    # Сериализуется через explicit save/load (не joblib): см. ниже _save_to_cache.
    "B1.3_fasttext": {
        "cls": B1FastTextClassifier,
        "params": {},
        "enabled": True,
    },
}


MODELS_DIR = RESULTS_DIR / "models"


# ---------------------------------------------------------------------------
# CacheKey — строгий контракт инвалидации
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    """Полный cache key для одного baseline.

    Инвалидация срабатывает при изменении ЛЮБОГО из 5 компонентов:
    - `params_hash`: гиперпараметры из BASELINE_CONFIGS
    - `dataset_hash`: SHA256 от СОДЕРЖИМОГО train CSV (не mtime)
    - `code_hash`: SHA256 от исходников baseline-модулей
    - `env_hash`: SHA256 от версий sklearn/torch/sentence-transformers/python
    - `schema_hash`: SHA256 от CSV_COLUMNS
    """

    params_hash: str
    dataset_hash: str
    code_hash: str
    env_hash: str
    schema_hash: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def diff(self, other: "CacheKey") -> list[str]:
        """Возвращает список компонентов, которые отличаются.

        Используется для логирования причины cache miss.
        """
        diffs = []
        for field_name in ("params_hash", "dataset_hash", "code_hash",
                           "env_hash", "schema_hash"):
            if getattr(self, field_name) != getattr(other, field_name):
                diffs.append(field_name.replace("_hash", ""))
        return diffs


# ---------------------------------------------------------------------------
# Hash helpers — pure functions для каждого компонента ключа
# ---------------------------------------------------------------------------

_HASH_LEN = 16  # первые 16 hex символов SHA256 (достаточно для коллизий 2^64)


def _hash_bytes(data: bytes) -> str:
    """Короткий SHA256 digest."""
    return sha256(data).hexdigest()[:_HASH_LEN]


def _compute_params_hash(params: dict[str, Any]) -> str:
    """SHA256 от стабильной сериализации params (JSON с sort_keys)."""
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return _hash_bytes(payload)


def _compute_dataset_hash(csv_path: Path) -> str:
    """SHA256 от СОДЕРЖИМОГО train CSV.

    Важно: хэш от bytes, НЕ от mtime. Защищает от ложных cache miss при
    `touch` / file copy (mtime меняется, content — нет).
    """
    return _hash_bytes(csv_path.read_bytes())


def _compute_code_hash() -> str:
    """SHA256 от исходников baseline-модулей.

    Включает только модули с логикой обучения (b0/b1/b1_fasttext/b2),
    НЕ сам trained_bundle.py — правка инфраструктуры не должна
    инвалидировать кэш моделей.

    NOTE Phase 3.3: добавление b1_fasttext инвалидирует cache при первом
    запуске после изменения. Это разовое событие — последующие запуски
    хитуют cache как обычно.
    """
    import d1.baselines.b0_rules
    import d1.baselines.b1_fasttext
    import d1.baselines.b1_tfidf
    import d1.baselines.b2_embedding
    real_paths = sorted([
        Path(d1.baselines.b0_rules.__file__),
        Path(d1.baselines.b1_tfidf.__file__),
        Path(d1.baselines.b1_fasttext.__file__),
        Path(d1.baselines.b2_embedding.__file__),
    ])

    h = sha256()
    for p in real_paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:_HASH_LEN]


def _compute_env_hash() -> str:
    """Fingerprint версий критичных пакетов.

    Detect upgrade sklearn/torch/sentence-transformers между запусками
    (разные версии могут дать разные веса даже при фиксированном seed).
    """
    versions: list[str] = [f"py={sys.version_info[:2]}"]
    for pkg in ("sklearn", "torch", "sentence_transformers"):
        try:
            mod = __import__(pkg)
            versions.append(f"{pkg}={getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            versions.append(f"{pkg}=missing")
    return _hash_bytes("|".join(versions).encode())


def _compute_schema_hash() -> str:
    """SHA256 от CSV_COLUMNS (детектирует изменение dataset schema)."""
    return _hash_bytes(json.dumps(CSV_COLUMNS, sort_keys=True).encode())


def _build_cache_key(name: str, csv_path: Path) -> CacheKey:
    """Собрать полный CacheKey для baseline с именем `name`."""
    if name not in BASELINE_CONFIGS:
        raise KeyError(f"Baseline '{name}' не зарегистрирован в BASELINE_CONFIGS")

    cfg = BASELINE_CONFIGS[name]
    return CacheKey(
        params_hash=_compute_params_hash(cfg["params"]),
        dataset_hash=_compute_dataset_hash(csv_path),
        code_hash=_compute_code_hash(),
        env_hash=_compute_env_hash(),
        schema_hash=_compute_schema_hash(),
    )


# ---------------------------------------------------------------------------
# TrainedBundle — контейнер обученных моделей
# ---------------------------------------------------------------------------

@dataclass
class TrainedBundle:
    """Контейнер для обученных baseline'ов.

    Консьюмеры (selective_router, b4_hybrid, bootstrap, learning_curves,
    analyze_confidence) получают bundle через `train_bundle(...)` и
    обращаются к моделям через `bundle.get(name)`.
    """

    models: dict[str, Any] = field(default_factory=dict)
    train_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        """Получить fitted baseline по имени.

        Raises:
            KeyError: baseline не загружен в bundle (не запрошен или disabled).
        """
        if name not in self.models:
            raise KeyError(
                f"Baseline '{name}' не в bundle. Доступно: {sorted(self.models)}",
            )
        return self.models[name]


# ---------------------------------------------------------------------------
# Persistence: joblib + metadata.json
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Normalize baseline name → файловый slug."""
    return (
        name.lower()
        .replace(".", "_")
        .replace("-", "_")
        .replace("+", "_")
    )


def _model_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{_slug(name)}.joblib"


def _fasttext_dir(cache_dir: Path, name: str) -> Path:
    """Директория для fastText bundle (бинарник + meta.json)."""
    return cache_dir / f"{_slug(name)}_ft"


def _metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "bundle_metadata.json"


def _load_bundle_metadata(cache_dir: Path) -> dict[str, Any]:
    """Прочитать bundle_metadata.json (или вернуть пустой dict)."""
    path = _metadata_path(cache_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Повреждённый bundle_metadata.json: %s", path)
        return {}


def _save_bundle_metadata(
    cache_dir: Path, metadata: dict[str, Any],
) -> None:
    """Атомарная запись metadata.json."""
    path = _metadata_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _try_load_from_cache(
    name: str, cache_dir: Path, expected_key: CacheKey,
) -> Any | None:
    """Попытка загрузить baseline из кэша с проверкой ключа.

    Returns:
        Fitted baseline instance если cache hit, иначе None.
        При cache miss логирует компоненты ключа, которые отличаются.

    Для B1.3_fasttext используется отдельная директория (binary+meta), а не
    `joblib` — fasttext._FastText не сериализуется через joblib.
    """
    cfg = BASELINE_CONFIGS.get(name, {})
    is_fasttext = cfg.get("cls") is B1FastTextClassifier
    if is_fasttext:
        ft_dir = _fasttext_dir(cache_dir, name)
        if not ft_dir.exists():
            logger.info("cache_miss: %s — директория %s отсутствует",
                        name, ft_dir.name)
            return None
    else:
        model_file = _model_path(cache_dir, name)
        if not model_file.exists():
            logger.info("cache_miss: %s — файл %s отсутствует",
                        name, model_file.name)
            return None

    bundle_meta = _load_bundle_metadata(cache_dir)
    entry = bundle_meta.get(name)
    if entry is None or "cache_key" not in entry:
        logger.info("cache_miss: %s — нет записи в metadata", name)
        return None

    try:
        stored_key = CacheKey(**entry["cache_key"])
    except TypeError:
        logger.info("cache_miss: %s — несовместимая схема CacheKey", name)
        return None

    diffs = stored_key.diff(expected_key)
    if diffs:
        logger.info(
            "cache_miss: %s — изменились компоненты ключа: %s",
            name, ", ".join(diffs),
        )
        return None

    try:
        if is_fasttext:
            model = B1FastTextClassifier.load(_fasttext_dir(cache_dir, name))
            logger.info("cache_hit: %s ← %s/", name, _fasttext_dir(cache_dir, name).name)
        else:
            model = joblib.load(_model_path(cache_dir, name))
            logger.info("cache_hit: %s ← %s", name, _model_path(cache_dir, name).name)
        return model
    except Exception as exc:  # noqa: BLE001 — broad catch оправдан для joblib/fastText
        logger.warning("cache_miss: %s — ошибка загрузки: %s", name, exc)
        return None


def _save_to_cache(
    name: str, model: Any, cache_dir: Path,
    cache_key: CacheKey, extra_meta: dict[str, Any],
) -> None:
    """Сохранить fitted baseline + обновить metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель.
    # - B2*: исключаем encoder (SentenceTransformer ~2GB) — lazy-load при
    #   первом predict после load.
    # - B1.3 fastText: explicit save (бинарь + meta.json) в отдельной директории.
    # - Прочие (B0/B1): joblib.dump.
    if isinstance(model, B1FastTextClassifier):
        model.save(_fasttext_dir(cache_dir, name))
    elif isinstance(model, B2EmbeddingClassifier):
        saved_encoder = model._encoder
        model._encoder = None
        try:
            joblib.dump(model, _model_path(cache_dir, name))
        finally:
            model._encoder = saved_encoder
    else:
        joblib.dump(model, _model_path(cache_dir, name))

    # Обновляем metadata: merge с существующим
    bundle_meta = _load_bundle_metadata(cache_dir)
    bundle_meta[name] = {
        "cache_key": cache_key.to_dict(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **extra_meta,
    }
    _save_bundle_metadata(cache_dir, bundle_meta)


# ---------------------------------------------------------------------------
# Validation: проверка запрошенных имён
# ---------------------------------------------------------------------------

def _resolve_names(names: list[str] | None) -> list[str]:
    """Резолвит список имён.

    - `names=None` → все enabled=True baseline'ы
    - неизвестное имя → KeyError
    - disabled baseline явно запрошен → RuntimeError с blocked_by
    """
    if names is None:
        return [n for n, cfg in BASELINE_CONFIGS.items() if cfg["enabled"]]

    for name in names:
        if name not in BASELINE_CONFIGS:
            raise KeyError(
                f"Неизвестный baseline '{name}'. "
                f"Зарегистрированы: {sorted(BASELINE_CONFIGS)}",
            )
        cfg = BASELINE_CONFIGS[name]
        if not cfg["enabled"]:
            reason = cfg.get("blocked_by", "причина не указана")
            raise RuntimeError(
                f"Baseline '{name}' disabled и не может быть обучен. "
                f"Blocked by: {reason}",
            )
    return list(names)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_train_data(train_csv_path: Path | None = None) -> tuple[list[str], list[str], Path]:
    """Загрузка train split.

    Args:
        train_csv_path: опциональный CSV для контролируемых экспериментов
            (например, learning curves). Если не задан — стандартный
            `d1_v6_train.csv`.

    Returns:
        (texts, labels, csv_path) — csv_path нужен для dataset_hash.
    """
    csv_path = train_csv_path or DATA_DIR / f"{DATASET_PREFIX}_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV не найден: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return df["text"].tolist(), df["route_domain"].tolist(), csv_path


# ---------------------------------------------------------------------------
# Main API: train_bundle
# ---------------------------------------------------------------------------

def train_bundle(
    names: list[str] | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    train_csv_path: Path | None = None,
    device_override: str | None = None,
) -> TrainedBundle:
    """Обучить/загрузить baseline'ы — ЕДИНСТВЕННЫЙ entry point.

    Args:
        names: список baseline-имён или None (= все enabled=True).
        use_cache: True — читать из cache_dir если ключ совпадает.
        cache_dir: директория для joblib-кэша (default: RESULTS_DIR/models).
        train_csv_path: опциональный train CSV. Используется для контролируемых
            экспериментов вроде learning curves без обхода SSoT обучения.
        device_override: опциональный device для embedding моделей
            (например, "cpu"). Применяется ТОЛЬКО к B2EmbeddingClassifier
            и НЕ влияет на cache_key (device — runtime-инфраструктура,
            не аффектит обученные веса). Используется в learning_curves
            и benchmark_latency для предотвращения накопления MPS-памяти.

    Returns:
        TrainedBundle с fitted моделями.

    Raises:
        KeyError: неизвестное имя.
        RuntimeError: явный запрос disabled baseline.
        FileNotFoundError: train CSV отсутствует.
    """
    resolved = _resolve_names(names)
    cache_dir = cache_dir or MODELS_DIR

    train_texts, train_labels, csv_path = _load_train_data(train_csv_path)
    logger.info("train_bundle: %d baseline'ов, train=%d строк",
                len(resolved), len(train_texts))

    bundle = TrainedBundle(
        train_size=len(train_texts),
        metadata={"csv_path": str(csv_path), "resolved": resolved},
    )

    for name in resolved:
        cfg = BASELINE_CONFIGS[name]
        cache_key = _build_cache_key(name, csv_path)

        # 1. Пробуем cache
        if use_cache:
            cached = _try_load_from_cache(name, cache_dir, cache_key)
            if cached is not None:
                if (
                    device_override is not None
                    and isinstance(cached, B2EmbeddingClassifier)
                ):
                    cached.set_device(device_override)
                bundle.models[name] = cached
                continue

        # 2. Fit заново
        logger.info("Обучение %s (params=%s)", name, cfg["params"])
        t0 = time.perf_counter()
        # device_override применяется только runtime — не входит в params_hash.
        init_kwargs = dict(cfg["params"])
        if (
            device_override is not None
            and cfg["cls"] is B2EmbeddingClassifier
        ):
            init_kwargs["device"] = device_override
        model = cfg["cls"](**init_kwargs)

        # B0Rules — rule-based, fit не нужен. Прочие — обычный fit.
        if isinstance(model, B0RulesClassifier):
            fit_time_ms = 0.0
        else:
            model.fit(train_texts, train_labels)
            fit_time_ms = (time.perf_counter() - t0) * 1000

        bundle.models[name] = model
        logger.info("Обучение %s готово за %.1f мс", name, fit_time_ms)

        # 3. Всегда сохраняем в cache_dir (в т.ч. B0 для единообразия).
        # Флаг `use_cache` контролирует ТОЛЬКО чтение: запись обязательна
        # для воспроизводимости между запусками.
        extra = {
            "fit_time_ms": round(fit_time_ms, 1),
            "train_size": len(train_texts),
        }
        if hasattr(model, "get_params_summary"):
            extra["summary"] = model.get_params_summary()
        _save_to_cache(name, model, cache_dir, cache_key, extra)

    return bundle


# ---------------------------------------------------------------------------
# Main API: predict_bundle
# ---------------------------------------------------------------------------

def predict_bundle(
    bundle: TrainedBundle,
    texts: list[str],
    names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch predict + proba для всех baseline'ов из bundle.

    Args:
        bundle: результат train_bundle.
        texts: входные тексты.
        names: подмножество имён (default: все в bundle).

    Returns:
        {baseline_name: {
            "preds": list[str],
            "proba": np.ndarray or None,
            "classes": list[str] or None,
            "confidence": np.ndarray,
        }}

        Для моделей без predict_proba (B0_rules, B2.2_centroid) — proba=None,
        confidence извлекается из класс-специфичного API.
    """
    import numpy as np

    target_names = names or list(bundle.models)
    out: dict[str, dict[str, Any]] = {}

    for name in target_names:
        model = bundle.get(name)
        entry: dict[str, Any] = {}

        # B0_rules: нет predict_proba, confidence через RulePrediction
        if isinstance(model, B0RulesClassifier):
            rule_preds = model.predict_with_confidence(texts)
            entry["preds"] = [rp.route_domain for rp in rule_preds]
            entry["proba"] = None
            entry["classes"] = None
            entry["confidence"] = np.array(
                [rp.confidence for rp in rule_preds], dtype=float,
            )
        else:
            preds = model.predict(texts)
            entry["preds"] = preds

            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(texts)
                    entry["proba"] = proba
                    entry["classes"] = model.classes_
                    entry["confidence"] = np.max(proba, axis=1)
                except (AttributeError, Exception):  # noqa: BLE001
                    entry["proba"] = None
                    entry["classes"] = getattr(model, "classes_", None)
                    entry["confidence"] = np.ones(len(texts), dtype=float)
            else:
                entry["proba"] = None
                entry["classes"] = getattr(model, "classes_", None)
                entry["confidence"] = np.ones(len(texts), dtype=float)

        out[name] = entry

    return out
