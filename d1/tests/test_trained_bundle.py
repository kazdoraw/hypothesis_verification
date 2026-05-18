"""TDD-тесты для d1/baselines/trained_bundle.py.

Определяет контракт SSoT-обучения baseline'ов: строгий cache key,
loud failure на disabled configs, корректная сериализация/десериализация.

Запуск:
    cd study && python -m pytest d1/tests/test_trained_bundle.py -v
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Минимальный toy dataset для быстрых тестов.
# Используется вместо реального train CSV, чтобы тесты были < 30 сек.
_TOY_TRAIN = pd.DataFrame({
    "id": [f"t{i}" for i in range(16)],
    "text": [
        "болит зуб неделю", "опухла щека после удаления",
        "температура и кровь", "ноет челюсть постоянно",
        "сколько стоит чистка", "какие врачи по выходным",
        "где ваша клиника расположена", "есть ли рассрочка лечения",
        "хочу записаться на осмотр", "запишите к ортодонту завтра",
        "можно перенести запись", "нужен приём к терапевту",
        "привет как дела", "спасибо большое", "ок понятно", "расскажите про погоду",
    ],
    "route_domain": [
        "anamnesis", "anamnesis", "anamnesis", "anamnesis",
        "faq", "faq", "faq", "faq",
        "booking", "booking", "booking", "booking",
        "unsupported", "unsupported", "unsupported", "unsupported",
    ],
    "subtype": [""] * 16,
    "explicit_booking": [""] * 16,
    "urgency": [""] * 16,
    "is_offtopic": [""] * 16,
    "specialization_hint": [""] * 16,
    "feedback_flag": [""] * 16,
    "faq_category": [""] * 16,
    "style": [""] * 16,
    "source": ["seed"] * 16,
    "seed_id": [f"seed_{i:03d}" for i in range(16)],
})


@pytest.fixture
def tmp_train_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Подменяет DATA_DIR на tmp и записывает toy train CSV."""
    from d1 import config

    # Подмена DATA_DIR и RESULTS_DIR на временные
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "data" / f"{config.DATASET_PREFIX}_train.csv"
    _TOY_TRAIN.to_csv(csv_path, index=False)

    # Перезагрузить trained_bundle, чтобы MODELS_DIR пересчитался
    import d1.baselines.trained_bundle as tb
    importlib.reload(tb)

    return csv_path


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Временный cache dir для bundle."""
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return cache


# ---------------------------------------------------------------------------
# Контракт 1: Runnable configs — train, cache hit, cache miss
# ---------------------------------------------------------------------------

class TestRunnableConfigs:
    """Проверка базовой функциональности: обучение, кэш, загрузка."""

    def test_baseline_configs_contains_expected_keys(self) -> None:
        """BASELINE_CONFIGS содержит все документированные baseline'ы.

        Phase 3.2/3.3 (2026-05-01): добавлены B2.4/B2.5 (e5-base) и B1.3 (fastText).
        """
        from d1.baselines.trained_bundle import BASELINE_CONFIGS

        expected = {
            "B0_rules",
            "B1.1_tfidf_lr",
            "B1.3_fasttext",
            "B2.1_bge-m3_svc",
            "B2.5_e5-small_svc",
        }
        assert set(BASELINE_CONFIGS) == expected

    def test_each_config_has_enabled_flag(self) -> None:
        """Каждый config имеет явный `enabled` флаг (контракт SSoT)."""
        from d1.baselines.trained_bundle import BASELINE_CONFIGS

        for name, cfg in BASELINE_CONFIGS.items():
            assert "enabled" in cfg, f"{name}: отсутствует `enabled` флаг"
            assert isinstance(cfg["enabled"], bool)

    def test_five_canonical_baselines_enabled(self) -> None:
        """Оставшиеся 5 baseline'ов — все enabled."""
        from d1.baselines.trained_bundle import BASELINE_CONFIGS

        for name in (
            "B0_rules",
            "B1.1_tfidf_lr",
            "B1.3_fasttext",
            "B2.1_bge-m3_svc",
            "B2.5_e5-small_svc",
        ):
            assert BASELINE_CONFIGS[name]["enabled"] is True

    def test_train_bundle_returns_fitted_b0_b1(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """train_bundle обучает B0 (без fit) и B1.1 на toy данных."""
        from d1.baselines.trained_bundle import train_bundle

        bundle = train_bundle(
            names=["B0_rules", "B1.1_tfidf_lr"],
            use_cache=False,
            cache_dir=tmp_cache_dir,
        )
        assert set(bundle.models) == {"B0_rules", "B1.1_tfidf_lr"}
        # B0 не требует fit, B1.1 должен быть fitted
        assert bundle.get("B1.1_tfidf_lr")._is_fitted

    def test_bundle_get_raises_on_missing(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """bundle.get('X') → KeyError с понятным сообщением."""
        from d1.baselines.trained_bundle import train_bundle

        bundle = train_bundle(
            names=["B0_rules"], use_cache=False, cache_dir=tmp_cache_dir,
        )
        with pytest.raises(KeyError, match=r"B1\.1_tfidf_lr"):
            bundle.get("B1.1_tfidf_lr")

    def test_cache_hit_on_second_call(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Второй вызов train_bundle с use_cache=True загружает из кэша."""
        from d1.baselines.trained_bundle import train_bundle

        # Первый вызов: полное обучение
        bundle1 = train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )
        first_fit_time = bundle1.get("B1.1_tfidf_lr").train_time_ms

        # Второй вызов: должен загрузить из кэша (train_time_ms исходный, ≠ 0)
        bundle2 = train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )
        # Cache hit означает, что модель загружена с сохранённым train_time_ms,
        # а не переобучена. Проверяем идентичность состояния.
        assert bundle2.get("B1.1_tfidf_lr").train_time_ms == first_fit_time

    def test_cache_files_created(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """train_bundle создаёт {slug}.joblib и metadata.json."""
        from d1.baselines.trained_bundle import train_bundle

        train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )
        joblib_files = list(tmp_cache_dir.glob("*.joblib"))
        assert any("b1_1_tfidf_lr" in p.name.lower() for p in joblib_files)
        assert (tmp_cache_dir / "bundle_metadata.json").exists()


# ---------------------------------------------------------------------------
# Контракт 2: Disabled configs — loud failure
# ---------------------------------------------------------------------------

class TestDisabledConfigs:
    """Проверка жёсткого контракта: disabled baseline → явная ошибка."""

    def test_train_bundle_skips_disabled_by_default(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Без явного имени любые disabled baseline'ы НЕ попадают в bundle."""
        from d1.baselines.trained_bundle import BASELINE_CONFIGS, train_bundle

        disabled = [n for n, c in BASELINE_CONFIGS.items() if not c["enabled"]]
        if not disabled:
            pytest.skip("Все baseline'ы enabled — нечего проверять")

        bundle = train_bundle(
            names=None, use_cache=False, cache_dir=tmp_cache_dir,
        )
        for name in disabled:
            assert name not in bundle.models, (
                f"Disabled {name} попал в bundle при names=None"
            )

    def test_explicit_disabled_request_raises(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Явный запрос disabled baseline → RuntimeError."""
        from d1.baselines.trained_bundle import BASELINE_CONFIGS, train_bundle

        disabled = [n for n, c in BASELINE_CONFIGS.items() if not c["enabled"]]
        if not disabled:
            pytest.skip("Все baseline'ы enabled — нечего проверять")

        with pytest.raises(RuntimeError, match=r"disabled"):
            train_bundle(
                names=[disabled[0]],
                use_cache=False, cache_dir=tmp_cache_dir,
            )

    def test_unknown_baseline_name_raises(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Неизвестное имя → KeyError."""
        from d1.baselines.trained_bundle import train_bundle

        with pytest.raises(KeyError, match="B999"):
            train_bundle(
                names=["B999_fake"],
                use_cache=False, cache_dir=tmp_cache_dir,
            )


# ---------------------------------------------------------------------------
# Контракт 3: Cache invalidation по каждому компоненту ключа
# ---------------------------------------------------------------------------

class TestCacheContract:
    """Критерии готовности Task 0 из плана: строгий cache contract."""

    def test_dataset_content_change_invalidates_cache(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Изменение содержимого train CSV → cache miss."""
        from d1.baselines.trained_bundle import train_bundle

        train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )

        # Меняем содержимое CSV: добавляем строку
        df = pd.read_csv(tmp_train_csv, dtype=str).fillna("")
        new_row = df.iloc[0].to_dict()
        new_row["id"] = "t999"
        new_row["text"] = "совсем другой текст для изменения хэша"
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(tmp_train_csv, index=False)

        # Второй вызов должен переобучить (cache miss)
        with patch("d1.baselines.trained_bundle.logger") as mock_log:
            train_bundle(
                names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
            )
            # Проверяем что был залогирован cache_miss с причиной dataset
            miss_logs = [
                str(c) for c in mock_log.info.call_args_list
                if "cache_miss" in str(c).lower()
            ]
            assert any("dataset" in log for log in miss_logs), \
                f"Ожидается cache_miss:dataset, логи: {miss_logs}"

    def test_mtime_only_change_preserves_cache(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Изменение только mtime (file copy) → cache HIT (content hash unchanged)."""
        import os
        import time

        from d1.baselines.trained_bundle import train_bundle

        bundle1 = train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )
        original_time = bundle1.get("B1.1_tfidf_lr").train_time_ms

        # Имитация file copy: перезаписываем файл тем же содержимым
        content = tmp_train_csv.read_bytes()
        time.sleep(0.01)
        tmp_train_csv.write_bytes(content)
        # Дополнительно обновляем mtime явно
        now = time.time()
        os.utime(tmp_train_csv, (now, now))

        bundle2 = train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )
        # Cache hit — модель загружена, train_time сохранён
        assert bundle2.get("B1.1_tfidf_lr").train_time_ms == original_time

    def test_params_change_invalidates_cache(
        self, tmp_train_csv: Path, tmp_cache_dir: Path, monkeypatch,
    ) -> None:
        """Изменение BASELINE_CONFIGS[name]['params'] → cache miss."""
        import d1.baselines.trained_bundle as tb

        tb.train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )

        # Модифицируем params этого baseline
        new_configs = {
            k: {**v, "params": {**v["params"], "head_params": {"C": 99.0}}}
            if k == "B1.1_tfidf_lr" else v
            for k, v in tb.BASELINE_CONFIGS.items()
        }
        monkeypatch.setattr(tb, "BASELINE_CONFIGS", new_configs)

        with patch.object(tb, "logger") as mock_log:
            tb.train_bundle(
                names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
            )
            miss_logs = [
                str(c) for c in mock_log.info.call_args_list
                if "cache_miss" in str(c).lower()
            ]
            assert any("params" in log for log in miss_logs), \
                f"Ожидается cache_miss:params, логи: {miss_logs}"

    def test_env_change_invalidates_cache(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """Mock-изменение версии sklearn → cache miss."""
        import d1.baselines.trained_bundle as tb

        tb.train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=True, cache_dir=tmp_cache_dir,
        )

        # Mock-апгрейд версии sklearn через подмену env_hash
        with patch.object(tb, "_compute_env_hash", return_value="x" * 16):
            with patch.object(tb, "logger") as mock_log:
                tb.train_bundle(
                    names=["B1.1_tfidf_lr"], use_cache=True,
                    cache_dir=tmp_cache_dir,
                )
                miss_logs = [
                    str(c) for c in mock_log.info.call_args_list
                    if "cache_miss" in str(c).lower()
                ]
                assert any("env" in log for log in miss_logs), \
                    f"Ожидается cache_miss:env, логи: {miss_logs}"


# ---------------------------------------------------------------------------
# Контракт 4: predict_bundle API
# ---------------------------------------------------------------------------

class TestPredictBundle:
    """Проверка batch predict API."""

    def test_predict_bundle_returns_preds_and_proba(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """predict_bundle возвращает {name: {preds, proba, classes, confidence}}."""
        import numpy as np

        from d1.baselines.trained_bundle import predict_bundle, train_bundle

        bundle = train_bundle(
            names=["B1.1_tfidf_lr"], use_cache=False, cache_dir=tmp_cache_dir,
        )
        texts = ["болит зуб", "сколько стоит", "запишите меня"]
        out = predict_bundle(bundle, texts)

        assert "B1.1_tfidf_lr" in out
        entry = out["B1.1_tfidf_lr"]
        assert set(entry.keys()) >= {"preds", "proba", "classes", "confidence"}
        assert len(entry["preds"]) == 3
        assert entry["proba"].shape[0] == 3
        assert len(entry["classes"]) == entry["proba"].shape[1]
        assert np.all((entry["confidence"] >= 0) & (entry["confidence"] <= 1))

    def test_predict_bundle_b0_rules_no_proba(
        self, tmp_train_csv: Path, tmp_cache_dir: Path,
    ) -> None:
        """B0_rules не имеет predict_proba — proba=None, confidence из RulePrediction."""
        from d1.baselines.trained_bundle import predict_bundle, train_bundle

        bundle = train_bundle(
            names=["B0_rules"], use_cache=False, cache_dir=tmp_cache_dir,
        )
        out = predict_bundle(bundle, ["болит зуб сильно"])

        entry = out["B0_rules"]
        assert len(entry["preds"]) == 1
        # B0 не даёт proba (или даёт degenerate) — важен preds и confidence
        assert "confidence" in entry
