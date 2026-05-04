"""Конфигурация pytest для d4 smoke-тестов.

Обеспечивает корректный sys.path для импорта d4.* и
предоставляет общие fixtures: chunks, samples, gold_map, doctors.

Фикстуры используют frozen artifacts (chunks_frozen.json) для
воспроизводимости: изменения в живом KB не ломают тесты.

Маркеры:
  slow — тесты с тяжёлыми зависимостями (ML-модели). Запуск: pytest -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: тесты с тяжёлыми ML-зависимостями (запуск: pytest -m slow)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Пропускает @pytest.mark.slow тесты, если не указан -m slow."""
    if config.getoption("-m") and "slow" in config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow test — запуск: pytest -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

# d4 пакет лежит в study/, добавляем study/ в sys.path
_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d4.data_gen.query_generator import load_eval_set
from d4.evaluation.gold_map import build_gold_map
from d4.pipeline.chunker import load_chunks

# Пути к данным — frozen artifacts для воспроизводимости
_D4_ROOT = Path(__file__).resolve().parent.parent
_CHUNKS_FROZEN_PATH = _D4_ROOT / "data" / "chunks_frozen.json"
_KB_DIR = _D4_ROOT / "data" / "kb"
_DOCTORS_PATH = _KB_DIR / "doctors.yaml"
_TESTS_DIR = Path(__file__).resolve().parent
_MINI_EVAL_SET_PATH = _TESTS_DIR / "mini_eval_set.yaml"
_HARD_EVAL_SET_PATH = _TESTS_DIR / "mini_eval_set_hard.yaml"


@pytest.fixture(scope="session")
def chunks() -> list:
    """KB chunks из chunks_frozen.json (frozen snapshot для воспроизводимости)."""
    return load_chunks(_CHUNKS_FROZEN_PATH)


@pytest.fixture(scope="session")
def doctors() -> list[dict]:
    """Список врачей из doctors.yaml."""
    with open(_DOCTORS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("doctors", [])


@pytest.fixture(scope="session")
def mini_samples() -> list:
    """Representative samples из mini_eval_set.yaml (smoke)."""
    return load_eval_set(_MINI_EVAL_SET_PATH)


@pytest.fixture(scope="session")
def hard_samples() -> list:
    """Adversarial samples из mini_eval_set_hard.yaml (stress)."""
    if not _HARD_EVAL_SET_PATH.exists():
        pytest.skip("mini_eval_set_hard.yaml not found")
    return load_eval_set(_HARD_EVAL_SET_PATH)


@pytest.fixture(scope="session")
def gold_map(mini_samples) -> dict[str, list[list[str]]]:
    """Gold map для mini eval set (multi-gold compatible schema)."""
    return build_gold_map(mini_samples)
