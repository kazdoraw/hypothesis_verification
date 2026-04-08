"""Конфигурация pytest для d4 smoke-тестов.

Обеспечивает корректный sys.path для импорта d4.* и
предоставляет общие fixtures: chunks, samples, gold_map, doctors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# d4 пакет лежит в study/, добавляем study/ в sys.path
_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d4.data_gen.query_generator import load_eval_set
from d4.evaluation.gold_map import build_gold_map
from d4.pipeline.chunker import load_chunks

# Пути к данным
_D4_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _D4_ROOT / "data" / "kb"
_CHUNKS_PATH = _KB_DIR / "chunks.json"
_DOCTORS_PATH = _KB_DIR / "doctors.yaml"
_MINI_EVAL_SET_PATH = Path(__file__).resolve().parent / "mini_eval_set.yaml"


@pytest.fixture(scope="session")
def chunks() -> list:
    """KB chunks из chunks.json (session-scoped — загружается один раз)."""
    return load_chunks(_CHUNKS_PATH)


@pytest.fixture(scope="session")
def doctors() -> list[dict]:
    """Список врачей из doctors.yaml."""
    with open(_DOCTORS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("doctors", [])


@pytest.fixture(scope="session")
def mini_samples() -> list:
    """10 representative samples из mini_eval_set.yaml."""
    return load_eval_set(_MINI_EVAL_SET_PATH)


@pytest.fixture(scope="session")
def gold_map(mini_samples, doctors) -> dict[str, list[str]]:
    """Gold map для mini eval set."""
    return build_gold_map(mini_samples, doctors)
