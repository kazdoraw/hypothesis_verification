"""Типизированная конфигурация эксперимента D4v2.

Валидация experiment.yaml при загрузке — ошибки ловятся сразу,
а не глубоко в call chain (KeyError в factory/notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


class ExperimentMeta(BaseModel):
    """Метаданные эксперимента."""

    name: str
    version: int = 2
    seed: int = 42
    description: str = ""


class LLMConfig(BaseModel):
    """Конфигурация LLM (для S1-S4)."""

    provider: str = "openrouter"
    api_url: str = "https://openrouter.ai/api/v1"
    model: str
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 1024


class JudgeConfig(BaseModel):
    """Конфигурация LLM-as-Judge."""

    provider: str = "openrouter"
    api_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-5.4-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    """Конфигурация embedding-модели."""

    model: str = "BAAI/bge-m3"
    device: str = "mps"
    similarity_method: str = "cosine"
    top_k: int = 5


class RateLimitConfig(BaseModel):
    """Параметры параллелизма и rate limiting."""

    max_workers: int = Field(default=8, ge=1, le=64)
    request_timeout_sec: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    checkpoint_every_n: int = Field(default=25, ge=1)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Полная конфигурация эксперимента D4v2.

    Загружается из experiment.yaml, валидируется при создании.
    """

    experiment: ExperimentMeta
    llm: LLMConfig
    judge: JudgeConfig
    embedding: EmbeddingConfig
    rate_limit: RateLimitConfig

    # Секции, не требующие строгой типизации (документационные / редко меняются)
    strategies: list[dict[str, Any]] = Field(default_factory=list)
    eval_set: dict[str, Any] = Field(default_factory=dict)
    kb: dict[str, Any] = Field(default_factory=dict)
    kb_scale: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Загрузка и валидация experiment.yaml.

    Args:
        config_path: путь к YAML файлу

    Returns:
        валидированный ExperimentConfig

    Raises:
        FileNotFoundError: файл не найден
        pydantic.ValidationError: невалидная структура/значения
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig.model_validate(raw)
