"""Типизированная конфигурация эксперимента D4v2.

Валидация experiment.yaml при загрузке — ошибки ловятся сразу,
а не глубоко в call chain (KeyError в factory/notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


class ExperimentMeta(BaseModel):
    """Метаданные эксперимента."""

    name: str
    version: int = 2
    stage: int = 1
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
    revision: str = "main"
    device: str = "mps"
    similarity_method: str = "cosine"
    top_k: int = 5


class RateLimitConfig(BaseModel):
    """Параметры параллелизма и rate limiting."""

    max_workers: int = Field(default=8, ge=1, le=64)
    request_timeout_sec: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    checkpoint_every_n: int = Field(default=25, ge=1)


class RerankingConfig(BaseModel):
    """Конфигурация cross-encoder reranking (S4r, S5)."""

    model: str = "BAAI/bge-reranker-base"
    revision: str = "main"
    device: str = "mps"
    top_k_retrieve: int = Field(default=20, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)


class TokenizerConfig(BaseModel):
    """Конфигурация токенизатора для chunker.py."""

    model: str = "Qwen/Qwen3.5-35B-A3B"
    revision: str = "main"


class HubConfig(BaseModel):
    """Управление загрузкой HuggingFace-моделей.

    local_files_only=True гарантирует: embedding, reranker и tokenizer
    грузятся строго из локального кеша без HTTP к HuggingFace Hub.
    Это обеспечивает воспроизводимость и устойчивость к сетевым сбоям.
    """

    local_files_only: bool = True


class RepresentationConfig(BaseModel):
    """Stage 2A: конфигурация chunk representation (retrieval-side only).

    mode определяет способ обогащения текста чанков для embedding:
    - plain (C0): без обогащения — Stage 1 baseline
    - contextual (C1): детерминированный context_prefix
    - llm_enriched (C2): LLM-сгенерированный enrichment_prefix
    """

    mode: Literal["plain", "contextual", "llm_enriched"] = "plain"
    c2_prompt_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    c2_prompt_max_tokens: int = Field(default=64, ge=16, le=1024)


# ---------------------------------------------------------------------------
# Generation config (versioned artifact для воспроизводимости eval set)
# ---------------------------------------------------------------------------


class DedupConfig(BaseModel):
    """Параметры дедупликации запросов (Bias 4)."""

    model: str = "intfloat/multilingual-e5-base"
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class VariationConfig(BaseModel):
    """Параметры LLM-валидации intent вариаций (Bias 5)."""

    intent_validation_model: str = "openai/gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 10


class SplitConfig(BaseModel):
    """Параметры group-stratified dev/test split (Bias 7)."""

    method: str = "group_stratified"
    group_field: str = "seed_family_id"
    test_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    seed: int = 42


class GenerationConfig(BaseModel):
    """Конфигурация генерации eval set (generation_config.yaml).

    Все параметры, влияющие на состав eval set. Без этого конфига
    эксперимент невоспроизводим — параметры остаются в notebook state.
    """

    dedup: DedupConfig = Field(default_factory=DedupConfig)
    variation: VariationConfig = Field(default_factory=VariationConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class S5PolicyConfig(BaseModel):
    """Конфигурация S5 tiered policy (B0 → S4r каскад).

    confidence_threshold — минимальный confidence для direct path.
    Eligibility rules — дополнительные фильтры поверх threshold:
    какие match_type допустимы, запрет multi-fact / aftercare / OOS.
    """

    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    allowed_match_types: list[str] = Field(
        default=["topic_keyword", "price_query", "doctor_query"],
        description="match_type из B0 confidence_debug, допустимые для direct path",
    )
    allow_aftercare_direct: bool = Field(
        default=False,
        description="Пускать aftercare_query в direct (обычно нет — ответы B0 шумные)",
    )
    allow_multifact_direct: bool = Field(
        default=False,
        description="Пускать multi-fact запросы в direct (обычно нет — B0 не синтезирует)",
    )
    doctor_requires_exact_entity: bool = Field(
        default=True,
        description="Для doctor_query direct допустим только при entity_match=exact",
    )


class ExperimentConfig(BaseModel):
    """Полная конфигурация эксперимента D4v2.

    Загружается из experiment.yaml, валидируется при создании.
    """

    experiment: ExperimentMeta
    llm: LLMConfig
    judge: JudgeConfig
    embedding: EmbeddingConfig
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    s5_policy: S5PolicyConfig = Field(default_factory=S5PolicyConfig)
    rate_limit: RateLimitConfig

    # Секции стратегий (документационные, core/experimental/descriptive)
    core_strategies: list[dict[str, Any]] = Field(default_factory=list)
    experimental_branches: list[dict[str, Any]] = Field(default_factory=list)
    descriptive_baseline: list[dict[str, Any]] = Field(default_factory=list)

    # Прочие документационные секции
    eval_set: dict[str, Any] = Field(default_factory=dict)
    kb: dict[str, Any] = Field(default_factory=dict)
    kb_scale: dict[str, Any] = Field(default_factory=dict)
    s1_policy: dict[str, Any] = Field(default_factory=dict)
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


def load_generation_config(config_path: str | Path) -> GenerationConfig:
    """Загрузка и валидация generation_config.yaml.

    Args:
        config_path: путь к YAML файлу

    Returns:
        валидированный GenerationConfig

    Raises:
        FileNotFoundError: файл не найден
        pydantic.ValidationError: невалидная структура/значения
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return GenerationConfig.model_validate(raw)
