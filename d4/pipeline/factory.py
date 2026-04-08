"""Фабрика для создания стратегий, LLMRunner и LLMJudge из experiment.yaml.

Единая точка инициализации — устраняет дублирование между notebooks.
"""

from __future__ import annotations

from d4.config import ExperimentConfig
from d4.evaluation.llm_judge import LLMJudge
from d4.evaluation.nli_checker import NLIClaimChecker
from d4.models import KBChunk
from d4.pipeline.llm_runner import LLMRunner
from d4.strategies.base import BaseContextStrategy
from d4.strategies.full_context import FullContextStrategy
from d4.strategies.hybrid import HybridStrategy
from d4.strategies.keyword_template import KeywordTemplateStrategy
from d4.strategies.lexical import LexicalStrategy
from d4.strategies.vector import VectorStrategy


def build_strategies(
    config: ExperimentConfig,
    include_baseline: bool = True,
) -> list[BaseContextStrategy]:
    """Создание списка стратегий из конфигурации.

    Args:
        config: валидированный ExperimentConfig
        include_baseline: включать B0 (по умолчанию True)

    Returns:
        список инициализированных стратегий
    """
    emb = config.embedding
    top_k = emb.top_k
    model_name = emb.model
    device = emb.device

    strategies: list[BaseContextStrategy] = [
        FullContextStrategy(),
        LexicalStrategy(top_k=top_k),
        VectorStrategy(model_name=model_name, device=device, top_k=top_k),
        HybridStrategy(model_name=model_name, device=device, top_k=top_k),
    ]

    if include_baseline:
        strategies.append(KeywordTemplateStrategy())

    return strategies


def build_llm_runner(
    config: ExperimentConfig,
    api_key: str,
) -> LLMRunner:
    """Создание LLMRunner из конфигурации.

    Args:
        config: валидированный ExperimentConfig
        api_key: API ключ OpenRouter (из переменной окружения)

    Returns:
        инициализированный LLMRunner
    """
    return LLMRunner(
        api_url=config.llm.api_url,
        api_key=api_key,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_sec=config.rate_limit.request_timeout_sec,
    )


def build_llm_judge(
    config: ExperimentConfig,
    api_key: str,
) -> LLMJudge:
    """Создание LLMJudge из конфигурации.

    Args:
        config: валидированный ExperimentConfig
        api_key: API ключ OpenRouter (из переменной окружения)

    Returns:
        инициализированный LLMJudge
    """
    return LLMJudge(
        api_key=api_key,
        api_url=config.judge.api_url,
        model=config.judge.model,
        temperature=config.judge.temperature,
        max_tokens=config.judge.max_tokens,
        timeout_sec=config.rate_limit.request_timeout_sec,
        max_workers=config.rate_limit.max_workers,
    )


def build_nli_checker(
    config: ExperimentConfig,
    api_key: str,
) -> NLIClaimChecker:
    """Создание NLI claim checker из конфигурации.

    Использует judge config (fast-модель для проверки фактов).

    Args:
        config: валидированный ExperimentConfig
        api_key: API ключ OpenRouter

    Returns:
        инициализированный NLIClaimChecker
    """
    return NLIClaimChecker(
        api_url=config.judge.api_url,
        api_key=api_key,
        model=config.judge.model,
        temperature=0.0,
        max_tokens=256,
        timeout_sec=config.rate_limit.request_timeout_sec,
        max_workers=config.rate_limit.max_workers,
    )


def get_reference_llm_config(
    config: ExperimentConfig,
    api_key: str,
) -> dict[str, object]:
    """Параметры reference LLM для generate_reference_answers().

    Использует judge config: другая модель, temperature=0
    для детерминированной генерации эталонных ответов.

    Args:
        config: валидированный ExperimentConfig
        api_key: API ключ OpenRouter

    Returns:
        dict с kwargs для generate_reference_answers()
    """
    return {
        "api_url": config.judge.api_url,
        "api_key": api_key,
        "model": config.judge.model,
        "temperature": 0.0,
        "max_tokens": config.judge.max_tokens,
        "max_workers": config.rate_limit.max_workers,
        "timeout_sec": config.rate_limit.request_timeout_sec,
    }


def precompute_embeddings(
    strategies: list[BaseContextStrategy],
    chunks: list[KBChunk],
) -> None:
    """Предвычисление embeddings для vector-стратегий (S3, S4).

    Args:
        strategies: список стратегий
        chunks: KB chunks для предвычисления
    """
    for strategy in strategies:
        if hasattr(strategy, "precompute_embeddings"):
            strategy.precompute_embeddings(chunks)
