"""Фабрика для создания стратегий, LLMRunner и LLMJudge из experiment.yaml.

Единая точка инициализации — устраняет дублирование между notebooks.
"""

from __future__ import annotations

import os
from pathlib import Path

from d4.config import ExperimentConfig
from d4.pipeline.chunker import configure_tokenizer
from d4.evaluation.llm_judge import LLMJudge
from d4.evaluation.nli_checker import NLIClaimChecker
from d4.models import KBChunk
from d4.pipeline.llm_runner import LLMRunner
from d4.strategies.base import BaseContextStrategy
from d4.strategies.full_context import FullContextStrategy
from d4.strategies.hybrid import HybridStrategy
from d4.strategies.hybrid_rerank import HybridRerankStrategy
from d4.strategies.keyword_template import KeywordTemplateStrategy
from d4.strategies.lexical import LexicalStrategy
from d4.strategies.tiered import TieredStrategy
from d4.strategies.vector import VectorStrategy


def build_strategies(
    config: ExperimentConfig,
    include_baseline: bool = True,
    include_experimental: bool = False,
) -> list[BaseContextStrategy]:
    """Создание списка стратегий из конфигурации.

    По умолчанию — только core (S1-S4) + B0. Для research-прогонов
    include_experimental=True добавляет S4r и S5.

    Args:
        config: валидированный ExperimentConfig
        include_baseline: включать B0 (descriptive baseline)
        include_experimental: включать S4r, S5 (research branches)

    Returns:
        список инициализированных стратегий
    """
    local_files_only = config.hub.local_files_only

    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    tok = config.tokenizer
    tok_revision = tok.revision if tok.revision != "main" else None
    configure_tokenizer(model=tok.model, revision=tok_revision, local_files_only=local_files_only)

    emb = config.embedding
    top_k = emb.top_k
    model_name = emb.model
    device = emb.device
    emb_revision = emb.revision if emb.revision != "main" else None

    # Core strategies: S1, S2, S3, S4
    strategies: list[BaseContextStrategy] = [
        FullContextStrategy(),
        LexicalStrategy(top_k=top_k),
        VectorStrategy(
            model_name=model_name, device=device, top_k=top_k,
            model_revision=emb_revision, local_files_only=local_files_only,
        ),
        HybridStrategy(
            model_name=model_name, device=device, top_k=top_k,
            model_revision=emb_revision, local_files_only=local_files_only,
        ),
    ]

    b0 = KeywordTemplateStrategy() if include_baseline else None

    if include_experimental:
        rerank_cfg = config.reranking
        rerank_revision = rerank_cfg.revision if rerank_cfg.revision != "main" else None
        hybrid_wide = HybridStrategy(
            model_name=model_name, device=device,
            top_k=rerank_cfg.top_k_retrieve,
            model_revision=emb_revision, local_files_only=local_files_only,
        )
        s4r = HybridRerankStrategy(
            hybrid=hybrid_wide,
            reranker_model=rerank_cfg.model,
            reranker_device=rerank_cfg.device,
            top_k_rerank=rerank_cfg.top_k_rerank,
            reranker_revision=rerank_revision,
            local_files_only=local_files_only,
        )
        strategies.append(s4r)

        if b0:
            s5_cfg = config.s5_policy
            strategies.append(TieredStrategy(
                b0=b0,
                fallback=s4r,
                confidence_threshold=s5_cfg.confidence_threshold,
                allowed_match_types=s5_cfg.allowed_match_types,
                allow_aftercare_direct=s5_cfg.allow_aftercare_direct,
                allow_multifact_direct=s5_cfg.allow_multifact_direct,
                doctor_requires_exact_entity=s5_cfg.doctor_requires_exact_entity,
            ))

    if b0:
        strategies.append(b0)

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
    """Предвычисление embeddings для vector-стратегий (S3, S4, S4r, S5).

    Args:
        strategies: список стратегий
        chunks: KB chunks для предвычисления
    """
    for strategy in strategies:
        if hasattr(strategy, "precompute_embeddings"):
            strategy.precompute_embeddings(chunks)


def enrich_chunks_from_config(
    config: ExperimentConfig,
    chunks: list[KBChunk],
    api_key: str | None = None,
) -> list[KBChunk]:
    """Обогащение чанков по config.representation.mode (Stage 2A).

    Вызывает enrichment.enrich_chunks с параметрами из конфигурации.
    Для mode='plain' (C0) возвращает чанки без изменений.

    Args:
        config: валидированный ExperimentConfig
        chunks: исходные KB chunks
        api_key: API ключ (обязателен для mode='llm_enriched')

    Returns:
        обогащённые (или исходные для plain) KBChunk
    """
    from d4.pipeline.enrichment import (
        enrich_chunks,
        load_enrichment_cache,
        save_enrichment_cache,
    )

    rep = config.representation
    llm_model = config.llm.model if rep.mode == "llm_enriched" else ""
    cache_dir = Path(__file__).resolve().parent.parent / "outputs" / "enrichment_cache"

    cache_kwargs = dict(
        c2_max_tokens=rep.c2_prompt_max_tokens,
        c2_temperature=rep.c2_prompt_temperature,
        llm_model=llm_model,
    )

    cached = load_enrichment_cache(chunks, rep.mode, cache_dir, **cache_kwargs)
    if cached is not None:
        import logging
        logging.getLogger(__name__).info(
            "Using cached enrichment: mode=%s, %d chunks", rep.mode, len(cached),
        )
        return cached

    llm_runner = None
    if rep.mode == "llm_enriched" and api_key:
        llm_runner = build_llm_runner(config, api_key)

    enriched = enrich_chunks(
        chunks,
        mode=rep.mode,
        llm_runner=llm_runner,
        c2_temperature=rep.c2_prompt_temperature,
        c2_max_tokens=rep.c2_prompt_max_tokens,
    )

    if rep.mode != "plain":
        save_enrichment_cache(enriched, rep.mode, cache_dir, **cache_kwargs)

    return enriched
