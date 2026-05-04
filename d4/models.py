"""Pydantic data models for D4v2 FAQ Architecture Experiment.

All data types used across strategies, pipeline, and evaluation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyID(str, Enum):
    S1 = "S1"   # Full Context
    S2 = "S2"   # Lexical Retrieval
    S3 = "S3"   # Vector Retrieval
    S4 = "S4"   # Hybrid Retrieval
    S4r = "S4r" # Hybrid + Reranking
    S5 = "S5"   # Tiered (B0 → S4r каскад)
    B0 = "B0"   # Keyword + Template (baseline)


class Difficulty(int, Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


# ---------------------------------------------------------------------------
# KB & Chunking
# ---------------------------------------------------------------------------

class KBChunk(BaseModel):
    """Логическая retrieval unit (1 врач или 1 FAQ-секция).

    Stage 2 enrichment: context_prefix (C1) и enrichment_prefix (C2)
    влияют на embedding через embedding_text, но НЕ на LLM context
    (Stage 2A — retrieval-side only).
    """

    id: str
    title: str
    content: str
    source: str = Field(description="Исходный файл: clinic_info.yaml / doctors.yaml")
    source_type: str = Field(description="clinic_info / doctors")
    entity_type: str = Field(description="doctor / faq_section")
    entity_id: Optional[str] = Field(default=None, description="ID сущности (напр. doctor slug)")
    token_count: int = 0
    raw_data: dict = Field(default_factory=dict, description="Оригинальный dict из YAML для B0")

    # Stage 2: chunk representation enrichment
    context_prefix: str = Field(default="", description="C1: детерминированный контекстный заголовок")
    enrichment_prefix: str = Field(default="", description="C2: LLM-сгенерированный контекстный префикс")

    @property
    def embedding_text(self) -> str:
        """Текст для embedding-индексации (Stage 2A).

        Конкатенация всех слоёв: context_prefix (C1) + enrichment_prefix (C2)
        + title + content. В C2 metadata-header сохраняется как стабильный
        grounding-layer, LLM-enrichment дополняет, а не вытесняет его.
        Property не сериализуется в JSON — chunks.json остаётся чистым.
        """
        parts = [p for p in (self.context_prefix, self.enrichment_prefix) if p]
        base = f"{self.title}\n{self.content}"
        return f"{chr(10).join(parts)}\n\n{base}" if parts else base


# ---------------------------------------------------------------------------
# Gold Annotation
# ---------------------------------------------------------------------------

class GoldFact(BaseModel):
    """Атомарный проверяемый факт с типизированным нормализатором.

    fact_type определяет, какой нормализатор применяется при сравнении
    canonical_value с текстом ответа (phone → только цифры, price → без пробелов и т.д.)
    """

    fact_type: str = Field(
        description="phone | price | fio | address | schedule | text",
    )
    canonical_value: str = Field(description="Каноническая форма факта")
    label: str = Field(default="", description="Человекочитаемое описание для отладки")


# ---------------------------------------------------------------------------
# Eval Set
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """Запрос из eval set + экспертная разметка."""

    sample_id: str
    query: str
    category: str
    subtype: str = ""
    answerable: bool = True
    expected_answer: str = ""
    expected_doctor: Optional[str] = None
    expected_specialization: Optional[str] = None
    expected_branch: Optional[str] = None
    expected_service: Optional[str] = None
    difficulty: Difficulty = Difficulty.MEDIUM
    notes: str = ""
    seed_family_id: str = Field(
        default="",
        description="ID seed-запроса для group split (seed + вариации = одна группа)",
    )
    gold_chunk_ids: list[list[str]] = Field(default_factory=list)
    gold_facts: list[GoldFact] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Результат retrieval layer (для S2-S4, S4r).

    pre_rerank_chunk_ids заполняется только S4r — содержит top-N кандидатов
    до reranking для расчёта rerank lift (pre_rerank_hit@N vs post_rerank_hit@K).
    """

    chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    context_text: str = ""
    context_token_count: int = 0
    pre_rerank_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Top-N кандидатов до reranking (только S4r)",
    )


# ---------------------------------------------------------------------------
# LLM Answer
# ---------------------------------------------------------------------------

class FAQAnswer(BaseModel):
    """Единый output format для всех стратегий (S1-S4 + B0)."""

    answer: str = ""
    answerable: bool = True
    doctor: Optional[str] = None
    specialization: Optional[str] = None
    branch: Optional[str] = None
    service: Optional[str] = None
    suggest_booking: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_ids: list[str] = Field(default_factory=list, description="IDs retrieval units использованных в ответе")


# ---------------------------------------------------------------------------
# Direct Answer Result (S5, B0 — стратегии с answer_directly)
# ---------------------------------------------------------------------------

@dataclass
class DirectAnswerResult:
    """Результат стратегии с answer_directly (S5, B0).

    Инкапсулирует ответ + retrieval + tokens + route для корректного
    учёта cost/latency. Без этого orchestrator не отличает direct от fallback.
    """

    answer: FAQAnswer
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    route_taken: str = "direct"  # "direct" | "fallback"
    error: Optional[str] = None
    confidence_debug: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy Result (один запрос × одна стратегия)
# ---------------------------------------------------------------------------

class StrategyResult(BaseModel):
    """Полный результат обработки одного запроса одной стратегией."""

    sample_id: str
    strategy_id: StrategyID
    retrieval: RetrievalResult = Field(default_factory=RetrievalResult)
    answer: FAQAnswer = Field(default_factory=FAQAnswer)
    latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    context_length: int = 0
    route_taken: str = Field(
        default="",
        description="direct | fallback | '' (для LLM-стратегий без answer_directly)",
    )
    error: Optional[str] = None
    confidence_debug: Optional[dict] = Field(
        default=None,
        description="B0 confidence components (для B0/S5 диагностики)",
    )


# ---------------------------------------------------------------------------
# Evaluation Scores
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    """Оценка LLM-judge для одного ответа."""

    sample_id: str
    strategy_id: StrategyID
    factual_accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    hallucination: bool = False
    reasoning: str = ""


class DeterministicScore(BaseModel):
    """Автоматические deterministic checks для одного ответа."""

    sample_id: str
    strategy_id: StrategyID
    answerability_correct: bool = False
    doctor_match: bool = False
    specialization_match: bool = False
    branch_match: bool = False
    service_match: bool = False
    unsupported_claims: int = 0
    total_claims: int = 0
    fact_match_rate: float | None = None

    @property
    def unsupported_claim_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.unsupported_claims / self.total_claims


class RetrievalScore(BaseModel):
    """Retrieval-layer quality metrics для S2-S4r.

    pre_rerank_hit заполняется только для S4r — позволяет измерить
    вклад reranking (rerank_lift = post_hit - pre_hit).
    """

    sample_id: str
    strategy_id: StrategyID
    gold_chunk_ids: list[list[str]] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    hit_at_k: bool = False
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    gold_chunk_in_context: bool = False
    pre_rerank_hit: bool = Field(
        default=False,
        description="Gold chunk в pre-rerank кандидатах (только S4r)",
    )
    pre_rerank_recall: float = Field(
        default=0.0,
        description="Recall в pre-rerank кандидатах (только S4r)",
    )
