"""Pydantic data models for D4v2 FAQ Architecture Experiment.

All data types used across strategies, pipeline, and evaluation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyID(str, Enum):
    S1 = "S1"  # Full Context
    S2 = "S2"  # Lexical Retrieval
    S3 = "S3"  # Vector Retrieval
    S4 = "S4"  # Hybrid Retrieval
    B0 = "B0"  # Keyword + Template (baseline)


class Difficulty(int, Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


# ---------------------------------------------------------------------------
# KB & Chunking
# ---------------------------------------------------------------------------

class KBChunk(BaseModel):
    """Логическая retrieval unit (1 врач или 1 FAQ-секция)."""

    id: str
    title: str
    content: str
    source: str = Field(description="Исходный файл: clinic_info.yaml / doctors.yaml")
    source_type: str = Field(description="clinic_info / doctors")
    entity_type: str = Field(description="doctor / faq_section")
    entity_id: Optional[str] = Field(default=None, description="ID сущности (напр. doctor slug)")
    token_count: int = 0
    raw_data: dict = Field(default_factory=dict, description="Оригинальный dict из YAML для B0")


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
    gold_chunk_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Результат retrieval layer (для S2-S4)."""

    chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    context_text: str = ""
    context_token_count: int = 0


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
    error: Optional[str] = None


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

    @property
    def unsupported_claim_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.unsupported_claims / self.total_claims


class RetrievalScore(BaseModel):
    """Retrieval-layer quality metrics для S2-S4."""

    sample_id: str
    strategy_id: StrategyID
    gold_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    hit_at_k: bool = False
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    gold_chunk_in_context: bool = False
