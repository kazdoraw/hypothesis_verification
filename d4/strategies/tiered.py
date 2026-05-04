"""S5: Tiered Strategy (B0 → S4r каскад).

Каскадная архитектура для production:
1. B0 (keyword+template) отвечает на типовые запросы
   (confidence >= threshold AND eligibility rules pass)
2. При низкой confidence или non-eligible запросе → fallback на S4r+LLM

Eligibility rules (поверх threshold):
- multi_fact-подобные запросы → всегда fallback
- aftercare → fallback (B0 ответы шумные)
- doctor → direct только при entity_match=exact
- topic_keyword → direct допустим
- none / OOS-like → fallback
"""

from __future__ import annotations

import time

from d4.models import DirectAnswerResult, FAQAnswer, KBChunk, RetrievalResult
from d4.strategies._patterns import MULTI_FACT_RE
from d4.strategies.base import BaseContextStrategy
from d4.strategies.hybrid_rerank import HybridRerankStrategy
from d4.strategies.keyword_template import KeywordTemplateStrategy


class TieredStrategy(BaseContextStrategy):
    """S5: Tiered — B0 для типовых, S4r+LLM для сложных."""

    strategy_id = "S5"
    name = "Tiered (B0 → S4r)"
    uses_llm = True

    def __init__(
        self,
        b0: KeywordTemplateStrategy,
        fallback: HybridRerankStrategy,
        llm_runner: object | None = None,
        confidence_threshold: float = 0.7,
        allowed_match_types: list[str] | None = None,
        allow_aftercare_direct: bool = False,
        allow_multifact_direct: bool = False,
        doctor_requires_exact_entity: bool = True,
    ) -> None:
        self._b0 = b0
        self._fallback = fallback
        self._llm_runner = llm_runner
        self._threshold = confidence_threshold
        self._allowed_match_types = set(
            allowed_match_types or ["topic_keyword", "price_query", "doctor_query"],
        )
        self._allow_aftercare = allow_aftercare_direct
        self._allow_multifact = allow_multifact_direct
        self._doctor_requires_exact = doctor_requires_exact_entity

    def set_llm_runner(self, runner: object) -> None:
        """Инжекция LLMRunner после создания (из Orchestrator/notebook)."""
        self._llm_runner = runner

    def precompute_embeddings(self, chunks: list[KBChunk]) -> None:
        """Делегируется fallback-стратегии (S4r)."""
        self._fallback.precompute_embeddings(chunks)

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Retrieval через fallback (S4r) — используется Orchestrator'ом."""
        return self._fallback.select_context(query, chunks)

    _DOCTOR_TRIGGERS = frozenset(
        ["врач", "доктор", "стаж", "хирург", "ортодонт", "терапевт", "ортопед",
         "стоматолог", "имплантолог", "пародонтолог"],
    )

    def _is_eligible_for_direct(
        self,
        query: str,
        conf_debug: dict,
        b0_answerable: bool = True,
    ) -> bool:
        """Eligibility check поверх confidence threshold.

        Returns True только если B0 может безопасно ответить напрямую.

        Guards:
        - b0_answerable=False → always fallback (OOS, missing_info, no-match)
        - multi_fact regex → fallback
        - match_type not in allowed → fallback
        - doctor-like query but top chunk is not doctor → fallback
        - aftercare → fallback (unless explicitly allowed)
        - doctor match without exact entity → fallback
        """
        if not b0_answerable:
            return False

        match_type = conf_debug.get("match_type", "none")
        entity_match = conf_debug.get("entity_match", "none")

        if not self._allow_multifact and MULTI_FACT_RE.search(query):
            return False

        if match_type not in self._allowed_match_types:
            return False

        if match_type == "aftercare_query" and not self._allow_aftercare:
            return False

        if match_type == "doctor_query" and self._doctor_requires_exact:
            if entity_match != "exact":
                return False

        query_lower = query.lower()
        is_doctor_query = any(t in query_lower for t in self._DOCTOR_TRIGGERS)
        if is_doctor_query and match_type != "doctor_query":
            return False

        return True

    def answer_directly(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> DirectAnswerResult:
        """Каскадный ответ: B0 → (при низкой confidence / non-eligible) S4r+LLM.

        Два условия для direct path:
        1. b0_confidence >= threshold
        2. _is_eligible_for_direct() == True
        """
        start = time.perf_counter()
        b0_result = self._b0.answer_directly(query, chunks)
        b0_answer = b0_result.answer if isinstance(b0_result.answer, FAQAnswer) else b0_result.answer
        b0_retrieval = b0_result.retrieval
        b0_conf_debug = b0_result.confidence_debug

        passes_threshold = b0_answer.confidence >= self._threshold
        eligible = self._is_eligible_for_direct(
            query, b0_conf_debug, b0_answerable=b0_answer.answerable,
        ) if b0_conf_debug else False

        if passes_threshold and eligible:
            if b0_conf_debug:
                b0_conf_debug["fallback_reason"] = None
            elapsed = (time.perf_counter() - start) * 1000
            return DirectAnswerResult(
                answer=b0_answer,
                retrieval=b0_retrieval,
                latency_ms=elapsed,
                route_taken="direct",
                confidence_debug=b0_conf_debug,
            )

        if b0_conf_debug:
            if not passes_threshold and not eligible:
                b0_conf_debug["fallback_reason"] = "below_threshold+failed_eligibility"
            elif not passes_threshold:
                b0_conf_debug["fallback_reason"] = "below_threshold"
            else:
                b0_conf_debug["fallback_reason"] = "failed_eligibility"

        if self._llm_runner is None:
            elapsed = (time.perf_counter() - start) * 1000
            return DirectAnswerResult(
                answer=b0_answer,
                retrieval=b0_retrieval,
                latency_ms=elapsed,
                route_taken="direct",
                error="llm_runner не инжектирован",
                confidence_debug=b0_conf_debug,
            )

        retrieval = self._fallback.select_context(query, chunks)
        llm_result = self._llm_runner.run(query, retrieval.context_text)
        elapsed = (time.perf_counter() - start) * 1000

        return DirectAnswerResult(
            answer=llm_result["answer"],
            retrieval=retrieval,
            latency_ms=elapsed,
            tokens_prompt=llm_result.get("tokens_prompt", 0),
            tokens_completion=llm_result.get("tokens_completion", 0),
            route_taken="fallback",
            error=llm_result.get("error"),
            confidence_debug=b0_conf_debug,
        )
