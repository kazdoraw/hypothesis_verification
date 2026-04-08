"""S1: Full Context — вся KB целиком в prompt.

Детерминированная сериализация: фиксированный порядок секций
(clinic_info → doctors). Никакого retrieval — весь контекст передаётся.
"""

from __future__ import annotations

from d4.models import KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy

# Фиксированный порядок: clinic_info → прайс → doctors
SOURCE_ORDER = ("clinic_info", "price_list", "doctors")


class FullContextStrategy(BaseContextStrategy):
    """S1: передача всей KB целиком в контекст LLM."""

    strategy_id = "S1"
    name = "Full Context"
    uses_llm = True

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Возвращает ВСЕ chunks в детерминированном порядке.

        Порядок: clinic_info секции → doctors (по id).
        """
        # Сортировка: source_type → entity_id
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                SOURCE_ORDER.index(c.source_type) if c.source_type in SOURCE_ORDER else 99,
                c.entity_id or "",
            ),
        )

        context_text = self.format_context(sorted_chunks)
        total_tokens = sum(c.token_count for c in sorted_chunks)

        return RetrievalResult(
            chunk_ids=[c.id for c in sorted_chunks],
            scores=[1.0] * len(sorted_chunks),  # все релевантны
            context_text=context_text,
            context_token_count=total_tokens,
        )
