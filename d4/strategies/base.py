"""Базовый класс для стратегий формирования контекста.

Все стратегии (S1-S4, B0) наследуют BaseContextStrategy
и реализуют метод select_context().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from d4.models import FAQAnswer, KBChunk, RetrievalResult


class BaseContextStrategy(ABC):
    """Абстрактная стратегия формирования контекста для LLM.

    Единственная переменная в эксперименте — способ выбора контекста.
    LLM, prompt, output schema фиксированы для S1-S4.
    """

    strategy_id: str  # "S1", "S2", "S3", "S4", "B0"
    name: str
    uses_llm: bool = True

    @abstractmethod
    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Выбор релевантного контекста для запроса.

        Args:
            query: запрос пациента
            chunks: все логические retrieval units из KB

        Returns:
            RetrievalResult с chunk_ids, scores и собранным context_text
        """
        ...

    def format_context(self, chunks: list[KBChunk]) -> str:
        """Сериализация списка chunks в текст для LLM.

        Единый формат для всех стратегий — гарантирует
        сопоставимость результатов.
        """
        parts: list[str] = []
        for chunk in chunks:
            parts.append(f"[{chunk.id}] {chunk.title}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)
