"""S4: Hybrid Retrieval — Lexical + Vector через Reciprocal Rank Fusion (RRF).

Требования по плану:
- Допустимы union, RRF, weighted merge
- НЕ использовать тяжёлый agentic RAG-контур
- Комбинация lexical (BM25) и vector (embedding) результатов
"""

from __future__ import annotations

import re

from d4.models import KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy
from d4.strategies.lexical import LexicalStrategy
from d4.strategies.vector import VectorStrategy

# RRF параметр (стандартное значение из литературы)
RRF_K = 60

_PRICE_INTENT_RE = re.compile(
    r"(почём|скольк|стоит|стоимост|цен[аыу]|прайс|прейскурант|"
    r"расценк|тариф|за\s+сколько|дорого|дёшев|бюджет)",
    re.IGNORECASE,
)

PRICE_INTENT_BOOST = 1.3


def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion для объединения нескольких ранжирований.

    Args:
        rankings: список ранжирований (каждое — list[chunk_id] по убыванию)
        k: параметр сглаживания (стандарт = 60)

    Returns:
        список (chunk_id, rrf_score) отсортированный по убыванию score
    """
    rrf_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridStrategy(BaseContextStrategy):
    """S4: Hybrid retrieval через RRF(Lexical + Vector)."""

    strategy_id = "S4"
    name = "Hybrid Retrieval"
    uses_llm = True

    def __init__(
        self,
        model_name: str,
        device: str,
        top_k: int = 5,
        lexical_top_k: int = 10,
        vector_top_k: int = 10,
        model_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.top_k = top_k
        self._lexical = LexicalStrategy(top_k=lexical_top_k)
        self._vector = VectorStrategy(
            top_k=vector_top_k,
            model_name=model_name,
            device=device,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )

    def precompute_embeddings(self, chunks: list[KBChunk]) -> None:
        """Предвычисление embeddings (делегируется vector стратегии)."""
        self._vector.precompute_embeddings(chunks)

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """RRF(BM25 ranking + Vector ranking) → top-k chunks."""
        if not chunks:
            return RetrievalResult()

        # Получаем ранжирования от обеих стратегий
        lexical_result = self._lexical.select_context(query, chunks)
        vector_result = self._vector.select_context(query, chunks)

        # RRF fusion
        rankings = [lexical_result.chunk_ids, vector_result.chunk_ids]
        fused = _reciprocal_rank_fusion(rankings)

        # Query-intent prior: price-like запросы получают бонус для price_list chunks
        if _PRICE_INTENT_RE.search(query):
            price_ids = {c.id for c in chunks if c.source_type == "price_list"}
            fused = [
                (cid, score * PRICE_INTENT_BOOST if cid in price_ids else score)
                for cid, score in fused
            ]
            fused.sort(key=lambda x: x[1], reverse=True)

        # Top-k после fusion
        top_results = fused[: self.top_k]

        if not top_results:
            return RetrievalResult()

        chunk_map = {c.id: c for c in chunks}
        selected_ids = [cid for cid, _ in top_results]
        selected_scores = [score for _, score in top_results]
        selected_chunks = [chunk_map[cid] for cid in selected_ids if cid in chunk_map]

        context_text = self.format_context(selected_chunks)
        total_tokens = sum(c.token_count for c in selected_chunks)

        return RetrievalResult(
            chunk_ids=selected_ids,
            scores=selected_scores,
            context_text=context_text,
            context_token_count=total_tokens,
        )
