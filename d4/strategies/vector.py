"""S3: Vector Retrieval — embedding similarity → top-k.

Требования по плану:
- Одна фиксированная embedding model (bge-m3)
- Один similarity method (cosine)
- Фиксированный top_k для всех запросов
- Отсутствие lexical rerank
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from d4.models import KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy


class VectorStrategy(BaseContextStrategy):
    """S3: Vector retrieval через embedding similarity."""

    strategy_id = "S3"
    name = "Vector Retrieval"
    uses_llm = True

    def __init__(
        self,
        model_name: str,
        device: str,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None
        self._chunk_embeddings: Optional[np.ndarray] = None
        self._chunk_ids: list[str] = []

    def _get_model(self):
        """Ленивая загрузка embedding модели."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def precompute_embeddings(self, chunks: list[KBChunk]) -> None:
        """Предвычисление embeddings для всех chunks.

        Вызывается один раз перед прогоном всех запросов.
        """
        model = self._get_model()
        texts = [f"{c.title}\n{c.content}" for c in chunks]
        self._chunk_embeddings = model.encode(texts, normalize_embeddings=True)
        self._chunk_ids = [c.id for c in chunks]

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Cosine similarity → top-k chunks. Без lexical rerank."""
        if not chunks:
            return RetrievalResult()

        model = self._get_model()

        # Предвычислить embeddings если не готовы
        if self._chunk_embeddings is None or len(self._chunk_ids) != len(chunks):
            self.precompute_embeddings(chunks)

        # Embedding запроса
        query_embedding = model.encode([query], normalize_embeddings=True)[0]

        # Cosine similarity (нормализованные → dot product)
        similarities = np.dot(self._chunk_embeddings, query_embedding)

        # Top-k по similarity
        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        selected_ids = [self._chunk_ids[i] for i in top_indices]
        selected_scores = [float(similarities[i]) for i in top_indices]

        # Собираем chunks в порядке убывания score
        chunk_map = {c.id: c for c in chunks}
        selected_chunks = [chunk_map[cid] for cid in selected_ids]

        context_text = self.format_context(selected_chunks)
        total_tokens = sum(c.token_count for c in selected_chunks)

        return RetrievalResult(
            chunk_ids=selected_ids,
            scores=selected_scores,
            context_text=context_text,
            context_token_count=total_tokens,
        )
