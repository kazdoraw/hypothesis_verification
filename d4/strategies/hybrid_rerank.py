"""S4r: Hybrid + Cross-Encoder Reranking.

Двухстадийный pipeline по стандарту 2026:
1. HybridStrategy(top_k=20) → RRF top-20 кандидатов
2. CrossEncoder reranking → top-5 по relevance score

Композиция (не наследование) HybridStrategy — S4r создаёт
HybridStrategy с расширенным top_k и дополняет reranking.
"""

from __future__ import annotations

import threading
import time

from d4.models import KBChunk, RetrievalResult
from d4.strategies._patterns import MULTI_FACT_RE
from d4.strategies.base import BaseContextStrategy
from d4.strategies.hybrid import HybridStrategy

_ZONE_PREFIXES: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor_",),
    "price": ("price_",),
    "aftercare": ("aftercare_",),
}


class HybridRerankStrategy(BaseContextStrategy):
    """S4r: Hybrid retrieval + cross-encoder reranking."""

    strategy_id = "S4r"
    name = "Hybrid + Reranking"
    uses_llm = True

    RERANK_CONFIDENCE_FLOOR = 0.05

    def __init__(
        self,
        hybrid: HybridStrategy,
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_device: str = "cpu",
        top_k_rerank: int = 5,
        reranker_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._hybrid = hybrid
        self._reranker_model = reranker_model
        self._reranker_device = reranker_device
        self._reranker_revision = reranker_revision
        self._local_files_only = local_files_only
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._top_k_rerank = top_k_rerank

    def _get_reranker(self):
        """Thread-safe lazy init CrossEncoder."""
        if self._reranker is not None:
            return self._reranker
        with self._reranker_lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                t0 = time.perf_counter()
                print(f"  [S4r] Init CrossEncoder → RAM ({self._reranker_model})...", flush=True)
                kwargs: dict = {
                    "device": self._reranker_device,
                    "local_files_only": self._local_files_only,
                }
                if self._reranker_revision:
                    kwargs["revision"] = self._reranker_revision
                self._reranker = CrossEncoder(self._reranker_model, **kwargs)
                dt = time.perf_counter() - t0
                print(f"  [S4r] CrossEncoder ready ({dt:.1f}s)", flush=True)
        return self._reranker

    def precompute_embeddings(self, chunks: list[KBChunk]) -> None:
        """Делегируется внутренней HybridStrategy + eager-load reranker."""
        self._hybrid.precompute_embeddings(chunks)
        self._get_reranker()

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Hybrid top-N → CrossEncoder reranking → top-k."""
        if not chunks:
            return RetrievalResult()

        hybrid_result = self._hybrid.select_context(query, chunks)
        if not hybrid_result.chunk_ids:
            return hybrid_result

        pre_rerank_ids = list(hybrid_result.chunk_ids)

        chunk_map = {c.id: c for c in chunks}
        candidates = [
            (cid, chunk_map[cid])
            for cid in hybrid_result.chunk_ids
            if cid in chunk_map
        ]

        if not candidates:
            return RetrievalResult()

        reranker = self._get_reranker()
        pairs = [(query, chunk.content) for _, chunk in candidates]
        t0 = time.perf_counter()
        scores = reranker.predict(pairs)
        dt = time.perf_counter() - t0
        if dt > 5.0:
            print(f"  [S4r] rerank {len(pairs)} pairs: {dt:.1f}s (slow!)", flush=True)

        max_score = float(max(scores)) if len(scores) > 0 else 0.0

        if max_score < self.RERANK_CONFIDENCE_FLOOR:
            # Reranker не уверен ни в одном кандидате — сохраняем hybrid order
            hybrid_top = [
                (cid, chunk_map[cid])
                for cid in hybrid_result.chunk_ids[: self._top_k_rerank]
                if cid in chunk_map
            ]
            return RetrievalResult(
                chunk_ids=[cid for cid, _ in hybrid_top],
                scores=[0.0] * len(hybrid_top),
                context_text=self.format_context([c for _, c in hybrid_top]),
                context_token_count=sum(c.token_count for _, c in hybrid_top),
                pre_rerank_chunk_ids=pre_rerank_ids,
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = ranked[: self._top_k_rerank]

        top = self._apply_diversity_guard(query, top, candidates, scores, pre_rerank_ids, chunk_map)

        selected_chunks = [chunk for (_, chunk), _ in top]
        selected_ids = [cid for (cid, _), _ in top]
        selected_scores = [float(s) for _, s in top]

        return RetrievalResult(
            chunk_ids=selected_ids,
            scores=selected_scores,
            context_text=self.format_context(selected_chunks),
            context_token_count=sum(c.token_count for c in selected_chunks),
            pre_rerank_chunk_ids=pre_rerank_ids,
        )

    @staticmethod
    def _detect_zones(chunk_ids: list[str]) -> set[str]:
        """Определяет какие смысловые зоны покрыты набором chunk_ids."""
        zones: set[str] = set()
        for cid in chunk_ids:
            for zone, prefixes in _ZONE_PREFIXES.items():
                if any(cid.startswith(p) for p in prefixes):
                    zones.add(zone)
        return zones

    def _apply_diversity_guard(
        self,
        query: str,
        top: list,
        candidates: list,
        scores,
        pre_rerank_ids: list[str],
        chunk_map: dict[str, KBChunk],
    ) -> list:
        """Diversity guard: для multi-fact запросов восстанавливает потерянные зоны.

        Если pre-rerank содержал chunks из 2+ зон, а post-rerank потерял одну,
        вставляет лучший chunk из потерянной зоны вместо наименее релевантного.

        Ключевое решение: chunk для восстановления выбирается по **hybrid RRF rank**
        (позиции в pre_rerank_ids), а не по reranker score, потому что reranker —
        именно тот компонент, который потерял нужный chunk.
        """
        if not MULTI_FACT_RE.search(query):
            return top

        pre_top_ids = pre_rerank_ids[: self._top_k_rerank * 2]
        pre_zones = self._detect_zones(pre_top_ids)
        if len(pre_zones) < 2:
            return top

        post_ids = [cid for (cid, _), _ in top]
        post_zones = self._detect_zones(post_ids)
        lost_zones = pre_zones - post_zones
        if not lost_zones:
            return top

        _ZONE_PRIORITY = {"doctor": 0, "price": 1, "aftercare": 2}
        sorted_lost = sorted(lost_zones, key=lambda z: _ZONE_PRIORITY.get(z, 99))

        top_mutable = list(top)
        pre_rank = {cid: idx for idx, cid in enumerate(pre_rerank_ids)}
        rerank_scores = {cid: float(s) for (cid, _), s in zip(candidates, scores)}

        replace_idx = len(top_mutable) - 1
        for zone in sorted_lost:
            if replace_idx < 0:
                break
            prefixes = _ZONE_PREFIXES[zone]
            zone_candidates = [
                cid for cid in pre_top_ids
                if any(cid.startswith(p) for p in prefixes) and cid in chunk_map
            ]
            if not zone_candidates:
                continue
            best_cid = min(zone_candidates, key=lambda c: pre_rank.get(c, 999))
            best_chunk = chunk_map[best_cid]
            best_score = rerank_scores.get(best_cid, 0.0)
            top_mutable[replace_idx] = ((best_cid, best_chunk), best_score)
            replace_idx -= 1

        return top_mutable
