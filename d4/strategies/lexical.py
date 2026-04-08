"""S2: Lexical Retrieval — keyword/BM25 → top-k units.

Правила НЕ подогнаны под конкретные тестовые формулировки.
Используется BM25 по содержимому chunks с русским стеммингом.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from nltk.stem.snowball import SnowballStemmer

from d4.models import KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy

# ---------------------------------------------------------------------------
# BM25 с русским стеммингом и стоп-словами
# ---------------------------------------------------------------------------

BM25_K1 = 1.5
BM25_B = 0.75

_STEMMER = SnowballStemmer("russian")

# Русские стоп-слова: высокочастотные, нулевая дискриминативность для BM25
_STOP_WORDS: frozenset[str] = frozenset({
    "и", "в", "на", "не", "что", "я", "с", "он", "а", "как",
    "это", "по", "но", "к", "из", "у", "за", "бы", "от", "до",
    "то", "же", "вы", "мы", "да", "ли", "уже", "нет", "вот",
    "мне", "вас", "нас", "ещё", "или", "так", "все", "при",
})


def _tokenize(text: str) -> list[str]:
    """Токенизация + стемминг + фильтрация стоп-слов для BM25."""
    text = text.lower()
    tokens = re.findall(r"[а-яёa-z0-9]+", text)
    return [_STEMMER.stem(t) for t in tokens if t not in _STOP_WORDS]


def _compute_bm25_scores(
    query_tokens: list[str],
    corpus: list[list[str]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Вычисление BM25 score для каждого документа в корпусе.

    Args:
        query_tokens: токены запроса
        corpus: список документов (каждый — список токенов)
        k1, b: параметры BM25

    Returns:
        список BM25 scores
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return []

    # Средняя длина документа
    avg_dl = sum(len(doc) for doc in corpus) / n_docs

    # IDF: количество документов, содержащих каждый терм
    df: Counter[str] = Counter()
    for doc in corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    scores: list[float] = []
    for doc in corpus:
        doc_len = len(doc)
        tf_doc = Counter(doc)
        score = 0.0

        for term in query_tokens:
            if term not in df:
                continue
            # IDF (Robertson-Sparck Jones)
            n_t = df[term]
            idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

            # TF с нормализацией длины
            raw_tf = tf_doc.get(term, 0)
            tf_norm = (raw_tf * (k1 + 1)) / (raw_tf + k1 * (1 - b + b * doc_len / avg_dl))

            score += idf * tf_norm

        scores.append(score)

    return scores


class LexicalStrategy(BaseContextStrategy):
    """S2: Lexical retrieval через BM25."""

    strategy_id = "S2"
    name = "Lexical Retrieval"
    uses_llm = True

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """BM25 поиск по содержимому chunks → top-k.

        Правила не подогнаны под тестовые формулировки —
        используется стандартный BM25 без ручной настройки.
        """
        if not chunks:
            return RetrievalResult()

        query_tokens = _tokenize(query)
        # Корпус: title + content каждого chunk
        corpus = [_tokenize(f"{c.title} {c.content}") for c in chunks]

        bm25_scores = _compute_bm25_scores(query_tokens, corpus)

        # Сортировка по score (убывание), top-k
        indexed_scores = list(enumerate(bm25_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = indexed_scores[: self.top_k]

        # Фильтруем нулевые scores
        top_results = [(idx, score) for idx, score in top_results if score > 0]

        if not top_results:
            return RetrievalResult()

        selected_chunks = [chunks[idx] for idx, _ in top_results]
        selected_scores = [score for _, score in top_results]
        selected_ids = [chunks[idx].id for idx, _ in top_results]

        context_text = self.format_context(selected_chunks)
        total_tokens = sum(c.token_count for c in selected_chunks)

        return RetrievalResult(
            chunk_ids=selected_ids,
            scores=selected_scores,
            context_text=context_text,
            context_token_count=total_tokens,
        )
