"""B0: Keyword + Template — descriptive non-LLM baseline.

Не включается в inferential block. Descriptive accuracy floor:
показывает, чего можно достичь без LLM на keyword matching + templates.

Safe direct intents: phone, address, doctor, working hours, pricing.
Формулировки ответов намеренно отличаются от expected_answer eval set.
"""

from __future__ import annotations

import re
import time

from d4.models import DirectAnswerResult, FAQAnswer, KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy

# Общий паттерн токенизации (используется в _keyword_score и _build_price_answer)
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


class KeywordTemplateStrategy(BaseContextStrategy):
    """B0: Rule-based keyword matching → шаблонный ответ без LLM."""

    strategy_id = "B0"
    name = "Keyword + Template"
    uses_llm = False

    # --- Retrieval ---

    def select_context(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> RetrievalResult:
        """Keyword matching — выбирает chunk по ключевым словам."""
        query_lower = query.lower()
        matched_chunks: list[KBChunk] = []
        scores: list[float] = []

        for chunk in chunks:
            score = self._keyword_score(query_lower, chunk)
            if score > 0:
                matched_chunks.append(chunk)
                scores.append(score)

        pairs = sorted(zip(matched_chunks, scores), key=lambda x: x[1], reverse=True)
        if pairs:
            matched_chunks, scores = zip(*pairs)  # type: ignore
            matched_chunks = list(matched_chunks)
            scores = list(scores)

        context_text = self.format_context(matched_chunks) if matched_chunks else ""
        total_tokens = sum(c.token_count for c in matched_chunks)

        return RetrievalResult(
            chunk_ids=[c.id for c in matched_chunks],
            scores=scores,
            context_text=context_text,
            context_token_count=total_tokens,
        )

    # --- Answer ---

    def answer_directly(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> DirectAnswerResult:
        """Шаблонный ответ (без LLM). Thread-safe, без shared mutable state."""
        start = time.perf_counter()
        query_lower = query.lower()
        retrieval = self.select_context(query, chunks)

        if not retrieval.chunk_ids:
            return self._no_match_result(retrieval, start)

        chunk_map = {c.id: c for c in chunks}
        top_chunk = chunk_map.get(retrieval.chunk_ids[0])
        if top_chunk is None:
            return self._no_match_result(retrieval, start)

        answer, match_type, entity_match = self._template_answer(
            query_lower, top_chunk, chunk_map,
        )
        elapsed = (time.perf_counter() - start) * 1000

        return DirectAnswerResult(
            answer=answer,
            retrieval=retrieval,
            latency_ms=elapsed,
            route_taken="direct",
            confidence_debug={
                "confidence": answer.confidence,
                "match_type": match_type,
                "entity_match": entity_match,
                "top_chunk_id": top_chunk.id,
                "components": {},
            },
        )

    # --- Keyword scoring ---

    # Тематические ключевые слова → chunk_id → triggers
    _TOPIC_KEYWORDS: dict[str, list[str]] = {
        "clinic_location": ["адрес", "где", "находит", "расположен"],
        "clinic_contacts": ["телефон", "позвонить", "номер", "контакт", "связ"],
        "clinic_working_hours": ["работает", "время", "график", "часы", "выходн"],
        "clinic_patient_comfort_and_safety": ["седация", "наркоз", "сон", "закись"],
        "clinic_financing": ["рассрочка", "кредит", "оплата"],
    }

    _DOCTOR_TRIGGERS = ("врач", "доктор", "стаж", "опыт", "работает", "принимает",
                        "стоматолог", "хирург", "ортодонт", "терапевт", "ортопед")

    _PRICE_TRIGGERS = ("стоит", "цена", "стоимость", "прайс", "сколько", "почём")

    def _keyword_score(self, query_lower: str, chunk: KBChunk) -> float:
        """Keyword overlap score между запросом и chunk."""
        score = 0.0
        chunk_text = f"{chunk.title} {chunk.content}".lower()

        query_words = set(_TOKEN_RE.findall(query_lower))
        chunk_words = set(_TOKEN_RE.findall(chunk_text))
        score += len(query_words & chunk_words) * 0.5

        # Бонус за exact surname match (врачи)
        if chunk.entity_type == "doctor":
            surname = chunk.title.split()[0].lower() if chunk.title else ""
            if surname and surname in query_lower:
                score += 5.0

        # Бонус за тематические ключевые слова
        if chunk.id in self._TOPIC_KEYWORDS:
            for kw in self._TOPIC_KEYWORDS[chunk.id]:
                if kw in query_lower:
                    score += 3.0

        # Бонус для price chunks
        if chunk.source_type == "price_list":
            if any(t in query_lower for t in self._PRICE_TRIGGERS):
                score += 3.0
            services = chunk.raw_data.get("services", [])
            for svc in services:
                svc_words = set(_TOKEN_RE.findall(svc.get("Service", "").lower()))
                if query_words & svc_words:
                    score += 2.0
                    break

        return score

    # --- Template answers ---

    def _template_answer(
        self,
        query_lower: str,
        top_chunk: KBChunk,
        chunk_map: dict[str, KBChunk],
    ) -> tuple[FAQAnswer, str, str]:
        """Шаблонный ответ по top_chunk.

        Returns:
            (answer, match_type, entity_match)
        """
        # Price
        if top_chunk.source_type == "price_list":
            answer = self._build_price_answer(query_lower, top_chunk)
            return answer, "price_query", "none"

        # Doctor (exact surname only)
        if top_chunk.entity_type == "doctor":
            surname = top_chunk.title.split()[0].lower() if top_chunk.title else ""
            entity_match = "exact" if surname and surname in query_lower else "none"
            doctor_name = top_chunk.title.split(" — ")[0] if " — " in top_chunk.title else top_chunk.title
            spec = top_chunk.title.split(" — ")[1] if " — " in top_chunk.title else None
            faq = FAQAnswer(
                answer=f"В нашей клинике работает {top_chunk.title}. "
                       f"Вы можете записаться по телефону +7 (842) 231-45-55.",
                answerable=True,
                doctor=doctor_name,
                specialization=spec,
                suggest_booking=True,
                confidence=0.80 if entity_match == "exact" else 0.40,
                source_ids=[top_chunk.id],
            )
            return faq, "doctor_query", entity_match

        # FAQ (clinic info) — topic_keyword match
        match_type = "topic_keyword" if top_chunk.id in self._TOPIC_KEYWORDS else "none"
        answer_text = self._build_faq_answer(top_chunk)
        faq = FAQAnswer(
            answer=answer_text,
            answerable=True,
            confidence=0.70 if match_type == "topic_keyword" else 0.30,
            source_ids=[top_chunk.id],
        )
        return faq, match_type, "none"

    def _build_faq_answer(self, chunk: KBChunk) -> str:
        """Программная генерация ответа из raw_data chunk."""
        d = chunk.raw_data
        cid = chunk.id

        if cid == "clinic_location":
            city, address = d.get("city", ""), d.get("address", "")
            if city and address:
                return f"Клиника расположена: г. {city}, {address}."

        if cid == "clinic_contacts":
            phones = d.get("phones", [])
            if phones:
                return f"Связаться с нами: тел. {', '.join(phones)}."

        if cid == "clinic_working_hours":
            schedule = d.get("schedule_text", "")
            if schedule:
                return f"Режим работы клиники: {schedule}."

        if cid == "clinic_patient_comfort_and_safety":
            sedation = d.get("sedation", {})
            nitrous = sedation.get("nitrous_oxide", {})
            mixture, purpose = nitrous.get("mixture", ""), nitrous.get("purpose", "")
            if mixture and purpose:
                return f"В клинике доступна седация ({mixture}) — {purpose}."

        if cid == "clinic_financing":
            inst = d.get("installments", d)
            months = inst.get("months", [])
            if months:
                return f"Доступна рассрочка на {' и '.join(str(m) for m in months)} мес."

        if cid == "clinic_children_info":
            min_age = d.get("min_age", "")
            if min_age:
                return f"Принимаем детей {min_age}."

        # Fallback для прочих FAQ-секций
        return (
            f"По вашему вопросу: {chunk.title}. "
            f"Подробности по телефону: +7 (842) 231-45-55."
        )

    def _build_price_answer(self, query_lower: str, chunk: KBChunk) -> FAQAnswer:
        """Шаблонный ответ по ценам из raw_data."""
        services = chunk.raw_data.get("services", [])
        query_words = set(_TOKEN_RE.findall(query_lower))

        matched = [
            svc for svc in services
            if set(_TOKEN_RE.findall(svc.get("Service", "").lower())) & query_words
        ]
        if not matched:
            matched = services[:5]

        lines = [f"{s['Service']} — {s['Price']}" for s in matched[:5]]
        answer_text = "Стоимость услуг:\n" + "\n".join(lines)
        if len(services) > len(matched):
            answer_text += f"\nВсего в категории: {len(services)} услуг."

        return FAQAnswer(
            answer=answer_text,
            answerable=True,
            service=matched[0]["Service"] if len(matched) == 1 else None,
            confidence=0.60,
            source_ids=[chunk.id],
        )

    # --- Helpers ---

    def _no_match_result(self, retrieval: RetrievalResult, start: float) -> DirectAnswerResult:
        """Результат при отсутствии match."""
        return DirectAnswerResult(
            answer=FAQAnswer(
                answer="К сожалению, я не нашёл информацию по вашему вопросу. "
                       "Пожалуйста, позвоните нам: +7 (842) 231-45-55.",
                answerable=False,
                confidence=0.0,
            ),
            retrieval=retrieval,
            latency_ms=(time.perf_counter() - start) * 1000,
            route_taken="direct",
            confidence_debug={
                "confidence": 0.0,
                "match_type": "none",
                "entity_match": "none",
                "top_chunk_id": None,
                "components": {},
            },
        )
