"""B0: Keyword + Template — production baseline без LLM.

Только правила и шаблоны. Не включается в inferential block.
Используется как практический ориентир.

Ответы генерируются программно из данных chunk.content,
формулировки намеренно отличаются от expected_answer eval set
для устранения тавтологии при оценке.
"""

from __future__ import annotations

import re

from d4.models import FAQAnswer, KBChunk, RetrievalResult
from d4.strategies.base import BaseContextStrategy


class KeywordTemplateStrategy(BaseContextStrategy):
    """B0: Rule-based keyword matching → шаблонный ответ без LLM."""

    strategy_id = "B0"
    name = "Keyword + Template"
    uses_llm = False

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

        # Сортировка по score (убывание)
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

    def answer_directly(
        self,
        query: str,
        chunks: list[KBChunk],
    ) -> FAQAnswer:
        """Формирование ответа по шаблону (без LLM).

        Основная логика B0: keyword match → template response.
        """
        query_lower = query.lower()
        retrieval = self.select_context(query, chunks)

        if not retrieval.chunk_ids:
            return FAQAnswer(
                answer="К сожалению, я не нашёл информацию по вашему вопросу. "
                       "Пожалуйста, позвоните нам: +7 (842) 231-45-55.",
                answerable=False,
                confidence=0.0,
            )

        # Собираем matched chunks
        chunk_map = {c.id: c for c in chunks}
        top_chunk = chunk_map.get(retrieval.chunk_ids[0])

        if top_chunk is None:
            return FAQAnswer(answer="", answerable=False, confidence=0.0)

        # Шаблонные ответы по типу chunk
        answer = self._template_answer(query_lower, top_chunk, chunk_map, retrieval)

        return answer

    def _keyword_score(self, query_lower: str, chunk: KBChunk) -> float:
        """Подсчёт keyword совпадений между запросом и chunk."""
        score = 0.0
        chunk_text = f"{chunk.title} {chunk.content}".lower()

        # Точные совпадения ключевых слов
        query_words = set(re.findall(r"[а-яёa-z0-9]+", query_lower))
        chunk_words = set(re.findall(r"[а-яёa-z0-9]+", chunk_text))
        overlap = query_words & chunk_words
        score += len(overlap) * 0.5

        # Бонус за совпадение фамилии врача
        if chunk.entity_type == "doctor":
            surname = chunk.title.split()[0].lower() if chunk.title else ""
            if surname and surname in query_lower:
                score += 5.0

        # Бонус за тематические ключевые слова
        topic_keywords = {
            "clinic_location": ["адрес", "где", "находит", "расположен", "дорог"],
            "clinic_contacts": ["телефон", "позвонить", "номер", "контакт", "связ"],
            "clinic_working_hours": ["работает", "время", "график", "часы", "выходн", "ежедневно"],
            "clinic_patient_comfort_and_safety": ["седация", "наркоз", "сон", "боюсь", "страх", "закись"],
            "clinic_financing": ["рассрочка", "кредит", "оплата", "стоимость"],
            "clinic_documents": ["документ", "договор", "согласие", "анкета"],
            "clinic_licenses": ["лицензия", "сертификат"],
        }

        chunk_id = chunk.id
        if chunk_id in topic_keywords:
            for kw in topic_keywords[chunk_id]:
                if kw in query_lower:
                    score += 3.0

        # Бонус для price chunks при ценовых запросах
        if chunk.source_type == "price_list":
            price_triggers = ["стоит", "цена", "стоимость", "прайс", "сколько", "почём"]
            for trigger in price_triggers:
                if trigger in query_lower:
                    score += 3.0
                    break
            # Бонус за совпадение с названиями услуг в raw_data
            services = chunk.raw_data.get("services", [])
            for svc in services:
                svc_name = svc.get("Service", "").lower()
                svc_words = set(re.findall(r"[а-яёa-z0-9]+", svc_name))
                if query_words & svc_words:
                    score += 2.0
                    break

        return score

    def _template_answer(
        self,
        query_lower: str,
        top_chunk: KBChunk,
        chunk_map: dict[str, KBChunk],
        retrieval: RetrievalResult,
    ) -> FAQAnswer:
        """Шаблонный ответ на основе найденного chunk.

        Факты извлекаются программно из chunk.content.
        Формулировки намеренно отличаются от expected_answer eval set.
        """
        # Прайс-лист — ответ из raw_data
        if top_chunk.source_type == "price_list":
            return self._build_price_answer(query_lower, top_chunk)

        # Врач — ответ из заголовка chunk
        if top_chunk.entity_type == "doctor":
            return FAQAnswer(
                answer=f"В нашей клинике работает {top_chunk.title}. "
                       f"Вы можете записаться по телефону +7 (842) 231-45-55.",
                answerable=True,
                doctor=top_chunk.title.split(" — ")[0] if " — " in top_chunk.title else None,
                specialization=top_chunk.title.split(" — ")[1] if " — " in top_chunk.title else None,
                suggest_booking=True,
                confidence=0.7,
                source_ids=[top_chunk.id],
            )

        # FAQ секции — ответ из данных chunk
        answer_text = self._build_faq_answer(top_chunk)

        return FAQAnswer(
            answer=answer_text,
            answerable=True,
            confidence=0.6,
            source_ids=[top_chunk.id],
        )

    def _build_faq_answer(self, chunk: KBChunk) -> str:
        """Программная генерация ответа из raw_data chunk.

        Извлекает факты из chunk.raw_data (dict) и формирует ответ
        с формулировкой, отличной от expected_answer eval set.
        """
        d = chunk.raw_data
        cid = chunk.id

        if cid == "clinic_meta":
            name = d.get("name", "")
            return f"{name} — добро пожаловать!" if name else ""

        if cid == "clinic_location":
            city = d.get("city", "")
            address = d.get("address", "")
            if city and address:
                return f"Клиника расположена: г. {city}, {address}."

        if cid == "clinic_contacts":
            phones = d.get("phones", [])
            emails = d.get("emails", [])
            parts: list[str] = []
            if phones:
                parts.append(f"тел. {', '.join(phones)}")
            if emails:
                parts.append(f"email: {', '.join(emails)}")
            if parts:
                return f"Связаться с нами: {'; '.join(parts)}."

        if cid == "clinic_working_hours":
            schedule = d.get("schedule_text", "")
            if schedule:
                return f"Режим работы клиники: {schedule}."

        if cid == "clinic_patient_comfort_and_safety":
            sedation = d.get("sedation", {})
            nitrous = sedation.get("nitrous_oxide", {})
            mixture = nitrous.get("mixture", "")
            purpose = nitrous.get("purpose", "")
            if mixture and purpose:
                return f"В клинике доступна седация ({mixture}) — {purpose}."

        if cid == "clinic_financing":
            inst = d.get("installments", d)
            ftype = inst.get("type", "рассрочка")
            months = inst.get("months", [])
            if months:
                return f"Доступна {ftype} на {' и '.join(str(m) for m in months)} мес."

        if cid == "clinic_children_info":
            min_age = d.get("min_age", "")
            note = d.get("note", "")
            if min_age:
                return f"Принимаем детей {min_age}. {note}." if note else f"Принимаем детей {min_age}."

        if cid == "clinic_documents":
            url = d.get("page_url", "")
            if url:
                return f"Документы для пациентов: {url}"

        if cid == "clinic_licenses":
            url = d.get("page_url", "")
            if url:
                return f"Лицензии клиники: {url}"

        # Fallback для прочих FAQ-секций
        return (
            f"По вашему вопросу: {chunk.title}. "
            f"Подробности по телефону: +7 (842) 231-45-55."
        )

    def _build_price_answer(self, query_lower: str, chunk: KBChunk) -> FAQAnswer:
        """Шаблонный ответ по ценам из raw_data."""
        services = chunk.raw_data.get("services", [])
        query_words = set(re.findall(r"[а-яёa-z0-9]+", query_lower))

        # Фильтрация услуг по keyword overlap
        matched = []
        for svc in services:
            svc_words = set(re.findall(r"[а-яёa-z0-9]+", svc.get("Service", "").lower()))
            if query_words & svc_words:
                matched.append(svc)

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
            confidence=0.6,
            source_ids=[chunk.id],
        )
