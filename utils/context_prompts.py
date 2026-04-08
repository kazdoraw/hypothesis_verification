"""
Context-Aware Prompts for dental clinic AI router.

Provides specialized prompts for each L1/L2 classification combination,
enriched with extracted entities and dialog context.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# =============================================================================
# Context Data Classes
# =============================================================================

@dataclass
class DialogContext:
    """
    Context from current dialog for classification.
    
    Attributes:
        last_bot_question: Previous bot question
        expected_response_type: Type of expected response (yes_no, choice, free_text)
        current_flow: Current flow (anamnesis, booking, faq)
        entities_collected: Already collected entities
        turn_count: Number of dialog turns
    """
    last_bot_question: Optional[str] = None
    expected_response_type: Optional[str] = None  # yes_no, choice, free_text, date, time, confirmation
    current_flow: Optional[str] = None
    entities_collected: Dict[str, Any] = None
    turn_count: int = 0
    
    def __post_init__(self):
        if self.entities_collected is None:
            self.entities_collected = {}


@dataclass
class EnrichedContext:
    """
    Enriched context for LLM prompt.
    
    Contains classification result, entities, and dialog context
    formatted for prompt construction.
    """
    l1: str
    l2: str
    entities: Dict[str, Any]
    modifiers: List[str]
    tone: str
    dialog_context: Optional[DialogContext]
    original_text: str
    
    def to_prompt_context(self) -> str:
        """Format context for prompt insertion."""
        parts = []
        
        parts.append(f"Тип запроса: {self.l1}/{self.l2}")
        
        if self.entities:
            entity_strs = [f"{k}: {v}" for k, v in self.entities.items()]
            parts.append(f"Сущности: {', '.join(entity_strs)}")
        
        if self.modifiers:
            parts.append(f"Модификаторы: {', '.join(self.modifiers)}")
        
        if self.tone != "neutral":
            parts.append(f"Тон: {self.tone}")
        
        if self.dialog_context:
            if self.dialog_context.current_flow:
                parts.append(f"Текущий флоу: {self.dialog_context.current_flow}")
            if self.dialog_context.entities_collected:
                collected = [f"{k}: {v}" for k, v in self.dialog_context.entities_collected.items()]
                parts.append(f"Уже собрано: {', '.join(collected)}")
        
        return "\n".join(parts)


# =============================================================================
# Context-Aware Prompts by L1/L2
# =============================================================================

CONTEXT_PROMPTS = {
    # =========================================================================
    # ANAMNESIS PROMPTS
    # =========================================================================
    "anamnesis.symptom": """Пациент жалуется на симптом.

Сообщение пациента: "{text}"

{context}

Задача:
1. Уточнить локализацию боли/симптома
2. Выяснить длительность
3. Узнать интенсивность (1-10)
4. Спросить про триггеры (что усиливает/ослабляет)

Если информация уже предоставлена - не переспрашивать.
Ответь коротким уточняющим вопросом или переходи к следующему шагу.""",

    "anamnesis.symptom+doctor": """Пациент жалуется на симптом и упоминает врача.

Сообщение пациента: "{text}"

{context}

Врач: {doctor}

Задача:
1. Уточнить, связан ли симптом с прошлым лечением у этого врача
2. Если да - возможна эскалация на проверку качества
3. Если нет - продолжить стандартный сбор анамнеза

Ответь коротким уточняющим вопросом.""",

    "anamnesis.complaint": """Пациент описывает проблему (не симптом).

Сообщение пациента: "{text}"

{context}

Задача:
1. Подтвердить понимание проблемы
2. Уточнить срочность (когда это произошло)
3. Предложить запись на приём или дать рекомендации

Ответь кратко и по делу.""",

    "anamnesis.services": """Пациент интересуется услугой.

Сообщение пациента: "{text}"

{context}

Задача:
1. Подтвердить наличие услуги
2. Уточнить текущую ситуацию (есть ли проблема или профилактика)
3. Предложить консультацию или запись

Ответь информативно и предложи следующий шаг.""",

    # =========================================================================
    # BOOKING PROMPTS
    # =========================================================================
    "booking.new_appointment": """Пациент хочет записаться на приём.

Сообщение пациента: "{text}"

{context}

Задача:
1. Уточнить, к какому специалисту (если не указано)
2. Предложить доступные даты/время
3. Собрать контактные данные (если нет)

Если специалист указан - сразу предложить время.
Ответь предложением записи.""",

    "booking.new_appointment+doctor": """Пациент хочет записаться к конкретному врачу.

Сообщение пациента: "{text}"

{context}

Врач: {doctor}

Задача:
1. Проверить доступность врача
2. Предложить ближайшие слоты у этого врача
3. Если врач недоступен - предложить альтернативу

Ответь предложением записи к указанному врачу.""",

    "booking.new_appointment+date": """Пациент хочет записаться на конкретную дату.

Сообщение пациента: "{text}"

{context}

Дата: {date}

Задача:
1. Проверить доступность на указанную дату
2. Предложить конкретные слоты
3. Уточнить специалиста, если не указан

Ответь доступными временными слотами.""",

    "booking.reschedule": """Пациент хочет перенести запись.

Сообщение пациента: "{text}"

{context}

Задача:
1. Найти текущую запись пациента
2. Уточнить желаемую новую дату/время
3. Предложить доступные варианты переноса

Ответь подтверждением и предложением новых вариантов.""",

    "booking.cancel": """Пациент хочет отменить запись.

Сообщение пациента: "{text}"

{context}

Задача:
1. Найти текущую запись пациента
2. Подтвердить отмену
3. Предложить перезаписаться на другое время

Ответь подтверждением отмены и предложением.""",

    # =========================================================================
    # FAQ PROMPTS
    # =========================================================================
    "faq.price": """Пациент спрашивает о стоимости.

Сообщение пациента: "{text}"

{context}

Задача:
1. Дать информацию о стоимости запрошенной услуги
2. Упомянуть, что точная цена после осмотра
3. Предложить консультацию для уточнения

Ответь информацией о цене.""",

    "faq.price+procedure": """Пациент спрашивает о стоимости конкретной процедуры.

Сообщение пациента: "{text}"

{context}

Процедура: {procedure}

Задача:
1. Дать диапазон цен на указанную процедуру
2. Объяснить, от чего зависит цена
3. Предложить консультацию

Ответь конкретно по запрошенной процедуре.""",

    "faq.clinic_info": """Пациент спрашивает информацию о клинике.

Сообщение пациента: "{text}"

{context}

Задача:
1. Дать запрошенную информацию (адрес, часы работы и т.д.)
2. Если вопрос о работе в конкретный день - подтвердить/опровергнуть
3. Предложить записаться, если уместно

Ответь конкретной информацией о клинике.""",

    "faq.procedure": """Пациент спрашивает о процедуре.

Сообщение пациента: "{text}"

{context}

Задача:
1. Кратко объяснить процедуру
2. Ответить на конкретный вопрос (больно ли, сколько длится и т.д.)
3. Предложить консультацию для подробностей

Ответь информативно, но кратко.""",

    "faq.visit_prep": """Пациент спрашивает о подготовке или рекомендациях.

Сообщение пациента: "{text}"

{context}

Задача:
1. Дать конкретные рекомендации по подготовке/уходу
2. Упомянуть основные ограничения
3. Предложить связаться при возникновении вопросов

Ответь чёткими рекомендациями.""",

    "faq.followup": """Пациент задаёт уточняющий вопрос.

Сообщение пациента: "{text}"

{context}

Задача:
1. Ответить на уточняющий вопрос
2. Дать дополнительную информацию, если нужно
3. Предложить альтернативы, если спрашивает

Ответь на конкретный вопрос пациента.""",

    # =========================================================================
    # FEEDBACK PROMPTS (v4.0: was NEGATIVE FEEDBACK, now includes POSITIVE)
    # =========================================================================
    
    # NEGATIVE FEEDBACK (high priority - escalation required)
    "feedback.negative_service": """⚠️ ВНИМАНИЕ: Жалоба на сервис!

Сообщение пациента: "{text}"

{context}

Задача:
1. Извиниться за неудобства
2. Уточнить детали проблемы
3. Предложить решение
4. Передать информацию менеджеру

Ответь с эмпатией и предложи решение.""",

    "feedback.negative_quality": """⚠️ ВНИМАНИЕ: Жалоба на качество лечения!

Сообщение пациента: "{text}"

{context}

Врач: {doctor}
Процедура: {procedure}
Дата: {date}

Задача:
1. Выразить сожаление
2. Уточнить детали проблемы
3. Предложить повторный осмотр бесплатно
4. ОБЯЗАТЕЛЬНО эскалировать менеджеру

Ответь с заботой и предложи немедленное решение.""",

    "feedback.negative_staff": """⚠️ ВНИМАНИЕ: Жалоба на персонал!

Сообщение пациента: "{text}"

{context}

Задача:
1. Извиниться за поведение сотрудника
2. Уточнить детали инцидента
3. Заверить, что информация будет передана руководству
4. Предложить компенсацию, если уместно

Ответь с извинениями и конкретными действиями.""",

    "feedback.negative_general": """⚠️ ВНИМАНИЕ: Общая жалоба!

Сообщение пациента: "{text}"

{context}

Задача:
1. Поблагодарить за обратную связь
2. Уточнить, что именно не устроило
3. Предложить связаться с руководством
4. Предложить решение, если возможно

Ответь конструктивно и с готовностью помочь.""",

    # POSITIVE FEEDBACK (v4.0 NEW)
    "feedback.positive_service": """✅ Пациент оставляет положительный отзыв о сервисе.

Сообщение пациента: "{text}"

{context}

Задача:
1. Поблагодарить за отзыв
2. Выразить радость, что сервис понравился
3. Предложить оставить отзыв на сайте/картах
4. Пригласить обращаться снова

Ответь с теплотой и благодарностью.""",

    "feedback.positive_quality": """✅ Пациент доволен качеством лечения.

Сообщение пациента: "{text}"

{context}

Врач: {doctor}
Процедура: {procedure}

Задача:
1. Поблагодарить за обратную связь
2. Сказать, что передадим врачу
3. Предложить профилактический осмотр
4. Попросить оставить отзыв (опционально)

Ответь с радостью и предложи следующий визит.""",

    "feedback.positive_staff": """✅ Пациент хвалит персонал.

Сообщение пациента: "{text}"

{context}

Задача:
1. Поблагодарить за добрые слова
2. Заверить, что передадим сотруднику
3. Выразить радость, что понравилось
4. Пригласить обращаться ещё

Ответь с теплотой и искренностью.""",

    "feedback.positive_general": """✅ Пациент оставляет общий положительный отзыв.

Сообщение пациента: "{text}"

{context}

Задача:
1. Поблагодарить за обратную связь
2. Выразить радость
3. Предложить оставить отзыв на картах/сайте
4. Предложить рассказать знакомым :)

Ответь с благодарностью и теплотой.""",

    # =========================================================================
    # CONVERSATIONAL PROMPTS
    # =========================================================================
    "conversational.greeting": """Пациент приветствует.

Сообщение пациента: "{text}"

{context}

Задача:
1. Поприветствовать в ответ
2. Спросить, чем можем помочь
3. Быть дружелюбным, но профессиональным

Ответь коротким приветствием и вопросом.""",

    "conversational.gratitude": """Пациент благодарит.

Сообщение пациента: "{text}"

{context}

Задача:
1. Принять благодарность
2. Если диалог завершён - попрощаться
3. Если есть ещё вопросы - предложить помощь

Ответь кратко и дружелюбно.""",

    "conversational.confirmation": """Пациент подтверждает.

Сообщение пациента: "{text}"

{context}

Ожидаемый тип ответа: {expected_response_type}
Последний вопрос бота: {last_bot_question}

Задача:
1. Интерпретировать подтверждение в контексте диалога
2. Продолжить текущий флоу
3. Если это подтверждение записи - завершить бронирование

Продолжи текущий диалог.""",

    "conversational.farewell": """Пациент прощается.

Сообщение пациента: "{text}"

{context}

Задача:
1. Вежливо попрощаться
2. Напомнить о записи, если есть
3. Предложить связаться, если будут вопросы

Ответь коротким прощанием.""",

    "conversational.unclear": """Сообщение пациента неясно.

Сообщение пациента: "{text}"

{context}

Задача:
1. Вежливо попросить уточнить
2. Предложить варианты того, чем можем помочь
3. Не угадывать, а спросить напрямую

Попроси уточнить, чем можем помочь.""",
}


# =============================================================================
# Prompt Builder
# =============================================================================

class PromptBuilder:
    """
    Build context-aware prompts for LLM.
    
    Uses classification result and extracted entities to select
    and fill appropriate prompt template.
    
    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build_prompt(
        ...     text="Болит зуб после лечения у Иванова",
        ...     l1="anamnesis",
        ...     l2="symptom",
        ...     entities={"doctor": "Иванова"},
        ...     dialog_context=context
        ... )
    """
    
    def __init__(self, custom_prompts: Optional[Dict[str, str]] = None):
        """
        Initialize prompt builder.
        
        Args:
            custom_prompts: Additional or override prompts
        """
        self.prompts = dict(CONTEXT_PROMPTS)
        if custom_prompts:
            self.prompts.update(custom_prompts)
    
    def build_prompt(
        self,
        text: str,
        l1: str,
        l2: str,
        entities: Optional[Dict[str, Any]] = None,
        modifiers: Optional[List[str]] = None,
        tone: str = "neutral",
        dialog_context: Optional[DialogContext] = None,
    ) -> str:
        """
        Build context-aware prompt.
        
        Args:
            text: Original user message
            l1: L1 classification
            l2: L2 classification
            entities: Extracted entities
            modifiers: Detected modifiers
            tone: Message tone
            dialog_context: Dialog context
            
        Returns:
            Filled prompt template
        """
        entities = entities or {}
        modifiers = modifiers or []
        
        # Select prompt key
        prompt_key = self._select_prompt_key(l1, l2, entities)
        
        # Get template
        template = self.prompts.get(prompt_key, self.prompts.get(f"{l1}.{l2}", ""))
        
        if not template:
            # Fallback to generic prompt
            template = self._get_fallback_prompt(l1)
        
        # Build context string
        enriched = EnrichedContext(
            l1=l1,
            l2=l2,
            entities=entities,
            modifiers=modifiers,
            tone=tone,
            dialog_context=dialog_context,
            original_text=text,
        )
        context_str = enriched.to_prompt_context()
        
        # Fill template
        prompt = template.format(
            text=text,
            context=context_str,
            doctor=entities.get("doctor", "не указан"),
            date=entities.get("date", "не указана"),
            time=entities.get("time", "не указано"),
            procedure=entities.get("procedure", "не указана"),
            tooth=entities.get("tooth", "не указан"),
            amount=entities.get("amount", "не указана"),
            expected_response_type=dialog_context.expected_response_type if dialog_context else "",
            last_bot_question=dialog_context.last_bot_question if dialog_context else "",
        )
        
        return prompt
    
    def _select_prompt_key(self, l1: str, l2: str, entities: Dict[str, Any]) -> str:
        """Select the most specific prompt key based on entities."""
        base_key = f"{l1}.{l2}"
        
        # Check for entity-specific prompts
        if entities.get("doctor"):
            specific_key = f"{base_key}+doctor"
            if specific_key in self.prompts:
                return specific_key
        
        if entities.get("date"):
            specific_key = f"{base_key}+date"
            if specific_key in self.prompts:
                return specific_key
        
        if entities.get("procedure"):
            specific_key = f"{base_key}+procedure"
            if specific_key in self.prompts:
                return specific_key
        
        return base_key
    
    def _get_fallback_prompt(self, l1: str) -> str:
        """Get fallback prompt for L1 category."""
        fallbacks = {
            "anamnesis": """Пациент сообщает о своей проблеме.

Сообщение: "{text}"

{context}

Задача: Собрать необходимую информацию для анамнеза.
Ответь уточняющим вопросом.""",

            "booking": """Пациент хочет записаться.

Сообщение: "{text}"

{context}

Задача: Помочь с записью на приём.
Предложи доступные варианты.""",

            "faq": """Пациент задаёт вопрос.

Сообщение: "{text}"

{context}

Задача: Ответить на вопрос пациента.
Дай информативный ответ.""",

            "feedback": """📝 Пациент оставляет отзыв.

Сообщение: "{text}"

{context}

Задача: Обработать отзыв.
Если жалоба - извинись и предложи решение.
Если благодарность - поблагодари и пригласи снова.""",

            "conversational": """Сообщение пациента.

Сообщение: "{text}"

{context}

Задача: Ответить уместно.
Будь вежливым и профессиональным.""",
        }
        return fallbacks.get(l1, fallbacks["faq"])
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt keys."""
        return list(self.prompts.keys())


# =============================================================================
# Context Interpreter
# =============================================================================

class ContextInterpreter:
    """
    Interpret user message in dialog context.
    
    Handles cases where classification depends on context:
    - "Да" after "Хотите записаться?" → booking.confirmation
    - "Нет" after "Хотите записаться?" → booking.rejection
    """
    
    AFFIRMATIVE_WORDS = {"да", "ага", "угу", "конечно", "хорошо", "ок", "окей", "давайте", "буду"}
    NEGATIVE_WORDS = {"нет", "неа", "не", "не хочу", "не надо", "отказываюсь"}
    
    def is_affirmative(self, text: str) -> bool:
        """Check if text is an affirmative response."""
        text_lower = text.lower().strip()
        return text_lower in self.AFFIRMATIVE_WORDS or text_lower.startswith("да")
    
    def is_negative(self, text: str) -> bool:
        """Check if text is a negative response."""
        text_lower = text.lower().strip()
        return text_lower in self.NEGATIVE_WORDS or text_lower.startswith("нет")
    
    def interpret_in_context(
        self,
        text: str,
        classified_l1: str,
        classified_l2: str,
        context: DialogContext,
    ) -> tuple[str, str]:
        """
        Re-interpret classification based on dialog context.
        
        Args:
            text: User message
            classified_l1: ML-classified L1
            classified_l2: ML-classified L2
            context: Dialog context
            
        Returns:
            Tuple of (adjusted_l1, adjusted_l2)
        """
        # If no context or not a simple response, keep original
        if not context or not context.expected_response_type:
            return classified_l1, classified_l2
        
        # Short confirmations/negations in context
        if len(text.split()) <= 3:
            if context.expected_response_type == "yes_no":
                if self.is_affirmative(text):
                    # Keep current flow, mark as confirmation
                    return context.current_flow or classified_l1, "confirmation"
                elif self.is_negative(text):
                    # Might need different handling
                    return context.current_flow or classified_l1, "rejection"
            
            elif context.expected_response_type == "confirmation":
                if self.is_affirmative(text):
                    return context.current_flow or classified_l1, "confirmation"
        
        return classified_l1, classified_l2


# =============================================================================
# Singleton Instances
# =============================================================================

_prompt_builder: Optional[PromptBuilder] = None
_context_interpreter: Optional[ContextInterpreter] = None


def get_prompt_builder() -> PromptBuilder:
    """Get or create default prompt builder."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def get_context_interpreter() -> ContextInterpreter:
    """Get or create default context interpreter."""
    global _context_interpreter
    if _context_interpreter is None:
        _context_interpreter = ContextInterpreter()
    return _context_interpreter


if __name__ == "__main__":
    # Test prompt builder
    builder = PromptBuilder()
    
    # Test with entities
    prompt = builder.build_prompt(
        text="Болит зуб после лечения у Иванова вчера",
        l1="anamnesis",
        l2="symptom",
        entities={"doctor": "Иванова", "date": "вчера"},
    )
    print("Generated Prompt:")
    print(prompt)
    print()
    
    # Test context interpretation
    interpreter = ContextInterpreter()
    context = DialogContext(
        last_bot_question="Хотите записаться на приём?",
        expected_response_type="yes_no",
        current_flow="booking",
    )
    
    l1, l2 = interpreter.interpret_in_context(
        text="Да",
        classified_l1="conversational",
        classified_l2="confirmation",
        context=context,
    )
    print(f"Interpreted: {l1}/{l2}")
