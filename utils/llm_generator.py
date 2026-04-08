"""
LLM-based Data Generator for dental clinic message classification.

Uses Together AI (or other LLMs) to generate:
- Paraphrases of existing samples
- Edge cases and complex examples
- Rare patterns and unusual formulations
"""

import json
import random
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import (
    INTENT_LABELS_L1,
    INTENT_LABELS_L2,
    L2_TO_L1,
    get_l2_classes_for_l1,
)


# =============================================================================
# Generation Prompts
# =============================================================================

GENERATION_PROMPTS = {
    "paraphrase": """Перефразируй следующее сообщение пациента стоматологической клиники,
сохранив смысл, но изменив формулировку. Верни ровно {n} вариантов, каждый на новой строке.

Оригинал: "{text}"
Класс: {l1}/{l2}

Требования:
- Сохранить смысл и intent
- Использовать разговорный русский язык
- Добавить естественные вариации
- Каждый вариант должен быть уникальным
- НЕ нумеровать варианты

Варианты (по одному на строку):""",

    "edge_case": """Создай сложный пример сообщения пациента для класса {l1}/{l2}.

Описание класса: {description}

Сообщение должно содержать:
- Основной intent: {l2}
- Дополнительный контекст или второй intent
- Эмоциональную окраску или модификаторы
- Упоминание врача, даты или процедуры (опционально)

Примеры сложных случаев для справки:
- "Я был у Иванова вчера и теперь болит ещё сильнее"
- "Спасибо, но цена кажется завышенной..."
- "Не хочу к этому врачу, запишите к другому"

Верни ровно {n} сложных примеров, каждый на новой строке.
НЕ нумеровать варианты.

Примеры:""",

    "rare_pattern": """Создай редкий/нетипичный пример сообщения для класса {l1}/{l2}.

Описание класса: {description}

Это должно быть:
- Грамматически корректное
- Соответствующее классу
- Но с необычной формулировкой

Например для booking.new_appointment:
- "Хотелось бы попасть к вам на осмотр, если это возможно"
- "Есть ли шанс записаться на ближайшее время?"

Верни ровно {n} редких примеров, каждый на новой строке.
НЕ нумеровать варианты.

Примеры:""",

    "with_entities": """Создай сообщение пациента класса {l1}/{l2}, которое содержит указанные сущности.

Описание класса: {description}
Обязательные сущности: {entities}

Требования:
- Сообщение должно соответствовать классу {l2}
- Должно содержать все указанные сущности
- Должно быть естественным и разговорным

Верни ровно {n} примеров, каждый на новой строке.

Примеры:""",

    # v4.0: renamed to feedback, now supports positive/negative
    "feedback": """Создай примеры отзывов пациентов стоматологической клиники.

Тип отзыва: {l2}
Описание: {description}
Тональность: {sentiment}

Требования:
- Отзыв должен быть реалистичным
- Использовать соответствующий эмоциональный язык
- Может содержать конкретные детали (имена врачей, даты, процедуры)

Примеры типов отзывов:
НЕГАТИВНЫЕ:
- negative_service: долгое ожидание, проблемы со связью
- negative_quality: проблемы после лечения
- negative_staff: грубость персонала
- negative_general: общее недовольство

ПОЗИТИВНЫЕ (v4.0 NEW):
- positive_service: быстрая запись, удобный сервис
- positive_quality: отличный результат лечения
- positive_staff: внимательный персонал, профессионализм
- positive_general: общая благодарность, рекомендации

Верни ровно {n} примеров отзывов типа {l2}, каждый на новой строке.

Примеры:""",
}

# Class descriptions for prompts (v4.0: updated for feedback taxonomy)
CLASS_DESCRIPTIONS = {
    # Anamnesis
    "symptom": "Симптомы, боль, дискомфорт (болит зуб, кровоточат дёсны)",
    "complaint": "Описание проблемы (выпала пломба, откололся зуб)",
    "services": "Интерес к услугам (хочу отбеливание, нужна имплантация)",
    # Booking
    "new_appointment": "Запрос на новую запись (хочу записаться, есть время?)",
    "reschedule": "Перенос записи (перенесите на завтра)",
    "cancel": "Отмена записи (отмените запись)",
    # FAQ
    "price": "Вопрос о стоимости (сколько стоит?)",
    "clinic_info": "Вопрос о клинике (где находитесь?, работаете в субботу?)",
    "procedure": "Вопрос о процедуре (это больно?, как проходит?)",
    "visit_prep": "Вопрос о подготовке/рекомендациях (что нельзя после?)",
    "followup": "Уточняющий вопрос (а если не поможет?)",
    # Feedback - Negative (v4.0: renamed from service_issue/quality/staff/general)
    "negative_service": "Жалоба на сервис (долго ждал, не перезвонили)",
    "negative_quality": "Жалоба на качество лечения (после лечения стало хуже)",
    "negative_staff": "Жалоба на персонал (врач был груб)",
    "negative_general": "Общая претензия (недоволен обслуживанием)",
    # Feedback - Positive (v4.0 NEW)
    "positive_service": "Благодарность за сервис (быстро записали, сразу перезвонили)",
    "positive_quality": "Благодарность за качество (отличный результат, зуб не болит)",
    "positive_staff": "Благодарность персоналу (внимательный врач, профессионал)",
    "positive_general": "Общая благодарность (рекомендую, лучшая клиника)",
    # Conversational
    "greeting": "Приветствие (здравствуйте, добрый день)",
    "gratitude": "Благодарность (спасибо, благодарю)",
    "confirmation": "Подтверждение (хорошо, понял, ок)",
    "farewell": "Прощание (до свидания, пока)",
    "unclear": "Неопределённый ответ (не знаю, подумаю)",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeneratedSample:
    """A sample generated by LLM."""
    text: str
    label_l1: str
    label_l2: str
    generation_type: str  # paraphrase, edge_case, rare_pattern, etc.
    base_text: Optional[str] = None  # Original text if paraphrase
    confidence: float = 0.8
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label_l1": self.label_l1,
            "label_l2": self.label_l2,
            "source": f"llm_{self.generation_type}",
            "base_text": self.base_text,
            "confidence": self.confidence,
        }


@dataclass
class GenerationStats:
    """Statistics for generation."""
    total_requests: int = 0
    total_generated: int = 0
    total_tokens: int = 0
    failed_requests: int = 0
    
    def log_request(self, n_generated: int, tokens: int):
        self.total_requests += 1
        self.total_generated += n_generated
        self.total_tokens += tokens
    
    def log_failure(self):
        self.total_requests += 1
        self.failed_requests += 1


# =============================================================================
# Quality Checker
# =============================================================================

class QualityChecker:
    """
    Check quality of LLM-generated samples.
    
    Validates:
    - Minimum length
    - No duplicates
    - Contains class markers
    - Grammar check (basic)
    """
    
    def __init__(self, existing_texts: Optional[set] = None):
        """
        Initialize quality checker.
        
        Args:
            existing_texts: Set of existing texts to avoid duplicates
        """
        self.existing_texts = existing_texts or set()
        
        # Keywords for each L2 class (v4.0: updated for feedback taxonomy)
        self.class_markers = {
            "symptom": ["болит", "боль", "ноет", "кровоточит", "чувствую"],
            "complaint": ["выпала", "откололся", "сломался", "шатается"],
            "services": ["хочу", "интересует", "нужно", "планирую"],
            "new_appointment": ["записаться", "запишите", "приём", "время"],
            "reschedule": ["перенести", "перезаписать", "другой день"],
            "cancel": ["отменить", "отмена", "не приду"],
            "price": ["стоит", "цена", "стоимость", "почём", "прайс"],
            "clinic_info": ["где", "адрес", "работаете", "парковка"],
            "procedure": ["больно", "проходит", "риски", "длится"],
            "visit_prep": ["подготовиться", "нельзя", "после", "ограничения"],
            "followup": ["а если", "а что", "альтернатива", "подробнее"],
            # Feedback - Negative (v4.0: renamed)
            "negative_service": ["ждал", "не перезвонили", "дозвониться", "потеряли"],
            "negative_quality": ["хуже", "выпала", "не помогло", "снова болит"],
            "negative_staff": ["груб", "невежливо", "нахамил", "опоздал"],
            "negative_general": ["недоволен", "жалоба", "претензия", "возмутительно"],
            # Feedback - Positive (v4.0 NEW)
            "positive_service": ["быстро записали", "сразу перезвонили", "удобно"],
            "positive_quality": ["отличный результат", "качественно", "превзошёл"],
            "positive_staff": ["внимательный", "профессионал", "вежливый", "заботливый"],
            "positive_general": ["рекомендую", "доволен", "лучшая", "буду обращаться"],
            # Conversational
            "greeting": ["здравствуй", "добрый", "привет"],
            "gratitude": ["спасибо", "благодарю"],
            "confirmation": ["хорошо", "понял", "ок", "да"],
            "farewell": ["свидания", "пока", "всего доброго"],
            "unclear": ["не знаю", "подумаю", "не решил"],
        }
    
    def is_valid(
        self,
        text: str,
        expected_l1: str,
        expected_l2: str,
        check_markers: bool = True,
    ) -> bool:
        """
        Validate generated text.
        
        Args:
            text: Generated text
            expected_l1: Expected L1 class
            expected_l2: Expected L2 class
            check_markers: Whether to check for class markers
            
        Returns:
            True if valid
        """
        # Minimum length
        if len(text.strip()) < 5:
            return False
        
        # Maximum length
        if len(text) > 500:
            return False
        
        # No duplicates
        text_normalized = text.lower().strip()
        if text_normalized in self.existing_texts:
            return False
        
        # Basic grammar check (no obvious errors)
        if self._has_obvious_errors(text):
            return False
        
        # Check for class markers (optional, relaxed)
        if check_markers and not self._has_class_markers(text, expected_l2):
            # Allow some samples without strict markers
            return random.random() < 0.3
        
        return True
    
    def _has_class_markers(self, text: str, l2: str) -> bool:
        """Check if text contains markers for the class."""
        markers = self.class_markers.get(l2, [])
        if not markers:
            return True
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in markers)
    
    def _has_obvious_errors(self, text: str) -> bool:
        """Check for obvious grammar errors."""
        # Double spaces
        if "  " in text and text.count("  ") > 2:
            return True
        
        # Repeated words
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                return True
        
        # Too many punctuation marks
        if text.count("!") > 3 or text.count("?") > 3:
            return True
        
        return False
    
    def add_existing(self, text: str):
        """Add text to existing set."""
        self.existing_texts.add(text.lower().strip())


# =============================================================================
# LLM Data Generator
# =============================================================================

class LLMDataGenerator:
    """
    Generate synthetic data using LLM.
    
    Uses Together AI to generate:
    - Paraphrases
    - Edge cases
    - Rare patterns
    - Samples with specific entities
    
    Example:
        >>> generator = LLMDataGenerator(llm_client)
        >>> samples = generator.generate_paraphrases(
        ...     "Болит зуб", "anamnesis", "symptom", n=5
        ... )
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        existing_texts: Optional[set] = None,
        use_simulator: bool = False,
    ):
        """
        Initialize LLM data generator.
        
        Args:
            llm_client: LLM client (TogetherLLM or similar)
            existing_texts: Set of existing texts to avoid duplicates
            use_simulator: Use simulation instead of real LLM
        """
        self.llm = llm_client
        self.quality_checker = QualityChecker(existing_texts)
        self.stats = GenerationStats()
        self.use_simulator = use_simulator or (llm_client is None)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM and return response."""
        if self.use_simulator:
            return self._simulate_response(prompt)
        
        if self.llm is None:
            return None
        
        try:
            # Assume llm has a method like classify_intent but for generation
            # Using raw API call
            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=512,
                )
                self.stats.log_request(0, response.tokens_used)
                return response.content
            else:
                return None
        except Exception as e:
            self.stats.log_failure()
            return None
    
    def _simulate_response(self, prompt: str) -> str:
        """Simulate LLM response for testing."""
        # Extract class info from prompt
        l2_match = re.search(r'/(\w+)', prompt)
        l2 = l2_match.group(1) if l2_match else "other"
        
        # Simulated variations based on class
        simulated_templates = {
            "symptom": [
                "Болит зуб уже несколько дней",
                "Чувствую боль при жевании",
                "Десна воспалилась и болит",
            ],
            "complaint": [
                "Пломба выпала после еды",
                "Кусочек зуба откололся",
                "Коронка стала шататься",
            ],
            "new_appointment": [
                "Хотел бы записаться к врачу",
                "Когда можно попасть на приём?",
                "Подскажите свободное время",
            ],
            "price": [
                "Какова стоимость лечения?",
                "Во сколько обойдётся процедура?",
                "Есть ли у вас прайс-лист?",
            ],
            "service_issue": [
                "Ждал приёма очень долго",
                "Мне так и не перезвонили",
                "Запись куда-то пропала",
            ],
            "quality": [
                "После лечения стало только хуже",
                "Пломба выпала через неделю",
                "Боль так и не прошла",
            ],
        }
        
        templates = simulated_templates.get(l2, ["Тестовое сообщение"])
        n_match = re.search(r'ровно (\d+)', prompt)
        n = int(n_match.group(1)) if n_match else 3
        
        results = []
        for _ in range(n):
            template = random.choice(templates)
            # Add some variation
            if random.random() < 0.3:
                template = template.lower()
            if random.random() < 0.2:
                template += "!"
            results.append(template)
        
        return "\n".join(results)
    
    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into list of texts."""
        if not response:
            return []
        
        lines = response.strip().split("\n")
        results = []
        
        for line in lines:
            # Clean up the line
            line = line.strip()
            
            # Remove numbering if present
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            
            # Skip empty or too short lines
            if len(line) < 5:
                continue
            
            # Skip lines that look like instructions
            if any(word in line.lower() for word in ["верни", "пример", "вариант"]):
                continue
            
            results.append(line)
        
        return results
    
    def generate_paraphrases(
        self,
        text: str,
        l1: str,
        l2: str,
        n: int = 5,
    ) -> List[GeneratedSample]:
        """
        Generate paraphrases of a given text.
        
        Args:
            text: Original text
            l1: L1 label
            l2: L2 label
            n: Number of paraphrases to generate
            
        Returns:
            List of GeneratedSample objects
        """
        prompt = GENERATION_PROMPTS["paraphrase"].format(
            text=text,
            l1=l1,
            l2=l2,
            n=n,
        )
        
        response = self._call_llm(prompt)
        texts = self._parse_response(response)
        
        samples = []
        for generated_text in texts:
            if self.quality_checker.is_valid(generated_text, l1, l2):
                samples.append(GeneratedSample(
                    text=generated_text,
                    label_l1=l1,
                    label_l2=l2,
                    generation_type="paraphrase",
                    base_text=text,
                ))
                self.quality_checker.add_existing(generated_text)
        
        self.stats.total_generated += len(samples)
        return samples
    
    def generate_edge_cases(
        self,
        l1: str,
        l2: str,
        n: int = 5,
    ) -> List[GeneratedSample]:
        """
        Generate edge case examples for a class.
        
        Args:
            l1: L1 label
            l2: L2 label
            n: Number of examples to generate
            
        Returns:
            List of GeneratedSample objects
        """
        description = CLASS_DESCRIPTIONS.get(l2, "")
        
        prompt = GENERATION_PROMPTS["edge_case"].format(
            l1=l1,
            l2=l2,
            description=description,
            n=n,
        )
        
        response = self._call_llm(prompt)
        texts = self._parse_response(response)
        
        samples = []
        for generated_text in texts:
            if self.quality_checker.is_valid(generated_text, l1, l2, check_markers=False):
                samples.append(GeneratedSample(
                    text=generated_text,
                    label_l1=l1,
                    label_l2=l2,
                    generation_type="edge_case",
                    confidence=0.7,  # Lower confidence for edge cases
                ))
                self.quality_checker.add_existing(generated_text)
        
        self.stats.total_generated += len(samples)
        return samples
    
    def generate_rare_patterns(
        self,
        l1: str,
        l2: str,
        n: int = 5,
    ) -> List[GeneratedSample]:
        """
        Generate rare/unusual pattern examples.
        
        Args:
            l1: L1 label
            l2: L2 label
            n: Number of examples to generate
            
        Returns:
            List of GeneratedSample objects
        """
        description = CLASS_DESCRIPTIONS.get(l2, "")
        
        prompt = GENERATION_PROMPTS["rare_pattern"].format(
            l1=l1,
            l2=l2,
            description=description,
            n=n,
        )
        
        response = self._call_llm(prompt)
        texts = self._parse_response(response)
        
        samples = []
        for generated_text in texts:
            if self.quality_checker.is_valid(generated_text, l1, l2, check_markers=False):
                samples.append(GeneratedSample(
                    text=generated_text,
                    label_l1=l1,
                    label_l2=l2,
                    generation_type="rare_pattern",
                    confidence=0.75,
                ))
                self.quality_checker.add_existing(generated_text)
        
        self.stats.total_generated += len(samples)
        return samples
    
    def generate_feedback(
        self,
        l2: str,
        n: int = 10,
    ) -> List[GeneratedSample]:
        """
        Generate feedback examples (v4.0: supports both positive and negative).
        
        Args:
            l2: L2 label (negative_service/quality/staff/general or positive_service/quality/staff/general)
            n: Number of examples to generate
            
        Returns:
            List of GeneratedSample objects
        """
        valid_feedback_l2 = [
            "negative_service", "negative_quality", "negative_staff", "negative_general",
            "positive_service", "positive_quality", "positive_staff", "positive_general",
        ]
        if l2 not in valid_feedback_l2:
            return []
        
        description = CLASS_DESCRIPTIONS.get(l2, "")
        sentiment = "негативная (жалоба)" if l2.startswith("negative_") else "позитивная (благодарность)"
        
        prompt = GENERATION_PROMPTS["feedback"].format(
            l2=l2,
            description=description,
            sentiment=sentiment,
            n=n,
        )
        
        response = self._call_llm(prompt)
        texts = self._parse_response(response)
        
        samples = []
        for generated_text in texts:
            if self.quality_checker.is_valid(generated_text, "feedback", l2):
                samples.append(GeneratedSample(
                    text=generated_text,
                    label_l1="feedback",  # v4.0: was negative_feedback
                    label_l2=l2,
                    generation_type="feedback",
                    confidence=0.85,
                ))
                self.quality_checker.add_existing(generated_text)
        
        self.stats.total_generated += len(samples)
        return samples
    
    # Backward compatibility alias
    def generate_negative_feedback(self, l2: str, n: int = 10) -> List[GeneratedSample]:
        """Deprecated: use generate_feedback() instead."""
        # Map old labels to new
        label_map = {
            "service_issue": "negative_service",
            "quality": "negative_quality",
            "staff": "negative_staff",
            "general": "negative_general",
        }
        new_l2 = label_map.get(l2, l2)
        return self.generate_feedback(new_l2, n)
    
    def generate_batch_for_all_classes(
        self,
        n_per_class: int = 5,
        generation_types: Optional[List[str]] = None,
    ) -> List[GeneratedSample]:
        """
        Generate samples for all L2 classes.
        
        Args:
            n_per_class: Number of samples per class
            generation_types: Types of generation to use
            
        Returns:
            List of all generated samples
        """
        generation_types = generation_types or ["edge_case", "rare_pattern"]
        all_samples = []
        
        for l2 in INTENT_LABELS_L2:
            l1 = L2_TO_L1.get(l2, "faq")
            
            for gen_type in generation_types:
                if gen_type == "edge_case":
                    samples = self.generate_edge_cases(l1, l2, n=n_per_class)
                elif gen_type == "rare_pattern":
                    samples = self.generate_rare_patterns(l1, l2, n=n_per_class)
                elif gen_type == "feedback" and l1 == "feedback":
                    samples = self.generate_feedback(l2, n=n_per_class)
                else:
                    samples = []
                
                all_samples.extend(samples)
        
        return all_samples
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "total_generated": self.stats.total_generated,
            "total_tokens": self.stats.total_tokens,
            "failed_requests": self.stats.failed_requests,
            "success_rate": (
                (self.stats.total_requests - self.stats.failed_requests) 
                / max(1, self.stats.total_requests)
            ),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_llm_samples(
    base_samples: List[Dict[str, str]],
    n_per_sample: int = 3,
    llm_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Generate LLM samples from base samples.
    
    Args:
        base_samples: List of dicts with 'text', 'label_l1', 'label_l2'
        n_per_sample: Number of variations per sample
        llm_client: LLM client
        
    Returns:
        List of generated sample dicts
    """
    generator = LLMDataGenerator(
        llm_client=llm_client,
        existing_texts={s["text"].lower() for s in base_samples},
    )
    
    all_generated = []
    
    for sample in base_samples:
        generated = generator.generate_paraphrases(
            text=sample["text"],
            l1=sample["label_l1"],
            l2=sample["label_l2"],
            n=n_per_sample,
        )
        
        for g in generated:
            all_generated.append(g.to_dict())
    
    return all_generated


if __name__ == "__main__":
    # Test with simulator
    generator = LLMDataGenerator(use_simulator=True)
    
    # Test paraphrase generation
    samples = generator.generate_paraphrases(
        text="Болит зуб",
        l1="anamnesis",
        l2="symptom",
        n=5,
    )
    print("Paraphrases:")
    for s in samples:
        print(f"  - {s.text}")
    
    # Test edge case generation (v4.0: feedback taxonomy)
    samples = generator.generate_edge_cases(
        l1="feedback",
        l2="negative_quality",
        n=3,
    )
    print("\nEdge Cases:")
    for s in samples:
        print(f"  - {s.text}")
    
    # Print stats
    print("\nGeneration Stats:")
    print(generator.get_stats())
