"""
Smart Router Pipeline for dental clinic AI.

Implements multi-stage classification pipeline:
1. Preprocessor - Text normalization and cleaning
2. Entity Extractor - Extract doctors, dates, procedures, etc.
3. ML Classifier - L1/L2 classification with confidence
4. Confidence Router - Direct/Clarify/LLM Fallback
5. Context Builder - Build enriched context
6. Prompt Selector - Select appropriate prompt

Architecture:
    Input Text
        │
        ▼
    ┌──────────────────┐
    │  1. Preprocessor  │  Normalize, clean
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 2. Entity Extractor│  Doctors, dates, procedures
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 3. ML Classifier  │  L1 + L2 + confidence
    └────────┬─────────┘
             │
             ├── confidence >= 0.85 ──► Direct Routing
             │
             ├── confidence 0.65-0.85 ─► Clarification
             │
             └── confidence < 0.65 ──► LLM Fallback
             │
             ▼
    ┌──────────────────┐
    │ 4. Context Builder│  Enrich with entities + dialog
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 5. Prompt Selector│  Select prompt by L1/L2
    └────────┬─────────┘
             │
             ▼
        [LLM Node]
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .taxonomy import (
    INTENT_LABELS_L1,
    INTENT_LABELS_L2,
    L2_TO_L1,
    L1_TO_PRODUCTION_NODE,
    CONFIDENCE_THRESHOLDS,
    ClassificationResult,
)

from .entity_extractor import EntityExtractor, ExtractionResult
from .context_prompts import (
    DialogContext,
    EnrichedContext,
    PromptBuilder,
    ContextInterpreter,
)


# SSoT thresholds from taxonomy
_HIGH_THRESHOLD = CONFIDENCE_THRESHOLDS["high"]
_MEDIUM_THRESHOLD = CONFIDENCE_THRESHOLDS["medium"]


# =============================================================================
# Enums and Constants
# =============================================================================

class RoutingAction(Enum):
    """Routing action based on confidence."""
    DIRECT = "direct"           # High confidence, route directly
    CLARIFY = "clarify"         # Medium confidence, ask clarifying question
    LLM_FALLBACK = "llm_fallback"  # Low confidence, use LLM


# Modifiers that affect classification
MODIFIER_PATTERNS = {
    "negation": [
        r"\b(?:не|нет|без|никак|ни\s+за\s+что)\b",
    ],
    "doubt": [
        r"\b(?:точно|правда|а\s+если|вдруг|наверное|может\s+быть)\b",
    ],
    "urgency": [
        r"\b(?:срочно|быстро|немедленно|сейчас|экстренно|срочный)\b",
    ],
    "emotion_negative": [
        r"\b(?:опять|снова|уже|ещё|надоело|устал[аи]?)\b",
    ],
    "sarcasm": [
        r"\b(?:ну\s+и|отлично|замечательно|прекрасно|супер)\b(?!.*спасибо)",
    ],
}

# Tone detection patterns
TONE_PATTERNS = {
    "worried": [
        r"\b(?:боюсь|переживаю|страшно|волнуюсь|беспокоюсь)\b",
    ],
    "angry": [
        r"\b(?:возмутительно|безобразие|недопустимо|ужас|кошмар)\b",
    ],
    "grateful": [
        r"\b(?:очень\s+благодарен|спасибо\s+огромное|большое\s+спасибо)\b",
    ],
    "urgent": [
        r"\b(?:помогите|срочно|не\s+могу\s+терпеть|сильная\s+боль|невыносимо)\b",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PreprocessedText:
    """Result of text preprocessing."""
    original: str
    normalized: str
    cleaned: str
    tokens: List[str]
    
    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "cleaned": self.cleaned,
            "n_tokens": len(self.tokens),
        }


@dataclass
class RouterResult:
    """Full result from Smart Router."""
    # Classification
    l1: str
    l2: str
    confidence: float
    
    # Routing
    action: RoutingAction
    production_node: str
    
    # Entities
    entities: Dict[str, Any]
    
    # Modifiers and tone
    modifiers: List[str]
    tone: str
    
    # Additional info
    secondary_intents: List[str] = field(default_factory=list)
    clarification_question: Optional[str] = None
    selected_prompt: Optional[str] = None
    
    # Debug info
    preprocessing: Optional[PreprocessedText] = None
    raw_classifier_output: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "l1": self.l1,
            "l2": self.l2,
            "confidence": self.confidence,
            "action": self.action.value,
            "production_node": self.production_node,
            "entities": self.entities,
            "modifiers": self.modifiers,
            "tone": self.tone,
            "secondary_intents": self.secondary_intents,
            "clarification_question": self.clarification_question,
        }
    
    @property
    def is_escalation_needed(self) -> bool:
        """Check if escalation is needed (v4.0: negative feedback only)."""
        # v4.0: feedback can be positive or negative, check L2 for negative
        return self.l1 == "feedback" and (self.l2 or "").startswith("negative_")
    
    @property
    def is_urgent(self) -> bool:
        """Check if message is urgent."""
        return "urgency" in self.modifiers or self.tone == "urgent"


@dataclass 
class RouterStats:
    """Statistics for router usage."""
    total_processed: int = 0
    direct_routes: int = 0
    clarifications: int = 0
    llm_fallbacks: int = 0
    escalations: int = 0
    
    def log_route(self, result: RouterResult):
        self.total_processed += 1
        if result.action == RoutingAction.DIRECT:
            self.direct_routes += 1
        elif result.action == RoutingAction.CLARIFY:
            self.clarifications += 1
        else:
            self.llm_fallbacks += 1
        if result.is_escalation_needed:
            self.escalations += 1
    
    def to_dict(self) -> dict:
        total = max(1, self.total_processed)
        return {
            "total_processed": self.total_processed,
            "direct_routes": self.direct_routes,
            "clarifications": self.clarifications,
            "llm_fallbacks": self.llm_fallbacks,
            "escalations": self.escalations,
            "direct_rate": self.direct_routes / total,
            "clarification_rate": self.clarifications / total,
            "fallback_rate": self.llm_fallbacks / total,
        }


# =============================================================================
# Preprocessor
# =============================================================================

class TextPreprocessor:
    """
    Preprocess text for classification.
    
    Steps:
    1. Normalize whitespace
    2. Handle punctuation
    3. Lowercase (optional)
    4. Tokenize
    """
    
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
    
    def preprocess(self, text: str) -> PreprocessedText:
        """Preprocess text."""
        original = text
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        # Clean (remove excessive punctuation)
        cleaned = re.sub(r'([!?.])\1+', r'\1', normalized)
        cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)
        
        # Optionally lowercase
        if self.lowercase:
            cleaned = cleaned.lower()
        
        # Tokenize
        tokens = cleaned.split()
        
        return PreprocessedText(
            original=original,
            normalized=normalized,
            cleaned=cleaned,
            tokens=tokens,
        )


# =============================================================================
# Modifier and Tone Detector
# =============================================================================

class ModifierDetector:
    """Detect modifiers and tone in text."""
    
    def detect_modifiers(self, text: str) -> List[str]:
        """Detect modifiers in text."""
        text_lower = text.lower()
        modifiers = []
        
        for modifier_type, patterns in MODIFIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    modifiers.append(modifier_type)
                    break
        
        return modifiers
    
    def detect_tone(self, text: str) -> str:
        """Detect tone of text."""
        text_lower = text.lower()
        
        for tone, patterns in TONE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return tone
        
        return "neutral"


# =============================================================================
# Confidence Router
# =============================================================================

class ConfidenceRouter:
    """Route based on classification confidence."""
    
    def __init__(
        self,
        high_threshold: float = _HIGH_THRESHOLD,
        medium_threshold: float = _MEDIUM_THRESHOLD,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def route(self, confidence: float, l1: str, modifiers: List[str]) -> RoutingAction:
        """
        Determine routing action.
        
        Args:
            confidence: Classification confidence
            l1: L1 classification
            modifiers: Detected modifiers
            
        Returns:
            RoutingAction
        """
        # Feedback (v4.0: was negative_feedback) requires careful handling
        if l1 == "feedback":
            # Even high confidence needs verification for feedback
            if confidence >= self.high_threshold:
                return RoutingAction.DIRECT
            else:
                return RoutingAction.CLARIFY
        
        # Doubt modifier lowers effective confidence
        if "doubt" in modifiers:
            confidence *= 0.9
        
        # Route based on confidence
        if confidence >= self.high_threshold:
            return RoutingAction.DIRECT
        elif confidence >= self.medium_threshold:
            return RoutingAction.CLARIFY
        else:
            return RoutingAction.LLM_FALLBACK
    
    def get_clarification_question(self, l1: str, l2: str) -> str:
        """Get clarification question for ambiguous case."""
        clarifications = {
            "anamnesis": "Уточните, пожалуйста, вы хотите рассказать о проблеме или записаться на приём?",
            "booking": "Вы хотите записаться на приём, перенести или отменить запись?",
            "faq": "Уточните, вас интересует стоимость услуги или информация о процедуре?",
            "feedback": "Вы хотите оставить отзыв о нашей работе?",  # v4.0: было negative_feedback
            "conversational": "Чем я могу вам помочь?",
        }
        return clarifications.get(l1, "Уточните, пожалуйста, ваш запрос.")


# =============================================================================
# Smart Router Pipeline
# =============================================================================

class SmartRouter:
    """
    Smart Router Pipeline for dental clinic AI.
    
    Combines multiple stages:
    1. Preprocessing
    2. Entity extraction
    3. Classification (external classifier)
    4. Modifier/tone detection
    5. Confidence-based routing
    6. Context building
    7. Prompt selection
    
    Example:
        >>> router = SmartRouter(classifier=my_classifier)
        >>> result = router.process("Болит зуб, хочу к Иванову")
        >>> print(result.l1, result.l2, result.action)
        anamnesis symptom RoutingAction.DIRECT
    """
    
    def __init__(
        self,
        classifier: Optional[Any] = None,
        high_threshold: float = _HIGH_THRESHOLD,
        medium_threshold: float = _MEDIUM_THRESHOLD,
    ):
        """
        Initialize Smart Router.
        
        Args:
            classifier: ML classifier with predict_with_confidence method
            high_threshold: Threshold for direct routing (v4.0: 0.90)
            medium_threshold: Threshold for clarification (v4.0: 0.70)
        """
        self.classifier = classifier
        
        # Components
        self.preprocessor = TextPreprocessor(lowercase=False)
        self.entity_extractor = EntityExtractor()
        self.modifier_detector = ModifierDetector()
        self.confidence_router = ConfidenceRouter(high_threshold, medium_threshold)
        self.context_interpreter = ContextInterpreter()
        self.prompt_builder = PromptBuilder()
        
        # Statistics
        self.stats = RouterStats()
    
    def process(
        self,
        text: str,
        dialog_context: Optional[DialogContext] = None,
        include_prompt: bool = False,
    ) -> RouterResult:
        """
        Process text through the full pipeline.
        
        Args:
            text: Input text
            dialog_context: Optional dialog context
            include_prompt: Whether to include selected prompt
            
        Returns:
            RouterResult with all information
        """
        # 1. Preprocess
        preprocessed = self.preprocessor.preprocess(text)
        
        # 2. Extract entities
        entities_result = self.entity_extractor.extract(preprocessed.normalized)
        entities = entities_result.to_flat_dict()
        
        # 3. Classify
        l1, l2, confidence, raw_output = self._classify(preprocessed.cleaned)
        
        # 4. Detect modifiers and tone
        modifiers = self.modifier_detector.detect_modifiers(preprocessed.normalized)
        tone = self.modifier_detector.detect_tone(preprocessed.normalized)
        
        # 5. Interpret in context
        if dialog_context:
            l1, l2 = self.context_interpreter.interpret_in_context(
                text=preprocessed.normalized,
                classified_l1=l1,
                classified_l2=l2,
                context=dialog_context,
            )
        
        # 6. Route based on confidence
        action = self.confidence_router.route(confidence, l1, modifiers)
        
        # 7. Get production node
        production_node = L1_TO_PRODUCTION_NODE.get(l1, "faq_node")
        
        # Build result
        result = RouterResult(
            l1=l1,
            l2=l2,
            confidence=confidence,
            action=action,
            production_node=production_node,
            entities=entities,
            modifiers=modifiers,
            tone=tone,
            preprocessing=preprocessed,
            raw_classifier_output=raw_output,
        )
        
        # Add clarification question if needed
        if action == RoutingAction.CLARIFY:
            result.clarification_question = self.confidence_router.get_clarification_question(l1, l2)
        
        # Build prompt if requested
        if include_prompt:
            result.selected_prompt = self.prompt_builder.build_prompt(
                text=preprocessed.normalized,
                l1=l1,
                l2=l2,
                entities=entities,
                modifiers=modifiers,
                tone=tone,
                dialog_context=dialog_context,
            )
        
        # Log statistics
        self.stats.log_route(result)
        
        return result
    
    def _classify(self, text: str) -> Tuple[str, str, float, dict]:
        """
        Classify text using classifier.
        
        Returns:
            Tuple of (l1, l2, confidence, raw_output)
        """
        if self.classifier is None:
            # Fallback to rule-based if no classifier
            return self._rule_based_classify(text)
        
        try:
            # Try to use classifier
            if hasattr(self.classifier, 'predict_proba'):
                # sklearn-style classifier
                proba = self.classifier.predict_proba([text])[0]
                l2_idx = proba.argmax()
                confidence = float(proba[l2_idx])
                l2 = INTENT_LABELS_L2[l2_idx] if l2_idx < len(INTENT_LABELS_L2) else "unclear"
                l1 = L2_TO_L1.get(l2, "faq")
                return l1, l2, confidence, {"proba": proba.tolist()}
            
            elif hasattr(self.classifier, 'classify'):
                # Custom classifier
                result = self.classifier.classify(text)
                return (
                    result.get("l1", "faq"),
                    result.get("l2", "unclear"),
                    result.get("confidence", 0.5),
                    result,
                )
            
            else:
                return self._rule_based_classify(text)
                
        except Exception:
            return self._rule_based_classify(text)
    
    def _rule_based_classify(self, text: str) -> Tuple[str, str, float, dict]:
        """Fallback rule-based classification."""
        text_lower = text.lower()
        
        # Feedback patterns (v4.0: was negative_feedback, now supports positive too)
        # Negative patterns
        if any(w in text_lower for w in ["жалоба", "недоволен", "возмутительно", "груб", "хуже"]):
            return "feedback", "negative_general", 0.75, {"method": "rule_based"}
        # Positive patterns (v4.0: NEW)
        if any(w in text_lower for w in ["рекомендую", "доволен", "отличн", "спасибо за работу", "профессионал"]):
            return "feedback", "positive_general", 0.70, {"method": "rule_based"}
        
        # Booking patterns
        if any(w in text_lower for w in ["записаться", "запишите", "запись", "приём"]):
            if "перенес" in text_lower:
                return "booking", "reschedule", 0.8, {"method": "rule_based"}
            if "отмен" in text_lower:
                return "booking", "cancel", 0.8, {"method": "rule_based"}
            return "booking", "new_appointment", 0.8, {"method": "rule_based"}
        
        # Anamnesis patterns
        if any(w in text_lower for w in ["болит", "боль", "ноет", "кровоточит"]):
            return "anamnesis", "symptom", 0.8, {"method": "rule_based"}
        if any(w in text_lower for w in ["выпала", "откололся", "сломался"]):
            return "anamnesis", "complaint", 0.75, {"method": "rule_based"}
        
        # FAQ patterns
        if any(w in text_lower for w in ["сколько стоит", "цена", "стоимость"]):
            return "faq", "price", 0.85, {"method": "rule_based"}
        if any(w in text_lower for w in ["где", "адрес", "работаете"]):
            return "faq", "clinic_info", 0.8, {"method": "rule_based"}
        if any(w in text_lower for w in ["больно", "как проходит"]):
            return "faq", "procedure", 0.75, {"method": "rule_based"}
        
        # Conversational patterns
        if any(w in text_lower for w in ["здравствуйте", "добрый", "привет"]):
            return "conversational", "greeting", 0.9, {"method": "rule_based"}
        if any(w in text_lower for w in ["спасибо", "благодарю"]):
            return "conversational", "gratitude", 0.9, {"method": "rule_based"}
        if any(w in text_lower for w in ["до свидания", "пока"]):
            return "conversational", "farewell", 0.9, {"method": "rule_based"}
        if text_lower in ["да", "нет", "хорошо", "ок", "ладно"]:
            return "conversational", "confirmation", 0.85, {"method": "rule_based"}
        
        # Default
        return "faq", "unclear", 0.4, {"method": "rule_based"}
    
    def get_stats(self) -> dict:
        """Get router statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = RouterStats()


# =============================================================================
# Batch Processing
# =============================================================================

def process_batch(
    texts: List[str],
    router: Optional[SmartRouter] = None,
) -> List[RouterResult]:
    """
    Process a batch of texts.
    
    Args:
        texts: List of texts
        router: SmartRouter instance (creates default if None)
        
    Returns:
        List of RouterResult objects
    """
    if router is None:
        router = SmartRouter()
    
    return [router.process(text) for text in texts]


def evaluate_router(
    texts: List[str],
    true_l1: List[str],
    true_l2: List[str],
    router: Optional[SmartRouter] = None,
) -> dict:
    """
    Evaluate router performance.
    
    Args:
        texts: List of texts
        true_l1: True L1 labels
        true_l2: True L2 labels
        router: SmartRouter instance
        
    Returns:
        Evaluation metrics
    """
    if router is None:
        router = SmartRouter()
    
    results = process_batch(texts, router)
    
    # Calculate accuracy
    l1_correct = sum(1 for r, t in zip(results, true_l1) if r.l1 == t)
    l2_correct = sum(1 for r, t in zip(results, true_l2) if r.l2 == t)
    
    total = len(texts)
    
    return {
        "l1_accuracy": l1_correct / total,
        "l2_accuracy": l2_correct / total,
        "routing_stats": router.get_stats(),
    }


# =============================================================================
# Singleton Instance
# =============================================================================

_default_router: Optional[SmartRouter] = None


def get_smart_router() -> SmartRouter:
    """Get or create default Smart Router instance."""
    global _default_router
    if _default_router is None:
        _default_router = SmartRouter()
    return _default_router


if __name__ == "__main__":
    # Test Smart Router
    router = SmartRouter()
    
    test_texts = [
        "Здравствуйте, хочу записаться на приём",
        "Болит зуб уже неделю, был у Иванова",
        "Сколько стоит отбеливание?",
        "Пломба выпала через неделю после лечения!",
        "Ок, буду",
        "Не знаю, подумаю",
    ]
    
    print("Smart Router Test:\n")
    for text in test_texts:
        result = router.process(text)
        print(f"Text: {text}")
        print(f"  L1/L2: {result.l1}/{result.l2}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Action: {result.action.value}")
        print(f"  Entities: {result.entities}")
        print(f"  Modifiers: {result.modifiers}")
        print(f"  Node: {result.production_node}")
        print()
    
    print("Router Stats:")
    print(router.get_stats())
