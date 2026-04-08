"""
Taxonomy utilities for two-level intent classification.

Provides mappings between L1 (routing) and L2 (subtype) labels,
production node assignments, and taxonomy loading from YAML.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

import yaml


# =============================================================================
# Constants: L1 and L2 Labels
# =============================================================================

# Level 1: Routing classes (5)
INTENT_LABELS_L1 = [
    "anamnesis",
    "booking", 
    "faq",
    "feedback",  # v4.0: было negative_feedback
    "conversational"
]

# Level 2: Subtype classes (24) - v4.0: расширено с 20 до 24
INTENT_LABELS_L2 = [
    # Anamnesis (3)
    "symptom", "complaint", "services",
    # Booking (3)
    "new_appointment", "reschedule", "cancel",
    # FAQ (5)
    "price", "clinic_info", "procedure", "visit_prep", "followup",
    # Feedback (8) - v4.0: расширено с 4 до 8 (negative + positive)
    "negative_service", "negative_quality", "negative_staff", "negative_general",
    "positive_service", "positive_quality", "positive_staff", "positive_general",
    # Conversational (5)
    "greeting", "gratitude", "confirmation", "farewell", "unclear"
]

# L2 → L1 mapping (v4.0: обновлено для feedback)
L2_TO_L1 = {
    # Anamnesis
    "symptom": "anamnesis",
    "complaint": "anamnesis", 
    "services": "anamnesis",
    # Booking
    "new_appointment": "booking",
    "reschedule": "booking",
    "cancel": "booking",
    # FAQ
    "price": "faq",
    "clinic_info": "faq",
    "procedure": "faq",
    "visit_prep": "faq",
    "followup": "faq",
    # Feedback - Negative (v4.0: переименовано из negative_feedback)
    "negative_service": "feedback",
    "negative_quality": "feedback",
    "negative_staff": "feedback",
    "negative_general": "feedback",
    # Feedback - Positive (v4.0: новые классы)
    "positive_service": "feedback",
    "positive_quality": "feedback",
    "positive_staff": "feedback",
    "positive_general": "feedback",
    # Conversational
    "greeting": "conversational",
    "gratitude": "conversational",
    "confirmation": "conversational",
    "farewell": "conversational",
    "unclear": "conversational",
}

# L1 → Production node mapping (v4.0: feedback_node)
L1_TO_PRODUCTION_NODE = {
    "anamnesis": "anamnesis_node",
    "booking": "booking_node",
    "faq": "faq_node",
    "feedback": "feedback_node",  # v4.0: было escalation_node
    "conversational": "continue_current",
}

# L2 → Scenario mapping (legacy compatibility, v4.0 updated)
L2_TO_SCENARIO = {
    "symptom": "anamnesis_flow",
    "complaint": "anamnesis_flow",
    "services": "anamnesis_flow",
    "new_appointment": "booking_flow",
    "reschedule": "booking_flow",
    "cancel": "booking_flow",
    "price": "faq_flow",
    "clinic_info": "faq_flow",
    "procedure": "faq_flow",
    "visit_prep": "faq_flow",
    "followup": "faq_flow",
    # Feedback - Negative (escalation для жалоб)
    "negative_service": "escalation_flow",
    "negative_quality": "escalation_flow",
    "negative_staff": "escalation_flow",
    "negative_general": "escalation_flow",
    # Feedback - Positive (feedback для положительных отзывов)
    "positive_service": "feedback_flow",
    "positive_quality": "feedback_flow",
    "positive_staff": "feedback_flow",
    "positive_general": "feedback_flow",
    # Conversational
    "greeting": "conversation_flow",
    "gratitude": "conversation_flow",
    "confirmation": "conversation_flow",
    "farewell": "conversation_flow",
    "unclear": "clarification_flow",
}

# Class weights for training (v4.0: feedback - high priority for negative)
CLASS_WEIGHTS_L1 = {
    "anamnesis": 1.0,
    "booking": 1.0,
    "faq": 1.0,
    "feedback": 1.5,  # v4.0: важно, но не критично как было negative_feedback
    "conversational": 0.8,
}

CLASS_WEIGHTS_L2 = {
    # Critical classes - Negative feedback (высший приоритет - не пропустить жалобы!)
    "negative_service": 2.0,
    "negative_quality": 2.0,
    "negative_staff": 2.0,
    "negative_general": 1.8,
    # Positive feedback (средний приоритет)
    "positive_service": 1.2,
    "positive_quality": 1.2,
    "positive_staff": 1.2,
    "positive_general": 1.0,
    # Important classes
    "symptom": 1.2,
    "complaint": 1.1,
    # Default weight for others
    **{label: 1.0 for label in INTENT_LABELS_L2 if label not in [
        "negative_service", "negative_quality", "negative_staff", "negative_general",
        "positive_service", "positive_quality", "positive_staff", "positive_general",
        "symptom", "complaint"
    ]}
}

# Confidence thresholds for routing (v4.0: обновлённые пороги)
CONFIDENCE_THRESHOLDS = {
    "high": 0.90,      # Direct routing (v4.0: было 0.85)
    "medium": 0.70,    # Clarification needed (v4.0: было 0.65)
    "low": 0.50,       # LLM fallback (v4.0: было 0.40)
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of intent classification."""
    l1: str
    l2: str
    confidence: float
    modifiers: List[str] = field(default_factory=list)
    tone: str = "neutral"
    entities: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[str] = field(default_factory=list)
    
    @property
    def production_node(self) -> str:
        """Get production node for routing."""
        return L1_TO_PRODUCTION_NODE.get(self.l1, "faq_node")
    
    @property
    def scenario(self) -> str:
        """Get scenario (legacy compatibility)."""
        return L2_TO_SCENARIO.get(self.l2, "faq_flow")
    
    @property
    def routing_action(self) -> str:
        """Determine routing action based on confidence."""
        if self.confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "direct_route"
        elif self.confidence >= CONFIDENCE_THRESHOLDS["medium"]:
            return "clarify"
        else:
            return "llm_fallback"
    
    @property
    def is_escalation_needed(self) -> bool:
        """Check if escalation is needed (negative feedback)."""
        # v4.0: проверяем negative subtypes в feedback
        return self.l1 == "feedback" and self.l2.startswith("negative_")
    
    @property
    def is_positive_feedback(self) -> bool:
        """Check if this is positive feedback (v4.0)."""
        return self.l1 == "feedback" and self.l2.startswith("positive_")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "l1": self.l1,
            "l2": self.l2,
            "confidence": self.confidence,
            "modifiers": self.modifiers,
            "tone": self.tone,
            "entities": self.entities,
            "secondary_intents": self.secondary_intents,
            "production_node": self.production_node,
            "scenario": self.scenario,
            "routing_action": self.routing_action,
            "is_escalation_needed": self.is_escalation_needed,
            "is_positive_feedback": self.is_positive_feedback,  # v4.0
        }


@dataclass
class TaxonomyConfig:
    """Configuration loaded from taxonomy YAML."""
    version: str
    l1_classes: List[str]
    l2_classes: List[str]
    l2_to_l1: Dict[str, str]
    entity_types: Dict[str, Any]
    modifiers: Dict[str, Any]
    tones: Dict[str, Any]
    confidence_thresholds: Dict[str, float]
    class_weights: Dict[str, Dict[str, float]]
    
    @classmethod
    def from_yaml(cls, path: Path) -> "TaxonomyConfig":
        """Load taxonomy configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        taxonomy = data.get("taxonomy", {})
        
        # Extract L1 and L2 classes
        l1_classes = list(taxonomy.get("level1", {}).keys())
        l2_classes = []
        l2_to_l1 = {}
        
        for l1, l1_data in taxonomy.get("level1", {}).items():
            for l2 in l1_data.get("subtypes", {}).keys():
                l2_classes.append(l2)
                l2_to_l1[l2] = l1
        
        return cls(
            version=taxonomy.get("version", "1.0"),
            l1_classes=l1_classes,
            l2_classes=l2_classes,
            l2_to_l1=l2_to_l1,
            entity_types=taxonomy.get("entity_types", {}),
            modifiers=taxonomy.get("modifiers", {}),
            tones=taxonomy.get("tones", {}),
            confidence_thresholds=taxonomy.get("confidence_thresholds", CONFIDENCE_THRESHOLDS),
            class_weights=taxonomy.get("class_weights", {}),
        )


# =============================================================================
# Helper Functions
# =============================================================================

def get_l1_from_l2(l2_label: str) -> str:
    """
    Get L1 (routing) label from L2 (subtype) label.
    
    Args:
        l2_label: L2 subtype label
        
    Returns:
        L1 routing label
        
    Example:
        >>> get_l1_from_l2("symptom")
        "anamnesis"
    """
    return L2_TO_L1.get(l2_label, "faq")


def get_production_node(l1_label: str) -> str:
    """
    Get production node name from L1 label.
    
    Args:
        l1_label: L1 routing label
        
    Returns:
        Production node name
        
    Example:
        >>> get_production_node("anamnesis")
        "anamnesis_node"
    """
    return L1_TO_PRODUCTION_NODE.get(l1_label, "faq_node")


def get_scenario(l2_label: str) -> str:
    """
    Get scenario from L2 label (legacy compatibility).
    
    Args:
        l2_label: L2 subtype label
        
    Returns:
        Scenario name
        
    Example:
        >>> get_scenario("symptom")
        "anamnesis_flow"
    """
    return L2_TO_SCENARIO.get(l2_label, "faq_flow")


def get_class_weight_l1(l1_label: str) -> float:
    """Get training weight for L1 class."""
    return CLASS_WEIGHTS_L1.get(l1_label, 1.0)


def get_class_weight_l2(l2_label: str) -> float:
    """Get training weight for L2 class."""
    return CLASS_WEIGHTS_L2.get(l2_label, 1.0)


def get_l2_classes_for_l1(l1_label: str) -> List[str]:
    """
    Get all L2 classes belonging to an L1 category.
    
    Args:
        l1_label: L1 routing label
        
    Returns:
        List of L2 labels
        
    Example:
        >>> get_l2_classes_for_l1("anamnesis")
        ["symptom", "complaint", "services"]
    """
    return [l2 for l2, l1 in L2_TO_L1.items() if l1 == l1_label]


def l1_id_to_label(id: int) -> str:
    """Convert L1 class ID to label."""
    if 0 <= id < len(INTENT_LABELS_L1):
        return INTENT_LABELS_L1[id]
    return "faq"


def l2_id_to_label(id: int) -> str:
    """Convert L2 class ID to label."""
    if 0 <= id < len(INTENT_LABELS_L2):
        return INTENT_LABELS_L2[id]
    return "unclear"


def l1_label_to_id(label: str) -> int:
    """Convert L1 label to class ID."""
    try:
        return INTENT_LABELS_L1.index(label)
    except ValueError:
        return INTENT_LABELS_L1.index("faq")


def l2_label_to_id(label: str) -> int:
    """Convert L2 label to class ID."""
    try:
        return INTENT_LABELS_L2.index(label)
    except ValueError:
        return INTENT_LABELS_L2.index("unclear")


def load_taxonomy(config_path: Optional[Path] = None) -> TaxonomyConfig:
    """
    Load taxonomy configuration from YAML file.
    
    Args:
        config_path: Path to intent_taxonomy.yaml. 
                     If None, searches in default locations.
                     
    Returns:
        TaxonomyConfig object
    """
    if config_path is None:
        # Try to find config
        possible_paths = [
            Path(__file__).parent.parent / "configs" / "intent_taxonomy.yaml",
            Path("configs/intent_taxonomy.yaml"),
            Path("study/configs/intent_taxonomy.yaml"),
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
    
    if config_path is None or not config_path.exists():
        # Return default config
        return TaxonomyConfig(
            version="4.0",  # v4.0: обновлённая версия
            l1_classes=INTENT_LABELS_L1,
            l2_classes=INTENT_LABELS_L2,
            l2_to_l1=L2_TO_L1,
            entity_types={},
            modifiers={},
            tones={},
            confidence_thresholds=CONFIDENCE_THRESHOLDS,
            class_weights={"l1": CLASS_WEIGHTS_L1, "l2": CLASS_WEIGHTS_L2},
        )
    
    return TaxonomyConfig.from_yaml(config_path)


def get_sklearn_class_weights(level: str = "l1") -> dict:
    """
    Get class weights in sklearn format.
    
    Args:
        level: "l1" or "l2"
        
    Returns:
        Dict mapping class name to weight (sklearn format)
    """
    if level == "l1":
        labels = INTENT_LABELS_L1
        weights = CLASS_WEIGHTS_L1
    else:
        labels = INTENT_LABELS_L2
        weights = CLASS_WEIGHTS_L2
    
    # sklearn expects class names as keys, not indices
    return {label: weights.get(label, 1.0) for label in labels}


# =============================================================================
# Validation
# =============================================================================

def validate_l1_label(label: str) -> bool:
    """Check if label is valid L1."""
    return label in INTENT_LABELS_L1


def validate_l2_label(label: str) -> bool:
    """Check if label is valid L2."""
    return label in INTENT_LABELS_L2


def validate_classification(l1: str, l2: str) -> bool:
    """
    Validate that L1 and L2 labels are consistent.
    
    Args:
        l1: L1 label
        l2: L2 label
        
    Returns:
        True if L2 belongs to L1
    """
    expected_l1 = L2_TO_L1.get(l2)
    return expected_l1 == l1


# =============================================================================
# Statistics
# =============================================================================

def get_taxonomy_stats() -> dict:
    """Get taxonomy statistics."""
    return {
        "version": "4.0",  # v4.0: обновлённая версия
        "n_l1_classes": len(INTENT_LABELS_L1),
        "n_l2_classes": len(INTENT_LABELS_L2),
        "l1_distribution": {
            l1: len(get_l2_classes_for_l1(l1)) 
            for l1 in INTENT_LABELS_L1
        },
        "production_nodes": list(set(L1_TO_PRODUCTION_NODE.values())),
        "feedback_classes": {
            "negative": [l2 for l2 in INTENT_LABELS_L2 if l2.startswith("negative_")],
            "positive": [l2 for l2 in INTENT_LABELS_L2 if l2.startswith("positive_")],
        },
    }


if __name__ == "__main__":
    # Test taxonomy utilities (v4.0)
    print("Taxonomy Stats:")
    stats = get_taxonomy_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nL2 → L1 Examples:")
    for l2 in ["symptom", "price", "negative_service", "positive_staff", "greeting"]:
        l1 = get_l1_from_l2(l2)
        node = get_production_node(l1)
        print(f"  {l2} → {l1} → {node}")
    
    print("\nL1 → L2 Classes:")
    for l1 in INTENT_LABELS_L1:
        l2_list = get_l2_classes_for_l1(l1)
        print(f"  {l1}: {l2_list}")
    
    print("\nClass Weights (L1):")
    for l1, weight in CLASS_WEIGHTS_L1.items():
        print(f"  {l1}: {weight}")
    
    print("\nFeedback Classes:")
    print(f"  Negative: {stats['feedback_classes']['negative']}")
    print(f"  Positive: {stats['feedback_classes']['positive']}")
