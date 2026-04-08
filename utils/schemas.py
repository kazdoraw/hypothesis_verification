"""
JSON Schema definitions for dental intake.
"""

import json
from pathlib import Path
from typing import Any, Optional

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception


# ============================================================================
# Complaint Types
# ============================================================================

COMPLAINT_TYPES = [
    "acute_pain",      # Острая боль
    "chronic_pain",    # Хроническая боль
    "esthetics",       # Эстетика
    "ortho",           # Ортодонтия
    "therapy",         # Терапия
]


# ============================================================================
# Required Fields by Complaint Type
# ============================================================================

REQUIRED_BASE = [
    "chief_complaint",   # Основная жалоба (текст)
    "localization",      # Локализация (зуб/область)
    "duration",          # Длительность (как давно)
    "intensity",         # Интенсивность боли (1-10)
]

REQUIRED_BY_TYPE = {
    "acute_pain": REQUIRED_BASE + ["onset", "triggers"],
    "chronic_pain": REQUIRED_BASE + ["onset", "relievers"],
    "esthetics": ["chief_complaint", "localization", "desired_outcome"],
    "ortho": ["chief_complaint", "localization", "bite_issues"],
    "therapy": ["chief_complaint", "localization", "last_visit"],
}

OPTIONAL_FIELDS = [
    "temperature",
    "swelling",
    "bleeding",
    "allergies",
    "chronic_conditions",
    "medications",
    "pregnancy",
    "red_flags",
    "urgency",
    "previous_treatment",
    "budget_range",
    "previous_ortho",
    "caries_history",
]


def get_required_fields(complaint_type: str) -> list[str]:
    """Get required fields for a specific complaint type."""
    return REQUIRED_BY_TYPE.get(complaint_type, REQUIRED_BASE)


# ============================================================================
# JSON Schema (Draft-07)
# ============================================================================

INTAKE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DentalIntake",
    "description": "JSON Schema для преданамнеза стоматологического пациента",
    "type": "object",
    "properties": {
        "complaint_type": {
            "type": "string",
            "enum": COMPLAINT_TYPES,
            "description": "Тип жалобы"
        },
        "chief_complaint": {
            "type": "string",
            "minLength": 3,
            "description": "Основная жалоба пациента"
        },
        "localization": {
            "type": "string",
            "description": "Локализация проблемы (зуб, область)"
        },
        "duration": {
            "type": "string",
            "description": "Как давно беспокоит"
        },
        "intensity": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Интенсивность боли (1-10)"
        },
        "onset": {
            "type": "string",
            "description": "Когда и как началось"
        },
        "triggers": {
            "type": "string",
            "description": "Что усиливает боль/дискомфорт"
        },
        "relievers": {
            "type": "string",
            "description": "Что облегчает состояние"
        },
        "temperature": {
            "type": "boolean",
            "description": "Повышенная температура"
        },
        "swelling": {
            "type": "boolean",
            "description": "Отёк"
        },
        "bleeding": {
            "type": "boolean",
            "description": "Кровоточивость"
        },
        "allergies": {
            "type": "string",
            "description": "Аллергии"
        },
        "chronic_conditions": {
            "type": "string",
            "description": "Хронические заболевания"
        },
        "medications": {
            "type": "string",
            "description": "Текущие препараты"
        },
        "pregnancy": {
            "type": "boolean",
            "description": "Беременность"
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Красные флаги (требующие срочного внимания)"
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "medium", "high", "emergency"],
            "description": "Срочность"
        },
        "desired_outcome": {
            "type": "string",
            "description": "Желаемый результат (для эстетики)"
        },
        "last_visit": {
            "type": "string",
            "description": "Дата последнего визита"
        },
        "bite_issues": {
            "type": "string",
            "description": "Проблемы с прикусом"
        },
        "previous_treatment": {
            "type": "string",
            "description": "Предыдущее лечение"
        },
        "budget_range": {
            "type": "string",
            "description": "Бюджет"
        },
        "previous_ortho": {
            "type": "string",
            "description": "Предыдущее ортодонтическое лечение"
        },
        "caries_history": {
            "type": "string",
            "description": "История кариеса"
        },
    },
    "required": ["complaint_type", "chief_complaint"],
    
    # Conditional requirements based on complaint_type
    "allOf": [
        {
            "if": {
                "properties": {"complaint_type": {"const": "acute_pain"}}
            },
            "then": {
                "required": ["localization", "duration", "intensity", "onset", "triggers"]
            }
        },
        {
            "if": {
                "properties": {"complaint_type": {"const": "chronic_pain"}}
            },
            "then": {
                "required": ["localization", "duration", "intensity", "onset", "relievers"]
            }
        },
        {
            "if": {
                "properties": {"complaint_type": {"const": "esthetics"}}
            },
            "then": {
                "required": ["localization", "desired_outcome"]
            }
        },
        {
            "if": {
                "properties": {"complaint_type": {"const": "ortho"}}
            },
            "then": {
                "required": ["localization", "bite_issues"]
            }
        },
        {
            "if": {
                "properties": {"complaint_type": {"const": "therapy"}}
            },
            "then": {
                "required": ["localization", "last_visit"]
            }
        },
    ]
}


def validate_intake(data: dict, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """
    Validate intake data against schema.
    
    Args:
        data: Intake data dict
        raise_on_error: If True, raise ValidationError on failure
        
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if not HAS_JSONSCHEMA:
        return True, "jsonschema not installed, skipping validation"
    
    try:
        validate(instance=data, schema=INTAKE_SCHEMA)
        return True, None
    except ValidationError as e:
        if raise_on_error:
            raise
        return False, str(e.message)


def export_schema(path: str | Path) -> None:
    """Export schema to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(INTAKE_SCHEMA, f, ensure_ascii=False, indent=2)


def compute_completion_rate(data: dict, complaint_type: str) -> float:
    """
    Compute completion rate for required fields.
    
    Args:
        data: Filled intake data
        complaint_type: Type of complaint
        
    Returns:
        Completion rate 0.0 to 1.0
    """
    required = get_required_fields(complaint_type)
    if not required:
        return 1.0
    
    filled = sum(1 for field in required if data.get(field))
    return filled / len(required)


# ============================================================================
# Example Data
# ============================================================================

EXAMPLE_INTAKES = [
    {
        "complaint_type": "acute_pain",
        "chief_complaint": "Сильная зубная боль",
        "localization": "Нижняя челюсть справа, 46 зуб",
        "duration": "2 дня",
        "intensity": 8,
        "onset": "Началось вчера вечером после еды",
        "triggers": "Холодное, горячее, жевание",
    },
    {
        "complaint_type": "esthetics",
        "chief_complaint": "Хочу отбелить зубы",
        "localization": "Передние зубы",
        "desired_outcome": "Белоснежная улыбка",
    },
    {
        "complaint_type": "ortho",
        "chief_complaint": "Кривые зубы",
        "localization": "Верхняя челюсть",
        "bite_issues": "Скученность передних зубов",
    },
]


if __name__ == "__main__":
    # Test validation
    for example in EXAMPLE_INTAKES:
        is_valid, error = validate_intake(example)
        print(f"Type: {example['complaint_type']}, Valid: {is_valid}")
        if error:
            print(f"  Error: {error}")
        
        rate = compute_completion_rate(example, example["complaint_type"])
        print(f"  Completion rate: {rate:.0%}")
