"""Таксономия v6 — production-aligned domain routing.

SSoT: d1/ontology/route_domain.yaml
Этот модуль загружает онтологию из YAML и предоставляет:
- Константы (ROUTE_DOMAINS, DOMAIN_TO_NODE, DOMAIN_SUBTYPES и т.д.)
- ClassificationResultV6 — dataclass, aligned с production RouterPlan
- Функции валидации и преобразования

НЕ импортирует из taxonomy.py (полная изоляция от legacy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Путь к SSoT онтологии (по умолчанию — d1/ontology/route_domain.yaml)
# ---------------------------------------------------------------------------
_DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent / "d1" / "ontology" / "route_domain.yaml"


# ---------------------------------------------------------------------------
# Загрузка YAML
# ---------------------------------------------------------------------------

def load_ontology(path: Path | str | None = None) -> dict[str, Any]:
    """Загрузка онтологии из YAML.

    Args:
        path: путь к YAML. По умолчанию — d1/ontology/route_domain.yaml

    Returns:
        dict с полным содержимым YAML
    """
    p = Path(path) if path else _DEFAULT_ONTOLOGY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Онтология не найдена: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_constants(ontology: dict[str, Any]) -> dict[str, Any]:
    """Построение кэшированных констант из онтологии."""
    domains_raw = ontology.get("domains", {})

    route_domains: list[str] = list(domains_raw.keys())
    domain_to_node: dict[str, str] = {}
    domain_subtypes: dict[str, list[str]] = {}
    subtype_to_domain: dict[str, str] = {}
    domain_descriptions: dict[str, str] = {}

    for dom_name, dom_data in domains_raw.items():
        domain_to_node[dom_name] = dom_data.get("production_node", "__end__")
        domain_descriptions[dom_name] = dom_data.get("description", "")
        subs = list(dom_data.get("subtypes", {}).keys())
        domain_subtypes[dom_name] = subs
        for sub in subs:
            subtype_to_domain[sub] = dom_name

    thresholds = ontology.get("confidence_thresholds", {})

    return {
        "route_domains": route_domains,
        "domain_to_node": domain_to_node,
        "domain_subtypes": domain_subtypes,
        "subtype_to_domain": subtype_to_domain,
        "domain_descriptions": domain_descriptions,
        "confidence_thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Кэшированные константы (инициализируются при первом импорте)
# ---------------------------------------------------------------------------

try:
    _ONTOLOGY = load_ontology()
    _CACHE = _build_constants(_ONTOLOGY)
except FileNotFoundError:
    _ONTOLOGY = {}
    _CACHE = {
        "route_domains": ["anamnesis", "faq", "booking", "unsupported"],
        "domain_to_node": {
            "anamnesis": "anamnesis_node",
            "faq": "faq_node",
            "booking": "booking_node",
            "unsupported": "__end__",
        },
        "domain_subtypes": {
            "anamnesis": ["symptom", "complaint", "treatment_interest"],
            "faq": ["price", "clinic_info", "doctor_info", "procedure_info"],
            "booking": ["new_appointment", "reschedule", "cancel"],
            "unsupported": ["offtopic", "greeting_only", "farewell_only", "unclear_short", "feedback"],
        },
        "subtype_to_domain": {},
        "domain_descriptions": {},
        "confidence_thresholds": {"high": 0.90, "medium": 0.70, "low": 0.50},
    }
    # Достраиваем subtype_to_domain из fallback
    for _dom, _subs in _CACHE["domain_subtypes"].items():
        for _sub in _subs:
            _CACHE["subtype_to_domain"][_sub] = _dom

# Публичные константы
ROUTE_DOMAINS: list[str] = _CACHE["route_domains"]
DOMAIN_TO_NODE: dict[str, str] = _CACHE["domain_to_node"]
DOMAIN_SUBTYPES: dict[str, list[str]] = _CACHE["domain_subtypes"]
SUBTYPE_TO_DOMAIN: dict[str, str] = _CACHE["subtype_to_domain"]
DOMAIN_DESCRIPTIONS: dict[str, str] = _CACHE["domain_descriptions"]
CONFIDENCE_THRESHOLDS_V6: dict[str, float] = _CACHE["confidence_thresholds"]

ALL_SUBTYPES: list[str] = list(SUBTYPE_TO_DOMAIN.keys())
NUM_DOMAINS: int = len(ROUTE_DOMAINS)
NUM_SUBTYPES: int = len(ALL_SUBTYPES)


# ---------------------------------------------------------------------------
# ClassificationResultV6 — aligned с production RouterPlan
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResultV6:
    """Результат классификации, aligned с production RouterPlan.

    Поля соответствуют: schemas.py → RouterPlan (intents, explicit_booking,
    urgency, is_offtopic, confidence, specialization).
    """
    route_domain: str
    subtype: str
    confidence: float
    explicit_booking: bool = False
    urgency: str = "normal"
    is_offtopic: bool = False
    specialization_hint: Optional[str] = None
    feedback_flag: bool = False

    @property
    def production_node(self) -> str:
        """Production нода для этого домена."""
        return DOMAIN_TO_NODE.get(self.route_domain, "__end__")


# ---------------------------------------------------------------------------
# Функции-хелперы
# ---------------------------------------------------------------------------

def get_domain_from_subtype(subtype: str) -> str:
    """Домен по подтипу. Raises KeyError если subtype неизвестен."""
    if subtype not in SUBTYPE_TO_DOMAIN:
        raise KeyError(f"Неизвестный subtype: {subtype!r}. Допустимые: {ALL_SUBTYPES}")
    return SUBTYPE_TO_DOMAIN[subtype]


def validate_domain(domain: str) -> bool:
    """Проверка что домен валиден."""
    return domain in ROUTE_DOMAINS


def validate_subtype(subtype: str) -> bool:
    """Проверка что subtype валиден."""
    return subtype in SUBTYPE_TO_DOMAIN


def get_subtypes_for_domain(domain: str) -> list[str]:
    """Список подтипов для домена."""
    return DOMAIN_SUBTYPES.get(domain, [])


def get_all_subtypes() -> list[str]:
    """Полный список всех подтипов."""
    return list(ALL_SUBTYPES)


def get_examples_for_subtype(subtype: str, ontology: dict[str, Any] | None = None) -> list[str]:
    """Примеры из онтологии для подтипа.

    Args:
        subtype: название подтипа
        ontology: загруженная онтология (если None — используется кэш)
    """
    ont = ontology or _ONTOLOGY
    if not ont:
        return []
    domain = SUBTYPE_TO_DOMAIN.get(subtype)
    if not domain:
        return []
    return (
        ont.get("domains", {})
        .get(domain, {})
        .get("subtypes", {})
        .get(subtype, {})
        .get("examples", [])
    )


def get_routing_rules(ontology: dict[str, Any] | None = None) -> dict[str, Any]:
    """Правила маршрутизации из онтологии."""
    ont = ontology or _ONTOLOGY
    return ont.get("routing_rules", {})
