"""ComplexityGate: SSoT для complexity-маркеров и тэгов в D1 (Task 3 плана).

Назначение: per-text rule-based классификация сложности запроса с детерминированным
priority и структурным API (`ComplexityDecision`). SimpleRouter (Task 4) использует
gate как первый этап cascade: complex/multi-intent → defer, simple → ML policy.

**Семантика `mixed_intent` vs subtypes (важно для аналитики):**
По формулам `symptom_price ⊆ mixed_intent`, `symptom_booking ⊆ mixed_intent`,
`symptom_doctor ⊆ mixed_intent`. Поэтому при наличии любого symptom+other-intent
сочетания `primary_tag = "mixed_intent"`, а subtype остаётся в поле `tags`
(`tags["symptom_price"] = True`, etc).

- `RouteDecision.reason` для defer будет `"complexity:mixed_intent"`,
  **никогда** `"complexity:symptom_price"`.
- Аналитика, которой нужен subtype, читает его из `tags`, не из `primary_tag`.

Формулы тэгов (зафиксированы):
    short_ambiguous = tokens(text) <= _MAX_SHORT_TOKENS
    mixed_intent    = symptom AND (price OR booking OR doctor)
    symptom_price   = symptom AND price
    symptom_booking = symptom AND booking
    symptom_doctor  = symptom AND doctor
    booking_doctor_name = booking AND surname-like doctor mention
    simple_faq      = price AND NOT symptom AND NOT booking
    simple_booking  = booking AND NOT symptom AND NOT booking_doctor_name
    simple_symptom  = symptom AND NOT (price OR booking OR doctor)

Priority при выборе primary_tag (от высшего к низшему):
    1. mixed_intent
    2. short_ambiguous
    3. booking_doctor_name
    4. symptom_price / symptom_booking / symptom_doctor (детерминированный
       ordering — недостижим если mixed_intent=True)
    5. simple_faq / simple_booking / simple_symptom
    6. unclassified (default, action=allow_ml)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Markers — SSoT (раньше дублировались в interactive_inference.py)
# ---------------------------------------------------------------------------

_SYMPTOM_MARKERS: tuple[str, ...] = (
    "бол", "десн", "кров", "опух", "ноет", "пульсир", "гной", "флюс",
)
_PRICE_MARKERS: tuple[str, ...] = (
    "стоим", "цен", "сколько", "прайс",
)
_BOOKING_MARKERS: tuple[str, ...] = (
    "запис", "запиш", "прием", "приём", "прийти", "свободн", "окошк",
)
_DOCTOR_MARKERS: tuple[str, ...] = (
    "врач", "ортодонт", "хирург", "пародонтолог", "терапевт",
)
_SURNAME_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]{3,}(?:ов[ауы]?|ев[ауы]?|ин[ауы]?|"
    r"ск(?:ий|ая|ой|ую)|ко|ых|их)\b",
)

_MAX_SHORT_TOKENS: int = 2

# Порядок subtype'ов при выборе primary_tag (если mixed_intent=False, что по
# формулам недостижимо, но для полной детерминированности фиксируем).
_SUBTYPE_PRIORITY: tuple[str, ...] = (
    "booking_doctor_name",
    "symptom_price", "symptom_booking", "symptom_doctor",
)
_SIMPLE_PRIORITY: tuple[str, ...] = (
    "simple_faq", "simple_booking", "simple_symptom",
)

Action = Literal["allow_ml", "defer"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexityDecision:
    """Результат classification одного текста."""

    text: str
    tags: dict[str, bool]      # все active/inactive маркеры
    primary_tag: str           # выбранный по priority
    action: Action             # defer / allow_ml
    reason: str                # = primary_tag (для использования в RouteDecision.reason)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class ComplexityGate:
    """Rule-based classifier complexity-тэгов.

    Конструктор не принимает параметров: маркеры — SSoT module-level constants.
    Это намеренно: иначе teams могли бы переопределять локально, ломая
    consistency между gate, evaluation и аналитикой.
    """

    def decide(self, text: str) -> ComplexityDecision:
        """Один текст → ComplexityDecision."""
        tags = self._compute_tags(text)
        primary_tag = self._select_primary(tags)
        action: Action = "defer" if self._is_defer(primary_tag) else "allow_ml"
        return ComplexityDecision(
            text=text,
            tags=tags,
            primary_tag=primary_tag,
            action=action,
            reason=primary_tag,
        )

    def decide_batch(self, texts: list[str]) -> list[ComplexityDecision]:
        """Batch wrapper. Эквивалент `[decide(t) for t in texts]`."""
        return [self.decide(t) for t in texts]

    # -- internals --------------------------------------------------------

    @staticmethod
    def _compute_tags(text: str) -> dict[str, bool]:
        """Вычислить все 8 тэгов из формул.

        Returns:
            dict с фиксированным набором ключей (см. `_TAG_NAMES`).
            Значения — bool. Гарантия immutable-like API.
        """
        low = text.lower()
        tokens = re.findall(r"[a-zа-яё0-9]+", low, flags=re.IGNORECASE)

        has_symptom = _has_any(low, _SYMPTOM_MARKERS)
        has_price = _has_any(low, _PRICE_MARKERS)
        has_booking = _has_any(low, _BOOKING_MARKERS)
        has_doctor = _has_any(low, _DOCTOR_MARKERS)
        has_surname = bool(_SURNAME_RE.search(text))

        # Subtype формулы (symptom + other intent).
        symptom_price = has_symptom and has_price
        symptom_booking = has_symptom and has_booking
        symptom_doctor = has_symptom and has_doctor
        mixed_intent = symptom_price or symptom_booking or symptom_doctor
        booking_doctor_name = has_booking and has_surname

        # Simple формулы (single intent без mixing).
        simple_faq = has_price and not has_symptom and not has_booking
        simple_booking = has_booking and not has_symptom and not booking_doctor_name
        simple_symptom = has_symptom and not (has_price or has_booking or has_doctor)

        return {
            "short_ambiguous": len(tokens) <= _MAX_SHORT_TOKENS,
            "mixed_intent": mixed_intent,
            "symptom_price": symptom_price,
            "symptom_booking": symptom_booking,
            "symptom_doctor": symptom_doctor,
            "booking_doctor_name": booking_doctor_name,
            "simple_faq": simple_faq,
            "simple_booking": simple_booking,
            "simple_symptom": simple_symptom,
        }

    @staticmethod
    def _select_primary(tags: dict[str, bool]) -> str:
        """Выбрать primary_tag по фиксированной priority.

        Order: mixed_intent → short_ambiguous → symptom_* → simple_* → unclassified.
        """
        if tags.get("mixed_intent"):
            return "mixed_intent"
        if tags.get("short_ambiguous"):
            return "short_ambiguous"
        for subtype in _SUBTYPE_PRIORITY:
            if tags.get(subtype):
                return subtype
        for simple in _SIMPLE_PRIORITY:
            if tags.get(simple):
                return simple
        return "unclassified"

    @staticmethod
    def _is_defer(primary_tag: str) -> bool:
        """Action policy: какие primary_tag → defer."""
        return primary_tag in {
            "mixed_intent",
            "short_ambiguous",
            "symptom_price",
            "symptom_booking",
            "symptom_doctor",
            "booking_doctor_name",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    """Match хотя бы одного substring marker в text (lower-cased)."""
    return any(marker in text for marker in markers)


# Public API для consumer'ов (interactive_inference.py, simple_router.py).
__all__ = [
    "Action",
    "ComplexityDecision",
    "ComplexityGate",
]
