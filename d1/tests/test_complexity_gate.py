"""Тесты для ComplexityGate (Task 3 плана).

Контракт:
- `ComplexityGate.decide(text)` возвращает `ComplexityDecision` с полями
  `text`, `tags`, `primary_tag`, `action`, `reason`.
- Priority при выборе primary_tag (от высшего к низшему):
    1. mixed_intent
    2. short_ambiguous
    3. booking_doctor_name
    4. symptom_price / symptom_booking / symptom_doctor (детерминированный
       ordering — недостижим если mixed_intent=True)
    5. simple_faq / simple_booking / simple_symptom
    6. unclassified (default)
- При совпадении нескольких symptom+other-intent сочетаний:
  primary_tag="mixed_intent", subtype остаётся в `tags` (NEVER в primary).
- `decide_batch(texts)` ≡ `[decide(t) for t in texts]`.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gate():
    """ComplexityGate instance с дефолтными маркерами."""
    from d1.baselines.complexity_gate import ComplexityGate

    return ComplexityGate()


# ---------------------------------------------------------------------------
# Simple cases (allow_ml)
# ---------------------------------------------------------------------------

def test_simple_faq_examples(gate) -> None:
    """Цена без симптома и без booking → simple_faq, allow_ml."""
    decision = gate.decide("сколько стоит чистка")
    assert decision.primary_tag == "simple_faq"
    assert decision.action == "allow_ml"
    assert decision.tags["simple_faq"] is True
    assert decision.tags["mixed_intent"] is False
    assert decision.reason == "simple_faq"


def test_simple_booking_examples(gate) -> None:
    """Booking без симптома → simple_booking, allow_ml."""
    decision = gate.decide("запишите на чистку")
    assert decision.primary_tag == "simple_booking"
    assert decision.action == "allow_ml"
    assert decision.tags["simple_booking"] is True
    assert decision.tags["mixed_intent"] is False


def test_booking_doctor_name_is_deferred(gate) -> None:
    """Запись к фамилии врача — не simple booking, а verification/defer зона."""
    decision = gate.decide("запишите к Петрову на пятницу")
    assert decision.primary_tag == "booking_doctor_name"
    assert decision.action == "defer"
    assert decision.tags["booking_doctor_name"] is True
    assert decision.tags["simple_booking"] is False


def test_simple_symptom_examples(gate) -> None:
    """Симптом без других intent'ов → simple_symptom, allow_ml.

    `болит зуб` — 2 токена, но не считается short_ambiguous, потому что
    содержит явный симптомный маркер. Граница: short_ambiguous priority
    выше simple_*, поэтому 2-token simple_symptom может быть перекрыт
    short_ambiguous. Этот тест проверяет 3-token вариант.
    """
    decision = gate.decide("у меня болит зуб")
    assert decision.primary_tag == "simple_symptom"
    assert decision.action == "allow_ml"
    assert decision.tags["simple_symptom"] is True
    assert decision.tags["mixed_intent"] is False


# ---------------------------------------------------------------------------
# Complex / defer cases
# ---------------------------------------------------------------------------

def test_mixed_intent_priority_over_symptom_price(gate) -> None:
    """`болит зуб сколько стоит` → primary_tag="mixed_intent" (НЕ symptom_price).

    Subtype остаётся в tags для аналитики.
    """
    decision = gate.decide("болит зуб сколько стоит")
    assert decision.primary_tag == "mixed_intent"
    assert decision.action == "defer"
    assert decision.reason == "mixed_intent"
    # Subtype в tags:
    assert decision.tags["mixed_intent"] is True
    assert decision.tags["symptom_price"] is True


def test_short_ambiguous(gate) -> None:
    """≤2 токена без явных markers → short_ambiguous, defer."""
    decision = gate.decide("зуб")
    assert decision.primary_tag == "short_ambiguous"
    assert decision.action == "defer"

    decision2 = gate.decide("врач")
    assert decision2.primary_tag == "short_ambiguous"


def test_unclassified_default_allow(gate) -> None:
    """Текст без активных маркеров и >2 токенов → unclassified, allow_ml.

    Граница: ≤2 токена без markers → short_ambiguous (по priority).
    Чтобы получить unclassified, нужно одновременно:
        - >2 токенов;
        - 0 активных domain-маркеров (symptom/price/booking/doctor).
    """
    decision = gate.decide("здравствуйте пожалуйста подскажите")
    assert decision.primary_tag == "unclassified"
    assert decision.action == "allow_ml"
    # Все tags False:
    assert not any(decision.tags.values())


# ---------------------------------------------------------------------------
# API invariants
# ---------------------------------------------------------------------------

def test_decide_batch_matches_decide(gate) -> None:
    """`decide_batch(texts)` равен поэлементным `decide(t)`."""
    texts = [
        "болит зуб",
        "сколько стоит чистка",
        "болит зуб сколько стоит",
        "зуб",
        "запишите на чистку",
    ]
    batch = gate.decide_batch(texts)
    individual = [gate.decide(t) for t in texts]
    assert len(batch) == len(individual)
    for a, b in zip(batch, individual):
        assert a == b


def test_priority_ordering_is_deterministic(gate) -> None:
    """При нескольких active tags primary_tag — детерминирован.

    `болит зуб запишите запишите врач` имеет:
    - symptom (болит/зуб)
    - booking (запишите)
    - doctor (врач)
    → mixed_intent доминирует.

    Многократный вызов даёт ровно тот же primary_tag.
    """
    text = "болит зуб запишите врач"
    decisions = [gate.decide(text) for _ in range(5)]
    primaries = {d.primary_tag for d in decisions}
    assert len(primaries) == 1
    assert primaries.pop() == "mixed_intent"
    # symptom_booking + symptom_doctor оба активны в tags:
    d = decisions[0]
    assert d.tags["symptom_booking"] is True
    assert d.tags["symptom_doctor"] is True


# ---------------------------------------------------------------------------
# Frozen dataclass invariant (нельзя случайно мутировать decision)
# ---------------------------------------------------------------------------

def test_decision_is_frozen(gate) -> None:
    """`ComplexityDecision` — frozen dataclass: попытка мутации → FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    decision = gate.decide("болит зуб")
    with pytest.raises(FrozenInstanceError):
        decision.action = "defer"  # type: ignore[misc]
