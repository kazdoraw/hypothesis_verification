"""Тесты для SimpleRouter (Task 4 плана).

Контракт:
- `SimpleRouter` оборачивает `B4HybridRouter` cascade'ом `ComplexityGate → ML`.
- При `gate.action == "defer"`: возвращается `RouteDecision` с
  фиксированным defer-payload (см. плановой документ Task 4).
- При `gate.action == "allow_ml"`: делегирует в `B4HybridRouter`.
- Не вызывает `.fit()`, не импортирует LLM-клиенты.
- `route(text)` ≡ `route_batch([text])[0]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from d1.baselines.complexity_gate import ComplexityGate
from d1.baselines.selective_router import RouteDecision


# ---------------------------------------------------------------------------
# Фейки для изоляции от обучения моделей
# ---------------------------------------------------------------------------

class _FakeHybrid:
    """B4HybridRouter-stub: фиксированный RouteDecision на каждый текст.

    Использование: тесты SimpleRouter не должны зависеть от обучения и от
    реального B4 поведения. Достаточно проверить что allow → delegated to hybrid.
    """

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls: list[list[str]] = []

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        self.calls.append(list(texts))
        return [self.decision for _ in texts]


def _make_accepted_decision() -> RouteDecision:
    """Эталонный accept-decision для allow_ml ветки."""
    return RouteDecision(
        label="anamnesis",
        confidence=0.85,
        margin=0.4,
        action="accept",
        sparse_dense_agree=True,
        reason="anamnesis_confident",
        dense_label="anamnesis",
    )


def _make_booking_decision(confidence: float = 0.95) -> RouteDecision:
    return RouteDecision(
        label="booking",
        confidence=confidence,
        margin=0.8,
        action="accept",
        sparse_dense_agree=True,
        reason="general_confident",
        dense_label="booking",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_route_single_matches_route_batch() -> None:
    """`route(text)` ≡ `route_batch([text])[0]`."""
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_accepted_decision())
    router = SimpleRouter(hybrid=hybrid, complexity_gate=ComplexityGate())

    text = "у меня болит зуб"
    single = router.route(text)
    batch = router.route_batch([text])[0]
    assert single == batch


def test_allow_delegates_to_hybrid() -> None:
    """allow_ml → результат идентичен `B4HybridRouter.route_batch`."""
    from d1.baselines.simple_router import SimpleRouter

    expected = _make_accepted_decision()
    hybrid = _FakeHybrid(expected)
    router = SimpleRouter(hybrid=hybrid)

    # `у меня болит зуб` (3 токена, simple_symptom) → allow_ml.
    result = router.route("у меня болит зуб")
    assert result == expected
    # Hybrid вызван ровно один раз с одним текстом:
    assert hybrid.calls == [["у меня болит зуб"]]


def test_defer_payload_contract() -> None:
    """Complexity defer: точный payload по плановому контракту.

    label="", confidence=0.0, margin=0.0, action="defer",
    sparse_dense_agree=False, reason="complexity:<primary_tag>",
    dense_label=None.
    """
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_accepted_decision())
    router = SimpleRouter(hybrid=hybrid)

    # mixed_intent: симптом + цена.
    result = router.route("болит зуб сколько стоит")
    assert result.label == ""
    assert result.confidence == 0.0
    assert result.margin == 0.0
    assert result.action == "defer"
    assert result.sparse_dense_agree is False
    assert result.reason == "complexity:mixed_intent"
    assert result.dense_label is None


def test_defer_does_not_call_hybrid() -> None:
    """При complexity-defer hybrid НЕ должен вызываться (экономия encoder).

    Если все тексты complex/short — hybrid.route_batch не вызывается вовсе.
    """
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_accepted_decision())
    router = SimpleRouter(hybrid=hybrid)

    # Все три — defer (mixed/short/mixed).
    router.route_batch([
        "болит зуб сколько стоит",
        "зуб",
        "болит десна запишите",
    ])
    assert hybrid.calls == []  # hybrid не вызывался


def test_mixed_batch_routes_correctly() -> None:
    """Batch со смесью complex+simple: defer-elements кодируются gate,
    simple-elements делегируются hybrid единым batch'ем.

    Порядок результатов сохраняется относительно входа.
    """
    from d1.baselines.simple_router import SimpleRouter

    accepted = _make_accepted_decision()
    hybrid = _FakeHybrid(accepted)
    router = SimpleRouter(hybrid=hybrid)

    texts = [
        "болит зуб сколько стоит",  # mixed → defer
        "у меня болит зуб",          # simple_symptom → allow → hybrid
        "зуб",                       # short → defer
        "у меня ноет десна",         # simple_symptom → allow → hybrid
    ]
    results = router.route_batch(texts)

    assert len(results) == 4
    # Defer'ы:
    assert results[0].action == "defer"
    assert results[0].reason == "complexity:mixed_intent"
    assert results[2].action == "defer"
    assert results[2].reason == "complexity:short_ambiguous"
    # Allow'ы (от hybrid):
    assert results[1] == accepted
    assert results[3] == accepted

    # Hybrid вызван ОДИН раз с двумя текстами (allow_idx 1 и 3):
    assert len(hybrid.calls) == 1
    assert hybrid.calls[0] == ["у меня болит зуб", "у меня ноет десна"]


def test_simple_booking_tag_policy_requires_booking_confidence() -> None:
    """simple_booking принимает только confident booking-label."""
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_booking_decision(confidence=0.95))
    router = SimpleRouter(hybrid=hybrid)

    result = router.route("запишите на чистку")
    assert result.action == "accept"
    assert result.label == "booking"


def test_simple_faq_tag_policy_defers_weak_or_wrong_label() -> None:
    """simple_faq не должен принимать слабый/не-FAQ сигнал."""
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_accepted_decision())  # label=anamnesis
    router = SimpleRouter(hybrid=hybrid)

    result = router.route("сколько стоит чистка")
    assert result.action == "defer"
    assert result.label == ""
    assert result.reason == "tag_policy:simple_faq"


def test_booking_doctor_name_deferred_before_hybrid() -> None:
    """Запись к фамилии врача отсекается ComplexityGate до ML."""
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_booking_decision())
    router = SimpleRouter(hybrid=hybrid)

    result = router.route("запишите к Петрову на пятницу")
    assert result.action == "defer"
    assert result.reason == "complexity:booking_doctor_name"
    assert hybrid.calls == []


def test_empty_batch_returns_empty_list() -> None:
    """Edge case: пустой батч не падает и не вызывает hybrid."""
    from d1.baselines.simple_router import SimpleRouter

    hybrid = _FakeHybrid(_make_accepted_decision())
    router = SimpleRouter(hybrid=hybrid)

    assert router.route_batch([]) == []
    assert hybrid.calls == []


def test_no_fit_in_simple_router_source() -> None:
    """SimpleRouter не должен вызывать `.fit()` или содержать его в исходниках."""
    src = Path("d1/baselines/simple_router.py").read_text(encoding="utf-8")
    assert ".fit(" not in src, "SimpleRouter не должен обучать модели"


def test_no_llm_imports_in_simple_router_source() -> None:
    """SimpleRouter не должен импортировать LLM-клиенты."""
    src = Path("d1/baselines/simple_router.py").read_text(encoding="utf-8")
    forbidden = [
        "OpenRouterClient", "openai", "together", "litellm",
        "chat.completions", "LLM fallback",
    ]
    for needle in forbidden:
        assert needle not in src, f"Запрещённый паттерн в simple_router.py: {needle}"


def test_tag_policy_thresholds_pinned() -> None:
    """Snapshot pinned tag-policy thresholds (Phase 2, 2026-05-01).

    Изменение DEFAULT_TAG_POLICIES без обновления этого snapshot — индикатор
    несогласованного rollback. SSoT артефакты выбора:
        d1/results/tag_policy_pareto_candidates.csv
        d1/results/tag_policy_sweep_results.csv
    """
    from d1.baselines.simple_router import DEFAULT_TAG_POLICIES

    expected: dict[str, tuple[float, tuple[str, ...]]] = {
        "simple_faq":     (0.75, ("faq",)),
        "simple_booking": (0.75, ("booking",)),
        "simple_symptom": (0.70, ("anamnesis",)),
    }
    actual = {
        tag: (policy.min_confidence, policy.allowed_labels)
        for tag, policy in DEFAULT_TAG_POLICIES.items()
    }
    assert actual == expected, (
        f"Pinned tag-policy thresholds изменились без обновления snapshot:\n"
        f"  expected: {expected}\n  actual:   {actual}"
    )
