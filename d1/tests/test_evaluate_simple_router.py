"""Тесты для evaluate_simple_router (Task 5 плана).

Покрытие:
- pure helpers (`_decisions_to_trace_rows`, `_build_complexity_breakdown`,
  `_compute_summary_row`) — изолированы от train_bundle и I/O;
- integration smoke (опциональный, помечен `requires_bundle`) — создание
  артефактов на одном маленьком eval_set с use_cache=True.

Pure-helpers тестируются на синтетических `RouteDecision`/`ComplexityDecision`,
без обучения. Это держит regression suite быстрым (<1s).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from d1.baselines.complexity_gate import ComplexityDecision
from d1.baselines.selective_router import RouteDecision


# ---------------------------------------------------------------------------
# Fixtures: synthetic decisions
# ---------------------------------------------------------------------------

def _accept(label: str = "anamnesis") -> RouteDecision:
    return RouteDecision(
        label=label, confidence=0.85, margin=0.4,
        action="accept", sparse_dense_agree=True,
        reason="anamnesis_confident", dense_label=label,
    )


def _ml_defer(label: str = "faq") -> RouteDecision:
    """ML-defer (не complexity-defer): reason без префикса complexity:."""
    return RouteDecision(
        label=label, confidence=0.4, margin=0.05,
        action="defer", sparse_dense_agree=False,
        reason="low_confidence", dense_label="anamnesis",
    )


def _complexity_defer(primary_tag: str = "mixed_intent") -> RouteDecision:
    """Defer от ComplexityGate (как создаёт SimpleRouter)."""
    return RouteDecision(
        label="", confidence=0.0, margin=0.0,
        action="defer", sparse_dense_agree=False,
        reason=f"complexity:{primary_tag}", dense_label=None,
    )


def _gate_decision(text: str, primary_tag: str, action: str) -> ComplexityDecision:
    """Стуб ComplexityDecision (frozen, поэтому конструктор)."""
    # tags: только primary_tag = True для теста.
    tags = {
        "short_ambiguous": False, "mixed_intent": False,
        "symptom_price": False, "symptom_booking": False, "symptom_doctor": False,
        "booking_doctor_name": False,
        "simple_faq": False, "simple_booking": False, "simple_symptom": False,
    }
    if primary_tag in tags:
        tags[primary_tag] = True
    # Если subtype mixed_intent — добавим subtype для теста active_tags.
    if primary_tag == "mixed_intent":
        tags["symptom_price"] = True
    return ComplexityDecision(
        text=text, tags=tags, primary_tag=primary_tag,
        action=action, reason=primary_tag,
    )


# ---------------------------------------------------------------------------
# Pure helpers tests
# ---------------------------------------------------------------------------

class TestDecisionsToTraceRows:
    """`_decisions_to_trace_rows` расширяет hybrid trace формат двумя колонками."""

    def test_required_columns_present(self) -> None:
        from d1.scripts.evaluate_simple_router import _decisions_to_trace_rows

        texts = ["болит зуб сколько стоит", "у меня болит зуб"]
        gold = ["faq", "anamnesis"]
        decisions = [_complexity_defer("mixed_intent"), _accept()]
        gate_decisions = [
            _gate_decision(texts[0], "mixed_intent", "defer"),
            _gate_decision(texts[1], "simple_symptom", "allow_ml"),
        ]
        rows = _decisions_to_trace_rows(texts, gold, decisions, gate_decisions)

        assert len(rows) == 2
        required = {
            "text_preview", "gold", "predicted", "action", "reason",
            "is_rule_accept", "confidence", "margin", "correct",
            "primary_tag", "active_tags", "is_tag_policy_defer",
        }
        assert required <= set(rows[0].keys())

    def test_active_tags_semicolon_joined(self) -> None:
        """`mixed_intent` row должен включать subtype в `active_tags`."""
        from d1.scripts.evaluate_simple_router import _decisions_to_trace_rows

        texts = ["болит зуб сколько стоит"]
        gold = ["faq"]
        decisions = [_complexity_defer("mixed_intent")]
        gate_decisions = [_gate_decision(texts[0], "mixed_intent", "defer")]
        rows = _decisions_to_trace_rows(texts, gold, decisions, gate_decisions)

        active = rows[0]["active_tags"].split(";")
        assert "mixed_intent" in active
        assert "symptom_price" in active

    def test_correct_flag_true_only_for_accepted_match(self) -> None:
        from d1.scripts.evaluate_simple_router import _decisions_to_trace_rows

        texts = ["t1", "t2", "t3"]
        gold = ["anamnesis", "anamnesis", "anamnesis"]
        decisions = [
            _accept("anamnesis"),       # accepted + match → correct
            _accept("faq"),             # accepted + mismatch → not correct
            _complexity_defer(),        # defer → not correct (несмотря на label)
        ]
        gate_decisions = [
            _gate_decision("t1", "simple_symptom", "allow_ml"),
            _gate_decision("t2", "simple_symptom", "allow_ml"),
            _gate_decision("t3", "mixed_intent", "defer"),
        ]
        rows = _decisions_to_trace_rows(texts, gold, decisions, gate_decisions)

        assert rows[0]["correct"] is True
        assert rows[1]["correct"] is False
        assert rows[2]["correct"] is False


class TestSummaryRow:
    """`_compute_summary_row` — selective-style + complexity_defer_rate / hybrid_defer_rate."""

    def test_complexity_vs_hybrid_defer_rate_split(self) -> None:
        """7 sample: 1 accept, 2 complexity-defer, 1 tag-policy, 3 ml-defer."""
        from d1.scripts.evaluate_simple_router import _compute_summary_row

        gold = ["anamnesis"] * 7
        decisions = [
            _accept(),                    # 1 accept
            _complexity_defer(),          # complexity defer
            _complexity_defer("short_ambiguous"),
            RouteDecision(
                label="", confidence=0.8, margin=0.5,
                action="defer", sparse_dense_agree=True,
                reason="tag_policy:simple_faq", dense_label="faq",
            ),
            _ml_defer(),                  # ml defer × 3
            _ml_defer(),
            _ml_defer(),
        ]
        row = _compute_summary_row(
            eval_set="test", router_name="SimpleRouter",
            n=7, y_true=gold, decisions=decisions,
        )
        assert row["n"] == 7
        assert row["coverage"] == pytest.approx(1 / 7, abs=1e-4)
        assert row["defer_rate"] == pytest.approx(6 / 7, abs=1e-4)
        assert row["complexity_defer_rate"] == pytest.approx(2 / 7, abs=1e-4)
        assert row["tag_policy_defer_rate"] == pytest.approx(1 / 7, abs=1e-4)
        assert row["hybrid_defer_rate"] == pytest.approx(3 / 7, abs=1e-4)
        # FN_deferred: gold=anamnesis было 6 deferred → 6.
        assert row["false_negative_deferred"] == 6


class TestComplexityBreakdown:
    """`_build_complexity_breakdown` агрегирует по primary_tag."""

    def test_breakdown_columns_and_groups(self) -> None:
        from d1.scripts.evaluate_simple_router import _build_complexity_breakdown

        texts = ["t1", "t2", "t3", "t4"]
        gold = ["anamnesis", "anamnesis", "anamnesis", "faq"]
        decisions = [
            _complexity_defer("mixed_intent"),
            _complexity_defer("mixed_intent"),
            _accept("anamnesis"),
            _ml_defer("faq"),
        ]
        gates = [
            _gate_decision("t1", "mixed_intent", "defer"),
            _gate_decision("t2", "mixed_intent", "defer"),
            _gate_decision("t3", "simple_symptom", "allow_ml"),
            _gate_decision("t4", "simple_faq", "allow_ml"),
        ]
        df = _build_complexity_breakdown(gold, decisions, gates)

        required = {
            "primary_tag", "n", "n_accept", "n_defer_complexity",
            "n_defer_tag_policy", "n_defer_ml", "accepted_accuracy",
            "accepted_recall_anamnesis",
        }
        assert required <= set(df.columns)

        mixed_row = df[df["primary_tag"] == "mixed_intent"].iloc[0]
        assert int(mixed_row["n"]) == 2
        assert int(mixed_row["n_defer_complexity"]) == 2
        assert int(mixed_row["n_accept"]) == 0
        assert int(mixed_row["n_defer_ml"]) == 0

        simple_sym = df[df["primary_tag"] == "simple_symptom"].iloc[0]
        assert int(simple_sym["n"]) == 1
        assert int(simple_sym["n_accept"]) == 1
        assert simple_sym["accepted_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Subtype breakdown (`mixed_intent` includes `symptom_price` subtype в active_tags)
# ---------------------------------------------------------------------------

class TestSubtypeBreakdown:
    def test_subtype_counts_independent(self) -> None:
        """`symptom_price` row считает все samples где этот tag активен,
        независимо от primary_tag."""
        from d1.scripts.evaluate_simple_router import _build_subtype_breakdown

        gold = ["anamnesis", "anamnesis", "faq"]
        decisions = [
            _complexity_defer("mixed_intent"),
            _complexity_defer("mixed_intent"),
            _accept("faq"),
        ]
        gates = [
            _gate_decision("t1", "mixed_intent", "defer"),  # symptom_price=True
            _gate_decision("t2", "mixed_intent", "defer"),  # symptom_price=True
            _gate_decision("t3", "simple_faq", "allow_ml"),
        ]
        df = _build_subtype_breakdown(gold, decisions, gates)

        sp = df[df["tag"] == "symptom_price"]
        assert not sp.empty
        assert int(sp.iloc[0]["n"]) == 2
        assert int(sp.iloc[0]["n_defer_complexity"]) == 2


# ---------------------------------------------------------------------------
# Constraint tests (architectural invariants)
# ---------------------------------------------------------------------------

def test_no_llm_imports_in_evaluate_simple_router_source() -> None:
    src = Path("d1/scripts/evaluate_simple_router.py").read_text(encoding="utf-8")
    forbidden = ["OpenRouterClient", "openai", "together", "litellm", "chat.completions"]
    for needle in forbidden:
        assert needle not in src, f"Запрещённый паттерн: {needle}"


def test_no_fit_calls_in_evaluate_simple_router_source() -> None:
    """Скрипт оркестрирует, не обучает (использует train_bundle с use_cache)."""
    src = Path("d1/scripts/evaluate_simple_router.py").read_text(encoding="utf-8")
    # Само упоминание `.fit(` (вне docstrings) не допускается.
    # Делаем простую проверку: нет литеральных `.fit(`.
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    code_text = "\n".join(code_lines)
    assert ".fit(" not in code_text, "evaluate_simple_router не должен обучать модели"
