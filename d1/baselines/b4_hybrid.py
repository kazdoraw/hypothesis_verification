"""B4 Hybrid Router — rules-first cascade over SelectiveRouter (Task 5 roadmap).

Архитектура каскада:
    1. B0 rules → если (domain, hit_count) в whitelist → rule-accept.
    2. Иначе → SelectiveRouter (sparse primary + dense second opinion + policy).

Принципиальные решения (из плана Task 5):

- **Rule confidence НЕ probability.** `B0RulesClassifier` возвращает
  `confidence = min(0.5 + 0.1 * hit_count, 0.95)` — это эвристический score,
  НЕ калиброванная вероятность. Использовать `rule_confidence >= X`
  как threshold — архитектурная ошибка.

- **Whitelist вместо universal threshold.** `RULE_ACCEPT_POLICY` задаёт
  `min_hit_count` per-domain. `faq` и `unsupported` **никогда** не
  rule-accepted (safety-критический контракт, юнит-тест):
    * faq: самый ambiguous домен, всегда ML arbitration
    * unsupported: ML может дать лучший sparse/dense signal

- **Output unification.** B4 возвращает `RouteDecision` (тот же тип,
  что и `SelectiveRouter`). `reason` для rule-accept имеет формат
  `"rule:{domain}:hits={N}"`, что отличает его от ML-reason'ов.

- **SSoT обучения.** B4HybridRouter НЕ создаёт baseline'ы — принимает
  `TrainedBundle` (из Task 0) и готовый `SelectiveRouter` (из Task 4).
"""

from __future__ import annotations

from typing import Any

from d1.baselines.b0_rules import RulePrediction
from d1.baselines.selective_router import (
    RouteDecision,
    SelectiveRouter,
)

# ---------------------------------------------------------------------------
# RULE_ACCEPT_POLICY — whitelist per-domain
# ---------------------------------------------------------------------------

# `min_hit_count = None` означает "НИКОГДА не rule-accept" (всегда ML fallback).
# Изменение этих значений напрямую влияет на safety/coverage trade-off и должно
# быть обосновано через evaluation на val (а не hard-code).
RULE_ACCEPT_POLICY: dict[str, dict[str, Any]] = {
    "anamnesis": {
        "min_hit_count": 2,
        "rationale": (
            "Clinical symptoms: false negative дороже false positive. "
            "При 2+ anamnesis-паттернах сигнал достаточно сильный для "
            "safety-conservative accept без ML arbitration."
        ),
    },
    "booking": {
        "min_hit_count": 3,
        "rationale": (
            "Booking legal-sensitive: не хотим false positive (случайная "
            "запись). Более строгий порог чем anamnesis."
        ),
    },
    "faq": {
        "min_hit_count": None,
        "rationale": (
            "FAQ — самый ambiguous домен, часто пересекается с anamnesis "
            "и booking. Всегда ML arbitration, rule-accept запрещён."
        ),
    },
    "unsupported": {
        "min_hit_count": None,
        "rationale": (
            "Unsupported — rules fallback-класс. ML может дать лучший "
            "sparse/dense signal, поэтому всегда идём в arbitration."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hit_count(matched_rule: str) -> int:
    """Parse 'N_patterns' → N, 'no_match' → 0.

    Raises:
        ValueError: формат строки не распознан (защита от стохастики).
    """
    if matched_rule == "no_match":
        return 0
    if "_" not in matched_rule:
        raise ValueError(f"Unexpected matched_rule format: {matched_rule!r}")
    prefix = matched_rule.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse hit_count from {matched_rule!r}",
        ) from exc


def _rule_accept_decision(rp: RulePrediction, hit_count: int) -> RouteDecision:
    """Сборка RouteDecision для rule-accept кейса.

    `confidence` из RulePrediction сохраняется для трассировки, но НЕ
    используется как probability. `margin=0.0` (N/A для rule-based).
    `sparse_dense_agree=True` (N/A — rule byassed ML arbitration).
    """
    return RouteDecision(
        label=rp.route_domain,
        confidence=rp.confidence,
        margin=0.0,
        action="accept",
        sparse_dense_agree=True,
        reason=f"rule:{rp.route_domain}:hits={hit_count}",
        dense_label=None,
    )


# ---------------------------------------------------------------------------
# B4HybridRouter
# ---------------------------------------------------------------------------

class B4HybridRouter:
    """Rules-first cascade над SelectiveRouter.

    Не обучает модели — принимает fitted `TrainedBundle` и `SelectiveRouter`.
    """

    def __init__(
        self,
        bundle: Any,                                  # TrainedBundle (duck-typed)
        selective: SelectiveRouter,
        rule_policy: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.rules = bundle.get("B0_rules")
        if not hasattr(self.rules, "predict_with_confidence"):
            raise TypeError(
                "bundle.get('B0_rules') должен иметь predict_with_confidence",
            )
        self.selective = selective
        self.rule_policy = rule_policy or RULE_ACCEPT_POLICY

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        """Batch routing через rules-first cascade.

        1. Для всех texts сразу получаем rule predictions (дёшево).
        2. Делим на rule-accept (определяется whitelist'ом) и ML-fallback.
        3. ML-fallback прогоняется через SelectiveRouter одним batch'ем.
        4. Смешиваем результаты в исходном порядке.
        """
        if not texts:
            return []

        rule_preds = self.rules.predict_with_confidence(texts)

        # Pre-sort: индексы rule-accept vs ML-fallback.
        rule_accepted: dict[int, RouteDecision] = {}
        ml_fallback_indices: list[int] = []

        for i, rp in enumerate(rule_preds):
            policy = self.rule_policy.get(rp.route_domain, {})
            min_hits = policy.get("min_hit_count")

            if min_hits is None:
                # Домен в deny-list (faq/unsupported) → всегда ML
                ml_fallback_indices.append(i)
                continue

            hit_count = _parse_hit_count(rp.matched_rule)
            if hit_count >= min_hits:
                rule_accepted[i] = _rule_accept_decision(rp, hit_count)
            else:
                ml_fallback_indices.append(i)

        # ML-fallback одним batch'ем (экономия encoder прогонов для B2.1).
        ml_decisions: list[RouteDecision] = []
        if ml_fallback_indices:
            ml_texts = [texts[i] for i in ml_fallback_indices]
            ml_decisions = self.selective.route_batch(ml_texts)

        # Сборка в исходном порядке.
        out: list[RouteDecision] = []
        ml_iter = iter(zip(ml_fallback_indices, ml_decisions))
        next_ml = next(ml_iter, None)
        for i in range(len(texts)):
            if i in rule_accepted:
                out.append(rule_accepted[i])
            else:
                assert next_ml is not None and next_ml[0] == i, (
                    f"ml_iter mismatch at i={i}: next_ml={next_ml}"
                )
                out.append(next_ml[1])
                next_ml = next(ml_iter, None)
        return out


__all__ = [
    "B4HybridRouter",
    "RULE_ACCEPT_POLICY",
    "_parse_hit_count",
]
