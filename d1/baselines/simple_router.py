"""SimpleRouter: ComplexityGate → B4HybridRouter cascade (Task 4 плана).

Архитектура:
    text → ComplexityGate.decide
        if action == "defer":   RouteDecision(complexity-defer payload)
        if action == "allow_ml": B4HybridRouter.route_batch([text])[0]

Назначение: safety wrapper над B4 для simple-only ML routing. Complex /
multi-intent / short_ambiguous идут в `defer` сразу, минуя ML cascade.

**Defer-payload контракт (зафиксирован в плане Task 4):**
    label="", confidence=0.0, margin=0.0, action="defer",
    sparse_dense_agree=False, reason=f"complexity:{primary_tag}",
    dense_label=None.

`label=""` — намеренно: при `action=="defer"` `label` не используется
downstream'ом (см. `compute_selective_report`, `_attach_decision_trace`).
Установка `label="anamnesis"` была бы искусственным safety bump.

**Не делает:**
- Не обучает модели (SSoT обучения — `train_bundle`).
- Не импортирует LLM-клиенты (defer = abstain outcome, не handoff).
- Не маппит defer на anamnesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from d1.baselines.b4_hybrid import B4HybridRouter
from d1.baselines.complexity_gate import ComplexityGate
from d1.baselines.selective_router import RouteDecision


@dataclass(frozen=True)
class TagPolicy:
    """Ограничение accept для отдельных simple-тэгов.

    Это production-conservative слой: ComplexityGate может разрешить ML,
    но конкретный простой тип запроса принимается только при достаточной
    уверенности и ожидаемом домене. Так мы не учим mixed-intent, а сужаем
    зону ответственности ML до доказанно надёжных простых сообщений.
    """

    min_confidence: float = 0.0
    allowed_labels: tuple[str, ...] | None = None


DEFAULT_TAG_POLICIES: dict[str, TagPolicy] = {
    # Phase 2: pin'нутые после tag-policy sweep (Exit Criteria).
    # Selection rule (val): min(threshold) ∈ [0.55..0.95] с
    #     accepted_acc(val, tag) ≥ 0.95 AND accepted_recall_anam(val) ≥ 0.98.
    # Конкретные числа выбраны как middle-ground между bootstrap-min (0.55)
    # и предыдущим production (0.90) — для сохранения hard_test recall_anam.
    # SSoT для sweep артефактов: d1/results/tag_policy_pareto_candidates.csv.
    "simple_faq": TagPolicy(min_confidence=0.75, allowed_labels=("faq",)),
    "simple_booking": TagPolicy(min_confidence=0.75, allowed_labels=("booking",)),
    "simple_symptom": TagPolicy(min_confidence=0.70, allowed_labels=("anamnesis",)),
}


class SimpleRouter:
    """Safety wrapper над B4HybridRouter с pre-filter через ComplexityGate.

    Args:
        hybrid: уже сконфигурированный `B4HybridRouter` (содержит fitted bundle
            и SelectiveRouter). SimpleRouter не нуждается в bundle напрямую,
            потому что вся ML-маршрутизация делегируется hybrid.
        complexity_gate: опциональный gate. По умолчанию — `ComplexityGate()`
            с дефолтными маркерами (SSoT: `complexity_gate.py`).
    """

    def __init__(
        self,
        hybrid: B4HybridRouter,
        complexity_gate: ComplexityGate | None = None,
        tag_policies: dict[str, TagPolicy] | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.gate = complexity_gate or ComplexityGate()
        self.tag_policies = tag_policies or DEFAULT_TAG_POLICIES

    def route(self, text: str) -> RouteDecision:
        """Один текст → RouteDecision. Эквивалент `route_batch([text])[0]`."""
        return self.route_batch([text])[0]

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        """Batch routing.

        Алгоритм:
        1. `gate.decide_batch(texts)` — дешёвая rule-based классификация.
        2. Делим индексы на `defer_idx` (gate сказал defer) и `allow_idx`.
        3. allow_idx прогоняются через `hybrid.route_batch` ОДНИМ батчем —
           критично для encoder cost (B2.1 — bge-m3).
        4. Сборка результатов в исходном порядке.

        Returns:
            list `RouteDecision` в порядке `texts`.
        """
        if not texts:
            return []

        gate_decisions = self.gate.decide_batch(texts)
        defer_idx: list[int] = []
        allow_idx: list[int] = []
        for i, gd in enumerate(gate_decisions):
            if gd.action == "defer":
                defer_idx.append(i)
            else:
                allow_idx.append(i)

        # ML batch — только при наличии allow'ов (экономия encoder проходов).
        ml_results: list[RouteDecision] = []
        if allow_idx:
            ml_texts = [texts[i] for i in allow_idx]
            ml_results = self.hybrid.route_batch(ml_texts)

        # Сборка outputs в исходном порядке.
        # Pre-allocate list — используем None как sentinel; type ignore оправдан,
        # так как все позиции будут заполнены до return.
        out: list[RouteDecision] = [None] * len(texts)  # type: ignore[list-item]
        for i in defer_idx:
            primary_tag = gate_decisions[i].primary_tag
            out[i] = _make_complexity_defer_decision(primary_tag)
        for j, i in enumerate(allow_idx):
            out[i] = _apply_tag_policy(
                primary_tag=gate_decisions[i].primary_tag,
                decision=ml_results[j],
                policies=self.tag_policies,
            )
        return out


def _make_complexity_defer_decision(primary_tag: str) -> RouteDecision:
    """Зафиксированный defer-payload для complexity-defer.

    См. plan Task 4: контракт обязательный для совместимости с downstream
    consumers (`compute_selective_report`, `_attach_decision_trace`,
    `simple_router_decisions_*.csv`).
    """
    return RouteDecision(
        label="",
        confidence=0.0,
        margin=0.0,
        action="defer",
        sparse_dense_agree=False,
        reason=f"complexity:{primary_tag}",
        dense_label=None,
    )


def _apply_tag_policy(
    primary_tag: str,
    decision: RouteDecision,
    policies: dict[str, TagPolicy],
) -> RouteDecision:
    """Post-ML guard для слабых simple-тэгов.

    Если B4 уже вернул defer — сохраняем его без изменений. Если B4 принял
    решение, но tag-policy считает его недостаточно надёжным, возвращаем
    abstain с отдельной причиной `tag_policy:<tag>`.
    """
    if decision.action != "accept":
        return decision

    policy = policies.get(primary_tag)
    if policy is None:
        return decision

    label_allowed = (
        policy.allowed_labels is None
        or decision.label in policy.allowed_labels
    )
    confidence_allowed = decision.confidence >= policy.min_confidence
    if label_allowed and confidence_allowed:
        return decision

    return RouteDecision(
        label="",
        confidence=decision.confidence,
        margin=decision.margin,
        action="defer",
        sparse_dense_agree=decision.sparse_dense_agree,
        reason=f"tag_policy:{primary_tag}",
        dense_label=decision.dense_label,
    )


__all__ = [
    "DEFAULT_TAG_POLICIES",
    "SimpleRouter",
    "TagPolicy",
]
