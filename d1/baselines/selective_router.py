"""SelectiveRouter — policy поверх fitted baseline'ов (Task 4 roadmap).

Это **не baseline** (B3 зарезервировано за LLM-router), а routing policy,
работающая поверх уже обученных моделей из `TrainedBundle`.

Принципиальные решения:
- Роутер **не обучает** модели — принимает fitted sparse (B1.1) + dense (B2.1).
- Пороги — через `SelectiveThresholds` (frozen dataclass), не хардкод.
- Два отдельных report generator'а:
    * `compute_accepted_only_report` — метрики на accepted subset
      (defer **исключён**, НЕ маппится в anamnesis). Это НЕ классический
      closed-set benchmark по полному eval set — это accuracy/F1 только
      на тех примерах, которые роутер принял. Сравнение с closed-set
      baseline'ами (B0/B1/B2) требует осторожности: у них n_samples = |eval|,
      а здесь n_samples = |accepted| < |eval|.
    * `compute_selective_report` — selective metrics (coverage, defer_rate,
      false_negative_deferred, safety_on_accepted).

Policy rules (приоритет сверху вниз):
    R1  top1 == anamnesis  и  conf >= anamnesis_threshold      → accept
    R2  top1 == faq  и  anamnesis в top2  и  margin<faq_anamnesis_margin
                                                                → defer (safety-conservative)
    R3  sparse != dense                                         → defer
    R4  conf >= general_threshold                               → accept
    R5  всё остальное                                           → defer (low_confidence)

Начальные пороги подбираются на val через Pareto candidates из Task 3,
не автоматически (human-in-the-loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from d1.baselines.eval_metrics import (
    LABEL_ORDER,
    RoutingReport,
    compute_all_metrics,
)

Action = Literal["accept", "defer"]

# Причины (reason) — строковые константы, чтобы аналитика могла группировать.
REASON_ANAMNESIS_CONFIDENT = "anamnesis_confident"
REASON_FAQ_ANAMNESIS_BORDERLINE = "faq_anamnesis_borderline"
REASON_SPARSE_DENSE_DISAGREE = "sparse_dense_disagree"
REASON_GENERAL_CONFIDENT = "general_confident"
REASON_LOW_CONFIDENCE = "low_confidence"


# ---------------------------------------------------------------------------
# Config + decision types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectiveThresholds:
    """Пороги selective policy.

    Дефолты — стартовые (из плана Task 4), safety-conservative. Production
    значения см. `PRODUCTION_THRESHOLDS` ниже (retune по val Pareto, Task 5
    post-eval).
    """

    anamnesis_threshold: float = 0.55       # ниже, чтобы не терять clinical
    faq_anamnesis_margin: float = 0.15      # safety guard против faq-override
    general_threshold: float = 0.70         # confident accept для остальных


# SSoT для production thresholds после retune на val (Task 5, threshold_sweep).
# Обоснование (val/test): recall_anamnesis=1.000 без потерь, coverage +9-10pp
# vs дефолтов, FN_deferred: val 38→24, test 80→62. На hard_test: hybrid
# coverage 0.60→0.65, recall_anam 0.84→0.85. Aggressive config (0.43/0.55)
# отклонён: recall_anam на test падает до 0.994.
PRODUCTION_THRESHOLDS = SelectiveThresholds(
    anamnesis_threshold=0.48,
    faq_anamnesis_margin=0.15,
    general_threshold=0.63,
)


@dataclass
class RouteDecision:
    """Итоговое решение роутера по одному тексту."""

    label: str                               # predicted domain (sparse top1)
    confidence: float                        # sparse top1 probability
    margin: float                            # sparse top1 - top2
    action: Action
    sparse_dense_agree: bool
    reason: str                              # human-readable rule name
    dense_label: str | None = None           # для аналитики disagreement


# ---------------------------------------------------------------------------
# SelectiveRouter
# ---------------------------------------------------------------------------

class SelectiveRouter:
    """Policy wrapper над fitted sparse+dense моделями.

    Контракт:
    - Принимает **уже обученные** модели (имеют `predict_proba` и `classes_`).
    - НЕ вызывает `.fit()` и не имеет mutation семантики.
    - Решения зависят только от proba и thresholds — детерминированы.
    """

    def __init__(
        self,
        sparse_model: Any,
        dense_model: Any,
        thresholds: SelectiveThresholds | None = None,
    ) -> None:
        self._validate_fitted("sparse", sparse_model)
        self._validate_fitted("dense", dense_model)

        sparse_classes = list(sparse_model.classes_)
        dense_classes = list(dense_model.classes_)
        if sparse_classes != dense_classes:
            raise ValueError(
                f"sparse.classes_ != dense.classes_: "
                f"sparse={sparse_classes}, dense={dense_classes}",
            )
        self.sparse = sparse_model
        self.dense = dense_model
        self.classes = sparse_classes
        self.thresholds = thresholds or SelectiveThresholds()

    @staticmethod
    def _validate_fitted(tag: str, model: Any) -> None:
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{tag}_model must expose predict_proba")
        if not hasattr(model, "classes_"):
            raise TypeError(f"{tag}_model must expose classes_")

    def route_batch(self, texts: list[str]) -> list[RouteDecision]:
        """Batch prediction + policy."""
        if not texts:
            return []

        sparse_proba = np.asarray(self.sparse.predict_proba(texts), dtype=float)
        dense_proba = np.asarray(self.dense.predict_proba(texts), dtype=float)

        decisions: list[RouteDecision] = []
        for i in range(len(texts)):
            decisions.append(self._decide_one(sparse_proba[i], dense_proba[i]))
        return decisions

    def _decide_one(
        self, sparse_row: np.ndarray, dense_row: np.ndarray,
    ) -> RouteDecision:
        """Применение policy к одному примеру."""
        # Sparse top1/top2 + margin
        sparse_sorted_idx = np.argsort(sparse_row)[::-1]
        s_top1_idx = int(sparse_sorted_idx[0])
        s_top2_idx = int(sparse_sorted_idx[1])
        s_top1_label = self.classes[s_top1_idx]
        s_top2_label = self.classes[s_top2_idx]
        s_top1_conf = float(sparse_row[s_top1_idx])
        s_top2_conf = float(sparse_row[s_top2_idx])
        margin = s_top1_conf - s_top2_conf

        # Dense top1
        d_top1_idx = int(np.argmax(dense_row))
        d_top1_label = self.classes[d_top1_idx]
        agree = s_top1_label == d_top1_label

        # --- R1: anamnesis confident accept ---
        if (
            s_top1_label == "anamnesis"
            and s_top1_conf >= self.thresholds.anamnesis_threshold
        ):
            return RouteDecision(
                label=s_top1_label, confidence=s_top1_conf, margin=margin,
                action="accept", sparse_dense_agree=agree,
                reason=REASON_ANAMNESIS_CONFIDENT,
                dense_label=d_top1_label,
            )

        # --- R2: faq с anamnesis в top2, узкий margin → defer (safety) ---
        if (
            s_top1_label == "faq"
            and s_top2_label == "anamnesis"
            and margin < self.thresholds.faq_anamnesis_margin
        ):
            return RouteDecision(
                label=s_top1_label, confidence=s_top1_conf, margin=margin,
                action="defer", sparse_dense_agree=agree,
                reason=REASON_FAQ_ANAMNESIS_BORDERLINE,
                dense_label=d_top1_label,
            )

        # --- R3: sparse/dense disagree → defer ---
        if not agree:
            return RouteDecision(
                label=s_top1_label, confidence=s_top1_conf, margin=margin,
                action="defer", sparse_dense_agree=False,
                reason=REASON_SPARSE_DENSE_DISAGREE,
                dense_label=d_top1_label,
            )

        # --- R4: general confident accept ---
        if s_top1_conf >= self.thresholds.general_threshold:
            return RouteDecision(
                label=s_top1_label, confidence=s_top1_conf, margin=margin,
                action="accept", sparse_dense_agree=True,
                reason=REASON_GENERAL_CONFIDENT,
                dense_label=d_top1_label,
            )

        # --- R5: fallback defer ---
        return RouteDecision(
            label=s_top1_label, confidence=s_top1_conf, margin=margin,
            action="defer", sparse_dense_agree=agree,
            reason=REASON_LOW_CONFIDENCE,
            dense_label=d_top1_label,
        )


# ---------------------------------------------------------------------------
# Reports (Task 4: два режима eval)
# ---------------------------------------------------------------------------

@dataclass
class SelectiveReport:
    """Selective routing metrics.

    ВАЖНО: defer **не маппится** на anamnesis в этом отчёте — это был бы
    искусственный bump safety. `false_negative_deferred` просто считает,
    сколько urgent/anamnesis ушло в defer (для human-review).
    """

    router_name: str
    thresholds: dict[str, float]
    n_samples: int
    coverage: float
    accepted_accuracy: float
    accepted_recall_anamnesis: float
    defer_rate: float
    false_negative_deferred: int
    safety_on_accepted: float | None = None
    reasons_breakdown: dict[str, int] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "router_name": self.router_name,
            "thresholds": self.thresholds,
            "n_samples": self.n_samples,
            "coverage": round(self.coverage, 4),
            "accepted_accuracy": round(self.accepted_accuracy, 4),
            "accepted_recall_anamnesis": round(self.accepted_recall_anamnesis, 4),
            "defer_rate": round(self.defer_rate, 4),
            "false_negative_deferred": self.false_negative_deferred,
            "safety_on_accepted": (
                round(self.safety_on_accepted, 4)
                if self.safety_on_accepted is not None else None
            ),
            "reasons_breakdown": dict(self.reasons_breakdown),
        }


def compute_accepted_only_report(
    y_true: list[str],
    decisions: list[RouteDecision],
    baseline_name: str = "selective_accepted_only",
) -> RoutingReport:
    """Метрики на accepted subset. Defer **исключён** (не маппится в anamnesis).

    ВАЖНО: это НЕ classical closed-set benchmark. `n_samples` = число accepted,
    НЕ число примеров в eval set. Сравнение с closed-set baseline'ами (B0/B1/B2,
    где n_samples = |eval|) требует явного указания coverage из selective report.
    """
    if len(y_true) != len(decisions):
        raise ValueError(
            f"len(y_true)={len(y_true)} != len(decisions)={len(decisions)}",
        )
    accepted_true = [t for t, d in zip(y_true, decisions) if d.action == "accept"]
    accepted_pred = [d.label for d in decisions if d.action == "accept"]
    return compute_all_metrics(
        y_true=accepted_true, y_pred=accepted_pred, baseline_name=baseline_name,
    )


def compute_selective_report(
    y_true: list[str],
    decisions: list[RouteDecision],
    router_name: str,
    thresholds: SelectiveThresholds | None = None,
    urgent_flags: list[bool] | None = None,
) -> SelectiveReport:
    """Selective metrics + safety guard.

    Args:
        y_true: gold labels.
        decisions: выходы `SelectiveRouter.route_batch`.
        router_name: human-readable идентификатор роутера.
        thresholds: для сохранения в отчёте (иначе пусто).
        urgent_flags: опционально, per-sample флаг urgent (для safety_on_accepted).
            Если не передан, `safety_on_accepted = None`.

    Returns:
        `SelectiveReport` — defer НЕ маппится на anamnesis.
    """
    n = len(y_true)
    if n != len(decisions):
        raise ValueError(
            f"len(y_true)={n} != len(decisions)={len(decisions)}",
        )
    if urgent_flags is not None and len(urgent_flags) != n:
        raise ValueError(
            f"len(urgent_flags)={len(urgent_flags)} != n={n}",
        )

    n_accept = sum(1 for d in decisions if d.action == "accept")
    n_defer = n - n_accept

    # accepted accuracy
    if n_accept == 0:
        accepted_accuracy = 0.0
        accepted_recall_anamnesis = 0.0
    else:
        correct = sum(
            1 for t, d in zip(y_true, decisions)
            if d.action == "accept" and t == d.label
        )
        accepted_accuracy = correct / n_accept

        anamnesis_accepted_total = sum(
            1 for t, d in zip(y_true, decisions)
            if d.action == "accept" and t == "anamnesis"
        )
        anamnesis_correct = sum(
            1 for t, d in zip(y_true, decisions)
            if d.action == "accept" and t == "anamnesis" and d.label == "anamnesis"
        )
        accepted_recall_anamnesis = (
            anamnesis_correct / anamnesis_accepted_total
            if anamnesis_accepted_total > 0 else 0.0
        )

    # false_negative_deferred: urgent/anamnesis → defer
    false_negative_deferred = sum(
        1 for t, d in zip(y_true, decisions)
        if d.action == "defer" and t == "anamnesis"
    )

    # safety_on_accepted: recall_urgent среди accepted (если есть urgent_flags)
    safety_on_accepted: float | None = None
    if urgent_flags is not None:
        accepted_urgent_total = sum(
            1 for u, d in zip(urgent_flags, decisions)
            if d.action == "accept" and u
        )
        accepted_urgent_correct = sum(
            1 for t, u, d in zip(y_true, urgent_flags, decisions)
            if d.action == "accept" and u and t == d.label
        )
        safety_on_accepted = (
            accepted_urgent_correct / accepted_urgent_total
            if accepted_urgent_total > 0 else 0.0
        )

    # Reasons breakdown
    reasons_breakdown: dict[str, int] = {}
    for d in decisions:
        reasons_breakdown[d.reason] = reasons_breakdown.get(d.reason, 0) + 1

    th_dict = (
        {
            "anamnesis_threshold": thresholds.anamnesis_threshold,
            "faq_anamnesis_margin": thresholds.faq_anamnesis_margin,
            "general_threshold": thresholds.general_threshold,
        }
        if thresholds is not None
        else {}
    )

    return SelectiveReport(
        router_name=router_name,
        thresholds=th_dict,
        n_samples=n,
        coverage=n_accept / n if n > 0 else 0.0,
        accepted_accuracy=accepted_accuracy,
        accepted_recall_anamnesis=accepted_recall_anamnesis,
        defer_rate=n_defer / n if n > 0 else 0.0,
        false_negative_deferred=false_negative_deferred,
        safety_on_accepted=safety_on_accepted,
        reasons_breakdown=reasons_breakdown,
    )


# Deprecated alias: сохраняется до Task 7 для backward compatibility.
# Новый код должен использовать compute_accepted_only_report.
compute_closed_set_report = compute_accepted_only_report


__all__ = [
    "LABEL_ORDER",
    "PRODUCTION_THRESHOLDS",
    "REASON_ANAMNESIS_CONFIDENT",
    "REASON_FAQ_ANAMNESIS_BORDERLINE",
    "REASON_GENERAL_CONFIDENT",
    "REASON_LOW_CONFIDENCE",
    "REASON_SPARSE_DENSE_DISAGREE",
    "RouteDecision",
    "SelectiveReport",
    "SelectiveRouter",
    "SelectiveThresholds",
    "compute_accepted_only_report",
    "compute_selective_report",
    # Deprecated alias для backward compatibility (будет удалён после Task 7).
    "compute_closed_set_report",
]
