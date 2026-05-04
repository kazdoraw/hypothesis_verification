"""Единый модуль метрик для D1 v6 baselines.

Покрывает §10 ТЗ:
- §10.1 Routing quality: accuracy, macro-F1, balanced accuracy, per-class P/R/F1
- §10.2 Safety: recall(anamnesis), recall(urgent), false-faq-for-anamnesis rate
- §10.4 Operational: latency per request

Использование:
    from d1.baselines.eval_metrics import compute_all_metrics
    report = compute_all_metrics(y_true, y_pred, latency_ms=5.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABEL_ORDER = ["anamnesis", "faq", "booking", "unsupported"]


@dataclass
class RoutingReport:
    """Структурированный отчёт по метрикам одного baseline."""

    baseline_name: str
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    # Safety (§10.2)
    recall_anamnesis: float = 0.0
    false_faq_for_anamnesis: float = 0.0
    recall_urgent: float | None = None
    # Operational (§10.4)
    latency_ms: float | None = None
    n_samples: int = 0

    def summary_dict(self) -> dict[str, Any]:
        """Плоский dict для сводной таблицы."""
        d: dict[str, Any] = {
            "baseline": self.baseline_name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "recall_anamnesis": round(self.recall_anamnesis, 4),
            "false_faq_for_anamnesis": round(self.false_faq_for_anamnesis, 4),
            "n_samples": self.n_samples,
        }
        if self.recall_urgent is not None:
            d["recall_urgent"] = round(self.recall_urgent, 4)
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        for cls in LABEL_ORDER:
            if cls in self.per_class:
                for metric_name in ("precision", "recall", "f1"):
                    key = f"{cls}_{metric_name}"
                    d[key] = round(self.per_class[cls].get(metric_name, 0.0), 4)
        return d


def _compute_safety_metrics(
    y_true: list[str],
    y_pred: list[str],
    urgency: list[str] | None = None,
) -> dict[str, float]:
    """§10.2 Safety metrics."""
    true_arr = np.array(y_true)
    pred_arr = np.array(y_pred)

    # recall(anamnesis)
    anam_mask = true_arr == "anamnesis"
    recall_anam = 0.0
    if anam_mask.sum() > 0:
        recall_anam = float((pred_arr[anam_mask] == "anamnesis").mean())

    # false faq for anamnesis: P(pred=faq | true=anamnesis)
    false_faq = 0.0
    if anam_mask.sum() > 0:
        false_faq = float((pred_arr[anam_mask] == "faq").mean())

    result = {
        "recall_anamnesis": recall_anam,
        "false_faq_for_anamnesis": false_faq,
    }

    # recall(urgent/emergency)
    if urgency is not None:
        urg_arr = np.array(urgency)
        urg_mask = np.isin(urg_arr, ["urgent", "emergency", "high"])
        if urg_mask.sum() > 0:
            # urgent cases должны попадать в anamnesis
            result["recall_urgent"] = float(
                (pred_arr[urg_mask] == "anamnesis").mean()
            )

    return result


def compute_all_metrics(
    y_true: list[str],
    y_pred: list[str],
    baseline_name: str = "unknown",
    urgency: list[str] | None = None,
    latency_ms: float | None = None,
) -> RoutingReport:
    """Вычисление полного набора метрик §10 ТЗ.

    Args:
        y_true: истинные метки route_domain
        y_pred: предсказанные метки route_domain
        baseline_name: имя модели
        urgency: список urgency для safety metrics
        latency_ms: средняя задержка на запрос

    Returns:
        RoutingReport со всеми метриками
    """
    labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Per-class
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Safety
    safety = _compute_safety_metrics(y_true, y_pred, urgency)

    return RoutingReport(
        baseline_name=baseline_name,
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        balanced_accuracy=float(bal_acc),
        per_class=per_class,
        confusion=cm,
        recall_anamnesis=safety["recall_anamnesis"],
        false_faq_for_anamnesis=safety["false_faq_for_anamnesis"],
        recall_urgent=safety.get("recall_urgent"),
        latency_ms=latency_ms,
        n_samples=len(y_true),
    )


# ---------------------------------------------------------------------------
# Safety-specific report (§10.2)
# ---------------------------------------------------------------------------

@dataclass
class SafetyReport:
    """Отчёт только по safety-метрикам (§10.2).

    Для safety_set macro-F1 и balanced_accuracy вводят в заблуждение,
    т.к. набор почти одно-классовый (anamnesis dominant).
    """

    baseline_name: str
    n_samples: int
    n_urgent: int
    recall_anamnesis: float
    recall_urgent: float
    false_faq_for_anamnesis: float
    false_negative_urgent: int  # абсолютное число пропущенных urgent
    misrouted_to: dict[str, int]  # куда ушли misrouted urgent cases
    latency_ms: float | None = None

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "baseline": self.baseline_name,
            "n_samples": self.n_samples,
            "n_urgent": self.n_urgent,
            "recall_anamnesis": round(self.recall_anamnesis, 4),
            "recall_urgent": round(self.recall_urgent, 4),
            "false_faq_for_anamnesis": round(self.false_faq_for_anamnesis, 4),
            "false_negative_urgent": self.false_negative_urgent,
            "misrouted_to": self.misrouted_to,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        return d


def compute_safety_report(
    y_true: list[str],
    y_pred: list[str],
    urgency: list[str],
    baseline_name: str = "unknown",
    latency_ms: float | None = None,
) -> SafetyReport:
    """Safety-only метрики для safety_set.

    Не считает macro-F1 / balanced_accuracy — они бессмысленны
    на почти одно-классовом наборе.
    """
    true_arr = np.array(y_true)
    pred_arr = np.array(y_pred)
    urg_arr = np.array(urgency)

    # recall(anamnesis)
    anam_mask = true_arr == "anamnesis"
    recall_anam = float((pred_arr[anam_mask] == "anamnesis").mean()) if anam_mask.sum() > 0 else 0.0

    # false faq for anamnesis
    false_faq = float((pred_arr[anam_mask] == "faq").mean()) if anam_mask.sum() > 0 else 0.0

    # urgent/emergency recall
    urg_mask = np.isin(urg_arr, ["urgent", "emergency", "high"])
    n_urgent = int(urg_mask.sum())
    if n_urgent > 0:
        correct_urgent = pred_arr[urg_mask] == "anamnesis"
        recall_urg = float(correct_urgent.mean())
        fn_urgent = int((~correct_urgent).sum())
        # куда ушли misrouted
        misrouted_preds = pred_arr[urg_mask & (pred_arr != "anamnesis")]
        misrouted_to: dict[str, int] = {}
        for p in misrouted_preds:
            key = str(p)  # np.str_ → str для JSON serialization
            misrouted_to[key] = misrouted_to.get(key, 0) + 1
    else:
        recall_urg = 0.0
        fn_urgent = 0
        misrouted_to = {}

    return SafetyReport(
        baseline_name=baseline_name,
        n_samples=len(y_true),
        n_urgent=n_urgent,
        recall_anamnesis=recall_anam,
        recall_urgent=recall_urg,
        false_faq_for_anamnesis=false_faq,
        false_negative_urgent=fn_urgent,
        misrouted_to=misrouted_to,
        latency_ms=latency_ms,
    )


def print_safety_report(report: SafetyReport) -> None:
    """Вывод safety report."""
    print(f"\n{'='*60}")
    print(f"  {report.baseline_name} [SAFETY]")
    print(f"{'='*60}")
    print(f"  Samples:                  {report.n_samples}")
    print(f"  Urgent cases:             {report.n_urgent}")
    print(f"  recall(anamnesis):        {report.recall_anamnesis:.4f}")
    print(f"  recall(urgent/emergency): {report.recall_urgent:.4f}")
    print(f"  false_faq_for_anamnesis:  {report.false_faq_for_anamnesis:.4f}")
    print(f"  false_negative_urgent:    {report.false_negative_urgent}")
    if report.misrouted_to:
        print(f"  misrouted urgent →        {report.misrouted_to}")
    if report.latency_ms is not None:
        print(f"  latency:                  {report.latency_ms:.2f} ms/req")
    print()


# ---------------------------------------------------------------------------
# Switch stress report (§2 roadmap) — text-only stress test, НЕ switch detector
# ---------------------------------------------------------------------------

# Константа-предупреждение, сохраняется в switch_results.json metadata и
# в docstring, чтобы консьюмеры (ВКРС, отчёты, dashboards) не
# интерпретировали эти метрики как context-aware switch detection.
SWITCH_INTERPRETATION_WARNING = (
    "Text-only stress test. NOT a switch detector. "
    "NOT comparable to context-aware routing. "
    "Router receives only `text` (no `active_domain`). "
    "Metric answers: 'On phrases that happen to be X->Y transitions, "
    "how often does router classify the new domain Y?' — it does NOT answer "
    "'is this a switch?' because there is no non-switch control set."
)


@dataclass
class SwitchStressReport:
    """Stress-test report для switch_test (text-only).

    ВНИМАНИЕ: `active_domain` НЕ подаётся в модель. Это артефакт разметки,
    используемый ТОЛЬКО для построения `per_transition` breakdown. См.
    :data:`SWITCH_INTERPRETATION_WARNING` — обязательная интерпретация.

    Attributes:
        baseline_name: имя модели (обычно "<baseline> @ switch_test").
        n_samples: общее число switch-кейсов.
        route_accuracy: доля предсказаний, где pred == new_domain (route_domain).
        per_transition: {"anamnesis->faq": {"correct": int, "total": int,
            "accuracy": float}, ...} — разбивка по парам (active, route).
        latency_ms: средняя задержка (ms/запрос), если замерена.
    """

    baseline_name: str
    n_samples: int
    route_accuracy: float
    per_transition: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_ms: float | None = None

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "baseline": self.baseline_name,
            "n_samples": self.n_samples,
            "route_accuracy": round(self.route_accuracy, 4),
            "per_transition": {
                key: {
                    "correct": entry["correct"],
                    "total": entry["total"],
                    "accuracy": round(entry["accuracy"], 4),
                }
                for key, entry in self.per_transition.items()
            },
            "interpretation_warning": SWITCH_INTERPRETATION_WARNING,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        return d


def compute_switch_stress_report(
    y_true: list[str],
    y_pred: list[str],
    active_domain: list[str],
    baseline_name: str = "unknown",
    latency_ms: float | None = None,
) -> SwitchStressReport:
    """Text-only switch stress test (§2 roadmap).

    Args:
        y_true: новый домен (route_domain из CSV).
        y_pred: предсказание router'а по text.
        active_domain: контекст предыдущей реплики (для breakdown only).
        baseline_name: имя модели.
        latency_ms: средняя задержка.

    Returns:
        SwitchStressReport с overall route_accuracy и per_transition.

    Raises:
        ValueError: если длины входов не совпадают.
    """
    if not (len(y_true) == len(y_pred) == len(active_domain)):
        raise ValueError(
            f"Длины не совпадают: y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, active_domain={len(active_domain)}",
        )

    true_arr = np.array(y_true)
    pred_arr = np.array(y_pred)
    active_arr = np.array(active_domain)

    n = len(y_true)
    route_acc = float((true_arr == pred_arr).mean()) if n > 0 else 0.0

    per_transition: dict[str, dict[str, Any]] = {}
    for i in range(n):
        active = active_arr[i] or "unknown"
        route = true_arr[i]
        key = f"{active}->{route}"
        entry = per_transition.setdefault(
            key, {"correct": 0, "total": 0, "accuracy": 0.0},
        )
        entry["total"] += 1
        if pred_arr[i] == route:
            entry["correct"] += 1

    for entry in per_transition.values():
        entry["accuracy"] = (
            entry["correct"] / entry["total"] if entry["total"] > 0 else 0.0
        )

    return SwitchStressReport(
        baseline_name=baseline_name,
        n_samples=n,
        route_accuracy=route_acc,
        per_transition=per_transition,
        latency_ms=latency_ms,
    )


def print_switch_stress_report(report: SwitchStressReport) -> None:
    """Человекочитаемый вывод switch stress report."""
    print(f"\n{'='*60}")
    print(f"  {report.baseline_name} [SWITCH STRESS — text-only]")
    print(f"{'='*60}")
    print(f"  Samples:           {report.n_samples}")
    print(f"  Route accuracy:    {report.route_accuracy:.4f}")
    if report.latency_ms is not None:
        print(f"  Latency:           {report.latency_ms:.2f} ms/req")
    print("\n  Per-transition (active → new):")
    # Стабильная сортировка: сначала по active, потом по route
    for key in sorted(report.per_transition):
        entry = report.per_transition[key]
        print(
            f"    {key:28s}  acc={entry['accuracy']:.3f}  "
            f"({entry['correct']}/{entry['total']})"
        )
    print(f"\n  ⚠ {SWITCH_INTERPRETATION_WARNING}")
    print()


def print_report(report: RoutingReport) -> None:
    """Человекочитаемый вывод отчёта."""
    print(f"\n{'='*60}")
    print(f"  {report.baseline_name}")
    print(f"{'='*60}")
    print(f"  Samples:           {report.n_samples}")
    print(f"  Accuracy:          {report.accuracy:.4f}")
    print(f"  Macro-F1:          {report.macro_f1:.4f}")
    print(f"  Balanced Accuracy: {report.balanced_accuracy:.4f}")
    if report.latency_ms is not None:
        print(f"  Latency:           {report.latency_ms:.2f} ms/req")
    print(f"\n  Safety:")
    print(f"    recall(anamnesis):         {report.recall_anamnesis:.4f}")
    print(f"    false_faq_for_anamnesis:   {report.false_faq_for_anamnesis:.4f}")
    if report.recall_urgent is not None:
        print(f"    recall(urgent/emergency):  {report.recall_urgent:.4f}")
    print(f"\n  Per-class:")
    for cls in LABEL_ORDER:
        if cls not in report.per_class:
            continue
        m = report.per_class[cls]
        print(
            f"    {cls:15s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
