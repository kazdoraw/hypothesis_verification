"""Manual Query Sandbox API для D1 (Task 6 roadmap).

Модуль даёт единый программный интерфейс для ручной проверки запросов:
- closed-set prediction по выбранным baseline'ам;
- SelectiveRouter decision (`accept` / `defer`);
- B4HybridRouter decision + rule trace;
- простая диагностика mixed/complex паттернов.

Ноутбук должен только вызывать функции отсюда и отображать результат. Этот
модуль не меняет датасеты, не добавляет gold-разметку и не обучает отдельные
модели сложности.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from d1.baselines.b4_hybrid import (
    B4HybridRouter,
    RULE_ACCEPT_POLICY,
    _parse_hit_count,
)
from d1.baselines.b0_rules import RulePrediction
from d1.baselines.complexity_gate import ComplexityGate
from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    RouteDecision,
    SelectiveRouter,
)
from d1.baselines.trained_bundle import TrainedBundle, train_bundle
from d1.config import RESULTS_DIR

ManualMode = Literal["closed_set", "selective", "hybrid", "all"]

RULES_NAME = "B0_rules"
SPARSE_NAME = "B1.1_tfidf_lr"
SPARSE_TUNED_NAME = "B1.2_tfidf_lr_tuned"
DENSE_NAME = "B2.1_bge-m3_svc"

CLOSED_SET_BASELINES = [
    RULES_NAME,
    SPARSE_NAME,
    SPARSE_TUNED_NAME,
    DENSE_NAME,
]


@dataclass(frozen=True)
class ManualRouterBundle:
    """Готовые объекты для ручного inference.

    Все модели приходят из `train_bundle` — единственной точки обучения/кэша.
    """

    bundle: TrainedBundle
    selective: SelectiveRouter
    hybrid: B4HybridRouter


@dataclass(frozen=True)
class ManualInferenceResult:
    """Результат ручного inference по одному тексту."""

    text: str
    mode: str
    closed_set: dict[str, dict[str, Any]]
    selective: dict[str, Any] | None
    hybrid: dict[str, Any] | None
    rule_trace: dict[str, Any] | None
    gold_label: str | None = None
    correctness: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для notebook/display."""
        return asdict(self)


_MANUAL_ROUTER_CACHE: ManualRouterBundle | None = None


def build_manual_router_bundle(use_cache: bool = True) -> ManualRouterBundle:
    """Загрузить TrainedBundle + SelectiveRouter + B4HybridRouter.

    SSoT: `train_bundle`. Повторные вызовы переиспользуют готовые router'ы в
    памяти, чтобы notebook sandbox отвечал быстро после первого запуска.
    """
    global _MANUAL_ROUTER_CACHE
    if use_cache and _MANUAL_ROUTER_CACHE is not None:
        return _MANUAL_ROUTER_CACHE

    bundle = train_bundle(
        names=CLOSED_SET_BASELINES,
        use_cache=use_cache,
    )
    selective = SelectiveRouter(
        sparse_model=bundle.get(SPARSE_NAME),
        dense_model=bundle.get(DENSE_NAME),
        thresholds=PRODUCTION_THRESHOLDS,
    )
    hybrid = B4HybridRouter(bundle=bundle, selective=selective)
    manual_bundle = ManualRouterBundle(
        bundle=bundle,
        selective=selective,
        hybrid=hybrid,
    )
    if use_cache:
        _MANUAL_ROUTER_CACHE = manual_bundle
    return manual_bundle


def infer_text(
    text: str,
    mode: ManualMode = "all",
    gold_label: str | None = None,
) -> ManualInferenceResult:
    """Один ручной запрос → predictions/RouteDecision/trace.

    Используется только для sandbox: не пишет в датасеты и не меняет модели.
    """
    return infer_many([text], mode=mode, gold_labels=[gold_label])[0]


def infer_many(
    texts: list[str],
    mode: ManualMode = "all",
    gold_labels: list[str | None] | None = None,
) -> list[ManualInferenceResult]:
    """Batch версия для быстрой проверки списка запросов."""
    if mode not in {"closed_set", "selective", "hybrid", "all"}:
        raise ValueError(f"Unknown mode: {mode}")
    if gold_labels is not None and len(gold_labels) != len(texts):
        raise ValueError(
            f"len(gold_labels)={len(gold_labels)} != len(texts)={len(texts)}",
        )

    labels = gold_labels if gold_labels is not None else [None] * len(texts)
    router_bundle = build_manual_router_bundle(use_cache=True)

    need_closed = mode in {"closed_set", "all"}
    need_selective = mode in {"selective", "all"}
    need_hybrid = mode in {"hybrid", "all"}

    closed_by_text = _predict_closed_set(router_bundle.bundle, texts) if need_closed else [{} for _ in texts]
    selective_decisions = (
        router_bundle.selective.route_batch(texts)
        if need_selective else [None] * len(texts)
    )
    hybrid_decisions = (
        router_bundle.hybrid.route_batch(texts)
        if need_hybrid else [None] * len(texts)
    )
    rule_traces = _rule_traces(router_bundle.bundle, texts) if need_hybrid or mode == "all" else [None] * len(texts)

    results: list[ManualInferenceResult] = []
    for text, gold, closed, selective, hybrid, rule_trace in zip(
        texts, labels, closed_by_text, selective_decisions, hybrid_decisions,
        rule_traces,
    ):
        selective_dict = _decision_to_dict(selective) if selective is not None else None
        hybrid_dict = _decision_to_dict(hybrid) if hybrid is not None else None
        correctness = (
            _compute_correctness(closed, selective_dict, hybrid_dict, gold)
            if gold else None
        )
        results.append(ManualInferenceResult(
            text=text,
            mode=mode,
            closed_set=closed,
            selective=selective_dict,
            hybrid=hybrid_dict,
            rule_trace=rule_trace,
            gold_label=gold,
            correctness=correctness,
        ))
    return results


def _predict_closed_set(
    bundle: TrainedBundle,
    texts: list[str],
) -> list[dict[str, dict[str, Any]]]:
    """Closed-set predictions по baseline'ам для каждого текста."""
    per_baseline: dict[str, list[dict[str, Any]]] = {}
    for name in CLOSED_SET_BASELINES:
        model = bundle.get(name)
        if name == RULES_NAME:
            rule_preds = model.predict_with_confidence(texts)
            per_baseline[name] = [_rule_prediction_to_closed_set(rp) for rp in rule_preds]
            continue

        preds = model.predict(texts)
        proba = np.asarray(model.predict_proba(texts), dtype=float)
        classes = list(model.classes_)
        per_baseline[name] = [
            _proba_prediction_to_closed_set(pred, row, classes)
            for pred, row in zip(preds, proba)
        ]

    rows: list[dict[str, dict[str, Any]]] = []
    for i in range(len(texts)):
        rows.append({name: per_baseline[name][i] for name in CLOSED_SET_BASELINES})
    return rows


def _rule_prediction_to_closed_set(rp: RulePrediction) -> dict[str, Any]:
    """B0 closed-set trace без probabilistic proba_by_class."""
    return {
        "label": rp.route_domain,
        "confidence": round(float(rp.confidence), 4),
        "proba_by_class": None,
        "top2": [
            {"label": rp.route_domain, "probability": round(float(rp.confidence), 4)},
        ],
    }


def _proba_prediction_to_closed_set(
    pred: str,
    row: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Prediction + top-2 proba для probabilistic baseline."""
    order = np.argsort(row)[::-1]
    proba_by_class = {
        cls: round(float(row[i]), 4)
        for i, cls in enumerate(classes)
    }
    top2 = [
        {"label": classes[int(i)], "probability": round(float(row[int(i)]), 4)}
        for i in order[:2]
    ]
    return {
        "label": str(pred),
        "confidence": top2[0]["probability"],
        "proba_by_class": proba_by_class,
        "top2": top2,
    }


def _decision_to_dict(decision: RouteDecision) -> dict[str, Any]:
    """RouteDecision → dict с округлением чисел для display."""
    return {
        "label": decision.label,
        "confidence": round(float(decision.confidence), 4),
        "margin": round(float(decision.margin), 4),
        "action": decision.action,
        "sparse_dense_agree": bool(decision.sparse_dense_agree),
        "reason": decision.reason,
        "dense_label": decision.dense_label,
    }


def _rule_traces(
    bundle: TrainedBundle,
    texts: list[str],
) -> list[dict[str, Any]]:
    """B0 rule trace + whitelist verdict для B4."""
    rules = bundle.get(RULES_NAME)
    rule_preds = rules.predict_with_confidence(texts)
    out = []
    for rp in rule_preds:
        hit_count = _parse_hit_count(rp.matched_rule)
        policy = RULE_ACCEPT_POLICY.get(rp.route_domain, {})
        min_hits = policy.get("min_hit_count")
        rule_accept = min_hits is not None and hit_count >= min_hits
        out.append({
            "route_domain": rp.route_domain,
            "confidence": round(float(rp.confidence), 4),
            "matched_rule": rp.matched_rule,
            "hit_count": hit_count,
            "rule_accept": bool(rule_accept),
            "min_hit_count": min_hits,
        })
    return out


def _compute_correctness(
    closed_set: dict[str, dict[str, Any]],
    selective: dict[str, Any] | None,
    hybrid: dict[str, Any] | None,
    gold_label: str,
) -> dict[str, Any]:
    """Correctness для gold label.

    Для selective/hybrid correctness считается только если action=accept.
    При defer возвращается None: это abstain outcome, а не label prediction.
    """
    closed_correct = {
        name: pred["label"] == gold_label
        for name, pred in closed_set.items()
    }

    def _policy_correct(decision: dict[str, Any] | None) -> bool | None:
        if decision is None or decision.get("action") != "accept":
            return None
        return decision["label"] == gold_label

    return {
        "gold_label": gold_label,
        "closed_set": closed_correct,
        "selective": _policy_correct(selective),
        "hybrid": _policy_correct(hybrid),
    }


# SSoT для маркеров и формул complexity-тэгов — d1/baselines/complexity_gate.py.
# Здесь — только thin wrappers для обратной совместимости публичного API.
_DEFAULT_GATE = ComplexityGate()


def tag_complexity(text: str) -> dict[str, bool]:
    """Backward-compat wrapper над `ComplexityGate.decide(text).tags`.

    Возвращает dict с фиксированным набором ключей (см. `ComplexityGate`).
    Для нового кода предпочитайте `ComplexityGate.decide(text)`, чтобы
    получить также `primary_tag` и `action`.
    """
    return _DEFAULT_GATE.decide(text).tags


def complexity_summary(decisions_csv: Path) -> pd.DataFrame:
    """Агрегировать complexity tags по decision trace CSV.

    Ожидает `hybrid_decisions_<eval_set>.csv` или совместимый файл с колонками
    `text_preview`, `action`, `correct`. Сохраняет CSV в `RESULTS_DIR`.

    Использует `ComplexityGate` как SSoT — формулы тэгов не дублируются.
    """
    decisions_csv = Path(decisions_csv)
    df = pd.read_csv(decisions_csv, dtype=str).fillna("")
    required = {"text_preview", "action", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{decisions_csv} missing columns: {sorted(missing)}")

    tagged = df.copy()
    tag_rows = [_DEFAULT_GATE.decide(t).tags for t in tagged["text_preview"]]
    tag_names = sorted({k for row in tag_rows for k in row})
    for tag in tag_names:
        tagged[tag] = [bool(row.get(tag, False)) for row in tag_rows]

    rows: list[dict[str, Any]] = []
    for tag in tag_names:
        subset = tagged[tagged[tag]]
        rows.append(_complexity_row(tag, subset))
    rows.append(_complexity_row("_all", tagged))

    summary = pd.DataFrame(rows).sort_values(["tag"]).reset_index(drop=True)
    out_path = RESULTS_DIR / f"complexity_summary_{_eval_set_from_decisions_path(decisions_csv)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return summary


def _complexity_row(tag: str, subset: pd.DataFrame) -> dict[str, Any]:
    n = len(subset)
    accepted = int((subset["action"] == "accept").sum()) if n else 0
    deferred = int((subset["action"] == "defer").sum()) if n else 0
    accepted_subset = subset[subset["action"] == "accept"]
    if len(accepted_subset) == 0:
        accepted_accuracy = 0.0
    else:
        correct_bool = accepted_subset["correct"].map(_to_bool)
        accepted_accuracy = float(correct_bool.mean())
    return {
        "tag": tag,
        "n": n,
        "accepted": accepted,
        "deferred": deferred,
        "coverage": round(accepted / n, 4) if n else 0.0,
        "accepted_accuracy": round(accepted_accuracy, 4),
    }


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _eval_set_from_decisions_path(path: Path) -> str:
    stem = path.stem
    prefix = "hybrid_decisions_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


__all__ = [
    "CLOSED_SET_BASELINES",
    "ManualInferenceResult",
    "ManualMode",
    "ManualRouterBundle",
    "build_manual_router_bundle",
    "complexity_summary",
    "infer_many",
    "infer_text",
    "tag_complexity",
]
