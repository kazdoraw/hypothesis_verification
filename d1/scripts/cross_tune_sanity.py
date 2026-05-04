"""Phase 0 sanity: bootstrap stability tag-policy thresholds на val.

Цель — убедиться, что правило выбора threshold не переобучено на конкретный
val sample. Если IQR выбранного threshold > 0.10 (= 1 шаг grid), это сигнал
о нестабильности и/или малом размере val. В таком случае нужно расширить
val или отбросить tag-policy подход.

Алгоритм (independent per tag):
    1. Один раз получить pre-policy decisions на val:
       - hybrid.route_batch(texts) — ML cascade (B4) БЕЗ tag-policy.
       - gate.decide_batch(texts) — primary_tag для каждого текста.
    2. n_resamples × 100 bootstrap (random_state=42) индексов val
       (with replacement).
    3. Для каждого resample и каждого tag ∈ {simple_faq, simple_booking,
       simple_symptom} — выбрать min(threshold) ∈ GRID, при котором:
           accepted_acc(resample, tag)        ≥ 0.95
           AND accepted_recall_anam(resample) ≥ 0.98
       Остальные tag-policy зафиксированы на DEFAULT_TAG_POLICIES.
    4. Сводка: median / IQR (P75 - P25) / min / max по 100 resample'ам.

Allowed labels (assumption для simple_symptom — отсутствует в DEFAULT):
    simple_faq      → ("faq",)
    simple_booking  → ("booking",)
    simple_symptom  → ("anamnesis",)

Output:
    d1/results/phase0_threshold_stability.csv с колонками:
        tag, n_bootstrap, threshold_median, threshold_iqr,
        threshold_min, threshold_max, n_unsolved.

Pass criteria: threshold_iqr ≤ 0.10 для каждого tag.

Запуск:
    cd study && python3 -m d1.scripts.cross_tune_sanity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from d1.baselines.b4_hybrid import B4HybridRouter
from d1.baselines.complexity_gate import ComplexityGate
from d1.baselines.selective_router import (
    PRODUCTION_THRESHOLDS,
    SelectiveRouter,
)
from d1.baselines.simple_router import DEFAULT_TAG_POLICIES, TagPolicy
from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

GRID: tuple[float, ...] = (
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)

TAGS_TO_TUNE: tuple[str, ...] = (
    "simple_faq", "simple_booking", "simple_symptom",
)

# Sanity assumption: simple_symptom → anamnesis (в DEFAULT_TAG_POLICIES policy
# для simple_symptom отсутствует; для тюнинга подразумеваем ожидаемый домен).
ALLOWED_LABELS_FOR_TUNING: dict[str, tuple[str, ...]] = {
    "simple_faq": ("faq",),
    "simple_booking": ("booking",),
    "simple_symptom": ("anamnesis",),
}

ACC_THRESHOLD = 0.95
RECALL_ANAM_THRESHOLD = 0.98
DEFAULT_N_RESAMPLES = 100
DEFAULT_RNG_SEED = 42

DEFAULT_SPARSE = "B1.1_tfidf_lr"
DEFAULT_DENSE = "B2.1_bge-m3_svc"
RULES_NAME = "B0_rules"


# ---------------------------------------------------------------------------
# Pre-policy decisions: snapshot ML cascade ДО применения tag-policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PrePolicyTrace:
    """In-memory снимок ML cascade без tag-policy (per-text arrays).

    Используется для быстрого пересчёта при bootstrap: оригинальный prediction
    модели не меняется, варьируется только tag-policy threshold.
    """

    gold: np.ndarray            # shape (n,) — gold route_domain
    primary_tag: np.ndarray     # shape (n,) — ComplexityGate primary
    ml_action: np.ndarray       # shape (n,) — accept | defer (после B4, до tag-policy)
    ml_label: np.ndarray        # shape (n,) — predicted label (только если accept)
    ml_confidence: np.ndarray   # shape (n,) — float confidence

    def __len__(self) -> int:
        return len(self.gold)


def _load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _build_hybrid_router(
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
) -> tuple[B4HybridRouter, ComplexityGate]:
    """Загрузить bundle один раз и собрать hybrid + gate (для reuse)."""
    logger.info(
        "Loading TrainedBundle: rules=%s, sparse=%s, dense=%s",
        RULES_NAME, sparse_name, dense_name,
    )
    bundle = train_bundle(
        names=[RULES_NAME, sparse_name, dense_name],
        use_cache=True,
    )
    selective = SelectiveRouter(
        sparse_model=bundle.get(sparse_name),
        dense_model=bundle.get(dense_name),
        thresholds=PRODUCTION_THRESHOLDS,
    )
    return B4HybridRouter(bundle=bundle, selective=selective), ComplexityGate()


def collect_pre_policy_trace(
    eval_set: str = "val",
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
    hybrid: B4HybridRouter | None = None,
    gate: ComplexityGate | None = None,
) -> _PrePolicyTrace:
    """Один проход hybrid + gate на split → arrays per-text без tag-policy.

    SimpleRouter не используется намеренно: его tag-policy искажает ml_label
    после применения. Вместо этого работаем с B4HybridRouter напрямую и
    собираем primary_tag через ComplexityGate, как делает SimpleRouter
    внутри.

    Args:
        eval_set: имя split (val, test, hard_test, blind_test, ...).
        hybrid, gate: если переданы, переиспользуются (экономия encoder load).
    """
    df = _load_split(eval_set)
    texts = df["text"].tolist()
    y_true = df["route_domain"].tolist()

    if hybrid is None or gate is None:
        hybrid, gate = _build_hybrid_router(sparse_name, dense_name)

    decisions = hybrid.route_batch(texts)
    gate_decisions = gate.decide_batch(texts)

    return _PrePolicyTrace(
        gold=np.asarray(y_true),
        primary_tag=np.asarray([gd.primary_tag for gd in gate_decisions]),
        ml_action=np.asarray([d.action for d in decisions]),
        ml_label=np.asarray([d.label for d in decisions]),
        ml_confidence=np.asarray([d.confidence for d in decisions], dtype=float),
    )


# ---------------------------------------------------------------------------
# Threshold sweep на bootstrap'ах.
# ---------------------------------------------------------------------------

def _apply_per_text_policies(
    trace: _PrePolicyTrace,
    indices: np.ndarray,
    overrides: dict[str, TagPolicy],
) -> tuple[np.ndarray, np.ndarray]:
    """Симулировать tag-policy на подмножестве indices.

    Для текстов, чей `primary_tag` имеет policy, переоценивается accept→defer.
    ComplexityGate-defer (action не определён здесь, поскольку trace берётся
    через hybrid; gate-defer симулируется отдельно: если primary_tag НЕ в
    {simple_*, unclassified, ...allow_ml-теги}, текст не попадает в hybrid).

    В нашем случае trace взят из hybrid.route_batch (без gate-фильтра), поэтому
    мы применяем tag-policy ко ВСЕМ текстам, для которых ml_action=accept.
    Это соответствует реальной семантике SimpleRouter: gate-defer бросается до
    ML и не зависит от threshold; тут тестируем только tag-policy слой.

    Returns:
        (final_action, final_correct) — два np.ndarray длины len(indices).
    """
    sub_action = trace.ml_action[indices].copy()
    sub_label = trace.ml_label[indices]
    sub_conf = trace.ml_confidence[indices]
    sub_tag = trace.primary_tag[indices]
    sub_gold = trace.gold[indices]

    for tag, policy in overrides.items():
        mask_tag = (sub_tag == tag) & (sub_action == "accept")
        if not mask_tag.any():
            continue
        allowed = policy.allowed_labels
        label_ok = (
            np.ones(mask_tag.shape, dtype=bool) if allowed is None
            else np.isin(sub_label, np.asarray(allowed))
        )
        conf_ok = sub_conf >= policy.min_confidence
        keep_accept = mask_tag & label_ok & conf_ok
        flip_to_defer = mask_tag & ~keep_accept
        sub_action = np.where(flip_to_defer, "defer", sub_action)

    accepted = sub_action == "accept"
    correct = accepted & (sub_label == sub_gold)
    return sub_action, correct


def _metrics_for_indices(
    trace: _PrePolicyTrace,
    indices: np.ndarray,
    overrides: dict[str, TagPolicy],
) -> tuple[float, float, dict[str, float]]:
    """Посчитать (accepted_recall_anam_global, _, accepted_acc_per_tag) на indices.

    Returns:
        accepted_recall_anam — глобальный recall(anamnesis) среди accepted.
        coverage             — global coverage (accepted / n).
        accepted_acc_per_tag — dict {tag: accuracy on accepted} для тегов из overrides.
    """
    sub_tag = trace.primary_tag[indices]
    sub_gold = trace.gold[indices]
    sub_label = trace.ml_label[indices]
    final_action, _ = _apply_per_text_policies(trace, indices, overrides)
    accepted = final_action == "accept"

    n_anam_accepted = int(np.sum(accepted & (sub_gold == "anamnesis")))
    n_anam_correct = int(np.sum(
        accepted & (sub_gold == "anamnesis") & (sub_label == "anamnesis")
    ))
    accepted_recall_anam = (
        n_anam_correct / n_anam_accepted if n_anam_accepted else 0.0
    )

    coverage = float(np.mean(accepted)) if len(indices) else 0.0

    per_tag: dict[str, float] = {}
    for tag in overrides:
        mask = (sub_tag == tag) & accepted
        n_acc = int(np.sum(mask))
        if n_acc == 0:
            per_tag[tag] = float("nan")
            continue
        n_corr = int(np.sum(mask & (sub_label == sub_gold)))
        per_tag[tag] = n_corr / n_acc

    return accepted_recall_anam, coverage, per_tag


def _select_threshold_for_tag(
    trace: _PrePolicyTrace,
    indices: np.ndarray,
    tag: str,
    grid: tuple[float, ...] = GRID,
    base_policies: dict[str, TagPolicy] | None = None,
) -> float | None:
    """Найти min(threshold) в grid с pass criteria, остальные tag-policy зафиксированы.

    Returns:
        Selected threshold ∈ grid или None, если pass criteria не выполнены ни
        для одного значения (unsolved случай).
    """
    base_policies = dict(base_policies or DEFAULT_TAG_POLICIES)
    allowed = ALLOWED_LABELS_FOR_TUNING[tag]

    for thr in sorted(grid):
        candidate = dict(base_policies)
        candidate[tag] = TagPolicy(min_confidence=thr, allowed_labels=allowed)
        recall_anam, _coverage, per_tag = _metrics_for_indices(
            trace, indices, candidate,
        )
        acc_tag = per_tag.get(tag, float("nan"))
        if (
            not np.isnan(acc_tag)
            and acc_tag >= ACC_THRESHOLD
            and recall_anam >= RECALL_ANAM_THRESHOLD
        ):
            return thr
    return None


def bootstrap_threshold_stability(
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    sparse_name: str = DEFAULT_SPARSE,
    dense_name: str = DEFAULT_DENSE,
) -> pd.DataFrame:
    """Основная функция Phase 0 sanity check.

    Возвращает DataFrame со сводкой по тэгам (median/IQR/min/max selected
    threshold по n_resamples bootstrap'ам val).
    """
    trace = collect_pre_policy_trace(
        sparse_name=sparse_name, dense_name=dense_name,
    )
    n = len(trace)
    if n == 0:
        raise RuntimeError("Val split пуст: нет данных для bootstrap")

    rng = np.random.default_rng(rng_seed)
    selections: dict[str, list[float]] = {tag: [] for tag in TAGS_TO_TUNE}
    unsolved: dict[str, int] = {tag: 0 for tag in TAGS_TO_TUNE}

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        for tag in TAGS_TO_TUNE:
            thr = _select_threshold_for_tag(trace, idx, tag)
            if thr is None:
                unsolved[tag] += 1
            else:
                selections[tag].append(thr)
        if (b + 1) % 25 == 0:
            logger.info("bootstrap %d/%d", b + 1, n_resamples)

    rows: list[dict[str, object]] = []
    for tag in TAGS_TO_TUNE:
        values = np.asarray(selections[tag], dtype=float)
        if values.size == 0:
            rows.append({
                "tag": tag,
                "n_bootstrap": n_resamples,
                "threshold_median": float("nan"),
                "threshold_iqr": float("nan"),
                "threshold_min": float("nan"),
                "threshold_max": float("nan"),
                "n_unsolved": unsolved[tag],
            })
            continue
        rows.append({
            "tag": tag,
            "n_bootstrap": n_resamples,
            "threshold_median": float(np.median(values)),
            "threshold_iqr": float(
                np.percentile(values, 75) - np.percentile(values, 25)
            ),
            "threshold_min": float(values.min()),
            "threshold_max": float(values.max()),
            "n_unsolved": unsolved[tag],
        })
    return pd.DataFrame(rows)


def main() -> Path:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = bootstrap_threshold_stability()
    out_path = RESULTS_DIR / "phase0_threshold_stability.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)

    logger.info("\nThreshold stability (pass = IQR ≤ 0.10):")
    for _, row in df.iterrows():
        marker = (
            "OK"
            if not np.isnan(row["threshold_iqr"]) and row["threshold_iqr"] <= 0.10
            else "FAIL"
        )
        logger.info(
            "  [%s] %s: median=%.2f iqr=%.2f range=[%.2f, %.2f] unsolved=%d",
            marker,
            row["tag"],
            row["threshold_median"],
            row["threshold_iqr"],
            row["threshold_min"],
            row["threshold_max"],
            row["n_unsolved"],
        )
    return out_path


if __name__ == "__main__":
    main()
