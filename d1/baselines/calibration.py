"""Pure calibration & threshold functions для D1 (Task 3a roadmap).

Модуль содержит ТОЛЬКО вычислительные функции — не импортирует matplotlib
и не пишет файлы. Orchestration + PNG-рендер живут в
`@/Users/kazdoraw/developer/med-agent/study/d1/scripts/analyze_confidence.py`.

API (стабильный контракт):
    compute_ece(confidence, correct, n_bins)             -> float
    compute_brier_ovr(y_true, proba, classes)            -> dict
    compute_reliability_table(confidence, correct, ...)  -> DataFrame
    compute_threshold_table(proba, preds, true, classes) -> DataFrame
    find_pareto_candidates(table, objectives, top_k)     -> DataFrame

Все функции детерминированные, без побочных эффектов.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (Guo et al., 2017).

    ECE = sum_m (|avg_acc(bin_m) - avg_conf(bin_m)|) * (count_m / N)

    Равномерное бининг по [0, 1]. Пустые бины игнорируются. sklearn не
    предоставляет ECE напрямую — реализуем руками.

    Args:
        confidence: 1D array, max predict_proba per sample, значения в [0, 1].
        correct: 1D bool array, pred == true per sample.
        n_bins: число равномерных бинов.

    Returns:
        ECE ∈ [0, 1]. 0 = идеально калибровано.

    Raises:
        ValueError: длины не совпадают или n_bins < 1.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError(
            f"Длины не совпадают: confidence={confidence.shape}, "
            f"correct={correct.shape}",
        )
    if n_bins < 1:
        raise ValueError(f"n_bins должен быть >= 1, получено {n_bins}")

    n = len(confidence)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Правый край включается только в последнем бине (чтобы conf=1.0 попал)
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += abs(avg_acc - avg_conf) * (count / n)

    return float(ece)


# ---------------------------------------------------------------------------
# Multiclass Brier score (One-vs-Rest)
# ---------------------------------------------------------------------------

def compute_brier_ovr(
    y_true: Sequence[str],
    proba: np.ndarray,
    classes: Sequence[str],
) -> dict[str, float | dict[str, float]]:
    """One-vs-Rest multiclass Brier score.

    sklearn не имеет multiclass Brier — строим через `brier_score_loss`
    в цикле по классам. Возвращаем per-class и macro среднее.

    Args:
        y_true: истинные метки (N строк).
        proba: (N, C) матрица вероятностей.
        classes: порядок классов в колонках proba.

    Returns:
        {"per_class": {cls: brier}, "macro": float}

    Raises:
        ValueError: размерности не совпадают.
    """
    y_true_arr = np.asarray(y_true)
    proba_arr = np.asarray(proba, dtype=float)
    classes_list = list(classes)

    if proba_arr.ndim != 2:
        raise ValueError(f"proba должна быть 2D, получено ndim={proba_arr.ndim}")
    if proba_arr.shape[0] != len(y_true_arr):
        raise ValueError(
            f"proba.shape[0]={proba_arr.shape[0]} != len(y_true)={len(y_true_arr)}",
        )
    if proba_arr.shape[1] != len(classes_list):
        raise ValueError(
            f"proba.shape[1]={proba_arr.shape[1]} != len(classes)={len(classes_list)}",
        )

    per_class: dict[str, float] = {}
    for i, cls in enumerate(classes_list):
        y_bin = (y_true_arr == cls).astype(int)
        per_class[cls] = float(brier_score_loss(y_bin, proba_arr[:, i]))

    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return {"per_class": per_class, "macro": macro}


# ---------------------------------------------------------------------------
# Reliability table (для reliability diagram)
# ---------------------------------------------------------------------------

def compute_reliability_table(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table: per-bin статистика для reliability diagram.

    Args:
        confidence: max predict_proba per sample.
        correct: pred == true per sample.
        n_bins: число равномерных бинов.

    Returns:
        DataFrame с колонками:
            - bin_lo, bin_hi, bin_center
            - count (число примеров в бине; бины с count=0 пропускаются)
            - avg_confidence
            - avg_accuracy
            - gap (avg_accuracy - avg_confidence; положительный = underconfident)
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError(
            f"Длины не совпадают: confidence={confidence.shape}, "
            f"correct={correct.shape}",
        )
    if n_bins < 1:
        raise ValueError(f"n_bins должен быть >= 1, получено {n_bins}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "bin_center": float((lo + hi) / 2),
            "count": count,
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
            "gap": avg_acc - avg_conf,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold table (coverage / selective metrics)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = np.linspace(0.3, 0.99, 70)


def compute_threshold_table(
    proba: np.ndarray,
    preds: Sequence[str],
    true_labels: Sequence[str],
    classes: Sequence[str],
    thresholds: np.ndarray | None = None,
    anamnesis_label: str = "anamnesis",
    faq_label: str = "faq",
) -> pd.DataFrame:
    """Threshold sweep для confidence-based reject policy.

    Для каждого threshold t вычисляется:
      - coverage: |{i : max_proba_i >= t}| / N
      - accepted_accuracy: acc среди принятых
      - overall_recall_anamnesis: TP_anam_among_accepted / total_anam (по ВСЕМ)
      - false_faq_for_anamnesis: (pred=faq & true=anam среди принятых) / total_anam
      - rejected_count: число отказов

    `overall_recall_anamnesis` считается от ВСЕГО числа anamnesis в датасете
    (не от принятых) — это важно для safety interpretation: отказ на anamnesis
    = пропущенный urgent, эквивалент FN в защитном сценарии.

    Args:
        proba: (N, C) вероятности.
        preds: argmax-предсказания (N строк).
        true_labels: истинные метки (N строк).
        classes: порядок классов в колонках proba (для консистентности, не
            используется напрямую — оставлен для будущих расширений).
        thresholds: 1D array порогов; default — 70 точек linspace(0.3, 0.99).

    Returns:
        DataFrame с колонками [threshold, coverage, accepted_accuracy,
        overall_recall_anamnesis, false_faq_for_anamnesis, rejected_count,
        accepted_count].
    """
    proba_arr = np.asarray(proba, dtype=float)
    pred_arr = np.asarray(preds)
    true_arr = np.asarray(true_labels)

    if proba_arr.ndim != 2:
        raise ValueError(f"proba должна быть 2D, получено ndim={proba_arr.ndim}")
    if not (proba_arr.shape[0] == len(pred_arr) == len(true_arr)):
        raise ValueError(
            f"Длины не совпадают: proba={proba_arr.shape[0]}, "
            f"preds={len(pred_arr)}, true={len(true_arr)}",
        )

    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError(
            f"thresholds должен быть непустым 1D array, "
            f"получено shape={thresholds.shape}",
        )
    if np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
        raise ValueError(
            f"thresholds должны лежать в [0, 1], "
            f"получен диапазон [{thresholds.min()}, {thresholds.max()}]",
        )

    confidence = proba_arr.max(axis=1)
    n_total = len(confidence)
    anam_mask = true_arr == anamnesis_label
    n_anam_total = int(anam_mask.sum())

    rows: list[dict[str, float]] = []
    for t in thresholds:
        accepted = confidence >= t
        n_accepted = int(accepted.sum())
        coverage = n_accepted / n_total if n_total > 0 else 0.0

        if n_accepted > 0:
            accepted_acc = float(
                (pred_arr[accepted] == true_arr[accepted]).mean(),
            )
        else:
            accepted_acc = 0.0

        # overall recall(anamnesis): из ВСЕХ anamnesis — принято И правильно
        if n_anam_total > 0:
            correct_anam_accepted = int(
                ((pred_arr == anamnesis_label)
                 & (true_arr == anamnesis_label)
                 & accepted).sum(),
            )
            overall_recall_anam = correct_anam_accepted / n_anam_total

            # false_faq_for_anamnesis (по принятым, нормируем на total anam)
            false_faq = int(
                ((pred_arr == faq_label)
                 & (true_arr == anamnesis_label)
                 & accepted).sum(),
            )
            false_faq_rate = false_faq / n_anam_total
        else:
            overall_recall_anam = 0.0
            false_faq_rate = 0.0

        rows.append({
            "threshold": float(t),
            "coverage": float(coverage),
            "accepted_accuracy": float(accepted_acc),
            "overall_recall_anamnesis": float(overall_recall_anam),
            "false_faq_for_anamnesis": float(false_faq_rate),
            "accepted_count": n_accepted,
            "rejected_count": n_total - n_accepted,
        })

    # Mute "classes" не используется напрямую — явно подавляем unused hint.
    del classes
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pareto candidate thresholds
# ---------------------------------------------------------------------------

def find_pareto_candidates(
    threshold_table: pd.DataFrame,
    objectives: tuple[str, str] = ("accepted_accuracy", "overall_recall_anamnesis"),
    top_k: int = 7,
) -> pd.DataFrame:
    """Pareto frontier на threshold_table по двум целям (maximize).

    Возвращает non-dominated threshold'ы (кандидаты). НИКОГДА не возвращает
    единичный "оптимальный" threshold — финальный выбор делается человеком
    (в ноутбуке/ВКРС) с учётом бизнес-целей.

    Args:
        threshold_table: результат compute_threshold_table.
        objectives: пара колонок для максимизации (acc и recall по умолчанию).
        top_k: максимальное число кандидатов после прореживания по coverage
            (чтобы не возвращать 70 почти идентичных точек). Если фронт
            содержит <= top_k точек — возвращаются все.

    Returns:
        DataFrame — подмножество строк threshold_table (Pareto frontier),
        отсортированных по `objectives[0]` (desc).

    Raises:
        KeyError: требуемая колонка отсутствует.
        ValueError: top_k < 1.
    """
    if top_k < 1:
        raise ValueError(f"top_k должен быть >= 1, получено {top_k}")
    for col in objectives:
        if col not in threshold_table.columns:
            raise KeyError(
                f"Колонка '{col}' отсутствует в threshold_table. "
                f"Доступные: {list(threshold_table.columns)}",
            )

    if threshold_table.empty:
        return threshold_table.copy()

    obj_a, obj_b = objectives
    values = threshold_table[[obj_a, obj_b]].to_numpy(dtype=float)

    # Non-dominated: для каждой точки нет другой, которая >= по обеим и > хотя бы по одной.
    n = len(values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            a_dom = values[j, 0] >= values[i, 0] and values[j, 1] >= values[i, 1]
            strict = values[j, 0] > values[i, 0] or values[j, 1] > values[i, 1]
            if a_dom and strict:
                is_pareto[i] = False
                break

    frontier = threshold_table.loc[is_pareto].copy()
    frontier = frontier.sort_values(obj_a, ascending=False).reset_index(drop=True)

    if len(frontier) <= top_k:
        return frontier

    # Прореживание: равномерные индексы вдоль отсортированного фронта
    idx = np.linspace(0, len(frontier) - 1, top_k).round().astype(int)
    return frontier.iloc[idx].reset_index(drop=True)
