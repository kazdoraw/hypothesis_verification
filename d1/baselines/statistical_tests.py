"""Bootstrap CI и paired significance для D1 (Task 8 roadmap).

Модуль содержит только статистические функции: не обучает модели, не читает
файлы и не пишет артефакты. Orchestration/CSV сохранение живёт в
`d1.scripts.run_statistical_tests`.

Главный контракт paired bootstrap: для пары моделей A/B каждая bootstrap
итерация использует ОДНИ И ТЕ ЖЕ resampled rows. Иначе paired structure
ломается, а p-value становится некорректным.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score

MetricFn = Callable[[pd.DataFrame, str], float]

LABEL_ORDER = ["anamnesis", "faq", "booking", "unsupported"]


def accuracy_metric(df: pd.DataFrame, pred_col: str) -> float:
    """Accuracy по колонке предсказаний."""
    if len(df) == 0:
        return 0.0
    return float((df["true_label"] == df[pred_col]).mean())


def macro_f1_metric(df: pd.DataFrame, pred_col: str) -> float:
    """Macro-F1 с фиксированным label order."""
    if len(df) == 0:
        return 0.0
    labels = [l for l in LABEL_ORDER if l in set(df["true_label"]) | set(df[pred_col])]
    return float(f1_score(
        df["true_label"],
        df[pred_col],
        labels=labels,
        average="macro",
        zero_division=0,
    ))


def recall_anamnesis_metric(df: pd.DataFrame, pred_col: str) -> float:
    """Recall(anamnesis)."""
    subset = df[df["true_label"] == "anamnesis"]
    if len(subset) == 0:
        return 0.0
    return float((subset[pred_col] == "anamnesis").mean())


def recall_urgent_metric(df: pd.DataFrame, pred_col: str) -> float:
    """Recall urgent: urgent/high/emergency должны попасть в anamnesis."""
    if "urgency" not in df.columns:
        return 0.0
    subset = df[df["urgency"].isin(["urgent", "emergency", "high"])]
    if len(subset) == 0:
        return 0.0
    return float((subset[pred_col] == "anamnesis").mean())


def _has_family_ids(seed_ids: list[str] | pd.Series) -> bool:
    """Есть ли usable seed_id families.

    Для hard_test/blind_test seed_id обычно пустой: тогда нужен row-level
    fallback с явной пометкой в output.
    """
    series = pd.Series(seed_ids, dtype=str).fillna("")
    return bool(series.str.strip().ne("").any())


def _effective_seed_ids(df: pd.DataFrame) -> pd.Series:
    """Вернуть seed_id или row-level surrogate для fallback."""
    if "seed_id" in df.columns and _has_family_ids(df["seed_id"]):
        return df["seed_id"].astype(str).replace("", np.nan).fillna(
            pd.Series([f"__row_{i}" for i in range(len(df))], index=df.index),
        )
    return pd.Series([f"__row_{i}" for i in range(len(df))], index=df.index)


def _generate_family_indices(
    seed_ids: list[str] | pd.Series,
    n_bootstrap: int,
    rng_seed: int,
) -> list[np.ndarray]:
    """Сгенерировать bootstrap row indices по family ids.

    `seed_ids` передаётся на уровне строк (len == n_rows). Sampling unit =
    уникальная family; если family выбрана, в sample попадают все её строки.
    Для row-level fallback каждая строка имеет уникальную synthetic family.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    seed_series = pd.Series(seed_ids, dtype=str).reset_index(drop=True)
    families = sorted(seed_series.unique().tolist())
    family_to_indices = {
        fam: seed_series.index[seed_series == fam].to_numpy(dtype=int)
        for fam in families
    }
    rng = np.random.default_rng(rng_seed)
    out: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        idx = np.concatenate([family_to_indices[fam] for fam in sampled_families])
        out.append(idx.astype(int))
    return out


def family_bootstrap_ci(
    df: pd.DataFrame,
    pred_col: str,
    metric_fn: MetricFn,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    method: Literal["percentile", "BCa"] = "BCa",
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """(point, ci_lower, ci_upper) через family-level bootstrap."""
    row = family_bootstrap_ci_report(
        df=df,
        baseline=pred_col,
        eval_set="unknown",
        metric_name=getattr(metric_fn, "__name__", "metric"),
        pred_col=pred_col,
        metric_fn=metric_fn,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        method=method,
        rng_seed=rng_seed,
    )
    return float(row["point"]), float(row["ci_lower"]), float(row["ci_upper"])


def family_bootstrap_ci_report(
    df: pd.DataFrame,
    baseline: str,
    eval_set: str,
    metric_name: str,
    pred_col: str,
    metric_fn: MetricFn,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    method: Literal["percentile", "BCa"] = "BCa",
    rng_seed: int = 42,
) -> dict[str, object]:
    """Bootstrap CI row для сохранения в `bootstrap_ci.csv`."""
    if pred_col not in df.columns:
        raise ValueError(f"pred_col not found: {pred_col}")
    row_level_fallback = not (
        "seed_id" in df.columns and _has_family_ids(df["seed_id"])
    )
    effective_seed_ids = _effective_seed_ids(df)
    indices_list = _generate_family_indices(effective_seed_ids, n_bootstrap, rng_seed)

    point = metric_fn(df, pred_col)
    samples = np.asarray([
        metric_fn(df.iloc[idx], pred_col)
        for idx in indices_list
    ], dtype=float)
    low, high = _bootstrap_interval(
        point=point,
        samples=samples,
        df=df,
        pred_col=pred_col,
        metric_fn=metric_fn,
        seed_ids=effective_seed_ids,
        alpha=alpha,
        method=method,
    )
    return {
        "baseline": baseline,
        "eval_set": eval_set,
        "metric": metric_name,
        "point": round(float(point), 6),
        "ci_lower": round(float(low), 6),
        "ci_upper": round(float(high), 6),
        "n_bootstrap": n_bootstrap,
        "method": method,
        "row_level_fallback": bool(row_level_fallback),
        "rng_seed": rng_seed,
    }


def paired_family_bootstrap(
    df: pd.DataFrame,
    pred_col_a: str,
    pred_col_b: str,
    metric_fn: MetricFn,
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
) -> dict[str, object]:
    """Paired bootstrap для delta = metric(A) - metric(B).

    Гарантия: внутри каждой bootstrap итерации A и B считаются на одном и том
    же `sub_df` (`indices`), то есть paired structure сохранён.
    """
    if pred_col_a not in df.columns or pred_col_b not in df.columns:
        raise ValueError("pred_col_a/pred_col_b must exist in df")
    row_level_fallback = not (
        "seed_id" in df.columns and _has_family_ids(df["seed_id"])
    )
    effective_seed_ids = _effective_seed_ids(df)
    indices_list = _generate_family_indices(effective_seed_ids, n_bootstrap, rng_seed)

    point_a = metric_fn(df, pred_col_a)
    point_b = metric_fn(df, pred_col_b)
    point_delta = point_a - point_b

    deltas: list[float] = []
    for idx in indices_list:
        sub_df = df.iloc[idx]
        m_a = metric_fn(sub_df, pred_col_a)
        m_b = metric_fn(sub_df, pred_col_b)
        deltas.append(m_a - m_b)

    delta_arr = np.asarray(deltas, dtype=float)
    low, high = np.quantile(delta_arr, [0.025, 0.975])
    p_value = float((delta_arr <= 0).mean())

    return {
        "delta_mean": round(float(point_delta), 6),
        "delta_bootstrap_mean": round(float(delta_arr.mean()), 6),
        "delta_ci_low": round(float(low), 6),
        "delta_ci_high": round(float(high), 6),
        "p_value_one_sided": round(p_value, 6),
        "n_resamples": n_bootstrap,
        "rng_seed": rng_seed,
        "row_level_fallback": bool(row_level_fallback),
    }


def _bootstrap_interval(
    point: float,
    samples: np.ndarray,
    df: pd.DataFrame,
    pred_col: str,
    metric_fn: MetricFn,
    seed_ids: pd.Series,
    alpha: float,
    method: str,
) -> tuple[float, float]:
    """Percentile или BCa interval по bootstrap samples."""
    method_norm = method.lower()
    if method_norm == "percentile":
        return tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2]))  # type: ignore[return-value]
    if method_norm != "bca":
        raise ValueError(f"Unsupported bootstrap method: {method}")

    low_q, high_q = _bca_quantiles(
        point=point,
        samples=samples,
        jackknife=_jackknife_values(df, pred_col, metric_fn, seed_ids),
        alpha=alpha,
    )
    low, high = np.quantile(samples, [low_q, high_q])
    return float(low), float(high)


def _jackknife_values(
    df: pd.DataFrame,
    pred_col: str,
    metric_fn: MetricFn,
    seed_ids: pd.Series,
) -> np.ndarray:
    """Leave-one-family-out jackknife estimates."""
    values = []
    for fam in sorted(seed_ids.unique().tolist()):
        mask = seed_ids != fam
        if mask.sum() == 0:
            continue
        values.append(metric_fn(df.loc[mask.to_numpy()], pred_col))
    return np.asarray(values, dtype=float)


def _bca_quantiles(
    point: float,
    samples: np.ndarray,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """BCa adjusted quantiles with safe fallbacks."""
    if len(samples) == 0:
        return alpha / 2, 1 - alpha / 2

    prop_less = float((samples < point).mean())
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    if len(jackknife) < 2 or np.allclose(jackknife, jackknife[0]):
        acc = 0.0
    else:
        jack_mean = float(jackknife.mean())
        diffs = jack_mean - jackknife
        denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
        acc = float(np.sum(diffs ** 3) / denom) if denom else 0.0

    def _adjust(q: float) -> float:
        z = norm.ppf(q)
        denom = 1 - acc * (z0 + z)
        if abs(denom) < 1e-9:
            return q
        adj = norm.cdf(z0 + (z0 + z) / denom)
        return float(min(max(adj, 0.0), 1.0))

    return _adjust(alpha / 2), _adjust(1 - alpha / 2)


METRICS: dict[str, MetricFn] = {
    "macro_f1": macro_f1_metric,
    "recall_anamnesis": recall_anamnesis_metric,
    "recall_urgent": recall_urgent_metric,
}


__all__ = [
    "METRICS",
    "accuracy_metric",
    "family_bootstrap_ci",
    "family_bootstrap_ci_report",
    "macro_f1_metric",
    "paired_family_bootstrap",
    "recall_anamnesis_metric",
    "recall_urgent_metric",
    "_generate_family_indices",
]
