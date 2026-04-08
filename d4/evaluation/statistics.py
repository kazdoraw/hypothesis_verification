"""Статистический анализ результатов эксперимента.

По плану §3.6:
- Bootstrap CI (95%, 10000 resamples)
- Paired comparisons: S1 vs S2, S1 vs S3, S1 vs S4, S2 vs S4, S3 vs S4
- Коррекция множественных сравнений: Holm
- Friedman test: S1-S4 одновременно
- Effect size: Cohen's d
- B0 НЕ включается в inferential block
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Результаты тестов
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Доверительный интервал по bootstrap."""
    mean: float
    ci_lower: float
    ci_upper: float
    n_samples: int


@dataclass
class PairedTestResult:
    """Результат парного сравнения двух стратегий."""
    strategy_a: str
    strategy_b: str
    metric: str
    mean_a: float
    mean_b: float
    diff: float
    p_value: float
    p_value_corrected: float
    effect_size_d: float
    significant: bool


@dataclass
class FriedmanResult:
    """Результат теста Фридмана для S1-S4."""
    metric: str
    statistic: float
    p_value: float
    significant: bool


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

# Парные сравнения по плану
PAIRED_COMPARISONS = [
    ("S1", "S2"),
    ("S1", "S3"),
    ("S1", "S4"),
    ("S2", "S4"),
    ("S3", "S4"),
]


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_N,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap доверительный интервал для среднего.

    Args:
        values: наблюдения
        n_resamples: число resamples
        confidence: уровень доверия (0.95)
        seed: seed для воспроизводимости
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0, n_samples=0)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        mean=float(np.mean(arr)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d — размер эффекта для двух групп.

    Pooled standard deviation.
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    n_a, n_b = len(a_arr), len(b_arr)
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------


def holm_correction(p_values: list[float]) -> list[float]:
    """Коррекция множественных сравнений методом Holm-Bonferroni.

    Args:
        p_values: исходные p-values

    Returns:
        скорректированные p-values (в том же порядке)
    """
    n = len(p_values)
    if n == 0:
        return []

    # Сортируем по возрастанию p-value, сохраняя индексы
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    cummax = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted = pval * (n - rank)
        adjusted = min(adjusted, 1.0)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = cummax

    return corrected


# ---------------------------------------------------------------------------
# Paired Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def paired_wilcoxon(
    values_a: list[float],
    values_b: list[float],
) -> float:
    """Paired Wilcoxon signed-rank test.

    Returns: p-value (two-sided)
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    diff = a - b

    # Убираем нулевые разницы
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return 1.0

    try:
        stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------


def friedman_test(
    strategy_values: dict[str, list[float]],
) -> FriedmanResult:
    """Friedman test для сравнения S1-S4 одновременно.

    Args:
        strategy_values: {strategy_id: [values]} — значения метрики по стратегиям.
                         Длины списков должны совпадать (paired).

    Returns:
        FriedmanResult
    """
    keys = sorted(strategy_values.keys())
    groups = [strategy_values[k] for k in keys]

    # Проверяем одинаковую длину
    lengths = {len(g) for g in groups}
    if len(lengths) != 1 or lengths.pop() < 3:
        return FriedmanResult(metric="", statistic=0.0, p_value=1.0, significant=False)

    try:
        stat, p = stats.friedmanchisquare(*groups)
        return FriedmanResult(
            metric="",
            statistic=float(stat),
            p_value=float(p),
            significant=p < ALPHA,
        )
    except ValueError:
        return FriedmanResult(metric="", statistic=0.0, p_value=1.0, significant=False)


# ---------------------------------------------------------------------------
# Полный анализ парных сравнений
# ---------------------------------------------------------------------------


def run_paired_analysis(
    strategy_values: dict[str, list[float]],
    metric_name: str,
    comparisons: list[tuple[str, str]] | None = None,
) -> list[PairedTestResult]:
    """Полный парный анализ: Wilcoxon + Holm + Cohen's d.

    Args:
        strategy_values: {strategy_id: [values]}
        metric_name: название метрики
        comparisons: пары для сравнения (None = по умолчанию из плана)

    Returns:
        список PairedTestResult
    """
    pairs = comparisons or PAIRED_COMPARISONS
    raw_p_values: list[float] = []
    results_raw: list[dict] = []

    for sa, sb in pairs:
        if sa not in strategy_values or sb not in strategy_values:
            continue

        vals_a = strategy_values[sa]
        vals_b = strategy_values[sb]

        # Paired: нужны одинаковой длины
        min_len = min(len(vals_a), len(vals_b))
        if min_len < 3:
            continue

        a_trimmed = vals_a[:min_len]
        b_trimmed = vals_b[:min_len]

        p = paired_wilcoxon(a_trimmed, b_trimmed)
        d = cohens_d(a_trimmed, b_trimmed)

        raw_p_values.append(p)
        results_raw.append({
            "strategy_a": sa,
            "strategy_b": sb,
            "metric": metric_name,
            "mean_a": float(np.mean(a_trimmed)),
            "mean_b": float(np.mean(b_trimmed)),
            "diff": float(np.mean(a_trimmed) - np.mean(b_trimmed)),
            "p_value": p,
            "effect_size_d": d,
        })

    # Holm correction
    corrected = holm_correction(raw_p_values)

    results: list[PairedTestResult] = []
    for i, item in enumerate(results_raw):
        p_corr = corrected[i] if i < len(corrected) else 1.0
        results.append(PairedTestResult(
            strategy_a=item["strategy_a"],
            strategy_b=item["strategy_b"],
            metric=item["metric"],
            mean_a=item["mean_a"],
            mean_b=item["mean_b"],
            diff=item["diff"],
            p_value=item["p_value"],
            p_value_corrected=p_corr,
            effect_size_d=item["effect_size_d"],
            significant=p_corr < ALPHA,
        ))

    return results
