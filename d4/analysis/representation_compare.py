"""Ядро Stage 2A анализа: сравнение representation modes.

Чистые функции трансформации DataFrame — не импортирует loaders,
не делает LLM-вызовов, не загружает файлы.
Связывание с loaders через notebook/caller.

Колонки wide-таблиц: MultiIndex (mode, metric) — это устраняет
проблемы парсинга mode из flat column names (llm_enriched_hit_rate).
Для удобства отображения flatten выполняется с разделителем «::».
"""

from __future__ import annotations

import pandas as pd

# Разделитель mode::metric в flat column names (не `_`, т.к. mode содержит `_`)
_SEP = "::"


# ---------------------------------------------------------------------------
# Helpers: pivot + delta
# ---------------------------------------------------------------------------

def _pivot_section(
    reports_df: pd.DataFrame,
    section: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Pivot long→wide для одной секции. Возвращает (df, modes, metrics)."""
    sub = reports_df[reports_df["section"] == section].copy()
    if sub.empty:
        return pd.DataFrame(), [], []

    pivot = sub.pivot_table(
        index="strategy",
        columns=["mode", "metric"],
        values="value",
        aggfunc="first",
    )
    modes = list(dict.fromkeys(m for m, _ in pivot.columns))
    metrics = list(dict.fromkeys(met for _, met in pivot.columns))

    # Flatten MultiIndex → "mode::metric"
    pivot.columns = [f"{m}{_SEP}{met}" for m, met in pivot.columns]
    return pivot.sort_index(), modes, metrics


def _add_deltas(
    pivot: pd.DataFrame,
    modes: list[str],
    metrics: list[str],
    baseline: str,
) -> pd.DataFrame:
    """Добавляет delta-колонки vs baseline для каждого (non-baseline mode, metric)."""
    if baseline not in modes:
        return pivot
    for mode in modes:
        if mode == baseline:
            continue
        for met in metrics:
            col = f"{mode}{_SEP}{met}"
            base_col = f"{baseline}{_SEP}{met}"
            if col in pivot.columns and base_col in pivot.columns:
                pivot[f"{mode}{_SEP}{met}{_SEP}delta"] = pivot[col] - pivot[base_col]
    return pivot


# ---------------------------------------------------------------------------
# Retrieval comparison
# ---------------------------------------------------------------------------

def build_retrieval_table(
    reports_df: pd.DataFrame,
    baseline: str = "plain",
) -> pd.DataFrame:
    """Таблица retrieval-метрик по mode × strategy + delta vs baseline.

    Колонки: {mode}::{metric}, {mode}::{metric}::delta
    Метрики из report.json: hit_rate, mrr, mean_recall, gold_in_context_rate, n_samples.
    """
    if reports_df.empty or "section" not in reports_df.columns:
        return pd.DataFrame()
    pivot, modes, metrics = _pivot_section(reports_df, "retrieval")
    if pivot.empty:
        return pivot
    return _add_deltas(pivot, modes, metrics, baseline)


# ---------------------------------------------------------------------------
# Quality (deterministic) comparison
# ---------------------------------------------------------------------------

def build_deterministic_table(
    reports_df: pd.DataFrame,
    baseline: str = "plain",
) -> pd.DataFrame:
    """Таблица quality-метрик по mode × strategy + delta vs baseline.

    Добавляет производные: answerability_rate, doctor_match_rate.
    """
    if reports_df.empty or "section" not in reports_df.columns:
        return pd.DataFrame()
    pivot, modes, metrics = _pivot_section(reports_df, "deterministic")
    if pivot.empty:
        return pivot

    # Производные rates
    for m in modes:
        n_col = f"{m}{_SEP}n"
        if n_col not in pivot.columns:
            continue
        ans_col = f"{m}{_SEP}answerability_correct"
        if ans_col in pivot.columns:
            new = f"{m}{_SEP}answerability_rate"
            pivot[new] = (pivot[ans_col] / pivot[n_col]).round(4)
            if "answerability_rate" not in metrics:
                metrics.append("answerability_rate")

        doc_col = f"{m}{_SEP}doctor_match"
        doc_total = f"{m}{_SEP}doctor_total"
        if doc_col in pivot.columns and doc_total in pivot.columns:
            new = f"{m}{_SEP}doctor_match_rate"
            pivot[new] = (pivot[doc_col] / pivot[doc_total].replace(0, float("nan"))).round(4)
            if "doctor_match_rate" not in metrics:
                metrics.append("doctor_match_rate")

    return _add_deltas(pivot, modes, metrics, baseline)


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

_GATE_THRESHOLDS = {
    "hit_rate_min_delta": 0.02,
    "mrr_min_delta": 0.01,
    "fmr_min_delta": 0.01,
    "fmr_max_drop": -0.03,
    # NEW: пороги, при которых засчитываем retrieval-регрессию
    "hit_rate_max_drop": -0.02,
    "mrr_max_drop": -0.01,
}


_NEXT_STEP_BY_RUN_NAME = {
    "smoke": "После smoke стоит запустить pilot_dev_30 для подтверждения сигнала.",
    "pilot_dev_30": "Сделайте полный full_dev (137 сэмплов) для проверки на репрезентативной выборке.",
    "pilot": "Сделайте полный full_dev (137 сэмплов) для проверки на репрезентативной выборке.",
    "full_dev": "Прогоните hard и blind eval-сеты, чтобы исключить переобучение под dev-распределение.",
    "hard": "Финальный шаг — blind set; убедитесь, что метрики держатся вне dev-распределения.",
    "blind": "Если результат подтверждается — оформляйте production-решение и фиксируйте baseline.",
}


def _next_step_hint(run_name: str | None) -> str:
    if run_name is None:
        return "Перепроверьте на следующем большем eval-сете."
    return _NEXT_STEP_BY_RUN_NAME.get(
        run_name,
        f"Перепроверьте на следующем большем eval-сете (run_name={run_name!r}).",
    )


def evaluate_gate(
    retrieval_table: pd.DataFrame,
    deterministic_table: pd.DataFrame,
    focus_strategy: str = "S3",
    run_name: str | None = None,
) -> dict:
    """Decision gate: proceed / stop / neutral.

    Расширен по сравнению с первой версией:
      * Учитывает retrieval-регрессии (hit_rate/MRR drop), а не только их рост —
        это закрывает дыру, когда FMR подрос на шумовую величину, а retrieval
        просел — раньше gate выдавал proceed.
      * Recommendation зависит от ``run_name``: предлагается следующий уровень
        eval-сета (smoke → pilot → full_dev → hard → blind) вместо общего
        «нужен full dev».

    Returns:
        {'verdict': str, 'signals': list[str], 'recommendation': str}
    """
    signals: list[str] = []

    if retrieval_table.empty:
        return {
            "verdict": "neutral",
            "signals": ["Нет данных для retrieval comparison"],
            "recommendation": "Перезапустите прогоны",
        }

    retrieval_gain = False
    retrieval_regression = False

    # Retrieval deltas
    delta_cols = [c for c in retrieval_table.columns if c.endswith(f"{_SEP}delta")]
    for col in delta_cols:
        if focus_strategy not in retrieval_table.index:
            continue
        val = retrieval_table.loc[focus_strategy, col]
        if pd.isna(val):
            continue
        # col = "llm_enriched::hit_rate::delta" → parts = [mode, metric, "delta"]
        parts = col.split(_SEP)
        mode, metric = parts[0], parts[1]
        if metric == "hit_rate":
            if val >= _GATE_THRESHOLDS["hit_rate_min_delta"]:
                signals.append(f"{mode}: hit_rate delta +{val:.3f} ≥ порог")
                retrieval_gain = True
            elif val <= _GATE_THRESHOLDS["hit_rate_max_drop"]:
                signals.append(f"{mode}: hit_rate drop {val:.3f} ≤ порог")
                retrieval_regression = True
        if metric == "mrr":
            if val >= _GATE_THRESHOLDS["mrr_min_delta"]:
                signals.append(f"{mode}: MRR delta +{val:.3f} ≥ порог")
                retrieval_gain = True
            elif val <= _GATE_THRESHOLDS["mrr_max_drop"]:
                signals.append(f"{mode}: MRR drop {val:.3f} ≤ порог")
                retrieval_regression = True

    # FMR deltas
    fmr_drop = False
    fmr_gain = False
    for col in deterministic_table.columns:
        if not col.endswith(f"{_SEP}delta"):
            continue
        parts = col.split(_SEP)
        if len(parts) < 3:
            continue
        mode, metric = parts[0], parts[1]
        if metric != "avg_fmr":
            continue
        if focus_strategy not in deterministic_table.index:
            continue
        val = deterministic_table.loc[focus_strategy, col]
        if pd.isna(val):
            continue
        if val < _GATE_THRESHOLDS["fmr_max_drop"]:
            signals.append(f"{mode}: FMR drop {val:.3f} < порог")
            fmr_drop = True
        elif val >= _GATE_THRESHOLDS["fmr_min_delta"]:
            signals.append(f"{mode}: FMR improvement +{val:.3f}")
            fmr_gain = True

    next_step = _next_step_hint(run_name)

    if retrieval_regression and not fmr_gain:
        verdict = "stop"
        recommendation = (
            "Retrieval просел без выигрыша по FMR — enrichment скорее вреден. "
            "Оставьте baseline (plain), не двигайтесь дальше по этой ветке."
        )
    elif retrieval_regression and fmr_gain:
        verdict = "neutral"
        recommendation = (
            "FMR немного подрос, но retrieval просел — сигнал противоречивый. "
            "Без bootstrap-CI / paired-теста выигрыш FMR может быть шумом. "
            f"{next_step}"
        )
    elif fmr_drop and not retrieval_gain:
        verdict = "stop"
        recommendation = "FMR упал без retrieval gains. Необходим анализ промптов."
    elif (retrieval_gain or fmr_gain) and not fmr_drop:
        verdict = "proceed"
        recommendation = (
            f"Representation enrichment даёт положительный сигнал. {next_step}"
        )
    else:
        verdict = "neutral"
        recommendation = (
            f"Нет явного сигнала по порогам gate. {next_step} "
            "Параллельно посмотрите bootstrap-CI и paired-тесты "
            "(см. d4/analysis/significance.py)."
        )

    return {"verdict": verdict, "signals": signals, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_modes(
    retrieval_table: pd.DataFrame,
    deterministic_table: pd.DataFrame,
    focus_strategy: str = "S3",
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Ранжирование modes по composite score.

    Composite = w_hit*hit_rate + w_mrr*MRR + w_fmr*FMR.
    Modes извлекаются из column names через «::» разделитель.
    """
    w = weights or {"hit_rate": 0.3, "mrr": 0.3, "avg_fmr": 0.4}

    # Собираем modes из обеих таблиц
    modes_set: set[str] = set()
    for table in (retrieval_table, deterministic_table):
        for col in table.columns:
            if _SEP in col and not col.endswith(f"{_SEP}delta"):
                mode = col.split(_SEP)[0]
                modes_set.add(mode)

    scores: list[tuple[str, float]] = []
    for mode in modes_set:
        composite = 0.0
        found = False
        for metric, weight in w.items():
            col = f"{mode}{_SEP}{metric}"
            for table in (retrieval_table, deterministic_table):
                if col in table.columns and focus_strategy in table.index:
                    val = table.loc[focus_strategy, col]
                    if pd.notna(val):
                        composite += weight * val
                        found = True
                    break
        if found:
            scores.append((mode, round(composite, 4)))

    return sorted(scores, key=lambda x: x[1], reverse=True)
