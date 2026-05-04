"""Загрузка versioned артефактов прогонов в DataFrames.

Чистые функции загрузки — никакой логики сравнения или трансформации.
Связывание с compare/plots происходит на уровне notebook/caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from d4.analysis.artifacts import RunPaths, make_run_paths

logger = logging.getLogger(__name__)


def load_report(run_id: str, mode: str, run_name: str = "pilot_dev_30") -> dict:
    """Загружает один report.json.

    Args:
        run_id: идентификатор прогона (YYYYMMDD_HHMMSS)
        mode: representation mode (plain/contextual/llm_enriched)
        run_name: имя eval set (pilot_dev_30, smoke, hard, blind, full_dev)

    Returns:
        dict с секциями: meta, quality, retrieval, deterministic, rank_analysis
    """
    rp = make_run_paths(run_id)
    path = rp.report_json(mode, run_name)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(
    run_id: str,
    modes: list[str],
    run_name: str = "pilot_dev_30",
) -> pd.DataFrame:
    """Собирает report.json всех modes в единый long-format DataFrame.

    Columns: mode, strategy, section, metric, value

    Извлекает секции: quality, retrieval, deterministic.

    Raises:
        FileNotFoundError: если не найден ни один report для всех запрошенных
            modes. Содержит список ожидавшихся путей — чтобы сразу было видно,
            где искать причину (неверный run_id, run_name или mode).
    """
    rows: list[dict] = []
    missing: list[Path] = []
    found_modes: list[str] = []

    for mode in modes:
        try:
            report = load_report(run_id, mode, run_name)
        except FileNotFoundError:
            missing.append(make_run_paths(run_id).report_json(mode, run_name))
            logger.warning("Report для mode=%s, run_name=%s не найден, пропускаю",
                           mode, run_name)
            continue

        found_modes.append(mode)

        for section in ("quality", "retrieval", "deterministic"):
            section_data = report.get(section, {})
            for strategy, metrics in section_data.items():
                for metric_name, value in metrics.items():
                    if value is None:
                        continue
                    rows.append({
                        "mode": mode,
                        "strategy": strategy,
                        "section": section,
                        "metric": metric_name,
                        "value": value,
                    })

    if not found_modes:
        expected = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Не найден ни один report для run_id={run_id!r}, "
            f"run_name={run_name!r}, modes={modes!r}. Ожидались файлы:\n  {expected}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Пустой DataFrame — проверьте run_id=%s, modes=%s, run_name=%s",
                        run_id, modes, run_name)
    return df


def load_rank_analysis(
    run_id: str,
    modes: list[str],
    run_name: str = "pilot_dev_30",
) -> pd.DataFrame:
    """Собирает rank_analysis из report.json в wide-format DataFrame.

    Columns: sample_id, {mode}_{strategy}_rr
    Значения — reciprocal rank (0 = miss).

    Raises:
        FileNotFoundError: если не найден ни один report для всех запрошенных
            modes.
    """
    merged: dict[str, dict[str, float]] = {}
    missing: list[Path] = []
    found_modes: list[str] = []

    for mode in modes:
        try:
            report = load_report(run_id, mode, run_name)
        except FileNotFoundError:
            missing.append(make_run_paths(run_id).report_json(mode, run_name))
            continue
        found_modes.append(mode)
        rank_data = report.get("rank_analysis", {})
        for sample_id, strat_rr in rank_data.items():
            if sample_id not in merged:
                merged[sample_id] = {}
            for strategy, rr in strat_rr.items():
                merged[sample_id][f"{mode}_{strategy}_rr"] = rr

    if not found_modes:
        expected = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Не найден ни один report для run_id={run_id!r}, "
            f"run_name={run_name!r}, modes={modes!r}. Ожидались файлы:\n  {expected}"
        )

    if not merged:
        return pd.DataFrame()

    rows = [{"sample_id": sid, **vals} for sid, vals in merged.items()]
    return pd.DataFrame(rows).set_index("sample_id").sort_index()
