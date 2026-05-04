"""SSoT для read-only загрузки reporting-артефактов D1.

Scope (зафиксирован в plan Task 1):
- Загрузка артефактов из `d1/results/` для нужд notebook / plotting / reports.

НЕ scope:
- Загрузка eval-датасетов (`d1/data/*.csv`) в orchestration-скриптах — они
  продолжают использовать свои локальные `load_split` helpers.
- Загрузка decision-trace CSV внутри `run_statistical_tests` — там это input
  eval-pipeline, не reporting.

Все функции строго read-only: не пишут, не мутируют файлы.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from d1.config import RESULTS_DIR


# ---------------------------------------------------------------------------
# Baselines / safety (источник: run_baselines → save_results)
# ---------------------------------------------------------------------------

def load_routing_results() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Загрузить `baseline_results.csv` + `baseline_results.json`.

    Returns:
        (df, data): DataFrame метрик + JSON с confusion matrices и per-class F1.
    """
    csv_path = RESULTS_DIR / "baseline_results.csv"
    json_path = RESULTS_DIR / "baseline_results.json"
    df = pd.read_csv(csv_path, dtype={"eval_set": str})
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return df, data


def load_safety_results() -> dict[str, Any]:
    """Загрузить `safety_results.json`."""
    with open(RESULTS_DIR / "safety_results.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data splits (только split summary — НЕ полный dataset)
# ---------------------------------------------------------------------------

def load_split_sizes() -> pd.DataFrame:
    """Размеры каждого split'а и распределение по классам.

    Читает `d1/data/{DATASET_PREFIX}_{name}.csv` только для выборки длины +
    counts по `route_domain`. Это reporting-функция (не data input), поэтому
    она легитимно живёт здесь.
    """
    # Локальный импорт, чтобы не тащить data-config в чисто-reporting API.
    from d1.baselines.eval_metrics import LABEL_ORDER
    from d1.config import DATA_DIR, DATASET_PREFIX

    split_names = [
        "train", "val", "test", "hard_test", "safety_set",
        "blind_test", "entity_held_out", "extended_eval", "switch_test",
    ]
    rows: list[dict[str, Any]] = []
    for name in split_names:
        path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        row: dict[str, Any] = {"split": name, "n": len(df)}
        for label in LABEL_ORDER:
            row[label] = int((df["route_domain"] == label).sum())
        row["has_active_domain"] = "active_domain" in df.columns
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------
# Calibration (Task 3 roadmap артефакты)
# ---------------------------------------------------------------------------

def load_calibration_metrics() -> pd.DataFrame:
    """Свести все `calibration_metrics_<model>_<eval>.json` в один DataFrame.

    Колонки: `baseline`, `eval_set`, `n`, `ECE`, `Brier(macro)`, `Brier(<class>)`.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("calibration_metrics_*.json")):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "baseline": m["baseline"],
            "eval_set": m["eval_set"],
            "n": m["n_samples"],
            "ECE": m["ece"],
            "Brier(macro)": m["brier_ovr"]["macro"],
        }
        for k, v in m["brier_ovr"]["per_class"].items():
            row[f"Brier({k})"] = v
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["eval_set", "baseline"])
        .reset_index(drop=True)
    )


def load_pareto_candidates() -> dict[str, pd.DataFrame]:
    """Все `pareto_candidates_*_val.csv` (только val — overfitting guard).

    Returns:
        mapping `имя_файла → DataFrame`. Если Pareto отсутствует — пустой dict.
    """
    result: dict[str, pd.DataFrame] = {}
    for path in sorted(RESULTS_DIR.glob("pareto_candidates_*_val.csv")):
        result[path.name] = pd.read_csv(path)
    return result


# ---------------------------------------------------------------------------
# Selective router (Task 4 roadmap)
# ---------------------------------------------------------------------------

def load_selective_results() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Загрузить `selective_results.csv` + `selective_results.json`."""
    csv_path = RESULTS_DIR / "selective_results.csv"
    json_path = RESULTS_DIR / "selective_results.json"
    df = pd.read_csv(csv_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return df, data


# ---------------------------------------------------------------------------
# Hybrid router + threshold sweep (Task 5 roadmap)
# ---------------------------------------------------------------------------

def load_hybrid_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить `hybrid_results.csv` + `hybrid_vs_selective.csv`."""
    hybrid_df = pd.read_csv(RESULTS_DIR / "hybrid_results.csv")
    vs_df = pd.read_csv(RESULTS_DIR / "hybrid_vs_selective.csv")
    return hybrid_df, vs_df


def load_threshold_sweep() -> pd.DataFrame:
    """Загрузить `threshold_sweep_results.csv`."""
    return pd.read_csv(RESULTS_DIR / "threshold_sweep_results.csv")


# ---------------------------------------------------------------------------
# Final-report composition helpers (Cell 31 dependencies)
# ---------------------------------------------------------------------------

def load_switch_results() -> pd.DataFrame:
    """Загрузить `switch_results.csv`."""
    return pd.read_csv(RESULTS_DIR / "switch_results.csv")


def load_bootstrap_ci() -> pd.DataFrame | None:
    """Загрузить `bootstrap_ci.csv` если доступно (для CI error bars)."""
    path = RESULTS_DIR / "bootstrap_ci.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Simple router (Task 5 плана — плейсхолдеры для мягкого skip)
# ---------------------------------------------------------------------------

def load_simple_router_results() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Загрузить SimpleRouter артефакты, если они уже сгенерированы.

    Возвращает None, если файлов нет — сигнал для `show_*` функции сделать
    graceful skip до выполнения Task 5.
    """
    csv_path = RESULTS_DIR / "simple_router_results.csv"
    vs_path = RESULTS_DIR / "simple_vs_hybrid.csv"
    if not (csv_path.exists() and vs_path.exists()):
        return None
    return pd.read_csv(csv_path), pd.read_csv(vs_path)


def load_complexity_breakdown(eval_set: str) -> pd.DataFrame | None:
    """Загрузить `complexity_breakdown_<eval_set>.csv` если есть."""
    path = RESULTS_DIR / f"complexity_breakdown_{eval_set}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_complexity_subtype_breakdown(eval_set: str) -> pd.DataFrame | None:
    """Загрузить `complexity_subtype_breakdown_<eval_set>.csv` если есть."""
    path = RESULTS_DIR / f"complexity_subtype_breakdown_{eval_set}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_complexity_gate_audit() -> pd.DataFrame | None:
    """Загрузить `complexity_gate_audit.csv` если gold-audit выполнен."""
    path = RESULTS_DIR / "complexity_gate_audit.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


__all__ = [
    "load_bootstrap_ci",
    "load_calibration_metrics",
    "load_complexity_breakdown",
    "load_complexity_gate_audit",
    "load_complexity_subtype_breakdown",
    "load_hybrid_results",
    "load_pareto_candidates",
    "load_routing_results",
    "load_safety_results",
    "load_selective_results",
    "load_simple_router_results",
    "load_split_sizes",
    "load_switch_results",
    "load_threshold_sweep",
]
