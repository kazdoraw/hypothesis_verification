"""Запуск полного pipeline D1 эксперимента из notebook.

Каждый шаг — это python-модуль из `d1.scripts.*`, исполняемый через
`python -m` текущим интерпретатором (`sys.executable`). Это гарантирует, что
при запуске из venv-jupyter все шаги используют именно этот venv, а не
системный `python3` из `PATH`.

Порядок шагов фиксирован: каждый последующий читает артефакты предыдущего
из `d1/results/`. Шаги, которые не входят в основной pipeline (latency,
lightweight cascade, learning curves, error taxonomy), управляются отдельными
RUN_* флагами в notebook.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Основной pipeline D1 (порядок имеет значение).
PIPELINE_STEPS: tuple[str, ...] = (
    "d1.scripts.run_baselines",          # обучение и оценка всех baselines
    "d1.scripts.analyze_confidence",     # калибровка confidence + Pareto
    "d1.scripts.evaluate_selective",     # selective router metrics
    "d1.scripts.evaluate_hybrid",        # hybrid (rules + selective ML)
    "d1.scripts.threshold_sweep",        # подбор thresholds для hybrid
    "d1.scripts.evaluate_simple_router", # production cascade simple router
    "d1.scripts.run_statistical_tests",  # bootstrap CI + paired tests
    "d1.scripts.plot_results",           # генерация всех figures
)


def run_d1_pipeline(steps: tuple[str, ...] | list[str] | None = None) -> None:
    """Выполнить D1 pipeline (полный или подмножество шагов).

    Args:
        steps: последовательность модулей вида ``"d1.scripts.X"``.
            При ``None`` выполняется полный ``PIPELINE_STEPS``.

    Raises:
        subprocess.CalledProcessError: если какой-либо шаг завершился ошибкой.
    """
    study_root = Path(__file__).resolve().parents[2]
    pipeline = tuple(steps) if steps is not None else PIPELINE_STEPS

    print(f"D1 pipeline: {len(pipeline)} шагов")
    print(f"Интерпретатор: {sys.executable}")
    print(f"Рабочая директория: {study_root}\n")

    for i, step in enumerate(pipeline, start=1):
        print(f"[{i}/{len(pipeline)}] {step}")
        subprocess.run(
            [sys.executable, "-m", step],
            cwd=str(study_root),
            check=True,
        )

    print(f"\n✓ D1 pipeline complete. Артефакты: {study_root / 'd1' / 'results'}")


__all__ = ["run_d1_pipeline", "PIPELINE_STEPS"]
