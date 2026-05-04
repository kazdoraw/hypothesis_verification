"""Тесты для threshold_sweep extension с complexity_gate flag (Task 6 плана).

Контракт:
- `sweep(with_complexity_gate=True/False/None)` возвращает DataFrame с
  колонкой `with_complexity_gate: bool`.
- При `None` (default) прогоняется обе ветки → 2× rows.
- При gate=True используется SimpleRouter wrapper (selective/hybrid).
- CLI флаг `--no-complexity-gate` отключает gate-вариант.
- pivot summary индексируется по (eval_set, router, with_complexity_gate).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_sweep_signature_has_with_complexity_gate() -> None:
    """`sweep()` должен принимать опциональный параметр `with_complexity_gate`."""
    from d1.scripts.threshold_sweep import sweep

    sig = inspect.signature(sweep)
    assert "with_complexity_gate" in sig.parameters, (
        "sweep() должен принимать with_complexity_gate"
    )


def test_module_exposes_constants() -> None:
    """Модуль предоставляет CLI-friendly константы для использования."""
    from d1.scripts import threshold_sweep as ts

    assert hasattr(ts, "THRESHOLD_CONFIGS")
    assert hasattr(ts, "EVAL_SETS")


def test_sweep_source_no_llm_imports() -> None:
    src = Path("d1/scripts/threshold_sweep.py").read_text(encoding="utf-8")
    forbidden = ["openai", "together", "litellm", "OpenRouterClient"]
    for needle in forbidden:
        assert needle not in src


def test_sweep_source_uses_simple_router_when_gate_enabled() -> None:
    """Source должен импортировать SimpleRouter (gate cascade)."""
    src = Path("d1/scripts/threshold_sweep.py").read_text(encoding="utf-8")
    assert "SimpleRouter" in src


def test_cli_supports_no_complexity_gate_flag() -> None:
    """CLI должен принимать `--no-complexity-gate`."""
    src = Path("d1/scripts/threshold_sweep.py").read_text(encoding="utf-8")
    assert "--no-complexity-gate" in src or "no_complexity_gate" in src


# ---------------------------------------------------------------------------
# Functional smoke (быстрый, на одном маленьком eval_set)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sweep_smoke_with_gate_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: sweep на 1 config × 1 eval_set × оба gate-варианта.

    Проверяет наличие колонки `with_complexity_gate` и удвоение rows
    при `with_complexity_gate=None` (оба варианта).
    """
    from d1.baselines.selective_router import SelectiveThresholds
    from d1.scripts import threshold_sweep as ts

    # Один config, один eval_set — ускоряем smoke.
    monkeypatch.setattr(ts, "THRESHOLD_CONFIGS", [
        ("smoke_cfg", SelectiveThresholds(
            anamnesis_threshold=0.5, faq_anamnesis_margin=0.15, general_threshold=0.65,
        )),
    ])
    monkeypatch.setattr(ts, "EVAL_SETS", ["test"])

    df = ts.sweep(with_complexity_gate=None)
    assert "with_complexity_gate" in df.columns
    # 1 cfg × 1 eval × (gate=False: selective+hybrid → 2 rows) +
    # (gate=True: только hybrid через SimpleRouter → 1 row) = 3 rows.
    # selective+gate пропускается: SimpleRouter оборачивает только hybrid.
    assert len(df) == 3
    assert set(df["with_complexity_gate"].unique()) == {False, True}
    # Hybrid-row при gate=True существует и отличается coverage от gate=False.
    hybrid_no_gate = df[(df["router"] == "hybrid") & (~df["with_complexity_gate"])].iloc[0]
    hybrid_gate = df[(df["router"] == "hybrid") & (df["with_complexity_gate"])].iloc[0]
    assert hybrid_gate["coverage"] <= hybrid_no_gate["coverage"], (
        "SimpleRouter добавляет defers → coverage не должно расти"
    )
