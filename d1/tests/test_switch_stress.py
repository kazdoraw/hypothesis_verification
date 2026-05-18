"""Unit-тесты SwitchStressReport (Task 2 roadmap).

Проверяем:
- Базовый расчёт route_accuracy и per_transition breakdown.
- Валидация несовпадающих длин.
- Обработка пустого active_domain (fallback "unknown").
- Интеграция с run_baselines.load_split (валидация SWITCH_CSV_COLUMNS).
- summary_dict содержит interpretation_warning (acceptance criteria).

Запуск:
    cd study && python -m pytest d1/tests/test_switch_stress.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from d1.baselines.eval_metrics import (
    SWITCH_INTERPRETATION_WARNING,
    SwitchStressReport,
    compute_switch_stress_report,
)
from d1.config import CSV_COLUMNS, SWITCH_CSV_COLUMNS


# ---------------------------------------------------------------------------
# compute_switch_stress_report: базовый расчёт
# ---------------------------------------------------------------------------

def test_route_accuracy_all_correct() -> None:
    """Все предсказания верны → route_accuracy == 1.0."""
    y_true = ["faq", "booking", "anamnesis"]
    y_pred = ["faq", "booking", "anamnesis"]
    active = ["anamnesis", "faq", "booking"]

    report = compute_switch_stress_report(y_true, y_pred, active, "B_test")

    assert report.route_accuracy == 1.0
    assert report.n_samples == 3
    assert report.baseline_name == "B_test"


def test_route_accuracy_half_correct() -> None:
    """Половина верна → route_accuracy == 0.5."""
    y_true = ["faq", "booking", "anamnesis", "faq"]
    y_pred = ["faq", "anamnesis", "anamnesis", "booking"]
    active = ["anamnesis"] * 4

    report = compute_switch_stress_report(y_true, y_pred, active, "B")

    assert report.route_accuracy == 0.5


def test_per_transition_breakdown() -> None:
    """per_transition корректно агрегирует по парам (active, route)."""
    # Три транзиции: 2× anamnesis->faq (1 верная), 1× faq->booking (верная)
    y_true = ["faq", "faq", "booking"]
    y_pred = ["faq", "anamnesis", "booking"]
    active = ["anamnesis", "anamnesis", "faq"]

    report = compute_switch_stress_report(y_true, y_pred, active, "B")

    assert "anamnesis->faq" in report.per_transition
    assert "faq->booking" in report.per_transition

    anam_faq = report.per_transition["anamnesis->faq"]
    assert anam_faq["total"] == 2
    assert anam_faq["correct"] == 1
    assert anam_faq["accuracy"] == 0.5

    faq_book = report.per_transition["faq->booking"]
    assert faq_book["total"] == 1
    assert faq_book["correct"] == 1
    assert faq_book["accuracy"] == 1.0


def test_empty_active_domain_fallback() -> None:
    """Пустая строка active_domain → ключ 'unknown->X'."""
    y_true = ["faq"]
    y_pred = ["faq"]
    active = [""]

    report = compute_switch_stress_report(y_true, y_pred, active, "B")

    assert "unknown->faq" in report.per_transition


def test_mismatched_lengths_raises() -> None:
    """Разные длины входов → ValueError."""
    with pytest.raises(ValueError, match="Длины не совпадают"):
        compute_switch_stress_report(
            y_true=["faq", "booking"],
            y_pred=["faq"],
            active_domain=["anamnesis"],
            baseline_name="B",
        )


def test_empty_inputs() -> None:
    """Пустые входы → route_accuracy=0.0, пустой breakdown."""
    report = compute_switch_stress_report([], [], [], "B")
    assert report.route_accuracy == 0.0
    assert report.n_samples == 0
    assert report.per_transition == {}


# ---------------------------------------------------------------------------
# summary_dict: interpretation_warning (acceptance criterion)
# ---------------------------------------------------------------------------

def test_summary_dict_contains_interpretation_warning() -> None:
    """summary_dict обязан содержать interpretation_warning с semantic markers.

    Проверяется наличие ключевых фраз, не exact string — тогда переформулировка
    warning'а не ломает тест, но искажение смысла (удаление "NOT") — ломает.
    """
    report = compute_switch_stress_report(
        ["faq"], ["faq"], ["anamnesis"], "B",
    )
    d = report.summary_dict()

    assert "interpretation_warning" in d
    warning = d["interpretation_warning"]
    # Semantic markers, обязательные для корректной интерпретации в ВКРС
    assert "NOT a switch detector" in warning
    assert "NOT comparable to context-aware routing" in warning
    # Источник константы не расходится с summary_dict (защита от копипасты)
    assert warning == SWITCH_INTERPRETATION_WARNING


def test_summary_dict_rounding() -> None:
    """Числовые поля округлены до 4 знаков."""
    report = compute_switch_stress_report(
        y_true=["faq"] * 3 + ["booking"] * 7,
        y_pred=["faq"] * 2 + ["anamnesis"] + ["booking"] * 5 + ["faq"] * 2,
        active_domain=["anamnesis"] * 10,
        baseline_name="B",
        latency_ms=1.23456,
    )
    d = report.summary_dict()

    # 7/10 верных → 0.7 (точное)
    assert d["route_accuracy"] == 0.7
    assert d["latency_ms"] == 1.23


# ---------------------------------------------------------------------------
# Schema: SWITCH_CSV_COLUMNS extension pattern
# ---------------------------------------------------------------------------

def test_switch_csv_columns_extends_base() -> None:
    """SWITCH_CSV_COLUMNS = CSV_COLUMNS + ['active_domain'], порядок сохранён."""
    assert SWITCH_CSV_COLUMNS[: len(CSV_COLUMNS)] == CSV_COLUMNS
    assert SWITCH_CSV_COLUMNS[-1] == "active_domain"
    assert len(SWITCH_CSV_COLUMNS) == len(CSV_COLUMNS) + 1


# ---------------------------------------------------------------------------
# Integration: load_split для switch_test валидирует колонки
# ---------------------------------------------------------------------------

def test_load_split_switch_missing_active_domain_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_split('switch_test') без active_domain → ValueError."""
    from d1.scripts import run_baselines

    # Создаём urезанный switch_test без active_domain
    bad_df = pd.DataFrame(
        {col: ["x"] for col in CSV_COLUMNS},  # без active_domain
    )
    bad_path = tmp_path / "d1_v6_switch_test.csv"
    bad_df.to_csv(bad_path, index=False)

    monkeypatch.setattr(run_baselines, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="active_domain"):
        run_baselines.load_split("switch_test")


def test_load_split_switch_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_split('switch_test') с полным SWITCH_CSV_COLUMNS → OK."""
    from d1.scripts import run_baselines

    good_df = pd.DataFrame(
        {col: ["x"] for col in SWITCH_CSV_COLUMNS},
    )
    good_path = tmp_path / "d1_v6_switch_test.csv"
    good_df.to_csv(good_path, index=False)

    monkeypatch.setattr(run_baselines, "DATA_DIR", tmp_path)

    df = run_baselines.load_split("switch_test")
    assert "active_domain" in df.columns
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Integration: orchestration-слой (run_all_baselines → save_results)
# ---------------------------------------------------------------------------

class _MockModel:
    """Минимальный baseline для orchestration-тестов.

    Возвращает фиксированную метку, независимо от входа. Этого достаточно,
    чтобы проверить, что путь исполнения switch/safety/routing корректно
    разветвляется в run_all_baselines, без реального обучения.
    """

    def __init__(self, label: str = "anamnesis") -> None:
        self._label = label

    def predict(self, texts: list[str]) -> list[str]:
        return [self._label] * len(texts)


def _mock_bundle(labels: list[str]) -> Any:
    """Простейший bundle-stub для monkeypatch train_bundle."""
    from d1.baselines.trained_bundle import TrainedBundle

    models = {name: _MockModel() for name in labels}
    return TrainedBundle(models=models, train_size=0, metadata={})


def _write_switch_csv(path: Path, n: int = 4) -> None:
    """Сгенерировать мини switch_test.csv со всеми SWITCH_CSV_COLUMNS."""
    rows = []
    transitions = [("anamnesis", "faq"), ("faq", "booking"),
                   ("booking", "anamnesis"), ("anamnesis", "booking")]
    for i in range(n):
        active, route = transitions[i % len(transitions)]
        row = {col: "" for col in SWITCH_CSV_COLUMNS}
        row["id"] = f"sw_{i:03d}"
        row["text"] = f"текст кейса {i}"
        row["route_domain"] = route
        row["active_domain"] = active
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_routing_csv(path: Path) -> None:
    """Мини test.csv для orchestration-теста (стандартный routing)."""
    rows = []
    for i, route in enumerate(["anamnesis", "faq", "booking"]):
        row = {col: "" for col in CSV_COLUMNS}
        row["id"] = f"t_{i}"
        row["text"] = f"тест {i}"
        row["route_domain"] = route
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def orchestration_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Подготовить изолированное окружение для run_all_baselines.

    Monkeypatch'ит DATA_DIR + train_bundle чтобы избежать реального
    обучения baseline'ов в тесте (они шумят через SentenceTransformer).
    """
    from d1.scripts import run_baselines

    _write_switch_csv(tmp_path / "d1_v6_switch_test.csv")
    _write_routing_csv(tmp_path / "d1_v6_test.csv")
    monkeypatch.setattr(run_baselines, "DATA_DIR", tmp_path)

    # Mock train_bundle — возвращает фиксированный список моделей.
    mock_models = ["B0_rules", "B1.1_tfidf_lr"]
    monkeypatch.setattr(
        run_baselines, "train_bundle",
        lambda names, use_cache: _mock_bundle(mock_models),
    )
    # Подменяем BASELINE_CONFIGS, чтобы enabled_names совпал с mock_models
    # (защитный check в run_all_baselines проверяет _BASELINE_ORDER).
    monkeypatch.setattr(
        run_baselines, "BASELINE_CONFIGS",
        {name: {"enabled": True, "cls": object, "params": {}}
         for name in mock_models},
    )
    return tmp_path


def test_run_all_baselines_switch_goes_to_all_switch(
    orchestration_env: Path,
) -> None:
    """switch_test попадает ТОЛЬКО в all_switch, не в all_routing/all_safety."""
    from d1.scripts.run_baselines import run_all_baselines

    routing, safety, switch = run_all_baselines(eval_sets=["switch_test"])

    assert "switch_test" in switch
    assert "switch_test" not in routing
    assert "switch_test" not in safety
    # Обе модели должны отчитаться по switch_test
    assert len(switch["switch_test"]) == 2


def test_run_all_baselines_routing_goes_to_all_routing(
    orchestration_env: Path,
) -> None:
    """test eval-set попадает в all_routing, не в all_switch."""
    from d1.scripts.run_baselines import run_all_baselines

    routing, safety, switch = run_all_baselines(eval_sets=["test"])

    assert "test" in routing
    assert "test" not in switch
    assert "test" not in safety


def test_save_results_separates_switch_and_routing(
    orchestration_env: Path, tmp_path: Path,
) -> None:
    """save_results создаёт switch_results.* и НЕ загрязняет baseline_results.*."""
    from d1.scripts.run_baselines import run_all_baselines, save_results

    out_dir = tmp_path / "results"
    routing, safety, switch = run_all_baselines(
        eval_sets=["test", "switch_test"],
    )
    save_results(routing, safety, switch, output_dir=out_dir)

    # Обязательные артефакты
    baseline_csv = out_dir / "baseline_results.csv"
    baseline_json = out_dir / "baseline_results.json"
    switch_csv = out_dir / "switch_results.csv"
    switch_json = out_dir / "switch_results.json"
    assert baseline_csv.exists()
    assert baseline_json.exists()
    assert switch_csv.exists()
    assert switch_json.exists()

    # baseline_results НЕ содержит switch-специфичных колонок / ключей
    import json as _json

    base_df = pd.read_csv(baseline_csv)
    assert "route_accuracy" not in base_df.columns
    assert not any(c.startswith("acc__") for c in base_df.columns)
    assert "switch_test" not in base_df["eval_set"].tolist()

    base_json = _json.loads(baseline_json.read_text(encoding="utf-8"))
    assert "switch_test" not in base_json

    # switch_results содержит interpretation_warning на верхнем уровне
    switch_json_data = _json.loads(switch_json.read_text(encoding="utf-8"))
    assert "interpretation_warning" in switch_json_data
    assert "NOT a switch detector" in switch_json_data["interpretation_warning"]


# ---------------------------------------------------------------------------
# CLI contract: дефолтный набор eval-сетов (защита от "тихой потери" набора)
# ---------------------------------------------------------------------------

def test_default_eval_sets_contract() -> None:
    """CLI дефолт покрывает все 8 eval-сетов согласно Task 2 roadmap."""
    from d1.scripts.run_baselines import _DEFAULT_EVAL_SETS

    expected = {
        "test", "val", "hard_test", "safety_set",
        "blind_test", "entity_held_out", "extended_eval", "switch_test",
    }
    assert set(_DEFAULT_EVAL_SETS) == expected, (
        f"Дефолт CLI потерял eval-сет: {expected - set(_DEFAULT_EVAL_SETS)}"
    )
