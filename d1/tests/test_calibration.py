"""Unit-тесты pure calibration functions (Task 3a roadmap).

Проверяемые контракты:
- ECE: идеальная калибровка → 0, полный disconnect → 1.
- Brier OvR: совпадает с ручным расчётом на synthetic data.
- Reliability table: пустые бины пропускаются, gap = acc - conf.
- Threshold table: coverage монотонна убывающая, overall_recall анти-корр. с t.
- Pareto frontier: не возвращает единичный threshold, только non-dominated.
- calibration.py НЕ импортирует matplotlib (acceptance criterion).

Запуск:
    cd study && python -m pytest d1/tests/test_calibration.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from d1.baselines.calibration import (
    compute_brier_ovr,
    compute_ece,
    compute_reliability_table,
    compute_threshold_table,
    find_pareto_candidates,
)


# ---------------------------------------------------------------------------
# Acceptance: pure module, без matplotlib
# ---------------------------------------------------------------------------

def test_calibration_module_has_no_matplotlib() -> None:
    """calibration.py НЕ должен импортировать matplotlib (разделение слоёв).

    Проверяем что ни один публичный атрибут модуля не пришёл из matplotlib
    (защита от кейса `from matplotlib ... import X`).
    """
    import d1.baselines.calibration as cal

    for name, obj in vars(cal).items():
        if name.startswith("_"):
            continue
        module = getattr(obj, "__module__", "")
        assert not module.startswith("matplotlib"), (
            f"calibration.py импортирует matplotlib через {name} ({module})"
        )


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration_zero() -> None:
    """Идеальная калибровка: в каждом бине avg_conf == avg_acc → ECE = 0."""
    # 2 бина: один с conf=0.2 и 20% accuracy, второй с conf=0.8 и 80% accuracy.
    conf = np.array([0.2] * 10 + [0.8] * 10)
    correct = np.array([True] * 2 + [False] * 8 + [True] * 8 + [False] * 2)
    ece = compute_ece(conf, correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_full_mismatch_high() -> None:
    """Conf=1 но всё wrong → ECE близко к 1 (в пределах разрешения бинов)."""
    conf = np.ones(100)
    correct = np.zeros(100, dtype=bool)
    ece = compute_ece(conf, correct, n_bins=10)
    assert ece == pytest.approx(1.0, abs=1e-6)


def test_ece_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="Длины не совпадают"):
        compute_ece(np.array([0.5, 0.6]), np.array([True]), n_bins=10)


def test_ece_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        compute_ece(np.array([0.5]), np.array([True]), n_bins=0)


def test_ece_empty_input() -> None:
    """Пустой вход → 0.0 (edge case)."""
    assert compute_ece(np.array([]), np.array([], dtype=bool)) == 0.0


# ---------------------------------------------------------------------------
# compute_brier_ovr
# ---------------------------------------------------------------------------

def test_brier_ovr_perfect() -> None:
    """Идеальные proba (one-hot по true) → Brier=0 по каждому классу."""
    y_true = ["a", "b", "c"]
    classes = ["a", "b", "c"]
    proba = np.eye(3)

    result = compute_brier_ovr(y_true, proba, classes)

    assert result["per_class"]["a"] == pytest.approx(0.0)
    assert result["per_class"]["b"] == pytest.approx(0.0)
    assert result["per_class"]["c"] == pytest.approx(0.0)
    assert result["macro"] == pytest.approx(0.0)


def test_brier_ovr_manual_match() -> None:
    """Brier совпадает с ручным расчётом для простого примера."""
    y_true = ["a", "b"]
    classes = ["a", "b"]
    # Для класса a: y_bin=[1,0], proba_a=[0.8, 0.3]
    # brier_a = mean((1-0.8)^2 + (0-0.3)^2) = mean(0.04 + 0.09) = 0.065
    # Для класса b: y_bin=[0,1], proba_b=[0.2, 0.7]
    # brier_b = mean((0-0.2)^2 + (1-0.7)^2) = mean(0.04 + 0.09) = 0.065
    proba = np.array([[0.8, 0.2], [0.3, 0.7]])

    result = compute_brier_ovr(y_true, proba, classes)
    assert result["per_class"]["a"] == pytest.approx(0.065)
    assert result["per_class"]["b"] == pytest.approx(0.065)
    assert result["macro"] == pytest.approx(0.065)


def test_brier_ovr_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="proba.shape"):
        compute_brier_ovr(
            y_true=["a", "b"],
            proba=np.array([[0.5, 0.5]]),  # только 1 строка
            classes=["a", "b"],
        )


# ---------------------------------------------------------------------------
# compute_reliability_table
# ---------------------------------------------------------------------------

def test_reliability_table_skips_empty_bins() -> None:
    """Бины без данных пропущены (count=0 не в результате)."""
    # Кластеризуем 4 значения в 2 бина из 10: [0.15, 0.15] → bin 0.1-0.2,
    # [0.85, 0.88] → bin 0.8-0.9. Остальные 8 бинов должны быть пропущены.
    conf = np.array([0.15, 0.15, 0.85, 0.88])
    correct = np.array([False, True, True, True])

    table = compute_reliability_table(conf, correct, n_bins=10)

    assert "count" in table.columns
    assert (table["count"] > 0).all()
    assert len(table) == 2


def test_reliability_table_invalid_n_bins() -> None:
    """n_bins < 1 → ValueError (API симметричен compute_ece)."""
    with pytest.raises(ValueError, match="n_bins"):
        compute_reliability_table(np.array([0.5]), np.array([True]), n_bins=0)


def test_reliability_table_gap_computation() -> None:
    """gap = avg_accuracy - avg_confidence."""
    conf = np.array([0.85, 0.85])
    correct = np.array([True, True])  # acc=1.0

    table = compute_reliability_table(conf, correct, n_bins=10)
    assert len(table) == 1
    assert table.iloc[0]["gap"] == pytest.approx(1.0 - 0.85)


# ---------------------------------------------------------------------------
# compute_threshold_table
# ---------------------------------------------------------------------------

def _simple_threshold_inputs() -> (
    tuple[np.ndarray, list[str], list[str], list[str]]
):
    """5 сэмплов, 3 класса (anamnesis/faq/booking)."""
    classes = ["anamnesis", "faq", "booking"]
    # high conf правильные: anam(0.9), faq(0.9), booking(0.9)
    # mid conf неправильный: anam-как-faq(0.6)
    # low conf правильный: anam(0.4)
    proba = np.array([
        [0.9, 0.05, 0.05],  # true=anam, pred=anam (correct)
        [0.05, 0.9, 0.05],  # true=faq, pred=faq
        [0.05, 0.05, 0.9],  # true=booking, pred=booking
        [0.3, 0.6, 0.1],    # true=anam, pred=faq (ошибка → false_faq_for_anam)
        [0.4, 0.3, 0.3],    # true=anam, pred=anam (correct, low conf)
    ])
    preds = ["anamnesis", "faq", "booking", "faq", "anamnesis"]
    true = ["anamnesis", "faq", "booking", "anamnesis", "anamnesis"]
    return proba, preds, true, classes


def test_threshold_table_contract_columns() -> None:
    proba, preds, true, classes = _simple_threshold_inputs()
    table = compute_threshold_table(proba, preds, true, classes)

    expected_cols = {
        "threshold", "coverage", "accepted_accuracy",
        "overall_recall_anamnesis", "false_faq_for_anamnesis",
        "accepted_count", "rejected_count",
    }
    assert expected_cols.issubset(set(table.columns))


def test_threshold_table_coverage_monotone_decreasing() -> None:
    """Coverage не возрастает при росте threshold."""
    proba, preds, true, classes = _simple_threshold_inputs()
    thresholds = np.linspace(0.3, 0.99, 20)
    table = compute_threshold_table(
        proba, preds, true, classes, thresholds=thresholds,
    )
    cov = table["coverage"].to_numpy()
    assert np.all(cov[1:] <= cov[:-1] + 1e-9)


def test_threshold_table_empty_thresholds_raises() -> None:
    """Пустой массив thresholds → ValueError."""
    proba, preds, true, classes = _simple_threshold_inputs()
    with pytest.raises(ValueError, match="thresholds"):
        compute_threshold_table(
            proba, preds, true, classes, thresholds=np.array([]),
        )


def test_threshold_table_out_of_range_thresholds_raises() -> None:
    """Thresholds вне [0, 1] → ValueError (защита от опечатки типа 1.5 или -0.1)."""
    proba, preds, true, classes = _simple_threshold_inputs()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_threshold_table(
            proba, preds, true, classes, thresholds=np.array([0.5, 1.5]),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_threshold_table(
            proba, preds, true, classes, thresholds=np.array([-0.1, 0.5]),
        )


def test_threshold_table_unsorted_thresholds_ok() -> None:
    """Неотсортированный thresholds — допустим, результат в порядке входа."""
    proba, preds, true, classes = _simple_threshold_inputs()
    table = compute_threshold_table(
        proba, preds, true, classes,
        thresholds=np.array([0.9, 0.3, 0.6]),
    )
    # Порядок строк в таблице соответствует порядку thresholds
    assert list(table["threshold"]) == [0.9, 0.3, 0.6]


def test_threshold_table_low_vs_high_threshold_recall() -> None:
    """t=0.3 покрывает всех → overall_recall_anam = 2/3 (2 correct anam)."""
    proba, preds, true, classes = _simple_threshold_inputs()

    table = compute_threshold_table(
        proba, preds, true, classes, thresholds=np.array([0.3, 0.95]),
    )
    # При t=0.3: 3 anam в датасете, из них 2 correct И accepted (0.9, 0.4) → 2/3
    assert table.iloc[0]["overall_recall_anamnesis"] == pytest.approx(2 / 3)
    # false_faq_for_anam: 1 случай (proba=0.6, pred=faq, true=anam, accepted)
    assert table.iloc[0]["false_faq_for_anamnesis"] == pytest.approx(1 / 3)

    # При t=0.95: никто не принят → recall=0
    assert table.iloc[1]["coverage"] == 0.0
    assert table.iloc[1]["accepted_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# find_pareto_candidates
# ---------------------------------------------------------------------------

def test_pareto_never_single_point_when_multiple_nondominated() -> None:
    """Когда есть несколько non-dominated точек — возвращается > 1."""
    df = pd.DataFrame({
        "threshold": [0.3, 0.5, 0.7, 0.9],
        "accepted_accuracy": [0.5, 0.7, 0.85, 0.95],
        "overall_recall_anamnesis": [0.95, 0.8, 0.6, 0.3],
        # Четыре точки, каждая выигрывает другу по одной оси → все non-dominated
    })

    frontier = find_pareto_candidates(df, top_k=10)
    assert len(frontier) == 4


def test_pareto_removes_dominated_point() -> None:
    """Доминируемая точка отфильтрована."""
    df = pd.DataFrame({
        "threshold": [0.3, 0.5, 0.7],
        "accepted_accuracy": [0.5, 0.6, 0.7],
        "overall_recall_anamnesis": [0.9, 0.5, 0.8],
        # Точка t=0.5: acc=0.6, rec=0.5 — доминируется t=0.7 (acc=0.7, rec=0.8)
    })

    frontier = find_pareto_candidates(df, top_k=10)
    thresholds = set(frontier["threshold"].tolist())
    assert 0.5 not in thresholds
    assert 0.3 in thresholds  # выигрывает по recall
    assert 0.7 in thresholds  # выигрывает по acc


def test_pareto_respects_top_k_subsampling() -> None:
    """При frontier > top_k возвращается ровно top_k точек."""
    df = pd.DataFrame({
        "threshold": np.linspace(0.3, 0.99, 50),
        # строго нарастающая accuracy, строго падающий recall → все non-dominated
        "accepted_accuracy": np.linspace(0.5, 0.99, 50),
        "overall_recall_anamnesis": np.linspace(0.95, 0.3, 50),
    })
    frontier = find_pareto_candidates(df, top_k=5)
    assert len(frontier) == 5


def test_pareto_missing_column_raises() -> None:
    df = pd.DataFrame({"threshold": [0.5], "accepted_accuracy": [0.9]})
    with pytest.raises(KeyError, match="overall_recall_anamnesis"):
        find_pareto_candidates(df)


def test_pareto_empty_input() -> None:
    df = pd.DataFrame(
        columns=["threshold", "accepted_accuracy", "overall_recall_anamnesis"],
    )
    frontier = find_pareto_candidates(df)
    assert frontier.empty


# ---------------------------------------------------------------------------
# Acceptance (Task 3 roadmap): grep-regression на analyze_confidence.py
# ---------------------------------------------------------------------------

def _read_analyze_confidence_source() -> str:
    from pathlib import Path as _P
    path = (
        _P(__file__).resolve().parent.parent / "scripts" / "analyze_confidence.py"
    )
    return path.read_text(encoding="utf-8")


def test_reliability_png_via_calibration_display() -> None:
    """Acceptance: reliability PNG строится через CalibrationDisplay.from_predictions.

    Grep-regression — защита от случайной замены на кастомный plot без
    sklearn-контракта.
    """
    src = _read_analyze_confidence_source()
    assert "CalibrationDisplay.from_predictions" in src, (
        "analyze_confidence.py должен строить reliability через "
        "sklearn.calibration.CalibrationDisplay.from_predictions"
    )


def test_pareto_candidates_only_on_val() -> None:
    """Acceptance: Pareto candidates строятся ТОЛЬКО на val (overfitting guard).

    Grep-проверка: в analyze_confidence.py есть условие `eval_set == "val"`
    вокруг накопления pareto bundles и отсутствует прямой вызов
    find_pareto_candidates внутри test/hard_test ветки.
    """
    src = _read_analyze_confidence_source()
    # Накопление val bundles для Pareto выполняется под условием eval_set == "val"
    assert 'eval_set == "val"' in src, (
        "analyze_confidence.py должен ограничивать Pareto накопление eval_set == 'val'"
    )
    # find_pareto_candidates НЕ должен вызываться напрямую в analyze_confidence —
    # только через save_val_pareto_candidates. Это защищает от случайного
    # применения Pareto к test/hard_test.
    # Разрешаем import; запрещаем прямой вызов find_pareto_candidates(...)
    # вне save_val_pareto_candidates (который сам по имени *_val).
    import re as _re
    direct_calls = _re.findall(r"find_pareto_candidates\s*\(", src)
    # Ожидаем ровно один вызов — внутри save_val_pareto_candidates.
    assert len(direct_calls) == 1, (
        f"find_pareto_candidates должен вызываться только в "
        f"save_val_pareto_candidates, найдено {len(direct_calls)} вызовов"
    )


# ---------------------------------------------------------------------------
# Integration: save_calibration_artifacts создаёт контрактные файлы
# ---------------------------------------------------------------------------

def test_save_calibration_artifacts_creates_files(tmp_path) -> None:
    """save_calibration_artifacts → 3 файла с корректной структурой."""
    from d1.scripts.analyze_confidence import (
        PredictionBundle,
        save_calibration_artifacts,
    )
    import json as _json

    # Минимальный bundle: 4 сэмпла, 3 класса
    classes = ["anamnesis", "faq", "booking"]
    proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.3, 0.4, 0.3],
        [0.05, 0.05, 0.9],
    ])
    preds = ["anamnesis", "faq", "faq", "booking"]
    true = ["anamnesis", "faq", "anamnesis", "booking"]
    confidence = proba.max(axis=1)

    bundle = PredictionBundle(
        name="TestModel",
        preds=preds,
        proba=proba,
        classes=classes,
        confidence=confidence,
        true_labels=true,
    )

    paths = save_calibration_artifacts(bundle, "test_synthetic", results_dir=tmp_path)

    # Все три файла созданы
    assert paths["threshold_table"].exists()
    assert paths["reliability_table"].exists()
    assert paths["metrics"].exists()

    # JSON содержит обязательные ключи
    metrics = _json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["baseline"] == "TestModel"
    assert metrics["eval_set"] == "test_synthetic"
    assert "ece" in metrics
    assert "brier_ovr" in metrics
    assert set(metrics["brier_ovr"]["per_class"].keys()) == set(classes)

    # Threshold table имеет контрактные колонки
    tt = pd.read_csv(paths["threshold_table"])
    for col in ["threshold", "coverage", "accepted_accuracy",
                "overall_recall_anamnesis", "false_faq_for_anamnesis"]:
        assert col in tt.columns
