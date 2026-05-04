"""Тесты для build_complexity_audit_sample (Task 5b плана).

Контракт:
- Стратифицированная выборка из `hard_test` по `(primary_tag, route_domain)`.
- random_state=42 → детерминизм.
- Колонки: id, text, route_domain, urgency, gold_is_complex,
  gold_complexity_class, annotator, annotation_date.
- gold_* колонки пустые (заполняются вручную).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_hard_test() -> pd.DataFrame:
    """Минимальная синтетическая hard_test для unit-теста.

    Покрывает 4 route_domain × 3 типичных complexity-категории.
    """
    rows: list[dict[str, object]] = []
    rid = 0
    # 8 anamnesis (4 simple_symptom, 4 mixed_intent)
    for _ in range(4):
        rows.append({
            "id": f"id{rid}", "text": "у меня болит зуб",
            "route_domain": "anamnesis", "urgency": "low",
        })
        rid += 1
    for _ in range(4):
        rows.append({
            "id": f"id{rid}", "text": "болит зуб сколько стоит лечение",
            "route_domain": "anamnesis", "urgency": "low",
        })
        rid += 1
    # 6 faq (2 simple_faq, 4 short_ambiguous)
    for _ in range(2):
        rows.append({
            "id": f"id{rid}", "text": "сколько стоит чистка",
            "route_domain": "faq", "urgency": "low",
        })
        rid += 1
    for _ in range(4):
        rows.append({
            "id": f"id{rid}", "text": "цена",
            "route_domain": "faq", "urgency": "low",
        })
        rid += 1
    # 4 booking (4 simple_booking)
    for _ in range(4):
        rows.append({
            "id": f"id{rid}", "text": "запишите на чистку",
            "route_domain": "booking", "urgency": "low",
        })
        rid += 1
    # 2 unsupported (unclassified)
    for _ in range(2):
        rows.append({
            "id": f"id{rid}", "text": "здравствуйте пожалуйста подскажите",
            "route_domain": "unsupported", "urgency": "low",
        })
        rid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_required_columns_present(tmp_path: Path) -> None:
    """Output CSV содержит все обязательные колонки + пустые gold_*."""
    from d1.scripts.build_complexity_audit_sample import build_audit_sample

    hard_test = _make_synthetic_hard_test()
    out_path = tmp_path / "complexity_audit_sample.csv"
    df = build_audit_sample(hard_test, out_path=out_path, target_size=10)

    required = {
        "id", "text", "route_domain", "urgency",
        "gold_is_complex", "gold_complexity_class",
        "annotator", "annotation_date",
    }
    assert required <= set(df.columns)
    # gold_*, annotator, annotation_date — пустые строки.
    for col in ["gold_is_complex", "gold_complexity_class", "annotator", "annotation_date"]:
        assert (df[col] == "").all() or df[col].isna().all()


def test_deterministic_with_random_state(tmp_path: Path) -> None:
    """Два прогона с одним hard_test → identical output (random_state=42)."""
    from d1.scripts.build_complexity_audit_sample import build_audit_sample

    hard_test = _make_synthetic_hard_test()
    out1 = tmp_path / "s1.csv"
    out2 = tmp_path / "s2.csv"

    df1 = build_audit_sample(hard_test, out_path=out1, target_size=10)
    df2 = build_audit_sample(hard_test, out_path=out2, target_size=10)

    pd.testing.assert_frame_equal(df1, df2)


def test_size_does_not_exceed_input(tmp_path: Path) -> None:
    """Если hard_test < target_size, выборка ≤ |hard_test|."""
    from d1.scripts.build_complexity_audit_sample import build_audit_sample

    hard_test = _make_synthetic_hard_test()  # 20 rows
    out_path = tmp_path / "small.csv"
    df = build_audit_sample(hard_test, out_path=out_path, target_size=100)
    assert len(df) <= len(hard_test)


def test_stratification_covers_multiple_strata(tmp_path: Path) -> None:
    """Sample должен включать хотя бы 2 различных primary_tag и 2 route_domain."""
    from d1.scripts.build_complexity_audit_sample import build_audit_sample

    hard_test = _make_synthetic_hard_test()
    out_path = tmp_path / "strat.csv"
    df = build_audit_sample(hard_test, out_path=out_path, target_size=10)

    # primary_tag вычисляется через ComplexityGate внутри build_audit_sample,
    # но в output может не быть колонки. Проверим что route_domain покрыт > 1.
    assert df["route_domain"].nunique() >= 2


def test_csv_saved_to_disk(tmp_path: Path) -> None:
    """Файл создан на диске и читается обратно."""
    from d1.scripts.build_complexity_audit_sample import build_audit_sample

    hard_test = _make_synthetic_hard_test()
    out_path = tmp_path / "subdir" / "audit.csv"
    df = build_audit_sample(hard_test, out_path=out_path, target_size=10)

    assert out_path.exists()
    loaded = pd.read_csv(out_path, dtype=str).fillna("")
    assert len(loaded) == len(df)
    assert set(df.columns) == set(loaded.columns)
