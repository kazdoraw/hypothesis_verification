"""Phase 0 sanity: распределение `primary_tag` по всем splits.

Цель — выявить test-specific guard'ы. Если `booking_doctor_name`,
`mixed_intent` или `short_ambiguous` присутствуют только в `hard_test`,
это значит ComplexityGate настроена на конкретный split, а не на общий
паттерн. В таком случае tag-policy будет переобучена под hard_test.

Выход: `d1/results/phase0_tag_distribution_per_split.csv` с колонками:
    split, primary_tag, count, share

Запуск:
    python -m d1.scripts.diagnose_tag_distribution
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from d1.baselines.complexity_gate import ComplexityGate
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

# Все splits включая train (DEFAULT_EVAL_SETS из evaluate_simple_router
# не содержит train; здесь нужен train для сравнения).
ALL_SPLITS: tuple[str, ...] = (
    "train", "val", "test", "hard_test",
    "blind_test", "entity_held_out", "extended_eval",
)

# Тэги, для которых критична общая представленность в нескольких splits.
TEST_SPECIFIC_GUARDS: tuple[str, ...] = (
    "booking_doctor_name", "mixed_intent", "short_ambiguous",
)


def _load_split_texts(name: str) -> list[str]:
    """Прочитать тексты из split CSV."""
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    return df["text"].tolist()


def compute_tag_distribution(
    splits: tuple[str, ...] = ALL_SPLITS,
) -> pd.DataFrame:
    """Прогнать `ComplexityGate` на каждом split и собрать распределение.

    Returns:
        DataFrame с колонками `split, primary_tag, count, share`.
        Доли нормализованы внутри каждого split.
    """
    gate = ComplexityGate()
    rows: list[dict[str, object]] = []

    for split in splits:
        try:
            texts = _load_split_texts(split)
        except FileNotFoundError as exc:
            logger.warning("Skip %s: %s", split, exc)
            continue
        decisions = gate.decide_batch(texts)
        n = len(decisions)
        counts: dict[str, int] = {}
        for d in decisions:
            counts[d.primary_tag] = counts.get(d.primary_tag, 0) + 1
        for tag, count in sorted(counts.items()):
            rows.append({
                "split": split,
                "primary_tag": tag,
                "count": count,
                "share": round(count / n, 4) if n else 0.0,
            })

    return pd.DataFrame(rows)


def check_guard_coverage(
    df: pd.DataFrame,
    guards: tuple[str, ...] = TEST_SPECIFIC_GUARDS,
) -> dict[str, list[str]]:
    """Для каждого guard-tag вернуть список splits, где он встречается.

    Pass criteria плана: каждый guard должен присутствовать в ≥ 2 splits.
    """
    coverage: dict[str, list[str]] = {}
    for guard in guards:
        rows = df[df["primary_tag"] == guard]
        coverage[guard] = sorted(rows["split"].unique().tolist())
    return coverage


def main() -> Path:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = compute_tag_distribution()
    out_path = RESULTS_DIR / "phase0_tag_distribution_per_split.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)

    coverage = check_guard_coverage(df)
    logger.info("\nTest-specific guard coverage (pass = ≥ 2 splits):")
    for guard, splits in coverage.items():
        marker = "OK" if len(splits) >= 2 else "FAIL"
        logger.info("  [%s] %s: %s", marker, guard, splits or "<none>")
    return out_path


if __name__ == "__main__":
    main()
