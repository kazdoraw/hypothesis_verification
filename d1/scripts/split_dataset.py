"""Split датасета D1 v6 по semantic families (seed_id).

Использует GroupShuffleSplit для разделения так, чтобы все вариации
одного seed оказались в одном split (предотвращение data leakage).

Выход: d1/data/d1_v6_{full,train,val,test,hard_test,switch_test}.csv

Запуск:
    cd study && python -m d1.scripts.split_dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import (
    DATA_DIR,
    HARD_CASES_FILE,
    SPLIT_RANDOM_STATE,
    TEST_RATIO,
    VAL_RATIO,
    CSV_COLUMNS,
    DATASET_PREFIX,
)
from utils.taxonomy_v6 import ROUTE_DOMAINS

logger = logging.getLogger(__name__)

_FULL_CSV = DATA_DIR / f"{DATASET_PREFIX}_full.csv"


def load_hard_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка hard cases и разделение на hard_test и switch_test.

    Returns:
        (hard_test_df, switch_test_df)
    """
    with open(HARD_CASES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = data["hard_cases"]

    regular = []
    switch = []
    for c in cases:
        row = {col: c.get(col, "") for col in CSV_COLUMNS}
        row["source"] = c.get("source", "manual_hard")
        row["seed_id"] = ""
        if c["id"].startswith("switch_"):
            row["active_domain"] = c.get("active_domain", "")
            switch.append(row)
        else:
            regular.append(row)

    hard_df = pd.DataFrame(regular)
    switch_df = pd.DataFrame(switch)

    # Приводим к нужным колонкам
    for col in CSV_COLUMNS:
        if col not in hard_df.columns:
            hard_df[col] = ""
        if col not in switch_df.columns:
            switch_df[col] = ""

    return hard_df[CSV_COLUMNS], switch_df


def group_shuffle_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Один split по seed_id группам.

    Returns:
        (remaining_df, split_df) — remaining = всё кроме split
    """
    groups = df["seed_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    remain_idx, split_idx = next(gss.split(df, groups=groups))
    return df.iloc[remain_idx].reset_index(drop=True), df.iloc[split_idx].reset_index(drop=True)


def check_class_balance(df: pd.DataFrame, name: str, min_pct: float = 0.05) -> None:
    """Проверка минимального баланса классов."""
    counts = df["route_domain"].value_counts(normalize=True)
    for domain in ROUTE_DOMAINS:
        pct = counts.get(domain, 0.0)
        if pct < min_pct:
            logger.warning(
                "%s: домен '%s' = %.1f%% (< %.0f%% minimum)",
                name, domain, pct * 100, min_pct * 100,
            )


def run_split() -> dict[str, pd.DataFrame]:
    """Основной pipeline split.

    Returns:
        dict с ключами: full, train, val, test, hard_test, switch_test
    """
    if not _FULL_CSV.exists():
        logger.error("Файл %s не найден. Сначала запустите generate_dataset.py", _FULL_CSV)
        sys.exit(1)

    full_df = pd.read_csv(_FULL_CSV, dtype=str).fillna("")
    logger.info("Загружено %d строк из %s", len(full_df), _FULL_CSV.name)

    # Grok-вариации (source != seed-only; seed включены)
    grok_df = full_df.copy()

    # Split 1: выделяем test (20%)
    train_val_df, test_df = group_shuffle_split(grok_df, TEST_RATIO, SPLIT_RANDOM_STATE)
    logger.info("Test: %d строк (%.1f%%)", len(test_df), len(test_df) / len(grok_df) * 100)

    # Split 2: из оставшегося выделяем val (12.5% от остатка ≈ 10% от total)
    train_df, val_df = group_shuffle_split(train_val_df, VAL_RATIO, SPLIT_RANDOM_STATE + 1)
    logger.info("Val: %d строк", len(val_df))
    logger.info("Train: %d строк", len(train_df))

    # Hard cases
    hard_test_df, switch_test_df = load_hard_cases()
    logger.info("Hard test: %d строк", len(hard_test_df))
    logger.info("Switch test: %d строк", len(switch_test_df))

    # Проверка no-leakage: seed_id не пересекается между train и test
    train_seeds = set(train_df["seed_id"].unique())
    test_seeds = set(test_df["seed_id"].unique())
    val_seeds = set(val_df["seed_id"].unique())
    overlap_train_test = train_seeds & test_seeds
    overlap_train_val = train_seeds & val_seeds
    if overlap_train_test:
        logger.error("LEAKAGE: train ∩ test seeds: %s", overlap_train_test)
    if overlap_train_val:
        logger.error("LEAKAGE: train ∩ val seeds: %s", overlap_train_val)

    # Проверка баланса
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        check_class_balance(df, name)

    # Сохранение
    splits = {
        "full": full_df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "hard_test": hard_test_df,
        "switch_test": switch_test_df,
    }

    for name, df in splits.items():
        path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  {name}: {len(df)} строк → {path.name}")

    # Сводка
    print(f"\n{'='*50}")
    print(f"  Train:       {len(train_df):>6}")
    print(f"  Val:         {len(val_df):>6}")
    print(f"  Test:        {len(test_df):>6}")
    print(f"  Hard test:   {len(hard_test_df):>6}")
    print(f"  Switch test: {len(switch_test_df):>6}")
    print(f"  {'─'*30}")
    print(f"  Итого:       {sum(len(d) for d in splits.values()):>6}")

    return splits


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="D1 v6 dataset split")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_split()


if __name__ == "__main__":
    main()
