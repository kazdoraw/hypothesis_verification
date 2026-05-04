"""Split датасета D1 v6 по semantic families (seed_id).

Использует GroupShuffleSplit для разделения так, чтобы все вариации
одного seed оказались в одном split (предотвращение data leakage).

Выход:
  - d1/data/d1_v6_{full,train,val,test,hard_test,switch_test}.csv
  - d1/data/d1_v6_safety_set.csv (ургентные случаи)
  - d1/data/entity_held_out.json (манифест unseen сущностей)

Запуск:
    cd study && python -m d1.scripts.split_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import (
    CSV_COLUMNS,
    DATA_DIR,
    DATASET_PREFIX,
    ENTITY_HELD_OUT,
    EXTENDED_FAQ_CATEGORIES,
    HARD_CASES_FILE,
    SPLIT_RANDOM_STATE,
    TEST_RATIO,
    VAL_RATIO,
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


def stratified_group_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group split, стратифицированный по route_domain.

    Гарантирует пропорциональное представительство каждого домена
    в split-части. Группировка по seed_id (все вариации seed → один split).

    Returns:
        (remaining_df, split_df)
    """
    rng = random.Random(random_state)

    # Уникальные seed families с их доменом
    seed_domain = (
        df[df["seed_id"] != ""]
        .groupby("seed_id")["route_domain"]
        .first()
        .to_dict()
    )

    # Группируем seed_ids по домену
    domain_seeds: dict[str, list[str]] = {}
    for sid, dom in seed_domain.items():
        domain_seeds.setdefault(dom, []).append(sid)

    split_seed_ids: set[str] = set()
    for dom, sids in domain_seeds.items():
        rng.shuffle(sids)
        n_split = max(1, round(len(sids) * test_size))
        split_seed_ids.update(sids[:n_split])

    mask = df["seed_id"].isin(split_seed_ids)
    split_df = df[mask].reset_index(drop=True)
    remaining_df = df[~mask].reset_index(drop=True)
    return remaining_df, split_df


def _text_has_entity(text: str) -> tuple[bool, str]:
    """Проверяет, содержит ли текст unseen entity маркер.

    Returns:
        (has_entity, matched_marker)
    """
    text_lower = text.lower()
    for _cat, keywords in ENTITY_HELD_OUT.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return True, kw.lower()
    return False, ""


def separate_entity_seeds(
    full_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Разделение: seed'ы с unseen entities → force в test.

    Определяет seed_id'ы, чьи SEED-тексты (source=seed) содержат
    маркеры из ENTITY_HELD_OUT. ВСЕ строки этих seed_id (seed + вариации)
    отделяются от основного набора.

    Returns:
        (remaining_df, entity_df, manifest)
    """
    # Находим seed_id'ы, содержащие unseen entities
    seed_rows = full_df[full_df["source"] == "seed"]
    entity_seed_ids: set[str] = set()
    matched: dict[str, set[str]] = {cat: set() for cat in ENTITY_HELD_OUT}

    for _, row in seed_rows.iterrows():
        has, marker = _text_has_entity(row["text"])
        if has:
            entity_seed_ids.add(row["seed_id"])
            for cat, keywords in ENTITY_HELD_OUT.items():
                if marker in [k.lower() for k in keywords]:
                    matched[cat].add(marker)

    # Все строки этих seed_id → entity set
    mask = full_df["seed_id"].isin(entity_seed_ids)
    entity_df = full_df[mask].reset_index(drop=True)
    remaining_df = full_df[~mask].reset_index(drop=True)

    manifest = {
        cat: sorted(ents) for cat, ents in matched.items() if ents
    }
    manifest["entity_seed_ids"] = sorted(entity_seed_ids)
    manifest["total_samples"] = len(entity_df)
    manifest["total_seeds"] = len(entity_seed_ids)

    logger.info(
        "Entity separation: %d seed_ids (%d строк) → forced test",
        len(entity_seed_ids), len(entity_df),
    )
    return remaining_df, entity_df, manifest


def verify_no_entity_leak(train_df: pd.DataFrame) -> int:
    """Проверяет, что train не содержит unseen entity маркеров.

    Returns:
        Количество утечек (0 = OK).
    """
    leaks = 0
    for _, row in train_df.iterrows():
        has, marker = _text_has_entity(row["text"])
        if has:
            logger.error(
                "ENTITY LEAK in train: '%s' содержит '%s'",
                row["text"][:60], marker,
            )
            leaks += 1
    if leaks == 0:
        logger.info("OK: train не содержит unseen entity маркеров")
    return leaks


def extract_safety_set(hard_test_df: pd.DataFrame) -> pd.DataFrame:
    """Выделение safety set: clinical urgent/emergency случаи.

    Clinical-only: только urgent cases с gold route_domain=anamnesis.
    Non-anamnesis urgent (напр. "Срочно" → booking) остаются
    в hard_test, но не входят в safety-метрику recall_urgent.

    Источник: hard_cases с urgency != 'normal' и route_domain == 'anamnesis'.
    """
    urgent_mask = hard_test_df["urgency"].isin(["urgent", "emergency"])
    clinical_mask = hard_test_df["route_domain"] == "anamnesis"
    safety_df = hard_test_df[urgent_mask & clinical_mask].reset_index(drop=True)

    n_excluded = int(urgent_mask.sum()) - len(safety_df)
    if n_excluded > 0:
        logger.info(
            "Safety set: %d clinical urgent (excluded %d non-anamnesis urgent)",
            len(safety_df), n_excluded,
        )
    else:
        logger.info("Safety set: %d строк (из hard_test)", len(safety_df))
    return safety_df


def load_blind_test() -> pd.DataFrame:
    """Загрузка human-written blind test из YAML."""
    blind_path = DATA_DIR / "d1_v6_blind_test.yaml"
    if not blind_path.exists():
        logger.warning("Blind test не найден: %s", blind_path)
        return pd.DataFrame(columns=CSV_COLUMNS)

    with open(blind_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []
    for c in data.get("blind_test", []):
        row = {col: c.get(col, "") for col in CSV_COLUMNS}
        row["source"] = "human_blind"
        row["seed_id"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Blind test: %d строк", len(df))
    return df[CSV_COLUMNS] if not df.empty else df


def separate_extended_cases(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Отделяем policy_only/subjective seed families от core dataset.

    Все строки (seed + вариации) seed'ов с extended faq_category
    выносятся в extended_eval. Остальное → core.

    Returns:
        (core_df, extended_df)
    """
    # Находим seed_ids с extended faq_category
    extended_seed_ids: set[str] = set()
    for _, row in df.iterrows():
        if row.get("faq_category", "") in EXTENDED_FAQ_CATEGORIES:
            sid = row.get("seed_id", "")
            if sid:
                extended_seed_ids.add(sid)

    if not extended_seed_ids:
        return df, pd.DataFrame(columns=df.columns)

    mask = df["seed_id"].isin(extended_seed_ids)
    extended_df = df[mask].reset_index(drop=True)
    core_df = df[~mask].reset_index(drop=True)

    logger.info(
        "Extended separation: %d seed families (%d строк) → extended_eval",
        len(extended_seed_ids), len(extended_df),
    )
    return core_df, extended_df


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
        dict с ключами: full, train, val, test, hard_test, safety_set, switch_test
    """
    if not _FULL_CSV.exists():
        logger.error("Файл %s не найден. Сначала запустите generate_dataset.py", _FULL_CSV)
        sys.exit(1)

    full_df = pd.read_csv(_FULL_CSV, dtype=str).fillna("")
    logger.info("Загружено %d строк из %s", len(full_df), _FULL_CSV.name)

    # 0. Extended separation: policy_only/subjective → extended_eval
    core_df, extended_df = separate_extended_cases(full_df)

    # 1. Entity separation: seed'ы с unseen entities → forced в test
    remaining_df, entity_df, entity_manifest = separate_entity_seeds(core_df)

    # 2. Split remaining: test (20%)
    train_val_df, test_random_df = stratified_group_split(
        remaining_df, TEST_RATIO, SPLIT_RANDOM_STATE,
    )
    # Entity rows → объединяем с random test
    test_df = pd.concat([test_random_df, entity_df], ignore_index=True)
    logger.info(
        "Test: %d строк (%d random + %d entity-forced)",
        len(test_df), len(test_random_df), len(entity_df),
    )

    # 3. Split remaining: val (12.5% от остатка ≈ 10% от total)
    train_df, val_df = stratified_group_split(
        train_val_df, VAL_RATIO, SPLIT_RANDOM_STATE + 1,
    )
    logger.info("Val: %d строк", len(val_df))
    logger.info("Train: %d строк", len(train_df))

    # 4. Hard cases
    hard_test_df, switch_test_df = load_hard_cases()
    logger.info("Hard test: %d строк", len(hard_test_df))
    logger.info("Switch test: %d строк", len(switch_test_df))

    # 5. Проверки
    # 5a. Seed overlap
    train_seeds = set(train_df["seed_id"].unique()) - {""}
    test_seeds = set(test_df["seed_id"].unique()) - {""}
    val_seeds = set(val_df["seed_id"].unique()) - {""}
    overlap_train_test = train_seeds & test_seeds
    overlap_train_val = train_seeds & val_seeds
    if overlap_train_test:
        logger.error("LEAKAGE: train ∩ test seeds: %s", overlap_train_test)
    if overlap_train_val:
        logger.error("LEAKAGE: train ∩ val seeds: %s", overlap_train_val)

    # 5b. Entity leak verification
    verify_no_entity_leak(train_df)

    # 5c. Class balance
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        check_class_balance(df, name)

    # 6. Safety set: urgent/emergency из hard_cases
    safety_set_df = extract_safety_set(hard_test_df)

    # 7. Blind test (human-written)
    blind_test_df = load_blind_test()

    # Сохранение CSV
    splits = {
        "full": full_df,
        "core": core_df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "hard_test": hard_test_df,
        "safety_set": safety_set_df,
        "blind_test": blind_test_df,
        "switch_test": switch_test_df,
        "extended_eval": extended_df,
    }

    for name, df in splits.items():
        path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  {name}: {len(df)} строк → {path.name}")

    # Entity-held-out (subset, не отдельный split)
    entity_path = DATA_DIR / f"{DATASET_PREFIX}_entity_held_out.csv"
    entity_df.to_csv(entity_path, index=False, encoding="utf-8")
    print(f"  entity_held_out: {len(entity_df)} строк → {entity_path.name}")

    # Entity manifest JSON
    manifest_path = DATA_DIR / "entity_held_out.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entity_manifest, f, ensure_ascii=False, indent=2)
    print(f"  manifest → {manifest_path.name}")

    # Сводка
    primary = ["train", "val", "test", "hard_test", "blind_test", "switch_test"]
    total_primary = sum(len(splits[k]) for k in primary)
    print(f"\n{'='*50}")
    print(f"  Core corpus: {len(core_df)} строк (из {len(full_df)} full)")
    print("  Primary splits (disjoint, from core):")
    print(f"    Train:           {len(train_df):>6}")
    print(f"    Val:             {len(val_df):>6}")
    print(f"    Test:            {len(test_df):>6}")
    print(f"    Hard test:       {len(hard_test_df):>6}")
    print(f"    Blind test:      {len(blind_test_df):>6} (human-written)")
    print(f"    Switch test:     {len(switch_test_df):>6}")
    print(f"    {'─'*28}")
    print(f"    Итого:         {total_primary:>6}")
    print(f"  Separate sets:")
    print(f"    extended_eval:   {len(extended_df):>6} (policy_only + subjective)")
    print(f"  Subsets (overlap with above):")
    print(f"    entity_held_out: {len(entity_df):>6} (⊂ test)")
    print(f"    safety_set:      {len(safety_set_df):>6} (⊂ hard_test)")

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
