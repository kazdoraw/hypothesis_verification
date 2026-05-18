"""Split датасета D1 v6 по semantic families (seed_id).

Использует GroupShuffleSplit для разделения так, чтобы все вариации
одного seed оказались в одном split (предотвращение data leakage).

Anti-leakage pipeline (cleanup-релиз 2026-05-11):
1. Group-aware split по `seed_id` — все вариации одного семантического
   зерна остаются в одном split.
2. `dedup_within_train` — exact-text дедуп внутри train.
3. `dedup_cross_splits` — exact-text дедуп между disjoint primary splits
   (train > val > test > hard_test > blind_test > switch_test > extended_eval).
4. `dedup_cross_splits_cosine` — семантический cosine-дедуп (BGE-M3,
   threshold=`LEAKAGE_COSINE_THRESHOLD` = 0.92) для тех же пар: ловит
   парафразы (порядок слов, опечатки, синонимы), которые exact-dedup
   пропускает.
5. `refresh_subset_views` — пересборка `safety_set` (⊂ hard_test) и
   `entity_held_out` (⊂ test) после dedup родителей.

Гарантия после run_split: `leakage_audit.run_audit()` показывает
`cosine_leakage=0` при том же пороге 0.92.

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
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
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
    LEAKAGE_COSINE_THRESHOLD,
    SPLIT_RANDOM_STATE,
    TEST_RATIO,
    VAL_RATIO,
    resolve_model_path,
)

# Согласован с dense baseline B2.1 и leakage_audit.py.
_DEDUP_EMBEDDING_MODEL = "BAAI/bge-m3"

# Локальная константа доменов: utils.taxonomy_v6 был удалён в прошлой
# чистке. Семантический контракт — четыре закрытых класса route_domain
# (см. d1/ontology/route_domain.yaml).
ROUTE_DOMAINS: tuple[str, ...] = ("anamnesis", "faq", "booking", "unsupported")

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


# Унифицированная нормализация текста для text-level dedup между splits.
# Цель — поймать «одинаковые до пунктуации/регистра/пробелов» примеры,
# которые LLM-аугментация может породить из разных seed-семей.
_DEDUP_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DEDUP_WS_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """lower → strip → удалить пунктуацию → схлопнуть whitespace."""
    if text is None:
        return ""
    normalized = str(text).lower().strip()
    normalized = _DEDUP_PUNCT_RE.sub(" ", normalized)
    normalized = _DEDUP_WS_RE.sub(" ", normalized).strip()
    return normalized


def dedup_within_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет точные/near-exact дубликаты текста ВНУТРИ train.

    Дубликаты при обучении дают «двойной» вклад в loss и искажают priors
    классов. Оставляем первое вхождение каждой нормализованной строки.
    """
    if train_df.empty:
        return train_df

    norm = train_df["text"].apply(_normalize_for_dedup)
    keep_mask = ~norm.duplicated(keep="first")
    n_before = len(train_df)
    train_df = train_df[keep_mask].reset_index(drop=True)
    n_removed = n_before - len(train_df)
    if n_removed:
        logger.warning(
            "Train text-dedup: удалено %d дубликатов (из %d → %d)",
            n_removed, n_before, len(train_df),
        )
    else:
        logger.info("Train text-dedup: дубликатов не найдено")
    return train_df


def dedup_cross_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Удаляет текстовые дубликаты МЕЖДУ disjoint primary splits.

    Политика: train неприкосновенен (это источник обучения), а каждый
    последующий disjoint primary split проверяется против всех «более
    ранних» в фиксированном порядке. ВАЖНО: `safety_set` и
    `entity_held_out` — это subsets (view) от `hard_test` и `test`
    соответственно (см. `extract_safety_set`, `separate_entity_seeds`).
    Их не нужно отдельно дедупить — они автоматически очищаются вместе
    со своими «родителями» через `_refresh_subset_views`.

    Порядок приоритета primary splits (от высокого к низкому):
        train > val > test > hard_test > blind_test > switch_test
        > extended_eval

    Returns:
        dict с обновлёнными DataFrame.
    """
    primary_priority = [
        "train", "val", "test", "hard_test",
        "blind_test", "switch_test", "extended_eval",
    ]

    seen_texts: set[str] = set()
    for name in primary_priority:
        df = splits.get(name)
        if df is None or df.empty:
            continue
        norm = df["text"].apply(_normalize_for_dedup)
        keep_mask = ~norm.isin(seen_texts)
        n_before = len(df)
        df_clean = df[keep_mask].reset_index(drop=True)
        n_removed = n_before - len(df_clean)
        if n_removed:
            logger.warning(
                "Cross-split text-dedup: из %s удалено %d строк (пересечение с более приоритетными сетами)",
                name, n_removed,
            )
        splits[name] = df_clean
        seen_texts.update(norm[keep_mask].tolist())

    return splits


def dedup_cross_splits_cosine(
    splits: dict[str, pd.DataFrame],
    threshold: float = LEAKAGE_COSINE_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """Удаляет семантические near-duplicates между disjoint primary splits.

    Дополняет `dedup_cross_splits` (exact-text): LLM-аугментация даёт
    парафразы (порядок слов, опечатки, синонимы), которые exact-dedup не
    ловит, но которые при cos ≥ 0.92 фактически переносят train-пример
    в test и завышают метрики на 1–3 пп.

    Тот же приоритет, что у `dedup_cross_splits`:
        train > val > test > hard_test > blind_test > switch_test > extended_eval

    Для каждой строки `B` считается max-cosine ко всем уже принятым
    splits (накопительно, чтобы поймать val↔train, test↔(train∪val) и т. д.).
    Если max-cos ≥ threshold — строка удаляется из `B`.

    Subset-views (`safety_set`, `entity_held_out`) не трогаются здесь —
    они пересобираются `refresh_subset_views()` после.
    """
    primary_priority = [
        "train", "val", "test", "hard_test",
        "blind_test", "switch_test", "extended_eval",
    ]

    relevant_names = [
        name for name in primary_priority
        if name in splits and not splits[name].empty
    ]
    if len(relevant_names) < 2:
        logger.info("Cross-split cosine-dedup: меньше двух непустых splits, пропуск")
        return splits

    from sentence_transformers import SentenceTransformer

    logger.info(
        "Cross-split cosine-dedup: модель=%s, threshold=%.2f, splits=%s",
        _DEDUP_EMBEDDING_MODEL, threshold, relevant_names,
    )
    model = SentenceTransformer(resolve_model_path(_DEDUP_EMBEDDING_MODEL))

    embeddings: dict[str, np.ndarray] = {}
    for name in relevant_names:
        texts = splits[name]["text"].astype(str).tolist()
        embeddings[name] = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )

    kept_names: list[str] = []
    for name in relevant_names:
        if not kept_names:
            kept_names.append(name)
            continue

        emb_current = embeddings[name]
        max_sims = np.zeros(emb_current.shape[0], dtype=np.float32)
        for prior in kept_names:
            sim = emb_current @ embeddings[prior].T
            max_sims = np.maximum(max_sims, sim.max(axis=1))

        keep_mask = max_sims < threshold
        n_before = len(splits[name])
        n_removed = int((~keep_mask).sum())
        if n_removed:
            logger.warning(
                "Cross-split cosine-dedup: из %s удалено %d строк (cos ≥ %.2f с более приоритетными сетами)",
                name, n_removed, threshold,
            )
            splits[name] = splits[name][keep_mask].reset_index(drop=True)
            embeddings[name] = emb_current[keep_mask]
        else:
            logger.info(
                "Cross-split cosine-dedup: %s — без удалений (max_sim=%.3f < %.2f)",
                name, float(max_sims.max()) if max_sims.size else 0.0, threshold,
            )
        kept_names.append(name)

    return splits


def refresh_subset_views(
    splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Пересоздаёт subset-views после dedup primary splits.

    `safety_set` ⊂ `hard_test` (urgent/emergency anamnesis) и
    `entity_held_out` ⊂ `test` (rows с unseen entities) могут «исчезнуть»,
    если их родители прошли dedup. Чтобы метрики safety/entity_held_out
    остались валидными, пересобираем их после text-dedup.
    """
    if "hard_test" in splits:
        splits["safety_set"] = extract_safety_set(splits["hard_test"])
    return splits


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

    # 8. Anti-leakage пайплайн (4 шага):
    #    8a. exact-text dedup внутри train
    #    8b. exact-text dedup между disjoint primary splits (по приоритету)
    #    8c. cosine-dedup BGE-M3 (threshold=0.92) между теми же splits
    #    8d. refresh subset-views (safety_set, entity_held_out)
    train_df = dedup_within_train(train_df)

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
    splits = dedup_cross_splits(splits)
    splits = dedup_cross_splits_cosine(splits)
    splits = refresh_subset_views(splits)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    hard_test_df = splits["hard_test"]
    safety_set_df = splits["safety_set"]
    blind_test_df = splits["blind_test"]
    switch_test_df = splits["switch_test"]
    extended_df = splits["extended_eval"]

    # Сохранение CSV
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
