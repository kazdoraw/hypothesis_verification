"""Build mini gold-audit sample for ComplexityGate (Task 5b плана).

Стратифицированная выборка ≈100 примеров из `hard_test` для ручной разметки
gold complexity класса. random_state=42 → детерминизм.

Output `d1/data/complexity_audit_sample.csv` колонки:
- `id`, `text`, `route_domain`, `urgency` — из hard_test;
- `gold_is_complex` (str, empty), `gold_complexity_class` (str, empty),
  `annotator` (str, empty), `annotation_date` (str, empty) — заполняются
  вручную автором работы.

Стратификация: `(primary_tag, route_domain)` × ≥10 на страту по возможности.
Если страта меньше 10 — берём всю.

Запуск:
    cd study && python -m d1.scripts.build_complexity_audit_sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from d1.baselines.complexity_gate import ComplexityGate
from d1.config import DATA_DIR, DATASET_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SIZE = 100
DEFAULT_RANDOM_STATE = 42
DEFAULT_PER_STRATUM = 10  # минимум на страту, если она достаточно большая

# Empty gold columns — заполняются вручную.
_GOLD_COLUMNS: tuple[str, ...] = (
    "gold_is_complex",
    "gold_complexity_class",
    "annotator",
    "annotation_date",
)


def build_audit_sample(
    hard_test_df: pd.DataFrame,
    out_path: Path,
    target_size: int = DEFAULT_TARGET_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    per_stratum: int = DEFAULT_PER_STRATUM,
) -> pd.DataFrame:
    """Построить sample CSV с пустыми gold-колонками.

    Args:
        hard_test_df: DataFrame со столбцами `text`, `route_domain`, `urgency`,
            опционально `id`. Если `id` нет, генерируется из индекса.
        out_path: куда сохранить CSV (создаётся parent dir).
        target_size: верхняя граница (≈100); реальный размер может быть меньше.
        random_state: seed для `pandas.DataFrame.sample`.
        per_stratum: target кол-ва samples per (primary_tag, route_domain).

    Returns:
        DataFrame, который записан на диск.
    """
    df = _ensure_id(hard_test_df)
    df = _attach_primary_tag(df)

    sampled = _stratified_sample(
        df,
        target_size=target_size,
        per_stratum=per_stratum,
        random_state=random_state,
    )
    output = _attach_gold_columns(sampled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    logger.info(
        "Saved audit sample: %s (n=%d, target=%d)", out_path, len(output), target_size,
    )
    return output


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантировать колонку `id`."""
    if "id" in df.columns:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    out["id"] = [f"hard_test_{i:04d}" for i in range(len(out))]
    return out


def _attach_primary_tag(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить временную колонку `_primary_tag` (используется только для страт)."""
    gate = ComplexityGate()
    decisions = gate.decide_batch(df["text"].tolist())
    out = df.copy()
    out["_primary_tag"] = [d.primary_tag for d in decisions]
    return out


def _stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    per_stratum: int,
    random_state: int,
) -> pd.DataFrame:
    """Стратифицированная выборка по (`_primary_tag`, `route_domain`).

    Стратегия:
    1. Группируем по (`_primary_tag`, `route_domain`).
    2. Из каждой группы берём `min(per_stratum, len(group))` samples.
    3. Если получилось < target_size, добираем uniformly из остатка.
    """
    groups = df.groupby(["_primary_tag", "route_domain"], sort=True)
    parts: list[pd.DataFrame] = []
    for _, group in groups:
        n_take = min(per_stratum, len(group))
        if n_take == 0:
            continue
        parts.append(group.sample(n=n_take, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    sampled_ids = set(sampled["id"].tolist())

    if len(sampled) < target_size:
        remainder = df[~df["id"].isin(sampled_ids)]
        n_extra = min(target_size - len(sampled), len(remainder))
        if n_extra > 0:
            extra = remainder.sample(n=n_extra, random_state=random_state)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    if len(sampled) > target_size:
        sampled = sampled.sample(
            n=target_size, random_state=random_state,
        ).reset_index(drop=True)

    # Убираем служебную колонку.
    sampled = sampled.drop(columns=["_primary_tag"], errors="ignore")
    # Стабильный порядок строк по id для детерминизма output'а.
    sampled = sampled.sort_values("id").reset_index(drop=True)
    return sampled


def _attach_gold_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить пустые gold-* и annotator/annotation_date колонки."""
    out = df.copy()
    for col in _GOLD_COLUMNS:
        out[col] = ""
    # Reorder: meta first, gold last (легче ревьюить в spreadsheet).
    cols_meta = [c for c in ["id", "text", "route_domain", "urgency"] if c in out.columns]
    cols_other = [c for c in out.columns if c not in cols_meta and c not in _GOLD_COLUMNS]
    return out[cols_meta + cols_other + list(_GOLD_COLUMNS)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build complexity audit sample for ComplexityGate gold-eval",
    )
    parser.add_argument(
        "--hard-test-path",
        type=Path,
        default=DATA_DIR / f"{DATASET_PREFIX}_hard_test.csv",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DATA_DIR / "complexity_audit_sample.csv",
    )
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.hard_test_path.exists():
        raise FileNotFoundError(f"hard_test не найден: {args.hard_test_path}")
    hard_test = pd.read_csv(args.hard_test_path, dtype=str).fillna("")
    build_audit_sample(
        hard_test_df=hard_test,
        out_path=args.out_path,
        target_size=args.target_size,
        per_stratum=args.per_stratum,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
