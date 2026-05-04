"""Аудит утечек (leakage) между train и test splits D1 v6.

Проверки:
1. Seed overlap: пересечение seed_id между train/val/test
2. Exact duplicates: одинаковые тексты в разных splits
3. Cosine similarity: семантически близкие пары train↔test (> порог)

Запуск:
    cd study && python -m d1.scripts.leakage_audit [--threshold 0.92]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import (
    DATA_DIR,
    DATASET_PREFIX,
    EMBEDDING_MODEL_PRIMARY,
    LEAKAGE_COSINE_THRESHOLD,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

_PAIRS_TO_CHECK = [
    ("train", "test"),
    ("train", "val"),
    ("train", "hard_test"),
    ("train", "blind_test"),
]


def load_split(name: str) -> pd.DataFrame:
    """Загрузка CSV split по имени."""
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        logger.warning("Split не найден: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def check_seed_overlap(splits: dict[str, pd.DataFrame]) -> int:
    """Проверка пересечения seed_id между splits.

    Returns:
        Количество обнаруженных утечек.
    """
    issues = 0
    for split_a, split_b in _PAIRS_TO_CHECK:
        if split_a not in splits or split_b not in splits:
            continue
        seeds_a = set(splits[split_a]["seed_id"].unique()) - {""}
        seeds_b = set(splits[split_b]["seed_id"].unique()) - {""}
        overlap = seeds_a & seeds_b
        if overlap:
            logger.error(
                "SEED LEAK: %s ∩ %s = %d seeds: %s",
                split_a, split_b, len(overlap), sorted(overlap)[:10],
            )
            issues += len(overlap)
        else:
            logger.info("OK: %s ∩ %s seed overlap = 0", split_a, split_b)
    return issues


def check_exact_duplicates(splits: dict[str, pd.DataFrame]) -> int:
    """Проверка точных дублей текстов между splits.

    Returns:
        Количество дублированных текстов.
    """
    issues = 0
    for split_a, split_b in _PAIRS_TO_CHECK:
        if split_a not in splits or split_b not in splits:
            continue
        texts_a = set(splits[split_a]["text"].str.lower())
        texts_b = set(splits[split_b]["text"].str.lower())
        dups = texts_a & texts_b
        if dups:
            logger.error(
                "EXACT DUP: %s ∩ %s = %d текстов",
                split_a, split_b, len(dups),
            )
            for t in sorted(dups)[:5]:
                logger.error("  '%s'", t[:80])
            issues += len(dups)
        else:
            logger.info("OK: %s ∩ %s exact dups = 0", split_a, split_b)
    return issues


def check_cosine_leakage(
    splits: dict[str, pd.DataFrame],
    threshold: float = LEAKAGE_COSINE_THRESHOLD,
) -> int:
    """Cosine similarity leakage: пары train↔test с sim > threshold.

    Использует sentence-transformers для кодирования.

    Returns:
        Количество подозрительных пар.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(resolve_model_path(EMBEDDING_MODEL_PRIMARY))
    issues = 0

    for split_a, split_b in _PAIRS_TO_CHECK:
        if split_a not in splits or split_b not in splits:
            continue
        if splits[split_b].empty:
            continue

        texts_a = splits[split_a]["text"].tolist()
        texts_b = splits[split_b]["text"].tolist()

        if not texts_a or not texts_b:
            continue

        logger.info(
            "Cosine check: %s (%d) ↔ %s (%d)...",
            split_a, len(texts_a), split_b, len(texts_b),
        )

        emb_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False)
        emb_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False)

        # Матрица similarity: (len_a, len_b)
        sim_matrix = emb_a @ emb_b.T
        max_sims = sim_matrix.max(axis=0)  # для каждой строки test — макс из train

        suspicious = int(np.sum(max_sims >= threshold))
        if suspicious:
            logger.warning(
                "COSINE LEAK: %s → %s: %d строк с sim >= %.2f",
                split_a, split_b, suspicious, threshold,
            )
            # Топ-5 подозрительных пар
            top_indices = np.argsort(max_sims)[::-1][:5]
            for idx in top_indices:
                if max_sims[idx] < threshold:
                    break
                best_train_idx = sim_matrix[:, idx].argmax()
                logger.warning(
                    "  sim=%.3f: test='%s' ← train='%s'",
                    max_sims[idx],
                    texts_b[idx][:60],
                    texts_a[best_train_idx][:60],
                )
            issues += suspicious
        else:
            logger.info(
                "OK: %s ↔ %s max_sim=%.3f (< %.2f)",
                split_a, split_b, float(max_sims.max()), threshold,
            )

    return issues


def run_audit(threshold: float = LEAKAGE_COSINE_THRESHOLD) -> dict[str, int]:
    """Полный аудит утечек.

    Returns:
        dict с количеством issues по типам.
    """
    split_names = ["train", "val", "test", "hard_test", "blind_test"]
    splits = {}
    for name in split_names:
        df = load_split(name)
        if not df.empty:
            splits[name] = df

    if "train" not in splits:
        logger.error("Train split не найден — аудит невозможен")
        return {"seed_overlap": -1, "exact_duplicates": -1, "cosine_leakage": -1}

    print("=" * 50)
    print("D1 v6 Leakage Audit")
    print("=" * 50)

    seed_issues = check_seed_overlap(splits)
    dup_issues = check_exact_duplicates(splits)
    cosine_issues = check_cosine_leakage(splits, threshold)

    results = {
        "seed_overlap": seed_issues,
        "exact_duplicates": dup_issues,
        "cosine_leakage": cosine_issues,
    }

    print(f"\n{'─'*50}")
    total = sum(results.values())
    status = "PASS ✓" if total == 0 else f"FAIL ✗ ({total} issues)"
    print(f"Результат: {status}")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="D1 v6 leakage audit")
    parser.add_argument(
        "--threshold", type=float, default=LEAKAGE_COSINE_THRESHOLD,
        help=f"Cosine similarity порог (default: {LEAKAGE_COSINE_THRESHOLD})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_audit(args.threshold)


if __name__ == "__main__":
    main()
