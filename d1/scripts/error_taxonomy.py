"""Formal error taxonomy для D1 (Task 7 roadmap).

Задача модуля — объяснить оставшиеся ошибки B1.1/B2.1 через стабильные,
проверяемые rule-based категории. Это не новая модель и не замена метрик:
taxonomy помогает понять, где именно остаётся риск после closed-set,
selective и hybrid экспериментов.

Методологические ограничения:
- категории `mixed_intent_error`, `post_treatment_ambiguity`,
  `specialization_confusion` опираются на CSV-разметку `subtype`, а не на
  свободный NLP-анализ текста;
- один пример может иметь несколько error_type, поэтому output multi-label:
  одна строка = (sample, error_type);
- audit CSV — шаблон для ручной проверки rule-based категорий. Автоматически
  не проставляем `yes/no`, чтобы не имитировать human audit.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from d1.baselines.trained_bundle import train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SETS = ["hard_test", "blind_test", "test"]
SPARSE_NAME = "B1.1_tfidf_lr"
DENSE_NAME = "B2.1_bge-m3_svc"

POST_TREATMENT_SUBTYPES = {"faq_borderline", "procedure_info"}
SPECIALIZATION_SUBTYPES = {
    "booking_with_spec",
    "booking_with_doctor",
    "doctor_info",
}


def classify_error_types(row: dict[str, Any] | pd.Series) -> list[str]:
    """Вернуть список rule-based error categories для одного error sample.

    Ожидаемые поля:
    `text`, `route_domain`, `subtype`, `pred_sparse`, `pred_dense`.

    Возвращает минимум один тег (`generic_error`), если строка является
    ошибкой, но не попадает в специфичные категории.
    """
    true_label = str(row.get("route_domain", ""))
    pred_sparse = str(row.get("pred_sparse", ""))
    pred_dense = str(row.get("pred_dense", ""))
    subtype = str(row.get("subtype", ""))
    text = str(row.get("text", ""))

    tags: list[str] = []

    # Pair-based taxonomy по sparse primary prediction.
    if true_label == "anamnesis" and pred_sparse == "faq":
        tags.append("anamnesis_to_faq")
    if true_label == "anamnesis" and pred_sparse == "booking":
        tags.append("anamnesis_to_booking")
    if true_label == "faq" and pred_sparse == "anamnesis":
        tags.append("faq_to_anamnesis")

    # Subtype-based taxonomy (работает над размеченными eval CSV).
    if "mixed_intent" in subtype:
        tags.append("mixed_intent_error")
    if subtype in POST_TREATMENT_SUBTYPES:
        tags.append("post_treatment_ambiguity")
    if subtype in SPECIALIZATION_SUBTYPES:
        tags.append("specialization_confusion")

    # Text-length taxonomy — допустимо для test/blind/hard.
    if len(_tokens(text)) <= 3:
        tags.append("vague_short_error")

    # Both-models-wrong: обе модели ошиблись (любые предсказания, могут
    # совпадать или нет). Если хотим именно disagreement — отдельный тег.
    if pred_sparse != true_label and pred_dense != true_label:
        tags.append("both_wrong")
        if pred_sparse != pred_dense:
            tags.append("models_disagree_both_wrong")

    if not tags:
        tags.append("generic_error")
    return _dedupe_keep_order(tags)


def build_error_taxonomy(
    df: pd.DataFrame,
    eval_set: str,
    sparse_model: Any,
    dense_model: Any,
) -> pd.DataFrame:
    """Построить multi-label taxonomy для одного eval set.

    Строки без ошибок обеих моделей исключаются. Если пример попал сразу в
    несколько категорий, он появляется несколькими строками с разным
    `error_type`.
    """
    required = {"id", "text", "route_domain", "subtype", "urgency", "source", "seed_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{eval_set}: missing columns {sorted(missing)}")

    work = df.copy().fillna("")
    texts = work["text"].tolist()
    work["pred_sparse"] = sparse_model.predict(texts)
    work["pred_dense"] = dense_model.predict(texts)
    work["sparse_error"] = work["pred_sparse"] != work["route_domain"]
    work["dense_error"] = work["pred_dense"] != work["route_domain"]
    errors = work[work["sparse_error"] | work["dense_error"]].copy()

    rows: list[dict[str, Any]] = []
    for _, row in errors.iterrows():
        tags = classify_error_types(row)
        for tag in tags:
            rows.append({
                "eval_set": eval_set,
                "id": row["id"],
                "text": row["text"],
                "true": row["route_domain"],
                "pred_sparse": row["pred_sparse"],
                "pred_dense": row["pred_dense"],
                "error_type": tag,
                "urgency": row["urgency"],
                "subtype": row["subtype"],
                "source": row["source"],
                "seed_id": row["seed_id"],
                "sparse_error": bool(row["sparse_error"]),
                "dense_error": bool(row["dense_error"]),
                "sparse_dense_disagree": row["pred_sparse"] != row["pred_dense"],
            })

    return pd.DataFrame(rows, columns=[
        "eval_set", "id", "text", "true", "pred_sparse", "pred_dense",
        "error_type", "urgency", "subtype", "source", "seed_id",
        "sparse_error", "dense_error", "sparse_dense_disagree",
    ])


def summarize_taxonomy(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """Сводка `[error_type, eval_set, count, pct_of_errors, example_text]`.

    `pct_of_errors` считается от числа уникальных error samples в eval_set,
    поэтому при multi-label категориях сумма процентов может быть > 1.0.
    """
    if taxonomy_df.empty:
        return pd.DataFrame(columns=[
            "eval_set", "error_type", "count", "pct_of_errors", "example_text",
        ])

    rows: list[dict[str, Any]] = []
    for eval_set, eval_df in taxonomy_df.groupby("eval_set", sort=True):
        n_error_samples = eval_df["id"].nunique()
        for error_type, group in eval_df.groupby("error_type", sort=True):
            rows.append({
                "eval_set": eval_set,
                "error_type": error_type,
                "count": int(len(group)),
                "pct_of_errors": (
                    round(len(group) / n_error_samples, 4)
                    if n_error_samples else 0.0
                ),
                "example_text": str(group.iloc[0]["text"])[:160],
            })
    return pd.DataFrame(rows).sort_values(
        ["eval_set", "count", "error_type"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def run_error_taxonomy(
    eval_sets: list[str] | None = None,
    audit_sample_size: int = 50,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Построить taxonomy CSV для eval sets + summary + audit sample."""
    eval_sets = eval_sets or DEFAULT_EVAL_SETS
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = train_bundle(names=[SPARSE_NAME, DENSE_NAME], use_cache=True)
    sparse = bundle.get(SPARSE_NAME)
    dense = bundle.get(DENSE_NAME)

    outputs: dict[str, pd.DataFrame] = {}
    all_taxonomy: list[pd.DataFrame] = []

    for eval_set in eval_sets:
        df = _load_split(eval_set)
        taxonomy = build_error_taxonomy(
            df=df,
            eval_set=eval_set,
            sparse_model=sparse,
            dense_model=dense,
        )
        outputs[eval_set] = taxonomy
        all_taxonomy.append(taxonomy)
        path = out_dir / f"error_taxonomy_{eval_set}.csv"
        taxonomy.to_csv(path, index=False)
        logger.info("Saved: %s", path)

    merged = (
        pd.concat(all_taxonomy, ignore_index=True)
        if all_taxonomy else pd.DataFrame()
    )
    summary = summarize_taxonomy(merged)
    summary_path = out_dir / "error_taxonomy_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved: %s", summary_path)

    audit = make_audit_sample(merged, n=audit_sample_size)
    audit_path = out_dir / "error_taxonomy_audit_sample.csv"
    audit.to_csv(audit_path, index=False)
    logger.info("Saved: %s", audit_path)

    print("\n=== ERROR TAXONOMY SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nAudit sample: {audit_path}")
    return outputs


def make_audit_sample(
    taxonomy_df: pd.DataFrame,
    n: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Создать шаблон ручного audit rule-based категорий.

    ВАЖНО: `rule_category_correct` не заполняется автоматически. Это поле
    предназначено для ручного решения `yes/no`.
    """
    if taxonomy_df.empty:
        base = pd.DataFrame(columns=taxonomy_df.columns)
    else:
        # Семплируем по уникальным (id, error_type), чтобы multi-label cases
        # попадали как отдельные проверяемые утверждения.
        base = taxonomy_df.drop_duplicates(["eval_set", "id", "error_type"])
        if len(base) > n:
            base = base.sample(n=n, random_state=random_state)

    audit = base.copy().reset_index(drop=True)
    audit["rule_category_correct"] = ""
    audit["manual_error_type"] = ""
    audit["audit_comment"] = ""
    return audit


def _load_split(eval_set: str) -> pd.DataFrame:
    path = DATA_DIR / f"{DATASET_PREFIX}_{eval_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower(), flags=re.IGNORECASE)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 formal error taxonomy")
    parser.add_argument("--eval-sets", nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--audit-sample-size", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_error_taxonomy(
        eval_sets=args.eval_sets,
        audit_sample_size=args.audit_sample_size,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_EVAL_SETS",
    "DENSE_NAME",
    "SPARSE_NAME",
    "build_error_taxonomy",
    "classify_error_types",
    "make_audit_sample",
    "run_error_taxonomy",
    "summarize_taxonomy",
]
