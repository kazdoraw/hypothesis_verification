#!/usr/bin/env python3
"""Quick checks for current D1 ML models.

Supports:
- interactive single-text predictions
- batch evaluation on CSV with gold labels
- full test split evaluation with error export

Examples:
  .venv/bin/python scripts/quick_check_d1_model.py --level l1 --text "У меня болит зуб"
  .venv/bin/python scripts/quick_check_d1_model.py --level l2 --csv data/d1_messages_v5_test.csv
  .venv/bin/python scripts/quick_check_d1_model.py --level l2 --csv data/d1_messages_v5_test.csv --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
DEFAULT_EXPORT_DIR = ROOT / "outputs" / "tables"


MODEL_CONFIG = {
    "l1": {
        "model": MODELS_DIR / "l1_gridsearch_best_v5.joblib",
        "vectorizer": MODELS_DIR / "tfidf_vectorizer_v5.joblib",
        "label_col": "label_l1",
    },
    "l2": {
        "model": MODELS_DIR / "l2_flat_svc_calibrated_v5.joblib",
        "vectorizer": MODELS_DIR / "tfidf_vectorizer_final_v5.joblib",
        "label_col": "label_l2",
    },
}


def load_artifacts(level: str):
    config = MODEL_CONFIG[level]
    model = joblib.load(config["model"])
    vectorizer = joblib.load(config["vectorizer"])
    return model, vectorizer, config["label_col"]


def predict_texts(model, vectorizer, texts: Iterable[str]):
    x = vectorizer.transform(list(texts))
    preds = model.predict(x)

    confidences = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(x)
        confidences = probas.max(axis=1)
    elif hasattr(model, "decision_function"):
        # Not probabilistic, but useful as relative confidence.
        scores = model.decision_function(x)
        if getattr(scores, "ndim", 1) == 1:
            confidences = [abs(float(s)) for s in scores]
        else:
            confidences = scores.max(axis=1)

    return preds, confidences


def evaluate_csv(level: str, csv_path: Path, limit: int | None = None):
    model, vectorizer, label_col = load_artifacts(level)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    if label_col not in rows[0]:
        raise ValueError(f"Column {label_col!r} not found in {csv_path}")

    texts = [row["text"] for row in rows]
    gold = [row[label_col] for row in rows]
    preds, confidences = predict_texts(model, vectorizer, texts)

    accuracy = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro")
    weighted_f1 = f1_score(gold, preds, average="weighted")
    labels = sorted(set(gold) | set(preds))
    cm = confusion_matrix(gold, preds, labels=labels)

    result = {
        "level": level,
        "csv_path": str(csv_path),
        "rows": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "label_col": label_col,
        "labels": labels,
        "classification_report": classification_report(gold, preds, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }

    errors = []
    for idx, row in enumerate(rows):
        if gold[idx] != preds[idx]:
            errors.append(
                {
                    "id": row.get("id", str(idx + 1)),
                    "text": row["text"],
                    "gold": gold[idx],
                    "pred": preds[idx],
                    "confidence": None if confidences is None else float(confidences[idx]),
                    "source": row.get("source", ""),
                    "complexity": row.get("complexity", ""),
                }
            )

    return result, errors


def export_results(level: str, result: dict, errors: list[dict]):
    DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_EXPORT_DIR / f"quick_eval_{level}_summary.json"
    errors_path = DEFAULT_EXPORT_DIR / f"quick_eval_{level}_errors.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(errors_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "text", "gold", "pred", "confidence", "source", "complexity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in errors:
            writer.writerow(row)

    return summary_path, errors_path


def main():
    parser = argparse.ArgumentParser(description="Quick checks for current D1 model")
    parser.add_argument("--level", choices=["l1", "l2"], required=True, help="Which model to inspect")
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--csv", type=Path, help="CSV dataset with gold labels")
    parser.add_argument("--limit", type=int, help="Limit number of rows from CSV")
    parser.add_argument("--top-errors", type=int, default=15, help="How many sample errors to print")
    args = parser.parse_args()

    if not args.text and not args.csv:
        parser.error("Use either --text or --csv")

    model, vectorizer, label_col = load_artifacts(args.level)

    if args.text:
        preds, confidences = predict_texts(model, vectorizer, [args.text])
        print(f"level: {args.level}")
        print(f"text: {args.text}")
        print(f"pred: {preds[0]}")
        if confidences is not None:
            print(f"confidence: {float(confidences[0]):.4f}")
        print(f"label_col: {label_col}")
        return

    result, errors = evaluate_csv(args.level, args.csv, args.limit)
    summary_path, errors_path = export_results(args.level, result, errors)

    print(f"level: {result['level']}")
    print(f"rows: {result['rows']}")
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro_f1: {result['macro_f1']:.4f}")
    print(f"weighted_f1: {result['weighted_f1']:.4f}")
    print(f"errors: {len(errors)}")
    print(f"summary: {summary_path}")
    print(f"errors_csv: {errors_path}")

    if errors:
        print("\nSample errors:")
        for row in errors[: args.top_errors]:
            conf = "n/a" if row["confidence"] is None else f"{row['confidence']:.4f}"
            print(f"- [{row['gold']} -> {row['pred']}] conf={conf} | {row['text']}")


if __name__ == "__main__":
    main()
