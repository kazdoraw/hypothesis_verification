#!/usr/bin/env python3
"""Generate reproducible artifacts for D1 hypothesis evaluation.

Outputs:
- metrics summary csv/json
- classification reports
- error analysis csv
- confusion matrices (L1, L2)
- per-class F1 bar charts (L1, L2)
- dataset distribution plots
- error rate by text length
- reliability diagram for calibrated L2 model
- markdown report

Run:
  .venv/bin/python scripts/generate_d1_hypothesis_artifacts.py
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-config-"))

import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "d1_messages_v5_test.csv"
MODELS_DIR = ROOT / "outputs" / "models"
OUT_DIR = ROOT / "outputs" / "hypothesis_d1"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"

L1_MODEL_PATH = MODELS_DIR / "l1_gridsearch_best_v5.joblib"
L1_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_v5.joblib"
L2_MODEL_PATH = MODELS_DIR / "l2_flat_svc_calibrated_v5.joblib"
L2_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_final_v5.joblib"
METHODS_TABLE_PATH = ROOT / "outputs" / "tables" / "v5_methods_comparison.csv"

L1_TARGET = 0.85
L2_TARGET = 0.90


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_model_and_vectorizer(model_path: Path, vectorizer_path: Path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict(model, vectorizer, texts: list[str]):
    x = vectorizer.transform(texts)
    preds = model.predict(x)
    probas = model.predict_proba(x) if hasattr(model, "predict_proba") else None
    decisions = model.decision_function(x) if hasattr(model, "decision_function") else None
    return preds, probas, decisions


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n": len(y_true),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path, figsize=(10, 8)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_bars(report: dict[str, Any], labels: list[str], title: str, path: Path, target: float) -> None:
    values = [report[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), values, color="#1f77b4", alpha=0.85)
    ax.axhline(target, color="green", linestyle="--", linewidth=1.5, label=f"Target {target:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(counter: Counter, title: str, path: Path, color: str) -> None:
    labels, values = zip(*counter.items())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01, str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_rate_by_length(rows: list[dict[str, Any]], path: Path) -> None:
    buckets = {
        "<=3 words": [],
        "4-6 words": [],
        ">=7 words": [],
    }
    for row in rows:
        words = len(row["text"].split())
        if words <= 3:
            bucket = "<=3 words"
        elif words <= 6:
            bucket = "4-6 words"
        else:
            bucket = ">=7 words"
        buckets[bucket].append(1 if row["label_l2"] != row["pred_l2"] else 0)

    labels = list(buckets.keys())
    values = [float(np.mean(buckets[label])) if buckets[label] else 0.0 for label in labels]
    counts = [len(buckets[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#d62728", alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Error rate")
    ax.set_title("L2 Error Rate by Text Length")
    for bar, val, count in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2%}\n(n={count})", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(y_true: list[str], y_pred: list[str], probas: np.ndarray, labels: list[str], path: Path) -> None:
    correct = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    confidences = probas.max(axis=1)
    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability Diagram (L2 calibrated model)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_error_rows(rows: list[dict[str, str]], pred_l1: list[str], pred_l2: list[str], probas_l2: np.ndarray | None) -> list[dict[str, Any]]:
    errors = []
    for idx, row in enumerate(rows):
        if row["label_l1"] != pred_l1[idx] or row["label_l2"] != pred_l2[idx]:
            confidence = None if probas_l2 is None else float(probas_l2[idx].max())
            errors.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "label_l1": row["label_l1"],
                    "pred_l1": pred_l1[idx],
                    "label_l2": row["label_l2"],
                    "pred_l2": pred_l2[idx],
                    "confidence_l2": confidence,
                    "source": row.get("source", ""),
                    "complexity": row.get("complexity", ""),
                    "word_count": len(row["text"].split()),
                }
            )
    return errors


def load_methods_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_methods_comparison(method_rows: list[dict[str, str]], path: Path) -> None:
    if not method_rows:
        return

    names = [row[""] for row in method_rows]
    accs = [float(row["accuracy"]) for row in method_rows]
    f1s = [float(row["f1_macro"]) for row in method_rows]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4c78a8")
    ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#f58518")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_title("D1 v5 Methods Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(path: Path, l1_metrics: dict[str, Any], l2_metrics: dict[str, Any], errors: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    l1_pass = l1_metrics["macro_f1"] >= L1_TARGET
    l2_pass = l2_metrics["macro_f1"] >= L2_TARGET

    top_pairs = Counter((e["label_l2"], e["pred_l2"]) for e in errors).most_common(10)
    l1_dist = Counter(r["label_l1"] for r in rows)
    l2_dist = Counter(r["label_l2"] for r in rows)

    lines = [
        "# D1 Hypothesis Report",
        "",
        f"- Dataset: `{DATA_PATH}`",
        f"- Samples: `{len(rows)}`",
        f"- L1 target macro-F1: `{L1_TARGET:.2f}`",
        f"- L2 target macro-F1: `{L2_TARGET:.2f}`",
        "",
        "## Verdict",
        "",
        f"- L1 macro-F1: `{l1_metrics['macro_f1']:.4f}` -> {'PASS' if l1_pass else 'FAIL'}",
        f"- L2 macro-F1: `{l2_metrics['macro_f1']:.4f}` -> {'PASS' if l2_pass else 'FAIL'}",
        f"- Total errors (L1 or L2): `{len(errors)}`",
        "",
        "## Metrics",
        "",
        "| Level | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
        f"| L1 | {l1_metrics['accuracy']:.4f} | {l1_metrics['macro_f1']:.4f} | {l1_metrics['weighted_f1']:.4f} |",
        f"| L2 | {l2_metrics['accuracy']:.4f} | {l2_metrics['macro_f1']:.4f} | {l2_metrics['weighted_f1']:.4f} |",
        "",
        "## Top L2 Error Pairs",
        "",
        "| Gold | Pred | Count |",
        "|---|---|---:|",
    ]

    for (gold, pred), count in top_pairs:
        lines.append(f"| {gold} | {pred} | {count} |")

    lines += [
        "",
        "## Dataset Distribution",
        "",
        f"- L1: `{dict(l1_dist)}`",
        f"- L2 classes: `{len(l2_dist)}`",
        "",
        "## Artifacts",
        "",
        f"- [l1_metrics.json]({TABLE_DIR / 'l1_metrics.json'})",
        f"- [l2_metrics.json]({TABLE_DIR / 'l2_metrics.json'})",
        f"- [d1_error_analysis.csv]({TABLE_DIR / 'd1_error_analysis.csv'})",
        f"- [d1_confusion_l1.png]({FIG_DIR / 'd1_confusion_l1.png'})",
        f"- [d1_confusion_l2.png]({FIG_DIR / 'd1_confusion_l2.png'})",
        f"- [d1_f1_l1.png]({FIG_DIR / 'd1_f1_l1.png'})",
        f"- [d1_f1_l2.png]({FIG_DIR / 'd1_f1_l2.png'})",
        f"- [d1_reliability_l2.png]({FIG_DIR / 'd1_reliability_l2.png'})",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()

    rows = load_dataset(DATA_PATH)
    texts = [row["text"] for row in rows]
    y_l1 = [row["label_l1"] for row in rows]
    y_l2 = [row["label_l2"] for row in rows]

    l1_model, l1_vectorizer = load_model_and_vectorizer(L1_MODEL_PATH, L1_VECTORIZER_PATH)
    l2_model, l2_vectorizer = load_model_and_vectorizer(L2_MODEL_PATH, L2_VECTORIZER_PATH)

    pred_l1, probas_l1, _ = predict(l1_model, l1_vectorizer, texts)
    pred_l2, probas_l2, _ = predict(l2_model, l2_vectorizer, texts)

    labels_l1 = sorted(set(y_l1) | set(pred_l1))
    labels_l2 = sorted(set(y_l2) | set(pred_l2))

    l1_metrics = compute_metrics(y_l1, list(pred_l1), labels_l1)
    l2_metrics = compute_metrics(y_l2, list(pred_l2), labels_l2)

    save_json(TABLE_DIR / "l1_metrics.json", l1_metrics)
    save_json(TABLE_DIR / "l2_metrics.json", l2_metrics)

    summary_rows = [
        {
            "level": "L1",
            "accuracy": f"{l1_metrics['accuracy']:.4f}",
            "macro_f1": f"{l1_metrics['macro_f1']:.4f}",
            "weighted_f1": f"{l1_metrics['weighted_f1']:.4f}",
            "target_macro_f1": f"{L1_TARGET:.2f}",
            "status": "PASS" if l1_metrics["macro_f1"] >= L1_TARGET else "FAIL",
        },
        {
            "level": "L2",
            "accuracy": f"{l2_metrics['accuracy']:.4f}",
            "macro_f1": f"{l2_metrics['macro_f1']:.4f}",
            "weighted_f1": f"{l2_metrics['weighted_f1']:.4f}",
            "target_macro_f1": f"{L2_TARGET:.2f}",
            "status": "PASS" if l2_metrics["macro_f1"] >= L2_TARGET else "FAIL",
        },
    ]
    save_csv(
        TABLE_DIR / "d1_hypothesis_summary.csv",
        summary_rows,
        ["level", "accuracy", "macro_f1", "weighted_f1", "target_macro_f1", "status"],
    )

    errors = build_error_rows(rows, list(pred_l1), list(pred_l2), probas_l2)
    save_csv(
        TABLE_DIR / "d1_error_analysis.csv",
        errors,
        ["id", "text", "label_l1", "pred_l1", "label_l2", "pred_l2", "confidence_l2", "source", "complexity", "word_count"],
    )

    plot_confusion(np.array(l1_metrics["confusion_matrix"]), labels_l1, "D1 Confusion Matrix (L1)", FIG_DIR / "d1_confusion_l1.png", figsize=(8, 6))
    plot_confusion(np.array(l2_metrics["confusion_matrix"]), labels_l2, "D1 Confusion Matrix (L2)", FIG_DIR / "d1_confusion_l2.png", figsize=(14, 10))
    plot_f1_bars(l1_metrics["classification_report"], labels_l1, "D1 Per-Class F1 (L1)", FIG_DIR / "d1_f1_l1.png", L1_TARGET)
    plot_f1_bars(l2_metrics["classification_report"], labels_l2, "D1 Per-Class F1 (L2)", FIG_DIR / "d1_f1_l2.png", L2_TARGET)
    plot_distribution(Counter(y_l1), "D1 v5 Test Distribution (L1)", FIG_DIR / "d1_distribution_l1.png", "#4c78a8")
    plot_distribution(Counter(y_l2), "D1 v5 Test Distribution (L2)", FIG_DIR / "d1_distribution_l2.png", "#f58518")
    plot_error_rate_by_length(
        [
            {
                "text": row["text"],
                "label_l2": row["label_l2"],
                "pred_l2": pred_l2[idx],
            }
            for idx, row in enumerate(rows)
        ],
        FIG_DIR / "d1_error_rate_by_length.png",
    )
    if probas_l2 is not None:
        plot_reliability(y_l2, list(pred_l2), probas_l2, labels_l2, FIG_DIR / "d1_reliability_l2.png")

    method_rows = load_methods_table(METHODS_TABLE_PATH)
    if method_rows:
        plot_methods_comparison(method_rows, FIG_DIR / "d1_methods_comparison.png")

    write_markdown_report(REPORT_DIR / "D1_HYPOTHESIS_REPORT.md", l1_metrics, l2_metrics, errors, rows)

    print("Generated D1 hypothesis artifacts:")
    print(f"- tables: {TABLE_DIR}")
    print(f"- figures: {FIG_DIR}")
    print(f"- report: {REPORT_DIR / 'D1_HYPOTHESIS_REPORT.md'}")


if __name__ == "__main__":
    main()
