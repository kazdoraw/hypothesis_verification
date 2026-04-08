"""
Metrics computation utilities for DS experiments.
"""

from typing import Optional
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, 
        classification_report, precision_score, recall_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    labels: Optional[list] = None,
    return_report: bool = False
) -> dict:
    """
    Compute classification metrics for D1.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names
        return_report: Include full classification report
        
    Returns:
        Dict with metrics: accuracy, macro_f1, weighted_f1, per_class_f1, confusion_matrix
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for metrics computation")
    
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    
    # Per-class F1
    per_class_f1 = {}
    for label in labels:
        f1 = f1_score(
            y_true, y_pred, 
            labels=[label], 
            average='macro', 
            zero_division=0
        )
        per_class_f1[label] = round(f1, 4)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    result = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_samples": len(y_true),
        "n_correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
    }
    
    if return_report:
        result["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )
    
    return result


def compute_intake_metrics(
    cases_a: list[dict],
    cases_b: list[dict],
    schema_getter: callable = None
) -> dict:
    """
    Compute intake/anamnesis metrics for D2.
    
    Args:
        cases_a: Baseline A results (list of case dicts with 'filled_fields', 'turns')
        cases_b: Proposed B results
        schema_getter: Function to get required fields for a complaint type
        
    Returns:
        Dict with metrics comparing A vs B
    """
    from .schemas import get_required_fields
    
    if schema_getter is None:
        schema_getter = get_required_fields
    
    def compute_completion_rate(case: dict) -> float:
        """Compute completion rate for a single case."""
        complaint_type = case.get("target_complaint_type", "acute_pain")
        required = schema_getter(complaint_type)
        filled = case.get("filled_fields", {})
        
        if not required:
            return 1.0
        
        filled_count = sum(1 for f in required if filled.get(f))
        return filled_count / len(required)
    
    def compute_redundant_rate(case: dict) -> float:
        """Compute redundant question rate."""
        questions_asked = case.get("questions_asked", [])
        filled_at_time = case.get("filled_at_time", {})
        
        if not questions_asked:
            return 0.0
        
        redundant = 0
        for i, q in enumerate(questions_asked):
            field = q.get("field")
            if field and filled_at_time.get(field, float('inf')) < i:
                redundant += 1
        
        return redundant / len(questions_asked)
    
    # Compute metrics for each group
    metrics_a = {
        "completion_rates": [compute_completion_rate(c) for c in cases_a],
        "turns": [c.get("turns", 0) for c in cases_a],
        "redundant_rates": [compute_redundant_rate(c) for c in cases_a],
    }
    
    metrics_b = {
        "completion_rates": [compute_completion_rate(c) for c in cases_b],
        "turns": [c.get("turns", 0) for c in cases_b],
        "redundant_rates": [compute_redundant_rate(c) for c in cases_b],
    }
    
    # Aggregate
    result = {
        "baseline_a": {
            "avg_completion_rate": round(np.mean(metrics_a["completion_rates"]), 4),
            "std_completion_rate": round(np.std(metrics_a["completion_rates"]), 4),
            "avg_turns": round(np.mean(metrics_a["turns"]), 2),
            "std_turns": round(np.std(metrics_a["turns"]), 2),
            "avg_redundant_rate": round(np.mean(metrics_a["redundant_rates"]), 4),
            "completion_rates": metrics_a["completion_rates"],
            "turns": metrics_a["turns"],
        },
        "proposed_b": {
            "avg_completion_rate": round(np.mean(metrics_b["completion_rates"]), 4),
            "std_completion_rate": round(np.std(metrics_b["completion_rates"]), 4),
            "avg_turns": round(np.mean(metrics_b["turns"]), 2),
            "std_turns": round(np.std(metrics_b["turns"]), 2),
            "avg_redundant_rate": round(np.mean(metrics_b["redundant_rates"]), 4),
            "completion_rates": metrics_b["completion_rates"],
            "turns": metrics_b["turns"],
        },
        "comparison": {
            "completion_rate_diff": round(
                np.mean(metrics_b["completion_rates"]) - np.mean(metrics_a["completion_rates"]), 
                4
            ),
            "turns_diff": round(
                np.mean(metrics_b["turns"]) - np.mean(metrics_a["turns"]), 
                2
            ),
            "redundant_rate_diff": round(
                np.mean(metrics_b["redundant_rates"]) - np.mean(metrics_a["redundant_rates"]), 
                4
            ),
        },
        "n_cases": len(cases_a),
    }
    
    # Expert sufficient rate (completion >= 0.9)
    threshold = 0.9
    result["baseline_a"]["expert_sufficient_rate"] = round(
        sum(1 for r in metrics_a["completion_rates"] if r >= threshold) / len(metrics_a["completion_rates"]),
        4
    )
    result["proposed_b"]["expert_sufficient_rate"] = round(
        sum(1 for r in metrics_b["completion_rates"] if r >= threshold) / len(metrics_b["completion_rates"]),
        4
    )
    
    return result


def compute_economics(
    llm_logs_a: list[dict],
    llm_logs_b: list[dict],
    cost_per_1k_tokens: float = 0.002
) -> dict:
    """
    Compute economic metrics (tokens, calls, cost).
    
    Args:
        llm_logs_a: LLM usage logs for baseline A
        llm_logs_b: LLM usage logs for proposed B
        cost_per_1k_tokens: Cost per 1000 tokens
        
    Returns:
        Dict with economic comparison
    """
    def aggregate_logs(logs: list[dict]) -> dict:
        if not logs:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "avg_tokens_per_call": 0,
                "estimated_cost": 0,
            }
        
        total_calls = len(logs)
        total_tokens = sum(log.get("tokens", 0) for log in logs)
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "avg_tokens_per_call": round(total_tokens / total_calls, 2) if total_calls > 0 else 0,
            "estimated_cost": round(total_tokens / 1000 * cost_per_1k_tokens, 4),
        }
    
    stats_a = aggregate_logs(llm_logs_a)
    stats_b = aggregate_logs(llm_logs_b)
    
    # Compute reduction percentages
    calls_reduction = 0
    tokens_reduction = 0
    if stats_a["total_calls"] > 0:
        calls_reduction = round(
            (stats_a["total_calls"] - stats_b["total_calls"]) / stats_a["total_calls"] * 100,
            2
        )
    if stats_a["total_tokens"] > 0:
        tokens_reduction = round(
            (stats_a["total_tokens"] - stats_b["total_tokens"]) / stats_a["total_tokens"] * 100,
            2
        )
    
    return {
        "baseline_a": stats_a,
        "proposed_b": stats_b,
        "comparison": {
            "calls_reduction_percent": calls_reduction,
            "tokens_reduction_percent": tokens_reduction,
            "cost_savings": round(stats_a["estimated_cost"] - stats_b["estimated_cost"], 4),
        }
    }


def format_metrics_table(metrics: dict, title: str = "Metrics Summary") -> str:
    """
    Format metrics as a markdown table.
    
    Args:
        metrics: Metrics dict from compute_classification_metrics
        title: Table title
        
    Returns:
        Markdown string
    """
    lines = [
        f"## {title}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics.get('accuracy', 'N/A')} |",
        f"| Macro F1 | {metrics.get('macro_f1', 'N/A')} |",
        f"| Weighted F1 | {metrics.get('weighted_f1', 'N/A')} |",
        f"| Samples | {metrics.get('n_samples', 'N/A')} |",
        "",
    ]
    
    if 'per_class_f1' in metrics:
        lines.extend([
            "### Per-class F1",
            "",
            "| Class | F1 |",
            "|-------|-----|",
        ])
        for label, f1 in metrics['per_class_f1'].items():
            lines.append(f"| {label} | {f1} |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    y_true = ["booking", "complaint_primary", "booking", "other", "price_question"]
    y_pred = ["booking", "other", "booking", "other", "price_question"]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    print("Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Macro F1: {metrics['macro_f1']}")
    print(f"  Per-class F1: {metrics['per_class_f1']}")
