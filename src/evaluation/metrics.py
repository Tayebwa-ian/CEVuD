"""
metrics.py
==========
Confusion-matrix and derived-metric computations shared by every downstream
analysis step (grid search, sensitivity analysis, the linearity check, and
the final comparative report).

Why F-beta with beta=2 as the default tuning objective, not F1:
------------------------------------------------------------------
In a security-triage setting, a false negative (a real vulnerability the
gate fails to escalate) is more costly than a false positive (an extra,
unnecessary LLM call on safe code) — a missed vulnerability can ship, an
extra LLM call just costs a few cents. F-beta with beta=2 weights recall
twice as heavily as precision, which better reflects that asymmetry than F1
(beta=1, which weights them equally). `beta` is a parameter everywhere in
this module, not hardcoded, so this choice can itself be reported and
sensitivity-tested rather than silently baked in.
"""

from __future__ import annotations

from typing import List, Dict, Any


def confusion_counts(predictions: List[bool], labels: List[int]) -> Dict[str, int]:
    """Computes TP/FP/FN/TN counts.

    Args:
        predictions: Per-sample escalation decisions (True = escalate/predicted positive).
        labels: Per-sample ground truth (1 = vulnerable, 0 = safe).

    Returns:
        Dict[str, int]: {"tp": ..., "fp": ..., "fn": ..., "tn": ...}
    """
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) length mismatch")

    tp = fp = fn = tn = 0
    for pred, label in zip(predictions, labels):
        is_vuln = label == 1
        if is_vuln and pred:
            tp += 1
        elif is_vuln and not pred:
            fn += 1
        elif not is_vuln and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(predictions: List[bool], labels: List[int], beta: float = 2.0) -> Dict[str, Any]:
    """Computes the full metric set from predictions vs. ground truth.

    Args:
        predictions: Per-sample escalation decisions.
        labels: Per-sample ground truth (1 = vulnerable, 0 = safe).
        beta: F-beta weighting; beta > 1 weights recall more heavily than
            precision (see module docstring). Default 2.0.

    Returns:
        Dict[str, Any]: confusion matrix counts plus precision, recall,
            f1, f_beta, accuracy, specificity, and escalation_rate
            (fraction of samples sent to the LLM — the pipeline's cost proxy).
    """
    cm = confusion_counts(predictions, labels)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    beta_sq = beta * beta
    f_beta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )

    escalations = tp + fp
    escalation_rate = escalations / total if total > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        f"f{beta:g}": round(f_beta, 4),
        "escalation_rate": round(escalation_rate, 4),
        "total_samples": total,
    }


def per_group_metrics(
    predictions: List[bool], labels: List[int], groups: List[str], beta: float = 2.0
) -> Dict[str, Dict[str, Any]]:
    """Breaks metrics down per group (typically `project`), so results can
    show whether performance is stable across projects or driven by one
    easy project — directly addressing the review's concern about
    single-curated-set evaluation.

    Args:
        predictions: Per-sample escalation decisions.
        labels: Per-sample ground truth.
        groups: Per-sample group label (e.g. project name), same length/order.
        beta: F-beta weighting, forwarded to compute_metrics.

    Returns:
        Dict[str, Dict[str, Any]]: group name -> compute_metrics(...) output,
            restricted to that group's samples.
    """
    if not (len(predictions) == len(labels) == len(groups)):
        raise ValueError("predictions, labels, and groups must be the same length")

    by_group: Dict[str, Dict[str, list]] = {}
    for pred, label, group in zip(predictions, labels, groups):
        bucket = by_group.setdefault(group, {"preds": [], "labels": []})
        bucket["preds"].append(pred)
        bucket["labels"].append(label)

    return {
        group: compute_metrics(bucket["preds"], bucket["labels"], beta=beta)
        for group, bucket in sorted(by_group.items())
    }
