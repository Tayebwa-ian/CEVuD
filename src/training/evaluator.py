"""evaluator.py
===============
Evaluates the fine-tuned vulnerability classifier on the held-out test split
and produces:

* scalar metrics: accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC
* confusion matrix
* ROC curve
* precision-recall curve
* calibration curve

All plots and the `metrics.json` are written to the run directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from training.config import TrainingConfig  # noqa: E402
from training.dataset_builder import load_jsonl  # noqa: E402


# ── Metric computation ──────────────────────────────────────────────────────

def compute_all_metrics(
    labels: List[int],
    preds: List[int],
    probs: np.ndarray,
    beta: float = 2.0,
) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for p, y in zip(preds, labels):
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    beta_sq = beta * beta
    f_beta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )

    vuln_probs = probs[:, 1] if probs.ndim > 1 else probs

    try:
        roc_auc = float(roc_auc_score(labels, vuln_probs))
    except Exception:
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(labels, vuln_probs))
    except Exception:
        pr_auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        f"f{beta:g}": round(f_beta, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "total_samples": len(labels),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: Dict[str, int], path: Path) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ticks = np.arange(2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["Pred Safe", "Pred Vuln"])
    ax.set_yticklabels(["True Safe", "True Vuln"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_roc(labels: List[int], probs: np.ndarray, path: Path) -> None:
    vuln_probs = probs[:, 1] if probs.ndim > 1 else probs
    fpr, tpr, _ = roc_curve(labels, vuln_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_pr(labels: List[int], probs: np.ndarray, path: Path) -> None:
    vuln_probs = probs[:, 1] if probs.ndim > 1 else probs
    precision, recall, _ = precision_recall_curve(labels, vuln_probs)
    pr_auc = average_precision_score(labels, vuln_probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_calibration(labels: List[int], probs: np.ndarray, path: Path) -> None:
    vuln_probs = probs[:, 1] if probs.ndim > 1 else probs
    prob_true, prob_pred = calibration_curve(labels, vuln_probs, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    test_path: str,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 8,
    beta: float = 2.0,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    texts, labels = load_jsonl(test_path)
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    preds = probs.argmax(axis=-1).tolist()

    metrics = compute_all_metrics(labels, preds, probs, beta=beta)
    cm = metrics["confusion_matrix"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_confusion_matrix(cm, out / "confusion_matrix.png")
    _plot_roc(labels, probs, out / "roc_curve.png")
    _plot_pr(labels, probs, out / "pr_curve.png")
    _plot_calibration(labels, probs, out / "calibration.png")

    metrics["model_path"] = model_path
    metrics["test_path"] = test_path
    metrics["plots"] = {
        "confusion_matrix": str(out / "confusion_matrix.png"),
        "roc_curve": str(out / "roc_curve.png"),
        "pr_curve": str(out / "pr_curve.png"),
        "calibration": str(out / "calibration.png"),
    }

    metrics_path = out / "metrics.json"
    with open(str(metrics_path), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[+] Evaluation complete. Metrics -> {metrics_path}")
    return metrics
