"""
sensitivity_analysis.py
========================
Turns the grid produced by `grid_search.run_grid_search` into the plots a
reviewer will actually ask for: "how sensitive are the results to threshold
and weight choices?"

Three artifacts are produced:
    1. A 2D heatmap of the selection metric over (weight_static, threshold).
    2. A line plot of the metric vs. threshold, at the best weight_static.
    3. A line plot of the metric vs. weight_static, at the best threshold.

All three are saved as PNGs (matplotlib, non-interactive "Agg" backend —
consistent with the existing `evaluate_pipeline.py`) and are meant to be
embedded directly in the paper's gating subsection.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    grid_results: List[Dict[str, Any]],
    selection_metric: str,
    out_path: str,
    best: Dict[str, Any] = None,
) -> None:
    """Saves a (weight_static x escalation_threshold) heatmap of `selection_metric`.

    Args:
        grid_results: Full grid from run_grid_search.
        selection_metric: Which metrics key to plot (e.g. "f2").
        out_path: PNG output path.
        best: If given, the best grid entry is marked with a star.
    """
    weights = sorted({g["weight_static"] for g in grid_results})
    thresholds = sorted({g["escalation_threshold"] for g in grid_results})

    grid_lookup = {(g["weight_static"], g["escalation_threshold"]): g["metrics"][selection_metric] for g in grid_results}
    matrix = np.array([[grid_lookup[(w, t)] for t in thresholds] for w in weights])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label=selection_metric)

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=90, fontsize=7)
    ax.set_yticks(range(len(weights)))
    ax.set_yticklabels([f"{w:.2f}" for w in weights], fontsize=7)
    ax.set_xlabel("escalation_threshold")
    ax.set_ylabel("weight_static (weight_slm = 1 - weight_static)")
    ax.set_title(f"Gate sensitivity: {selection_metric} on validation split")

    if best is not None:
        wi = weights.index(best["weight_static"])
        ti = thresholds.index(best["escalation_threshold"])
        ax.plot(ti, wi, marker="*", color="red", markersize=18, markeredgecolor="white",
                 label="Selected configuration")
        ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved sensitivity heatmap: {out_path}")


def plot_threshold_sensitivity(
    grid_results: List[Dict[str, Any]],
    fixed_weight_static: float,
    selection_metric: str,
    out_path: str,
) -> None:
    """Saves a line plot of `selection_metric` (and precision/recall) vs.
    escalation_threshold, holding weight_static fixed at the best value.
    """
    rows = sorted(
        [g for g in grid_results if g["weight_static"] == fixed_weight_static],
        key=lambda g: g["escalation_threshold"],
    )
    if not rows:
        raise ValueError(f"No grid entries found at weight_static={fixed_weight_static}")

    thresholds = [g["escalation_threshold"] for g in rows]
    metric_vals = [g["metrics"][selection_metric] for g in rows]
    precision_vals = [g["metrics"]["precision"] for g in rows]
    recall_vals = [g["metrics"]["recall"] for g in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, metric_vals, label=selection_metric, linewidth=2.5, color="darkorange")
    plt.plot(thresholds, precision_vals, label="precision", linestyle="--", alpha=0.7)
    plt.plot(thresholds, recall_vals, label="recall", linestyle="--", alpha=0.7)
    plt.xlabel("escalation_threshold")
    plt.ylabel("score")
    plt.title(f"Threshold sensitivity at weight_static={fixed_weight_static:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved threshold sensitivity plot: {out_path}")


def plot_weight_sensitivity(
    grid_results: List[Dict[str, Any]],
    fixed_threshold: float,
    selection_metric: str,
    out_path: str,
) -> None:
    """Saves a line plot of `selection_metric` (and precision/recall) vs.
    weight_static, holding escalation_threshold fixed at the best value.
    """
    rows = sorted(
        [g for g in grid_results if g["escalation_threshold"] == fixed_threshold],
        key=lambda g: g["weight_static"],
    )
    if not rows:
        raise ValueError(f"No grid entries found at escalation_threshold={fixed_threshold}")

    weights = [g["weight_static"] for g in rows]
    metric_vals = [g["metrics"][selection_metric] for g in rows]
    precision_vals = [g["metrics"]["precision"] for g in rows]
    recall_vals = [g["metrics"]["recall"] for g in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(weights, metric_vals, label=selection_metric, linewidth=2.5, color="darkorange")
    plt.plot(weights, precision_vals, label="precision", linestyle="--", alpha=0.7)
    plt.plot(weights, recall_vals, label="recall", linestyle="--", alpha=0.7)
    plt.xlabel("weight_static (weight_slm = 1 - weight_static)")
    plt.ylabel("score")
    plt.title(f"Weight sensitivity at escalation_threshold={fixed_threshold:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved weight sensitivity plot: {out_path}")


def generate_all_sensitivity_plots(
    grid_results: List[Dict[str, Any]],
    best: Dict[str, Any],
    selection_metric: str,
    out_dir: str,
) -> Dict[str, str]:
    """Convenience wrapper: generates all three plots into `out_dir` using
    `best`'s weight/threshold as the fixed value for the two line plots.

    Returns:
        Dict[str, str]: {"heatmap": path, "threshold_sensitivity": path, "weight_sensitivity": path}
    """
    paths = {
        "heatmap": os.path.join(out_dir, "gate_sensitivity_heatmap.png"),
        "threshold_sensitivity": os.path.join(out_dir, "gate_threshold_sensitivity.png"),
        "weight_sensitivity": os.path.join(out_dir, "gate_weight_sensitivity.png"),
    }
    plot_heatmap(grid_results, selection_metric, paths["heatmap"], best=best)
    plot_threshold_sensitivity(grid_results, best["weight_static"], selection_metric, paths["threshold_sensitivity"])
    plot_weight_sensitivity(grid_results, best["escalation_threshold"], selection_metric, paths["weight_sensitivity"])
    return paths
