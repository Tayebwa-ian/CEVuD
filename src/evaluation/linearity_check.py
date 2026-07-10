"""
linearity_check.py
====================
Answers "is a linear gate justified, or would a learned non-linear boundary
do meaningfully better?" by fitting a small logistic regression over the
same two signals (severity_weight, slm_score) on the VALIDATION split, then
comparing it against the tuned linear gate on the TEST split.

Deliberately implemented with plain numpy (already a dependency, via
`torch`'s ecosystem and `model_manager.py`/`triage_orchestrator.py`) rather
than adding scikit-learn: this is a 2-feature, convex logistic regression —
batch gradient descent converges in a few hundred iterations and keeps the
evaluation suite free of new dependency-resolution risk (see the wcmatch/
deepagents conflict this project already hit once).

Interpretation guidance (used by run_comparative_evaluation.py when writing
the report): if the logistic model's test-set F-beta is NOT meaningfully
higher than the tuned linear gate's, that is itself the evidence for
choosing the simpler linear form — it is monotonic in both inputs by
construction (a property a security team can reason about), cheaper to
compute, and carries no risk of overfitting a training/validation split the
way a learned boundary does.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np

from gate_strategies import logistic_regression_gate, linear_weighted_gate
from metrics import compute_metrics
from schema import RawScoreRecord


def fit_logistic_regression(
    records: List[RawScoreRecord],
    learning_rate: float = 0.5,
    epochs: int = 3000,
    l2_penalty: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Fits P(label=1 | severity_weight, slm_score) via batch gradient
    descent on the (ridge-regularized) logistic loss.

    Args:
        records: Training data — pass the VALIDATION split only, never test.
        learning_rate: Gradient descent step size.
        epochs: Number of full-batch gradient steps.
        l2_penalty: Ridge regularization strength (guards against the
            2-parameter model overfitting a small validation split).
        seed: Random seed for weight initialization.

    Returns:
        Tuple[float, float, float]: (bias, weight_severity, weight_slm)
    """
    if not records:
        raise ValueError("fit_logistic_regression requires a non-empty record list.")

    rng = np.random.default_rng(seed)
    X = np.array([[r.severity_weight, r.slm_score] for r in records], dtype=np.float64)
    y = np.array([r.label for r in records], dtype=np.float64)
    n = len(y)

    # Standard bias + 2 weights, small random init.
    bias = 0.0
    weights = rng.normal(scale=0.01, size=2)

    for _ in range(epochs):
        z = bias + X @ weights
        preds = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        error = preds - y

        grad_bias = np.mean(error)
        grad_weights = (X.T @ error) / n + l2_penalty * weights  # L2 on weights only, not bias

        bias -= learning_rate * grad_bias
        weights -= learning_rate * grad_weights

    return float(bias), float(weights[0]), float(weights[1])


def compare_linear_vs_logistic(
    validation_records: List[RawScoreRecord],
    test_records: List[RawScoreRecord],
    best_linear_params: Dict[str, Any],
    beta: float = 2.0,
) -> Dict[str, Any]:
    """Fits the logistic gate on validation, then evaluates BOTH the tuned
    linear gate and the logistic gate on the (held-out) test split.

    Args:
        validation_records: Used only to fit the logistic regression.
        test_records: Used only for the final comparison numbers.
        best_linear_params: The winning params dict from grid_search.run_grid_search
            (weight_static, weight_slm, escalation_threshold), evaluated here
            WITHOUT the override, to isolate the linear-vs-nonlinear question
            from the override question (see grid_search.py's module docstring).
        beta: F-beta weighting for compute_metrics.

    Returns:
        Dict[str, Any]: {
            "logistic_coefficients": {"bias":..., "weight_severity":..., "weight_slm":...},
            "linear_test_metrics": {...},
            "logistic_test_metrics": {...},
            "verdict": human-readable interpretation string.
        }
    """
    bias, w_sev, w_slm = fit_logistic_regression(validation_records)

    logistic_params = {"coefficients": (bias, w_sev, w_slm), "decision_threshold": 0.5}
    linear_params = {**best_linear_params, "override_enabled": False}

    labels = [r.label for r in test_records]
    severities = [r.severity_weight for r in test_records]
    slm_scores = [r.slm_score for r in test_records]

    linear_preds = [linear_weighted_gate(s, p, linear_params) for s, p in zip(severities, slm_scores)]
    logistic_preds = [logistic_regression_gate(s, p, logistic_params) for s, p in zip(severities, slm_scores)]

    linear_metrics = compute_metrics(linear_preds, labels, beta=beta)
    logistic_metrics = compute_metrics(logistic_preds, labels, beta=beta)

    metric_key = f"f{beta:g}"
    delta = logistic_metrics[metric_key] - linear_metrics[metric_key]

    if abs(delta) < 0.02:
        verdict = (
            f"Logistic regression's test-set {metric_key} ({logistic_metrics[metric_key]:.4f}) is "
            f"within 0.02 of the linear gate's ({linear_metrics[metric_key]:.4f}). The linear gate "
            f"is preferred: it is monotonic in both inputs by construction, requires no separate "
            f"training step, and is not at risk of overfitting the validation split."
        )
    elif delta > 0:
        verdict = (
            f"Logistic regression outperforms the linear gate by {delta:.4f} {metric_key} on the "
            f"test split ({logistic_metrics[metric_key]:.4f} vs {linear_metrics[metric_key]:.4f}). "
            f"This suggests the true decision boundary is not well captured by a simple weighted "
            f"sum; a non-linear gate may be worth adopting in a future revision."
        )
    else:
        verdict = (
            f"The linear gate outperforms logistic regression by {-delta:.4f} {metric_key} on the "
            f"test split. This supports the linear design outright."
        )

    return {
        "logistic_coefficients": {"bias": bias, "weight_severity": w_sev, "weight_slm": w_slm},
        "linear_test_metrics": linear_metrics,
        "logistic_test_metrics": logistic_metrics,
        "verdict": verdict,
    }
