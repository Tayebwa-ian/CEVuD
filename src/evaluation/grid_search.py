"""
grid_search.py
===============
Sweeps (weight_static, escalation_threshold) over the VALIDATION split only,
scoring each combination with `metrics.compute_metrics`, and returns both the
best combination and the full grid (needed by sensitivity_analysis.py for the
heatmap/line plots).

Methodological note on the override rule
------------------------------------------
The grid search here tunes ONLY the base linear gate
(`override_enabled=False` for every grid point). The static/SLM override is
evaluated separately, as an ablation, on top of the tuned linear gate (see
`gate_strategies.cevud_no_override` vs `cevud_full` in
`run_comparative_evaluation.py`). This is a deliberate methodological choice:
conflating "which weights/threshold are best" with "does the override help"
would make it impossible to attribute gains to one or the other. Tuning the
linear part first, then measuring the override's marginal effect on top of
the ALREADY-tuned gate, is what makes the override's contribution
interpretable — and is exactly the "explicit disclosure of the override
rule's provenance" the review asked for.

`weight_slm` is always `1 - weight_static` (the two are constrained to sum
to 1, matching the production formula), so the grid is effectively 2D:
(weight_static, escalation_threshold).
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from gate_strategies import linear_weighted_gate
from metrics import compute_metrics
from schema import RawScoreRecord


def run_grid_search(
    validation_records: List[RawScoreRecord],
    weight_grid: List[float] = None,
    threshold_grid: List[float] = None,
    beta: float = 2.0,
    selection_metric: str = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates `linear_weighted_gate` (override disabled) at every
    (weight_static, threshold) combination on the validation split.

    Args:
        validation_records: RawScoreRecords from the validation split ONLY.
            Never pass test-split records here — that would defeat the
            purpose of a held-out test set.
        weight_grid: Candidate weight_static values in [0, 1]. Defaults to
            0.0, 0.05, ..., 1.0 (21 points).
        threshold_grid: Candidate escalation_threshold values in [0, 1].
            Defaults to the same 21-point grid.
        beta: F-beta weighting passed to compute_metrics (see metrics.py
            docstring for why beta=2 is the default: false negatives cost
            more than false positives in security triage).
        selection_metric: Which key in compute_metrics' output to maximize
            when choosing the "best" combination. Defaults to f"f{beta:g}"
            (e.g. "f2" when beta=2.0).

    Returns:
        Tuple[List[Dict], Dict]:
            - Full grid: one dict per (weight_static, threshold) combination,
              containing the params and the full metrics dict. This is what
              sensitivity_analysis.py consumes for heatmaps/line plots.
            - Best combination: the single grid entry with the highest
              selection_metric score (ties broken by higher recall, then by
              lower escalation_rate, to prefer cheaper gates among equally
              good ones).
    """
    if not validation_records:
        raise ValueError("run_grid_search requires a non-empty validation split.")

    if weight_grid is None:
        weight_grid = [round(i * 0.05, 2) for i in range(21)]  # 0.0, 0.05, ..., 1.0
    if threshold_grid is None:
        threshold_grid = [round(i * 0.05, 2) for i in range(21)]
    if selection_metric is None:
        selection_metric = f"f{beta:g}"

    labels = [r.label for r in validation_records]
    severities = [r.severity_weight for r in validation_records]
    slm_scores = [r.slm_score for r in validation_records]

    grid_results: List[Dict[str, Any]] = []
    for weight_static in weight_grid:
        weight_slm = round(1.0 - weight_static, 4)
        for threshold in threshold_grid:
            params = {
                "weight_static": weight_static,
                "weight_slm": weight_slm,
                "escalation_threshold": threshold,
                "override_enabled": False,  # see module docstring
            }
            predictions = [
                linear_weighted_gate(sev, slm, params) for sev, slm in zip(severities, slm_scores)
            ]
            metrics = compute_metrics(predictions, labels, beta=beta)
            grid_results.append({
                "weight_static": weight_static,
                "weight_slm": weight_slm,
                "escalation_threshold": threshold,
                "metrics": metrics,
            })

    def sort_key(entry: Dict[str, Any]):
        m = entry["metrics"]
        return (m[selection_metric], m["recall"], -m["escalation_rate"])

    best = max(grid_results, key=sort_key)

    print(
        f"[+] Grid search complete: {len(grid_results)} combinations evaluated on "
        f"{len(validation_records)} validation samples. "
        f"Best {selection_metric}={best['metrics'][selection_metric]:.4f} at "
        f"weight_static={best['weight_static']}, threshold={best['escalation_threshold']}"
    )
    return grid_results, best
