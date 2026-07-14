"""
gate_strategies.py
===================
Every baseline, ablation, and the production gate itself, expressed as pure
functions of the form::

    strategy(severity_weight: float, slm_score: float, params: dict) -> bool

All of them take the SAME two cached numbers per sample (see
`raw_score_extractor.py`), so comparing them costs nothing beyond calling a
different function — no re-running Semgrep, no re-running the SLM, no LLM
calls.

Two of the review's requested baselines collapse onto existing rules, which
is itself worth stating explicitly in the paper:

    - "Semgrep only" and "Semgrep + LLM without Stage 2" are the SAME
      decision rule. Removing Stage 2 (the SLM gate) means every Semgrep
      finding goes straight to the LLM — which is exactly what a
      standalone "Semgrep only" detector does. See `semgrep_only`.

    - "SLM only" and "SLM + LLM without Stage 1" are
      likewise the same rule: without Stage 1 filtering, the SLM score
      alone decides escalation. See `small_model_only`.

This module intentionally has ZERO dependency on triage_orchestrator.py or
any I/O — `triage_orchestrator.py` imports `linear_weighted_gate` FROM here
(see the modified triage_orchestrator.py) so the production gate and the
evaluated gate can never drift apart.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, NamedTuple


GateFn = Callable[[float, float, Dict[str, Any]], bool]


# ---------------------------------------------------------------------------
# 1. The CEVuD linear gate (production logic, parametrized for grid search)
# ---------------------------------------------------------------------------

def linear_weighted_gate(severity_weight: float, slm_score: float, params: Dict[str, Any]) -> bool:
    """The CEVuD staged gate: a weighted linear combination of static
    severity and SLM probability, with an optional short-circuit override.

    R = (weight_static * severity_weight) + (weight_slm * slm_score)
    escalate = (R >= escalation_threshold) OR override_condition

    where, if `override_enabled` is True:
        override_condition = (severity_weight >= static_override_value)
                              OR (slm_score > slm_override_threshold)

    Args:
        severity_weight: Semgrep severity mapped to [0, 1].
        slm_score: SLM classifier's vulnerability probability, [0, 1].
        params: Dict with keys:
            weight_static (float): default 0.15
            weight_slm (float): default 0.85
            escalation_threshold (float): default 0.2
            override_enabled (bool): default False
            static_override_value (float): default 1.0 (i.e. ERROR severity)
            slm_override_threshold (float): default 0.90

    Returns:
        bool: True if the sample should be escalated to Stage 3.
    """
    weight_static = params.get("weight_static", 0.15)
    weight_slm = params.get("weight_slm", 0.85)
    threshold = params.get("escalation_threshold", 0.2)
    override_enabled = params.get("override_enabled", False)
    static_override_value = params.get("static_override_value", 1.0)
    slm_override_threshold = params.get("slm_override_threshold", 0.90)

    risk_score = (weight_static * severity_weight) + (weight_slm * slm_score)
    base_escalate = risk_score >= threshold

    if not override_enabled:
        return base_escalate

    static_override = severity_weight >= static_override_value
    slm_override = slm_score > slm_override_threshold
    return base_escalate or static_override or slm_override


# ---------------------------------------------------------------------------
# 2. Single-signal baselines
# ---------------------------------------------------------------------------

def semgrep_only(severity_weight: float, slm_score: float, params: Dict[str, Any] = None) -> bool:
    """Escalates whenever Semgrep produced ANY finding for this sample.

    This is both the "Semgrep only" detector baseline and the
    "Semgrep + LLM without Stage 2" pipeline ablation (see module docstring):
    with no SLM gate, every static finding is forwarded as-is.

    params:
        min_severity_weight (float): minimum severity_weight counted as a
            "finding" worth escalating. Default 0.0 (any finding at all,
            including INFO). Set higher (e.g. 0.7) to require WARNING/ERROR.
    """
    params = params or {}
    threshold = params.get("min_severity_weight", 0.0)
    return severity_weight > threshold


def small_model_only(severity_weight: float, slm_score: float, params: Dict[str, Any] = None) -> bool:
    """Escalates based solely on the SLM probability, ignoring Semgrep entirely.

    This is both the "SLM only" detector baseline and the
    "SLM + LLM without Stage 1" pipeline ablation (see module
    docstring): with no static pre-filter, the SLM score alone decides.

    params:
        threshold (float): default 0.5
    """
    params = params or {}
    threshold = params.get("threshold", 0.5)
    return slm_score >= threshold


def always_llm(severity_weight: float, slm_score: float, params: Dict[str, Any] = None) -> bool:
    """Escalates every single sample. Upper bound on recall/cost — the
    "send everything to the LLM" baseline the review specifically asked for.
    """
    return True


def semgrep_or_small_model(severity_weight: float, slm_score: float, params: Dict[str, Any] = None) -> bool:
    """OR-gate baseline: escalate if EITHER signal alone would escalate.
    Contrast with `linear_weighted_gate`, which combines the two signals
    rather than treating them as independent triggers.
    """
    params = params or {}
    return semgrep_only(severity_weight, slm_score, params) or small_model_only(severity_weight, slm_score, params)


# ---------------------------------------------------------------------------
# 3. Non-linear comparison gate (see linearity_check.py for how it's fit)
# ---------------------------------------------------------------------------

def logistic_regression_gate(severity_weight: float, slm_score: float, params: Dict[str, Any]) -> bool:
    """Decision rule from a logistic regression fit on
    [severity_weight, slm_score] -> label (see linearity_check.py). Used
    ONLY to answer "would a non-linear/learned boundary do meaningfully
    better than the hand-specified linear gate?" — never used in production.

    params must include:
        coefficients (Tuple[float, float, float]): (bias, w_severity, w_slm)
            learned on the validation split.
        decision_threshold (float): default 0.5, threshold on the sigmoid output.
    """
    bias, w_sev, w_slm = params["coefficients"]
    threshold = params.get("decision_threshold", 0.5)
    z = bias + w_sev * severity_weight + w_slm * slm_score
    # Manual sigmoid (no numpy/scipy dependency needed for a scalar).
    import math
    prob = 1.0 / (1.0 + math.exp(-z))
    return prob >= threshold


# ---------------------------------------------------------------------------
# Strategy registry — used by run_comparative_evaluation.py to iterate over
# every baseline uniformly, and to render the comparison table with a
# human-readable name/description for the paper.
# ---------------------------------------------------------------------------

class StrategySpec(NamedTuple):
    fn: GateFn
    display_name: str
    description: str


GATE_STRATEGIES: Dict[str, StrategySpec] = {
    "semgrep_only": StrategySpec(
        semgrep_only,
        "Semgrep only",
        "Escalate on any static finding. Equivalent to 'Semgrep + LLM without Stage 2'.",
    ),
    "small_model_only": StrategySpec(
        small_model_only,
        "Escalate if the local small model's P(vuln) exceeds the threshold. "
        "Pure neural baseline — no static signal.",
        {"threshold": 0.5},
    ),
    "always_llm": StrategySpec(
        always_llm,
        "Always escalate",
        "Send every sample to the LLM. Upper bound on recall and on LLM cost.",
    ),
    "semgrep_or_small_model": StrategySpec(
        semgrep_or_small_model,
        "OR-gate: escalate if EITHER the static severity OR the small model "
        "score triggers. Naive combination — no learned weighting.",
        {},
    ),
    "cevud_full": StrategySpec(
        linear_weighted_gate,
        "CEVuD (tuned weights, with override)",
        "Production linear gate with the static/SLM override enabled.",
    ),
    "cevud_no_override": StrategySpec(
        linear_weighted_gate,
        "CEVuD (tuned weights, no override)",
        "Ablation: same tuned linear gate with override_enabled=False, "
        "isolating the override rule's marginal contribution.",
    ),
    "logistic_regression": StrategySpec(
        logistic_regression_gate,
        "Logistic regression gate",
        "Learned non-linear boundary over the same two signals, fit on the "
        "validation split — used only to test whether linearity costs performance.",
    ),
}
