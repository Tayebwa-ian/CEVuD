"""
Unit Tests: gate_strategies
===========================
Validates every baseline, ablation, and the production linear gate itself.
All strategies are pure functions of (severity_weight, slm_score, params),
so no model loading or I/O is needed.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.gate_strategies import (
    linear_weighted_gate,
    semgrep_only,
    codesheriff_only,
    always_llm,
    semgrep_or_codesheriff,
    logistic_regression_gate,
)


class TestLinearWeightedGate:
    """Tests for the production CEVuD linear gate."""

    def test_standard_escalation_above_threshold(self):
        # weight_static=0.4, severity=1.0 (ERROR) -> 0.4
        # weight_slm=0.6, slm=0.5 -> 0.3
        # risk = 0.7 > threshold 0.52 -> escalate
        result = linear_weighted_gate(1.0, 0.5, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is True

    def test_standard_no_escalation_below_threshold(self):
        # severity=0.0, slm=0.1 -> risk = 0.06 < 0.52
        result = linear_weighted_gate(0.0, 0.1, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is False

    def test_static_override_forces_escalation(self):
        # severity=1.0 (ERROR) triggers static override regardless of SLM
        result = linear_weighted_gate(1.0, 0.0, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": True,
            "static_override_value": 1.0,
            "slm_override_threshold": 0.90,
        })
        assert result is True

    def test_slm_override_forces_escalation(self):
        # slm=0.95 > 0.90 triggers SLM override regardless of severity
        result = linear_weighted_gate(0.0, 0.95, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": True,
            "static_override_value": 1.0,
            "slm_override_threshold": 0.90,
        })
        assert result is True

    def test_slm_just_below_override_no_escalation(self):
        # slm=0.89 < 0.90, severity=0.0 -> risk = 0.534 (0.6*0.89)
        # 0.534 >= 0.52 -> escalates via base gate
        result = linear_weighted_gate(0.0, 0.89, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": True,
            "static_override_value": 1.0,
            "slm_override_threshold": 0.90,
        })
        assert result is True

    def test_override_disabled_ignores_high_slm(self):
        # slm=0.95 but override disabled -> risk = 0.6*0.95 = 0.57 >= 0.52
        result = linear_weighted_gate(0.0, 0.95, {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is True

    def test_default_params_used_when_missing(self):
        # Missing params -> defaults: w_static=0.4, w_slm=0.6, threshold=0.52
        # severity=0.7 -> 0.28, slm=0.5 -> 0.3, risk=0.58 >= 0.52
        result = linear_weighted_gate(0.7, 0.5, {})
        assert result is True

    def test_weight_symmetric(self):
        # weight_static=0.6, weight_slm=0.4 (inverse of default)
        # severity=0.0, slm=1.0 -> risk = 0.4 >= 0.52? No, 0.4 < 0.52
        result = linear_weighted_gate(0.0, 1.0, {
            "weight_static": 0.6,
            "weight_slm": 0.4,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is False

    def test_extreme_weights(self):
        # weight_static=1.0, weight_slm=0.0
        # severity=0.0 -> risk=0.0 < 0.52
        result = linear_weighted_gate(0.0, 1.0, {
            "weight_static": 1.0,
            "weight_slm": 0.0,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is False

    def test_extreme_weights_reverse(self):
        # weight_static=0.0, weight_slm=1.0
        # slm=1.0 -> risk=1.0 >= 0.52
        result = linear_weighted_gate(0.0, 1.0, {
            "weight_static": 0.0,
            "weight_slm": 1.0,
            "escalation_threshold": 0.52,
            "override_enabled": False,
        })
        assert result is True


class TestSemgrepOnly:
    def test_escalates_above_threshold(self):
        assert semgrep_only(0.7, 0.0, {"min_severity_weight": 0.5}) is True

    def test_no_escalate_below_threshold(self):
        assert semgrep_only(0.3, 0.9, {"min_severity_weight": 0.5}) is False

    def test_default_threshold_is_zero(self):
        # Any severity > 0 escalates with default threshold=0.0
        assert semgrep_only(0.3, 0.0, {}) is True
        assert semgrep_only(0.0, 0.9, {}) is False


class TestCodesheriffOnly:
    def test_escalates_above_threshold(self):
        assert codesheriff_only(0.0, 0.7, {"threshold": 0.5}) is True

    def test_no_escalate_below_threshold(self):
        assert codesheriff_only(1.0, 0.3, {"threshold": 0.5}) is False

    def test_default_threshold_is_half(self):
        assert codesheriff_only(0.0, 0.5, {}) is True
        assert codesheriff_only(0.0, 0.49, {}) is False


class TestAlwaysLLM:
    def test_always_returns_true(self):
        assert always_llm(0.0, 0.0, {}) is True
        assert always_llm(1.0, 1.0, {}) is True
        assert always_llm(0.5, 0.5, {}) is True


class TestSemgrepOrCodesheriff:
    def test_semgrep_triggers(self):
        assert semgrep_or_codesheriff(0.7, 0.0, {"min_severity_weight": 0.5}) is True

    def test_codesheriff_triggers(self):
        assert semgrep_or_codesheriff(0.0, 0.7, {"threshold": 0.5}) is True

    def test_neither_triggers(self):
        assert semgrep_or_codesheriff(0.0, 0.0, {"min_severity_weight": 0.5, "threshold": 0.5}) is False

    def test_both_triggers(self):
        assert semgrep_or_codesheriff(0.7, 0.7, {}) is True


class TestLogisticRegressionGate:
    def test_positive_coefficients_escalate_high_inputs(self):
        # bias=-0.5, w_sev=1.0, w_slm=1.0 -> z = -0.5 + 1.0 + 1.0 = 1.5 -> sigmoid > 0.5
        result = logistic_regression_gate(1.0, 1.0, {
            "coefficients": (-0.5, 1.0, 1.0),
            "decision_threshold": 0.5,
        })
        assert result is True

    def test_negative_coefficients_no_escalate(self):
        # bias=0.5, w_sev=-1.0, w_slm=-1.0 -> z = 0.5 - 1.0 - 1.0 = -1.5 -> sigmoid < 0.5
        result = logistic_regression_gate(1.0, 1.0, {
            "coefficients": (0.5, -1.0, -1.0),
            "decision_threshold": 0.5,
        })
        assert result is False

    def test_default_threshold(self):
        # Default decision_threshold=0.5
        result = logistic_regression_gate(0.0, 0.0, {
            "coefficients": (0.0, 0.0, 0.0),
        })
        # z = 0.0 -> sigmoid = 0.5 -> at threshold 0.5, >= means True
        assert result is True

    def test_custom_threshold(self):
        # z = 0.0 -> sigmoid = 0.5, threshold=0.6 -> no escalate
        result = logistic_regression_gate(0.0, 0.0, {
            "coefficients": (0.0, 0.0, 0.0),
            "decision_threshold": 0.6,
        })
        assert result is False
