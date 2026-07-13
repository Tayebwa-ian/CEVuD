"""
Unit Tests: linearity_check
============================
Validates the logistic regression fit and the linear-vs-logistic comparison
that answers "is a linear gate justified?"
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.schema import RawScoreRecord, RepoProvenance
from evaluation.linearity_check import fit_logistic_regression, compare_linear_vs_logistic


def _make_record(severity_weight, slm_score, label):
    return RawScoreRecord(
        sample_id=f"r_{severity_weight}_{slm_score}",
        project="test_proj",
        file_path="f.py",
        function_name="func",
        label=label,
        severity="ERROR" if severity_weight == 1.0 else "WARNING",
        severity_weight=severity_weight,
        slm_score=slm_score,
        vulnerability_type=None,
        provenance=RepoProvenance(
            project="test_proj",
            git_url="https://example.com/test.git",
            requested_ref="main",
            resolved_commit_sha="abc123",
            cloned_at_utc="2025-01-01T00:00:00Z",
        ),
    )


class TestFitLogisticRegression:
    def test_returns_three_floats(self):
        records = [_make_record(0.5, 0.5, 1), _make_record(0.5, 0.5, 0)]
        bias, w_sev, w_slm = fit_logistic_regression(records)
        assert isinstance(bias, float)
        assert isinstance(w_sev, float)
        assert isinstance(w_slm, float)

    def test_converges_on_clearly_separable_data(self):
        # Perfect separation: high inputs -> label 1, low inputs -> label 0
        records = [
            _make_record(1.0, 1.0, 1),
            _make_record(0.9, 0.9, 1),
            _make_record(0.0, 0.0, 0),
            _make_record(0.1, 0.1, 0),
        ]
        bias, w_sev, w_slm = fit_logistic_regression(records, learning_rate=1.0, epochs=5000)
        # Both weights should be positive (higher inputs -> higher P(label=1))
        assert w_sev > 0
        assert w_slm > 0

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="requires a non-empty record list"):
            fit_logistic_regression([])

    def test_reproducible_with_seed(self):
        records = [_make_record(0.5, 0.5, i % 2) for i in range(20)]
        r1 = fit_logistic_regression(records, seed=42)
        r2 = fit_logistic_regression(records, seed=42)
        assert r1 == pytest.approx(r2)


class TestCompareLinearVsLogistic:
    def _make_splits(self):
        val = [_make_record(1.0, 0.9, 1), _make_record(0.0, 0.1, 0)]
        test = [_make_record(0.9, 0.8, 1), _make_record(0.1, 0.2, 0)]
        return val, test

    def test_returns_expected_keys(self):
        val, test = self._make_splits()
        result = compare_linear_vs_logistic(val, test, {
            "weight_static": 0.5,
            "weight_slm": 0.5,
            "escalation_threshold": 0.5,
        })
        assert "logistic_coefficients" in result
        assert "linear_test_metrics" in result
        assert "logistic_test_metrics" in result
        assert "verdict" in result

    def test_logistic_coefficients_have_expected_fields(self):
        val, test = self._make_splits()
        result = compare_linear_vs_logistic(val, test, {
            "weight_static": 0.5,
            "weight_slm": 0.5,
            "escalation_threshold": 0.5,
        })
        coeffs = result["logistic_coefficients"]
        assert "bias" in coeffs
        assert "weight_severity" in coeffs
        assert "weight_slm" in coeffs

    def test_metrics_have_expected_fields(self):
        val, test = self._make_splits()
        result = compare_linear_vs_logistic(val, test, {
            "weight_static": 0.5,
            "weight_slm": 0.5,
            "escalation_threshold": 0.5,
        })
        for key in ["precision", "recall", "f1", "f2", "accuracy"]:
            assert key in result["linear_test_metrics"]
            assert key in result["logistic_test_metrics"]

    def test_verdict_is_string(self):
        val, test = self._make_splits()
        result = compare_linear_vs_logistic(val, test, {
            "weight_static": 0.5,
            "weight_slm": 0.5,
            "escalation_threshold": 0.5,
        })
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0
