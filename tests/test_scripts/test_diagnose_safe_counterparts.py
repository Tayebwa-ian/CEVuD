"""
Unit Tests: diagnose_safe_counterparts.py
==========================================
Validates the safe-counterpart diagnosis helpers: bucket assignment and
per-pair metrics.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from scripts.diagnose_safe_counterparts import _bucket, _pair_metrics


class TestBucket:
    def test_bucket_zero_to_point_one(self):
        assert _bucket(0.0) == "[0.00,0.10)"
        assert _bucket(0.05) == "[0.00,0.10)"
        assert _bucket(0.099) == "[0.00,0.10)"

    def test_bucket_point_one_to_point_two_five(self):
        assert _bucket(0.1) == "[0.10,0.25)"
        assert _bucket(0.2) == "[0.10,0.25)"

    def test_bucket_point_two_five_to_point_five(self):
        assert _bucket(0.25) == "[0.25,0.50)"
        assert _bucket(0.4) == "[0.25,0.50)"

    def test_bucket_point_five_to_point_seven_five(self):
        assert _bucket(0.5) == "[0.50,0.75)"
        assert _bucket(0.7) == "[0.50,0.75)"

    def test_bucket_point_seven_five_to_one(self):
        assert _bucket(0.75) == "[0.75,1.01)"
        assert _bucket(0.99) == "[0.75,1.01)"
        assert _bucket(1.0) == "[0.75,1.01)"

    def test_bucket_above_one_falls_to_last(self):
        assert _bucket(1.5) == "[0.75,1.01)"


class TestPairMetrics:
    def test_returns_all_expected_keys(self):
        vuln = "def vuln():\n    eval(input())\n"
        fixed = "def vuln():\n    safe(input())\n"
        metrics = _pair_metrics(vuln, fixed)
        expected_keys = {
            "changed_line_ratio",
            "is_trivial",
            "is_substantial",
            "vuln_code_lines",
            "fixed_code_lines",
            "vuln_signal_lines",
            "fixed_signal_lines",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_changed_line_ratio_in_range(self):
        vuln = "def vuln():\n    eval(input())\n"
        fixed = "def vuln():\n    safe(input())\n"
        ratio = _pair_metrics(vuln, fixed)["changed_line_ratio"]
        assert 0.0 <= ratio <= 1.0

    def test_trivial_change_detected(self):
        vuln = "def foo():\n    return 1\n"
        fixed = "def foo():\n    return 1\n"
        metrics = _pair_metrics(vuln, fixed)
        assert metrics["is_trivial"] is True
        assert metrics["changed_line_ratio"] == 0.0

    def test_substantial_change_detected(self):
        vuln = "def foo():\n    x = 1\n"
        fixed = "def bar():\n    y = 2\n    z = 3\n"
        metrics = _pair_metrics(vuln, fixed)
        assert metrics["is_substantial"] is True
        assert metrics["changed_line_ratio"] > 0.5

    def test_counts_code_lines(self):
        vuln = "# comment\n# comment\ndef foo():\n    return 1\n"
        fixed = "# comment\ndef bar():\n    return 2\n"
        metrics = _pair_metrics(vuln, fixed)
        assert metrics["vuln_code_lines"] == 4  # all non-empty lines
        assert metrics["fixed_code_lines"] == 3  # all non-empty lines
        assert metrics["vuln_signal_lines"] == 2  # only real code lines
        assert metrics["fixed_signal_lines"] == 2  # only real code lines

    def test_empty_inputs(self):
        metrics = _pair_metrics("", "")
        assert metrics["changed_line_ratio"] == 1.0
        assert metrics["is_trivial"] is True
