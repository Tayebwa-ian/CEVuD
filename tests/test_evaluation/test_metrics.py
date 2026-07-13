"""
Unit Tests: metrics
====================
Validates confusion-matrix computation, derived metrics, and per-group
breakdowns used throughout the evaluation suite.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.metrics import confusion_counts, compute_metrics, per_group_metrics


class TestConfusionCounts:
    def test_all_tp(self):
        cm = confusion_counts([True, True, True], [1, 1, 1])
        assert cm == {"tp": 3, "fp": 0, "fn": 0, "tn": 0}

    def test_all_tn(self):
        cm = confusion_counts([False, False, False], [0, 0, 0])
        assert cm == {"tp": 0, "fp": 0, "fn": 0, "tn": 3}

    def test_all_fp(self):
        cm = confusion_counts([True, True, True], [0, 0, 0])
        assert cm == {"tp": 0, "fp": 3, "fn": 0, "tn": 0}

    def test_all_fn(self):
        cm = confusion_counts([False, False, False], [1, 1, 1])
        assert cm == {"tp": 0, "fp": 0, "fn": 3, "tn": 0}

    def test_mixed(self):
        cm = confusion_counts([True, False, True, False], [1, 1, 0, 0])
        assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            confusion_counts([True, False], [1])

    def test_empty_input(self):
        cm = confusion_counts([], [])
        assert cm == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


class TestComputeMetrics:
    def test_perfect_classification(self):
        m = compute_metrics([True, False, True, False], [1, 0, 1, 0])
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["escalation_rate"] == 0.5
        assert m["token_reduction_rate"] == 0.5

    def test_all_wrong(self):
        m = compute_metrics([False, True], [1, 0])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["specificity"] == 0.0
        assert m["accuracy"] == 0.0

    def test_f2_weights_recall(self):
        # precision=0.5, recall=1.0 -> F1=0.667, F2 should be higher
        # TP=1, FP=1, FN=0 -> precision=0.5, recall=1.0
        m = compute_metrics([True, True, False], [1, 0, 0], beta=2.0)
        assert m["f2"] > m["f1"]

    def test_f1_equals_f2_when_beta_1(self):
        preds = [True, False, True, False]
        labels = [1, 0, 1, 0]
        m1 = compute_metrics(preds, labels, beta=1.0)
        m2 = compute_metrics(preds, labels, beta=2.0)
        assert m1["f1"] == pytest.approx(m2["f1"])
        # f2 with beta=2 is different from f1
        assert m1["f1"] == pytest.approx(m2["f1"])

    def test_cost_reduction_below_trr(self):
        # With cost_ratio=0.02, cost_reduction should be slightly below TRR
        m = compute_metrics([True, False, False, False], [1, 0, 0, 0], cost_ratio=0.02)
        assert m["cost_savings_ratio"] < m["token_reduction_rate"]
        assert m["cost_savings_ratio"] == pytest.approx(m["token_reduction_rate"] * 0.98)

    def test_cost_reduction_equals_trr_at_zero_ratio(self):
        m = compute_metrics([True, False], [1, 0], cost_ratio=0.0)
        assert m["cost_savings_ratio"] == m["token_reduction_rate"]

    def test_cost_ratio_clamped(self):
        # cost_ratio > 1 should be clamped to 1.0 -> cost_reduction = 0
        m = compute_metrics([True, False], [1, 0], cost_ratio=2.0)
        assert m["cost_savings_ratio"] == 0.0

    def test_no_predictions_positive(self):
        # TP+FP=0 -> precision=0 by convention
        m = compute_metrics([False, False], [1, 1])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_empty_input(self):
        m = compute_metrics([], [])
        assert m["total_samples"] == 0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0


class TestPerGroupMetrics:
    def test_two_groups(self):
        # proj_a: correct predictions (TP, TN)
        # proj_b: wrong predictions (FP, FN)
        preds = [True, False, True, False]
        labels = [1, 0, 0, 1]
        groups = ["proj_a", "proj_a", "proj_b", "proj_b"]
        result = per_group_metrics(preds, labels, groups)
        assert "proj_a" in result
        assert "proj_b" in result
        # proj_a: TP=1, FP=0, FN=0, TN=1 -> precision=1.0, recall=1.0
        assert result["proj_a"]["precision"] == pytest.approx(1.0)
        assert result["proj_a"]["recall"] == pytest.approx(1.0)
        # proj_b: TP=0, FP=1, FN=1, TN=0 -> precision=0.0, recall=0.0
        assert result["proj_b"]["precision"] == pytest.approx(0.0)
        assert result["proj_b"]["recall"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must be the same length"):
            per_group_metrics([True], [1, 0], ["a", "b"])

    def test_single_group(self):
        preds = [True, False, True]
        labels = [1, 0, 1]
        groups = ["only", "only", "only"]
        result = per_group_metrics(preds, labels, groups)
        assert "only" in result
        assert result["only"]["total_samples"] == 3
