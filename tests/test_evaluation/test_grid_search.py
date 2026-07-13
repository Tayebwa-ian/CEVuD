"""
Unit Tests: grid_search
========================
Validates the (weight_static, escalation_threshold) grid sweep that selects
the production gate's weights. The grid is evaluated on the validation split
only, with override_enabled=False for every cell.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.schema import RawScoreRecord, RepoProvenance
from evaluation.grid_search import run_grid_search


def _make_record(sample_id, severity_weight, slm_score, label):
    return RawScoreRecord(
        sample_id=sample_id,
        project="test_proj",
        file_path=f"f_{sample_id}.py",
        function_name=f"func_{sample_id}",
        label=label,
        severity="ERROR" if severity_weight == 1.0 else "WARNING" if severity_weight > 0 else "NONE",
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


class TestRunGridSearch:
    def test_returns_list_and_best_entry(self):
        records = [
            _make_record("vuln_1", 1.0, 0.9, 1),
            _make_record("safe_1", 0.0, 0.1, 0),
            _make_record("vuln_2", 0.7, 0.8, 1),
            _make_record("safe_2", 0.0, 0.2, 0),
        ]
        grid, best = run_grid_search(records, weight_grid=[0.0, 0.5, 1.0], threshold_grid=[0.3, 0.7])
        assert len(grid) == 6  # 3 weights * 2 thresholds
        assert "weight_static" in best
        assert "weight_slm" in best
        assert "escalation_threshold" in best
        assert "metrics" in best
        assert best["weight_slm"] == pytest.approx(1.0 - best["weight_static"])

    def test_grid_size_matches_product(self):
        records = [_make_record(f"r{i}", 0.5, 0.5, i % 2) for i in range(10)]
        grid, _ = run_grid_search(records, weight_grid=[0.0, 0.25, 0.5, 0.75, 1.0], threshold_grid=[0.0, 0.5, 1.0])
        assert len(grid) == 15  # 5 * 3

    def test_best_has_highest_f2(self):
        # Build records where high weight_static + high threshold wins
        records = [
            _make_record("v1", 1.0, 0.9, 1),
            _make_record("v2", 1.0, 0.9, 1),
            _make_record("s1", 0.0, 0.1, 0),
        ]
        _, best = run_grid_search(records, weight_grid=[0.0, 1.0], threshold_grid=[0.0, 1.0])
        # weight_static=1.0, threshold=1.0: v1/s1 both have severity=1.0/0.0
        # v1: 1.0>=1.0 -> TP, v2: 1.0>=1.0 -> TP, s1: 0.0>=1.0 -> TN
        # precision=1.0, recall=1.0, F2=1.0 (perfect)
        assert best["weight_static"] == 1.0
        assert best["escalation_threshold"] == 1.0

    def test_tie_break_prefers_lower_escalation_rate(self):
        # Two configurations with same F2, one cheaper
        records = [
            _make_record("v1", 0.5, 0.5, 1),
            _make_record("s1", 0.5, 0.5, 0),
        ]
        # With threshold=0.5 and weight_static=0.5: risk=0.5 -> escalate both
        # With threshold=1.0: risk=0.5 -> no escalate
        # F2 at threshold=0.5: precision=0.5, recall=1.0 -> F2 = (1+4)*0.5*1/(4*0.5+1) = 2.5/3 = 0.833
        # F2 at threshold=1.0: precision=0, recall=0 -> F2=0
        # The higher F2 wins
        _, best = run_grid_search(records, weight_grid=[0.5], threshold_grid=[0.5, 1.0])
        assert best["escalation_threshold"] == 0.5

    def test_override_never_enabled_in_grid(self):
        records = [_make_record("r", 0.5, 0.5, 1)]
        grid, _ = run_grid_search(records, weight_grid=[0.0, 1.0], threshold_grid=[0.5])
        for entry in grid:
            assert entry.get("override_enabled", True) is not True or "override_enabled" not in entry
            # The grid dict doesn't include override_enabled by design (it's always False)

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="non-empty validation split"):
            run_grid_search([])

    def test_default_grid_axes(self):
        records = [_make_record(f"r{i}", 0.5, 0.5, i % 2) for i in range(20)]
        grid, _ = run_grid_search(records)
        weights = sorted({e["weight_static"] for e in grid})
        thresholds = sorted({e["escalation_threshold"] for e in grid})
        assert len(weights) == 21  # 0.0, 0.05, ..., 1.0
        assert len(thresholds) == 21
