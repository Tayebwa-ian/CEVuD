"""
Unit Tests: dataset_splitter
=============================
Validates the held-out split strategies that keep validation/test data
separate from the grid search / weight selection process.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.schema import RawScoreRecord, RepoProvenance, SplitName, VALID_SPLITS
from evaluation.dataset_splitter import (
    split_by_project,
    split_stratified,
    apply_split,
)


def _make_record(sample_id, project, label):
    return RawScoreRecord(
        sample_id=sample_id,
        project=project,
        file_path=f"f_{sample_id}.py",
        function_name=f"func_{sample_id}",
        label=label,
        severity="ERROR",
        severity_weight=1.0 if label == 1 else 0.0,
        slm_score=0.5,
        vulnerability_type=None,
        provenance=RepoProvenance(
            project=project,
            git_url="https://example.com/test.git",
            requested_ref="main",
            resolved_commit_sha="abc123",
            cloned_at_utc="2025-01-01T00:00:00Z",
        ),
    )


class TestSplitByProject:
    def test_all_projects_assigned(self):
        records = [_make_record(f"p{i}_s{j}", f"proj_{i}", j % 2) for i in range(4) for j in range(5)]
        assignment = split_by_project(records, val_frac=0.25, test_frac=0.25, seed=42)
        all_projects = {"proj_0", "proj_1", "proj_2", "proj_3"}
        assigned_projects = {r.project for r in records if assignment[r.sample_id] == "train"} | \
                            {r.project for r in records if assignment[r.sample_id] == "validation"} | \
                            {r.project for r in records if assignment[r.sample_id] == "test"}
        assert assigned_projects == all_projects

    def test_no_project_in_multiple_splits(self):
        records = [_make_record(f"p{i}_s{j}", f"proj_{i}", j % 2) for i in range(6) for j in range(5)]
        assignment = split_by_project(records, val_frac=0.2, test_frac=0.2, seed=42)
        project_splits = {}
        for r in records:
            split = assignment[r.sample_id]
            project_splits.setdefault(r.project, set()).add(split)
        for proj, splits in project_splits.items():
            assert len(splits) == 1, f"Project {proj} appears in multiple splits: {splits}"

    def test_all_splits_non_empty(self):
        records = [_make_record(f"p{i}_s{j}", f"proj_{i}", j % 2) for i in range(5) for j in range(4)]
        assignment = split_by_project(records, val_frac=0.2, test_frac=0.2, seed=42)
        splits = apply_split(records, assignment)
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_reproducible_with_seed(self):
        records = [_make_record(f"p{i}_s{j}", f"proj_{i}", j % 2) for i in range(4) for j in range(5)]
        a1 = split_by_project(records, val_frac=0.25, test_frac=0.25, seed=42)
        a2 = split_by_project(records, val_frac=0.25, test_frac=0.25, seed=42)
        assert a1 == a2

    def test_too_few_projects_raises(self):
        records = [_make_record("s1", "only_proj", 1), _make_record("s2", "only_proj", 0)]
        with pytest.raises(ValueError, match="requires >= 3 distinct projects"):
            split_by_project(records)


class TestSplitStratified:
    def test_all_splits_non_empty(self):
        records = [_make_record(f"s{i}", f"proj_{i % 3}", i % 2) for i in range(30)]
        assignment = split_stratified(records, val_frac=0.2, test_frac=0.2, seed=42)
        splits = apply_split(records, assignment)
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_all_records_assigned(self):
        records = [_make_record(f"s{i}", f"proj_{i % 3}", i % 2) for i in range(20)]
        assignment = split_stratified(records, val_frac=0.2, test_frac=0.2, seed=42)
        assigned = {r.sample_id: assignment[r.sample_id] for r in records}
        assert len(assigned) == len(records)

    def test_reproducible_with_seed(self):
        records = [_make_record(f"s{i}", f"proj_{i % 3}", i % 2) for i in range(20)]
        a1 = split_stratified(records, val_frac=0.2, test_frac=0.2, seed=42)
        a2 = split_stratified(records, val_frac=0.2, test_frac=0.2, seed=42)
        assert a1 == a2

    def test_stratified_preserves_class_balance(self):
        # Equal vulnerable/safe, so each split should be roughly balanced
        records = [_make_record(f"s{i}", "proj", i % 2) for i in range(100)]
        assignment = split_stratified(records, val_frac=0.2, test_frac=0.2, seed=42)
        splits = apply_split(records, assignment)
        for name, split in splits.items():
            vuln = sum(1 for r in split if r.label == 1)
            safe = sum(1 for r in split if r.label == 0)
            total = vuln + safe
            if total > 0:
                ratio = vuln / total
                assert 0.2 < ratio < 0.8, f"Split {name} has unbalanced ratio: {ratio}"


class TestApplySplit:
    def test_correct_partitioning(self):
        records = [_make_record(f"s{i}", "proj", i % 2) for i in range(10)]
        assignment = {f"s{i}": "train" if i < 5 else "test" for i in range(10)}
        splits = apply_split(records, assignment)
        assert len(splits["train"]) == 5
        assert len(splits["test"]) == 5
        assert len(splits["validation"]) == 0

    def test_missing_assignment_raises(self):
        records = [_make_record("s1", "proj", 1)]
        assignment = {}
        with pytest.raises(KeyError, match="No split assignment found"):
            apply_split(records, assignment)

    def test_returns_all_valid_splits(self):
        splits = apply_split([], {})
        for s in VALID_SPLITS:
            assert s in splits
            assert splits[s] == []
