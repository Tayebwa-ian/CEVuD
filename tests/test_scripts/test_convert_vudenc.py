"""
Unit Tests: convert_vudenc.py
==============================
Validates the VUDENC → CEVuD manifest converter: row-to-sample conversion,
function-name inference, benign-manifest loading, and the end-to-end stream
with a mock HuggingFace dataset.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from scripts.convert_vudenc import (
    _row_to_sample,
    _infer_function_name,
    _load_benign_as_local_projects,
    convert_vudenc,
)


class TestRowToSample:
    def test_returns_none_for_empty_raw_lines(self):
        assert _row_to_sample({}, 0) is None

    def test_returns_none_for_missing_raw_lines_and_lines(self):
        assert _row_to_sample({"label": [1, 0]}, 0) is None

    def test_returns_sample_with_label_one(self):
        row = {
            "raw_lines": ["def foo():", "    return 1"],
            "label": [1, 0],
            "type": ["sql", "0"],
        }
        sample = _row_to_sample(row, 0)
        assert sample is not None
        assert sample["label"] == 1
        assert sample["vulnerability_type"] == "sql"
        assert sample["sample_subtype"] == "vulnerable"

    def test_returns_sample_with_label_zero(self):
        row = {
            "raw_lines": ["def foo():", "    return 0"],
            "label": [0, 0],
            "type": ["0", "0"],
        }
        sample = _row_to_sample(row, 1)
        assert sample is not None
        assert sample["label"] == 0
        assert sample["vulnerability_type"] == "unknown"
        assert sample["sample_subtype"] == "benign"

    def test_dominant_vulnerability_type(self):
        row = {
            "raw_lines": ["def foo():", "    return 1"],
            "label": [1, 1],
            "type": ["sql", "xss", "sql"],
        }
        sample = _row_to_sample(row, 0)
        assert sample["vulnerability_type"] == "sql"

    def test_prefers_raw_lines_over_tokenised(self):
        row = {
            "raw_lines": ["def foo():", "    return 1"],
            "lines": ["def", "foo", "():"],
            "label": [1],
            "type": ["sql"],
        }
        sample = _row_to_sample(row, 0)
        assert "def foo():" in sample["source_code"]

    def test_extracts_function_name(self):
        row = {
            "raw_lines": ["def target_func(x):", "    return x * 2"],
            "label": [1, 0],
            "type": ["sql", "0"],
        }
        sample = _row_to_sample(row, 0)
        assert sample["function_name"] == "target_func"

    def test_returns_unknown_function_name_when_no_def(self):
        row = {
            "raw_lines": ["x = 1", "y = 2"],
            "label": [0, 0],
            "type": ["0", "0"],
        }
        sample = _row_to_sample(row, 0)
        assert sample["function_name"] == "unknown_function"

    def test_line_range_covers_all_lines(self):
        raw_lines = ["line1", "line2", "line3"]
        row = {
            "raw_lines": raw_lines,
            "label": [0, 0, 0],
            "type": ["0", "0", "0"],
        }
        sample = _row_to_sample(row, 0)
        assert sample["start_line"] == 1
        assert sample["end_line"] == 3

    def test_empty_raw_lines_returns_none(self):
        assert _row_to_sample({"raw_lines": []}, 0) is None


class TestInferFunctionName:
    def test_extracts_def_name(self):
        code = "def my_func():\n    pass\n"
        assert _infer_function_name(code) == "my_func"

    def test_extracts_async_def_name(self):
        code = "async def my_func():\n    pass\n"
        assert _infer_function_name(code) == "my_func"

    def test_returns_unknown_when_no_def(self):
        code = "x = 1\n"
        assert _infer_function_name(code) == "unknown_function"

    def test_returns_unknown_when_empty(self):
        assert _infer_function_name("") == "unknown_function"


class TestLoadBenignAsLocalProjects:
    def _make_project(self, project_name="test_proj", samples=None):
        if samples is None:
            samples = [
                MagicMock(
                    sample_id="s1",
                    file_path="foo.py",
                    function_name="bar",
                    start_line=1,
                    end_line=5,
                    label=0,
                    vulnerability_type="benign",
                    sample_subtype="benign_control",
                    source_code="def bar():\n    return 0\n",
                    fixed_code=None,
                    repo_url="https://github.com/org/repo",
                    commit_id="abc123",
                    target_commit="abc123",
                    cve_id=None,
                    cvss_score=0.0,
                    diff_with_context="",
                )
            ]
        proj = MagicMock()
        proj.project = project_name
        proj.samples = samples
        return proj

    @patch("scripts.convert_vudenc.load_manifest")
    def test_returns_empty_for_empty_projects(self, mock_load_manifest):
        mock_load_manifest.return_value = []
        result = _load_benign_as_local_projects("/tmp/manifest.json")
        assert result == []

    @patch("scripts.convert_vudenc.load_manifest")
    def test_converts_project_samples(self, mock_load_manifest):
        mock_load_manifest.return_value = [self._make_project()]
        result = _load_benign_as_local_projects("/tmp/manifest.json")
        assert len(result) == 1
        assert result[0]["project"] == "vudenc_benign::test_proj"
        assert result[0]["local_source"] == {"root_path": "."}
        assert len(result[0]["samples"]) == 1
        assert result[0]["samples"][0]["label"] == 0
        assert result[0]["samples"][0]["sample_subtype"] == "benign_control"
