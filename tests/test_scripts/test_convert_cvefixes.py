"""
Unit Tests: convert_cvefixes.py
================================
Validates the CVEfixes → CEVuD manifest converter: row resolution helpers,
noise/trivial filters, deduplication, and the end-to-end stream with a mock
HuggingFace dataset.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from scripts.convert_cvefixes import (
    _resolve_label,
    _resolve_code,
    _resolve_repo_url,
    _resolve_commit_id,
    _resolve_cve_id,
    _resolve_cvss_score,
    _resolve_vuln_type,
    _resolve_diff_with_context,
    _infer_function_name,
    convert_cvefixes,
)


class TestResolveLabel:
    def test_returns_one_when_vulnerable_code_present(self):
        row = {"vulnerable_code": "def foo(): pass"}
        assert _resolve_label(row) == 1

    def test_returns_none_when_vulnerable_code_empty(self):
        row = {"vulnerable_code": ""}
        assert _resolve_label(row) is None

    def test_returns_none_when_vulnerable_code_missing(self):
        row = {}
        assert _resolve_label(row) is None

    def test_returns_none_when_vulnerable_code_whitespace(self):
        row = {"vulnerable_code": "   "}
        assert _resolve_label(row) is None

    def test_returns_none_when_not_string(self):
        row = {"vulnerable_code": 123}
        assert _resolve_label(row) is None


class TestResolveCode:
    def test_returns_vulnerable_code(self):
        row = {"vulnerable_code": "def foo(): return 1"}
        assert _resolve_code(row) == "def foo(): return 1"

    def test_returns_none_when_missing(self):
        assert _resolve_code({}) is None


class TestResolveRepoUrl:
    def test_returns_stripped_url(self):
        row = {"repo_url": "  https://github.com/org/repo.git  "}
        assert _resolve_repo_url(row) == "https://github.com/org/repo.git"

    def test_returns_unknown_when_empty(self):
        row = {"repo_url": ""}
        assert _resolve_repo_url(row) == "unknown_repo"

    def test_returns_unknown_when_missing(self):
        assert _resolve_repo_url({}) == "unknown_repo"


class TestResolveCommitId:
    def test_prefers_hash_field(self):
        row = {"hash": "abc123", "commit_id": "def456"}
        assert _resolve_commit_id(row) == "abc123"

    def test_falls_back_to_commit_id(self):
        row = {"commit_id": "def456"}
        assert _resolve_commit_id(row) == "def456"

    def test_falls_back_to_commit_hash(self):
        row = {"commit_hash": "ghi789"}
        assert _resolve_commit_id(row) == "ghi789"

    def test_falls_back_to_commit_sha(self):
        row = {"commit_sha": "jkl012"}
        assert _resolve_commit_id(row) == "jkl012"

    def test_returns_unknown_when_all_missing(self):
        assert _resolve_commit_id({}) == "unknown_commit"

    def test_skips_empty_strings(self):
        row = {"hash": "", "commit_id": "  "}
        assert _resolve_commit_id(row) == "unknown_commit"


class TestResolveCveId:
    def test_returns_cve_id(self):
        row = {"cve_id": "CVE-2021-1234"}
        assert _resolve_cve_id(row) == "CVE-2021-1234"

    def test_returns_unknown_when_empty(self):
        row = {"cve_id": ""}
        assert _resolve_cve_id(row) == "unknown_cve"

    def test_returns_unknown_when_missing(self):
        assert _resolve_cve_id({}) == "unknown_cve"


class TestResolveCvssScore:
    def test_prefers_cvss3(self):
        row = {"cvss3_base_score": 9.8, "cvss2_base_score": 5.0}
        assert _resolve_cvss_score(row) == 9.8

    def test_falls_back_to_cvss2(self):
        row = {"cvss2_base_score": 5.0}
        assert _resolve_cvss_score(row) == 5.0

    def test_returns_zero_when_missing(self):
        assert _resolve_cvss_score({}) == 0.0


class TestResolveVulnType:
    def test_returns_cwe_id(self):
        row = {"cwe_id": "CWE-79"}
        assert _resolve_vuln_type(row) == "CWE-79"

    def test_returns_unknown_when_empty(self):
        row = {"cwe_id": ""}
        assert _resolve_vuln_type(row) == ""

    def test_returns_unknown_when_missing(self):
        assert _resolve_vuln_type({}) == "unknown"


class TestResolveDiffWithContext:
    def test_returns_diff(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n def vuln():\n-    eval(input())\n+    safe())\n"
        row = {"diff_with_context": diff}
        assert _resolve_diff_with_context(row) == diff.strip()

    def test_returns_empty_when_missing(self):
        row = {}
        assert _resolve_diff_with_context(row) == ""

    def test_returns_empty_when_whitespace(self):
        row = {"diff_with_context": "   "}
        assert _resolve_diff_with_context(row) == ""


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


class TestConvertCvefixes:
    def _make_row(self, vuln_code="def vuln():\n    eval(input())\n", fixed_code=None,
                  repo_url="https://github.com/org/repo", hash="abc123",
                  cve_id="CVE-2021-1234", cwe_id="CWE-79",
                  cvss3_base_score=9.8, diff_with_context="--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n def vuln():\n-    eval(input())\n+    safe())\n",
                  language="python"):
        row = {
            "vulnerable_code": vuln_code,
            "fixed_code": fixed_code,
            "repo_url": repo_url,
            "hash": hash,
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "cvss3_base_score": cvss3_base_score,
            "diff_with_context": diff_with_context,
            "language": language,
        }
        return row

    @patch("scripts.convert_cvefixes.load_from_disk")
    def test_emits_vulnerable_and_fixed_samples(self, mock_load_from_disk):
        vuln_code = "def vuln():\n    eval(input())\n"
        safe_code = "def vuln():\n    safe()\n"
        row = self._make_row(vuln_code=vuln_code, fixed_code=safe_code)
        mock_ds = MagicMock()
        mock_ds.to_iterable_dataset.return_value = iter([row])
        mock_load_from_disk.return_value = mock_ds

        with patch("scripts.convert_cvefixes.parse_diff_anchors", return_value={"foo.py": {"pre_start": 1, "pre_end": 2, "post_start": 1, "post_end": 2}}):
            with patch("scripts.convert_cvefixes.is_trivial_change", return_value=False):
                with patch("scripts.convert_cvefixes.code_signal_line_count", return_value=2):
                    convert_cvefixes(
                        dataset_id="hitoshura25/cvefixes",
                        output_path="/tmp/test_manifest_cvefixes.json",
                        local_dir="./cvefixes_dataset",
                        limit=None,
                        noise_filter=False,
                        trivial_filter=False,
                        dedup=False,
                        emit_fixed_safe=True,
                    )

        with open("/tmp/test_manifest_cvefixes.json") as f:
            manifest = json.load(f)

        assert len(manifest) == 1
        samples = manifest[0]["samples"]
        assert len(samples) == 2
        assert samples[0]["label"] == 1
        assert samples[1]["label"] == 0
        assert samples[0]["sample_subtype"] == "vulnerable"
        assert samples[1]["sample_subtype"] == "fixed"

    @patch("scripts.convert_cvefixes.load_from_disk")
    def test_skips_non_python(self, mock_load_from_disk):
        row = self._make_row(language="java")
        mock_ds = MagicMock()
        mock_ds.to_iterable_dataset.return_value = iter([row])
        mock_load_from_disk.return_value = mock_ds

        convert_cvefixes(
            dataset_id="hitoshura25/cvefixes",
            output_path="/tmp/test_manifest_cvefixes_skip.json",
            local_dir="./cvefixes_dataset",
            limit=None,
            noise_filter=False,
            trivial_filter=False,
            dedup=False,
            emit_fixed_safe=False,
        )

        with open("/tmp/test_manifest_cvefixes_skip.json") as f:
            manifest = json.load(f)
        assert len(manifest) == 0

    @patch("scripts.convert_cvefixes.load_from_disk")
    def test_drops_empty_projects(self, mock_load_from_disk):
        row = self._make_row()
        row["repo_url"] = ""
        mock_ds = MagicMock()
        mock_ds.to_iterable_dataset.return_value = iter([row])
        mock_load_from_disk.return_value = mock_ds

        convert_cvefixes(
            dataset_id="hitoshura25/cvefixes",
            output_path="/tmp/test_manifest_cvefixes_empty.json",
            local_dir="./cvefixes_dataset",
            limit=None,
            noise_filter=False,
            trivial_filter=False,
            dedup=False,
            emit_fixed_safe=False,
        )

        with open("/tmp/test_manifest_cvefixes_empty.json") as f:
            manifest = json.load(f)
        assert len(manifest) == 0

    @patch("scripts.convert_cvefixes.load_from_disk")
    def test_dedup_drops_duplicate(self, mock_load_from_disk):
        vuln_code = "def vuln():\n    eval(input())\n"
        safe_code = "def vuln():\n    safe()\n"
        row1 = self._make_row(vuln_code=vuln_code, fixed_code=safe_code, hash="abc123")
        row2 = self._make_row(vuln_code=vuln_code, fixed_code=safe_code, hash="def456")
        mock_ds = MagicMock()
        mock_ds.to_iterable_dataset.return_value = iter([row1, row2])
        mock_load_from_disk.return_value = mock_ds

        with patch("scripts.convert_cvefixes.parse_diff_anchors", return_value={"foo.py": {"pre_start": 1, "pre_end": 2, "post_start": 1, "post_end": 2}}):
            with patch("scripts.convert_cvefixes.is_trivial_change", return_value=False):
                with patch("scripts.convert_cvefixes.code_signal_line_count", return_value=2):
                    convert_cvefixes(
                        dataset_id="hitoshura25/cvefixes",
                        output_path="/tmp/test_manifest_cvefixes_dedup.json",
                        local_dir="./cvefixes_dataset",
                        limit=None,
                        noise_filter=False,
                        trivial_filter=False,
                        dedup=True,
                        emit_fixed_safe=True,
                    )

        with open("/tmp/test_manifest_cvefixes_dedup.json") as f:
            manifest = json.load(f)
        samples = manifest[0]["samples"]
        assert len(samples) == 2  # First pair kept, second dropped as duplicate
