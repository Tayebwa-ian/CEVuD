"""
Unit Tests: raw_score_extractor
=================================
Validates that:
1. Git-source extraction does NOT include cross-file context in the SLM input
   (training and pipeline inference both use only imports + enclosing function).
2. SLM inference in _finalize applies chunking (matching training/pipeline).
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from evaluation.schema import BenchmarkSample, ProjectManifest, GitRepoSource, RepoProvenance
from evaluation.raw_score_extractor import RawScoreExtractor


SAMPLE_SOURCE = """\
import os
from utils import helper

def vulnerable_function():
    result = os.system("rm -rf /")
    return result

def safe_function():
    return "world"

def another_vulnerable(x):
    return eval(x)
"""


def _make_git_project(samples):
    return ProjectManifest(
        project="test_repo",
        samples=samples,
        git_source=GitRepoSource(git_url="https://example.com/test.git", ref="main"),
    )


def _make_sample(sample_id, file_path, start_line, end_line, label=1, source_code=None, commit_id="abc123"):
    return BenchmarkSample(
        sample_id=sample_id,
        file_path=file_path,
        function_name=f"func_{sample_id}",
        start_line=start_line,
        end_line=end_line,
        label=label,
        source_code=source_code,
        commit_id=commit_id,
    )


class TestGitSourceExtractionNoCrossFile:
    """Change 1: cross-file context must NOT appear in the SLM input text."""

    def test_slm_input_excludes_cross_file_content(self):
        """Directly test that build_context_snippet is called with empty cross
        dict, meaning no cross-file content is baked into the SLM text."""
        from code_context import build_context_snippet, collect_module_imports

        imports = collect_module_imports(SAMPLE_SOURCE)
        # vulnerable_function spans lines 4-6 in SAMPLE_SOURCE
        slm_code_with_empty_cross = build_context_snippet(
            SAMPLE_SOURCE, (4, 6), imports, {}
        )
        # The SLM input should contain the function signature and imports
        assert "def vulnerable_function" in slm_code_with_empty_cross
        assert "import os" in slm_code_with_empty_cross
        # Cross-file content (other functions from the same file) should be absent
        # because cross_file context is only for OTHER files, but the key test
        # is that no extra file content is included
        assert "def safe_function" not in slm_code_with_empty_cross
        assert "def another_vulnerable" not in slm_code_with_empty_cross

    @patch("evaluation.raw_score_extractor._run_semgrep")
    def test_extract_git_source_uses_empty_cross(self, mock_semgrep):
        """Verify that _extract_git_source builds SLM code without cross-file
        context by checking the snippets it produces."""
        mock_semgrep.return_value = {"results": []}

        project = _make_git_project([
            _make_sample("s1", "module.py", 5, 8, label=1, source_code=SAMPLE_SOURCE),
        ])

        config_path = _write_config()
        extractor = RawScoreExtractor(config_path=config_path)

        provenance = RepoProvenance(
            project="test_repo",
            git_url="https://example.com/test.git",
            requested_ref="main",
            resolved_commit_sha="abc123",
            cloned_at_utc="2025-01-01T00:00:00Z",
        )

        # Patch _finalize to capture snippets without running the real model
        captured_snippets = []
        def mock_finalize(proj, meta, severities, snippets, prov):
            captured_snippets.extend(snippets)
            return []

        # Mock _git_show to return the sample source so _extract_git_source
        # takes the non-inline branch (which uses expand_to_function).
        with patch.object(extractor, "_finalize", side_effect=mock_finalize), \
             patch.object(RawScoreExtractor, "_git_show", return_value=SAMPLE_SOURCE):
            records = extractor._extract_git_source(project, "/fake/path", provenance)

        # The snippet sent to _finalize should contain only the target function
        assert len(captured_snippets) == 1
        slm_text = captured_snippets[0]
        assert "def vulnerable_function" in slm_text
        # The snippet should NOT contain other functions from the same file
        # because expand_to_function returns only the enclosing function
        assert slm_text.count("def ") == 1, (
            f"Expected 1 function definition in SLM input, found {slm_text.count('def ')}: {slm_text!r}"
        )


class TestFinalizeAppliesChunking:
    """Change 2: _finalize must chunk snippets before calling the classifier."""

    def test_finalize_chunks_long_snippets(self):
        """A long snippet (>64 lines) should be split into multiple chunks
        before being passed to the classifier."""
        long_source = "\n".join(f"def f_{i}():\n    return {i}\n" for i in range(60))

        project = _make_git_project([
            _make_sample("long_1", "long.py", 2, 5, label=1, source_code=long_source, commit_id=None),
        ])

        config_path = _write_config()
        extractor = RawScoreExtractor(config_path=config_path)

        captured_texts = []

        def fake_inference(texts):
            captured_texts.extend(texts)
            return [0.1] * len(texts)

        with patch.object(extractor._get_model_manager(), "get_classifier_inference", side_effect=fake_inference):
            records = extractor._finalize(
                project,
                project.samples,
                ["NONE"] * len(project.samples),
                [long_source] * len(project.samples),
                RepoProvenance(project="test", git_url=None, requested_ref=None, resolved_commit_sha=None, cloned_at_utc=None),
            )

        # With ~180 lines and chunk_max_lines=64, we expect multiple chunks
        assert len(captured_texts) > 1, "Expected multiple chunks for a long snippet"
        # Each chunk should be shorter than the original
        for chunk in captured_texts:
            chunk_lines = len(chunk.splitlines())
            assert chunk_lines <= 72, f"Chunk has {chunk_lines} lines, expected ~64"

    def test_finalize_short_snippet_single_call(self):
        """A short snippet should still go through chunking but produce 1 chunk."""
        short_source = "def short():\n    return 1\n"
        project = _make_git_project([
            _make_sample("short_1", "short.py", 1, 2, label=1, source_code=short_source, commit_id=None),
        ])

        config_path = _write_config()
        extractor = RawScoreExtractor(config_path=config_path)

        captured_texts = []

        def fake_inference(texts):
            captured_texts.extend(texts)
            return [0.1] * len(texts)

        with patch.object(extractor._get_model_manager(), "get_classifier_inference", side_effect=fake_inference):
            records = extractor._finalize(
                project,
                project.samples,
                ["NONE"],
                [short_source],
                RepoProvenance(project="test", git_url=None, requested_ref=None, resolved_commit_sha=None, cloned_at_utc=None),
            )

        assert len(captured_texts) >= 1
        assert len(records) == 1
        assert records[0].slm_score == pytest.approx(0.1, abs=1e-3)

    def test_finalize_empty_snippets(self):
        """Empty snippets list should not call the classifier."""
        project = _make_git_project([])
        config_path = _write_config()
        extractor = RawScoreExtractor(config_path=config_path)

        with patch.object(extractor._get_model_manager(), "get_classifier_inference") as mock_inf:
            records = extractor._finalize(
                project, [], [], [],
                RepoProvenance(project="test", git_url=None, requested_ref=None, resolved_commit_sha=None, cloned_at_utc=None),
            )
            mock_inf.assert_not_called()

        assert records == []

    def test_finalize_chunk_scores_aggregated_with_max(self):
        """When a snippet has multiple chunks, the SLM score should be the
        max of per-chunk probabilities (matching pipeline aggregation)."""
        source = "\n".join(f"def f_{i}():\n    return {i}\n" for i in range(60))
        project = _make_git_project([
            _make_sample("long_1", "long.py", 2, 5, label=1, source_code=source, commit_id=None),
        ])

        config_path = _write_config()
        extractor = RawScoreExtractor(config_path=config_path)

        # Return rising probabilities so max = last chunk's probability
        def fake_inference(texts):
            n = len(texts)
            return [round((i + 1) / n, 4) for i in range(n)]

        with patch.object(extractor._get_model_manager(), "get_classifier_inference", side_effect=fake_inference):
            records = extractor._finalize(
                project,
                project.samples,
                ["NONE"],
                [source],
                RepoProvenance(project="test", git_url=None, requested_ref=None, resolved_commit_sha=None, cloned_at_utc=None),
            )

        assert len(records) == 1
        # Score should be the max of per-chunk probabilities (last chunk = 1.0)
        assert records[0].slm_score == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmpdir=None):
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    config_path = os.path.join(tmpdir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "paths": {"workspace_root": tmpdir},
            "semgrep_severity_map": {"NONE": 0.0, "ERROR": 1.0, "WARNING": 0.7, "INFO": 0.3},
            "slm_inference": {"chunk_max_lines": 64, "chunk_overlap": 8, "min_code_lines": 2, "aggregation": "max"},
        }, f)
    return config_path
