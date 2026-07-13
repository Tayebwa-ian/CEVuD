"""
Unit Tests: code_chunks
=======================
Validates the uniform code-window chunking used to feed the small Stage-2
classifier (and the aggregation that turns per-chunk scores into one score).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from code_chunks import chunk_code, aggregate_chunk_scores


def _code_lines(n, start=0):
    """Build ``n`` real code-signal lines (no comments/docstrings)."""
    return [f"def f_{start + i}():\n    return {start + i}\n" for i in range(n)]


def test_single_chunk_when_snippet_fits():
    code = "".join(_code_lines(3))
    chunks = chunk_code(code, max_lines=64, overlap=8, min_code_lines=2)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == len(code.splitlines())


def test_multiple_chunks_with_overlap():
    # ~180 lines -> several 64-line windows with 8-line overlap.
    code = "".join(_code_lines(60))
    chunks = chunk_code(code, max_lines=64, overlap=8, min_code_lines=2)
    assert len(chunks) >= 2
    # Windows overlap by `overlap` lines (boundaries are not strictly disjoint).
    for a, b in zip(chunks, chunks[1:]):
        # Next window starts before the previous one ends (overlap).
        assert b.start_line < a.end_line
    # The very last chunk covers the end of the snippet.
    assert chunks[-1].end_line == len(code.splitlines())


def test_low_signal_windows_dropped():
    # A long snippet that is entirely comments is dropped entirely once it
    # exceeds the single-chunk size (multi-chunk branch enforces min_code_lines).
    comment_blob = "\n".join(f"# comment {i}" for i in range(130))
    assert chunk_code(comment_blob, max_lines=64, overlap=8, min_code_lines=2) == []


def test_low_signal_chunk_kept_when_single():
    # A short comment-only snippet is a single chunk and is kept (it still gets
    # a score at inference; the function-level filter already handled noise).
    chunks = chunk_code("# just a docstring\n", max_lines=64, overlap=8, min_code_lines=2)
    assert len(chunks) == 1


def test_aggregate_max_and_mean():
    scores = [0.1, 0.9, 0.2]
    assert aggregate_chunk_scores(scores, "max") == pytest.approx(0.9)
    assert aggregate_chunk_scores(scores, "mean") == pytest.approx(0.4)
    # Default aggregation is max.
    assert aggregate_chunk_scores(scores) == pytest.approx(0.9)
    # Empty input is safe.
    assert aggregate_chunk_scores([]) == 0.0
