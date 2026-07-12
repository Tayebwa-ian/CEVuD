"""code_chunks.py
===============
Shared chunking utilities for the small Stage-2 classifier.

The classifier (CodeBERT, capped at 512 tokens) is trained and run on
*uniform code chunks* rather than whole functions. This:

  * keeps every input inside the model's context window, so the vulnerable
    code is never silently truncated away,
  * matches how we train (function-level label, chunk-level input) so there is
    no train/inference skew, and
  * lets the gate aggregate per-chunk scores (a function is vulnerable if *any*
    chunk is).

Rationale and the techniques we borrow are documented in
``docs/SLM_CHUNKING.md`` (sliding-window chunking for encoder models, the
VulDeePecker "code gadget" idea, LineVul's line-level scoring, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from data_quality import code_signal_line_count


@dataclass
class CodeChunk:
    """A uniform window of ``code`` with 1-based line offsets (relative to the
    original snippet it was cut from)."""

    text: str
    start_line: int
    end_line: int


def chunk_code(
    code: str,
    max_lines: int = 64,
    overlap: int = 8,
    min_code_lines: int = 2,
) -> List[CodeChunk]:
    """Split ``code`` into uniform, line-windowed chunks that fit the model.

    * A snippet that already fits in ``max_lines`` becomes a single chunk.
    * Otherwise a sliding window of ``max_lines`` with ``overlap`` overlapping
      lines walks the snippet, so a vulnerability sitting near a boundary is
      never split away from its surrounding context.
    * Chunks carrying fewer than ``min_code_lines`` real code-signal lines
      (comments / docstrings / version assignments do not count) are dropped,
      so the classifier never trains on signal-free text.

    Returns an empty list only when ``code`` is empty.
    """
    if max_lines < 1:
        max_lines = 1
    lines = code.splitlines()
    if not lines:
        return []

    if len(lines) <= max_lines:
        return [CodeChunk("\n".join(lines), 1, len(lines))]

    chunks: List[CodeChunk] = []
    step = max(1, max_lines - overlap)
    n = len(lines)
    start = 0
    while start < n:
        end = min(n, start + max_lines)
        window = lines[start:end]
        chunk = CodeChunk("\n".join(window), start + 1, end)
        if code_signal_line_count(chunk.text) >= min_code_lines:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def aggregate_chunk_scores(scores: List[float], method: str = "max") -> float:
    """Reduce per-chunk probabilities to a single snippet / function score.

    * ``max``  — the function is vulnerable if *any* chunk is (recommended: a
                 single dangerous statement makes the whole function unsafe).
    * ``mean`` — average over chunks (more conservative, smoother scores).
    """
    if not scores:
        return 0.0
    if method == "mean":
        return sum(scores) / len(scores)
    # default: max
    return max(scores)
