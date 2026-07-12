"""
convert_vudenc.py
=================
Converts the VUDENC dataset (DetectVul/Vudenc on HuggingFace, or the
original LauraWartschinski/VulnerabilityDetection GitHub clone) into the
CEVuD benchmark manifest format — using the SAME manifest schema and
project organisation as `convert_cvefixes.py` produced
`benchmark_manifest_cvefixes.json`, so both datasets plug into the same
evaluation harness (`src/evaluation/run_comparative_evaluation.py`) and
the same training pipeline (`src/training/`).

WHY THE TWO DATASETS HAVE DISTINCT ROLES
---------------------------------------
The custom Stage-2 classifier (src/training/) is developed entirely on
CVEfixes (benchmark_manifest_cvefixes.json): it is used for the model's
training, validation, and its own evaluation (project-level splits prevent
in-corpus leakage). VUDENC (benchmark_manifest_vudenc.json) is the corpus
for the gate study — the comparative evaluation of the full CEVuD pipeline
(Semgrep + the CVEfixes-trained classifier + the gating strategies) run by
src/evaluation/. The two are independently curated, so the gate study
measures the pipeline on code the classifier never trained on.

ABOUT VUDENC
------------
VUDENC (Vulnerability Detection with Deep Learning on a Natural Codebase,
Wartschinski et al., Information & Software Technology, 2022) contains
real-world Python functions mined from vulnerability-fixing commits, labeled
at a **per-line (statement) level** across seven vulnerability categories:
    - SQL injection (SQLi)
    - Cross-Site Scripting (XSS)
    - Command injection
    - Cross-Site Request Forgery (XSRF)
    - Remote Code Execution (RCE)
    - Path Disclosure
    - Open Redirect

HuggingFace schema (DetectVul/Vudenc):
    - lines     (List[str])  : Tokenised/normalised code lines
    - raw_lines (List[str])  : Original untouched source code lines
    - label     (List[int])  : Per-line binary vulnerability label (1/0)
    - type      (List[str])  : Per-line vulnerability category string

CEVuD operates at function level (not line level). This converter:
  1. Joins `raw_lines` back into a full function string (`source_code`).
  2. Derives a function-level label: 1 if ANY line is labeled 1, else 0.
     Functions whose every line is labeled 0 become the `label=0` (safe)
     class, so VUDENC can supply both classes when its line labels include
     non-vulnerable functions; check the printed class balance and, if it
     skews positive, restrict/pre-sample as needed for your experiment.
  3. Derives the vulnerability type from the most-common type in `type`.
  4. Groups samples into logical "projects" by vulnerability category
     (e.g. `vudenc_sql`, `vudenc_xss`) so the DatasetSplitter can enforce
     project-level train/val/test splits (preventing data leakage across
     the boundary).

MANIFEST SCHEMA (mirrors benchmark_manifest_cvefixes.json)
---------------------------------------------------------
Each emitted sample carries the same fields as the CVEfixes manifest so the
two are interchangeable downstream:
    sample_id, file_path, function_name, start_line, end_line, label,
    vulnerability_type, source_code, fixed_code, repo_url, commit_id,
    target_commit, cve_id, cvss_score, diff_with_context

VUDENC ships **no repository or commit metadata** (it is a curated corpus of
functions, not a commit database), so `repo_url`, `commit_id`,
`target_commit`, `cve_id`, `cvss_score`, `diff_with_context` and
`fixed_code` are emitted as `None`/`""`/`0.0` — they are present for schema
consistency and future traceability, not populated. Because there is no
repo to clone, each project uses a `local_source` entry and `source_code`
is embedded inline; the evaluation harness (`raw_score_extractor.py`)
recognises this shape and materialises each snippet to its own file for
scanning (the same inline path CVEfixes uses as its clone-failure fallback).

Usage (HuggingFace — recommended, no download):
    pip install datasets
    python src/scripts/convert_vudenc.py \\
        --output benchmark_manifest_vudenc.json

Usage (local GitHub clone — alternative):
    git clone --depth 1 https://github.com/LauraWartschinski/VulnerabilityDetection
    python src/scripts/convert_vudenc.py \\
        --local-dir VulnerabilityDetection/data \\
        --output benchmark_manifest_vudenc.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import uuid
from collections import Counter
from typing import Any, Dict, Generator, List, Optional

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# data_quality lives in src/ — shared helpers that keep label noise out of the
# corpus (duplicate / contradictory snippets, snippets with no real code).
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from data_quality import (  # noqa: E402
    code_signal_line_count,
    normalize_code,
)

# HuggingFace dataset ID for the VUDENC dataset
HF_DATASET_ID = "DetectVul/Vudenc"

# The seven vulnerability types covered by VUDENC
VUDENC_VULN_TYPES = {
    "sql", "xss", "command", "xsrf", "remote_code_execution",
    "path_disclosure", "open_redirect",
}


# ---------------------------------------------------------------------------
# Source A: HuggingFace streaming loader
# ---------------------------------------------------------------------------

def _stream_hf(split: str = "train") -> Generator[Dict[str, Any], None, None]:
    """Streams rows from HuggingFace DetectVul/Vudenc without full download."""
    if not HF_AVAILABLE:
        print(
            "[ERROR] `datasets` not installed. Run: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[*] Streaming '{HF_DATASET_ID}' (split='{split}') from HuggingFace ...")
    ds = load_dataset(HF_DATASET_ID, split=split, streaming=True, trust_remote_code=False)
    yield from ds


# ---------------------------------------------------------------------------
# Source B: Local directory loader (GitHub clone)
# ---------------------------------------------------------------------------

def _stream_local(local_dir: str) -> Generator[Dict[str, Any], None, None]:
    """Reads VUDENC raw text files from a local directory clone.

    Expects the LauraWartschinski/VulnerabilityDetection layout where each
    vulnerability type has a subdirectory containing .txt or .py files,
    each file being one function.
    """
    pattern = os.path.join(local_dir, "**", "*.py")
    files = glob.glob(pattern, recursive=True)
    if not files:
        # Also try .txt extension used by the original repo
        pattern = os.path.join(local_dir, "**", "*.txt")
        files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"[WARN] No .py or .txt files found under '{local_dir}'", file=sys.stderr)
        return

    for fpath in files:
        # Infer vulnerability type from parent directory name
        parent = os.path.basename(os.path.dirname(fpath)).lower()
        vuln_type = parent if parent in VUDENC_VULN_TYPES else "unknown"
        # Infer label: files in a "vul" or "vulnerable" sub-path are 1
        is_vuln = "vul" in fpath.lower() or "unsafe" in fpath.lower()

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except OSError:
            continue

        lines = raw_text.splitlines()
        yield {
            "raw_lines": lines,
            "label": [int(is_vuln)] * len(lines),
            "type": [vuln_type] * len(lines),
        }


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def _row_to_sample(row: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Converts a single VUDENC row into a CEVuD BenchmarkSample dict.

    Returns None if the row is malformed or empty.
    """
    # Prefer raw_lines (original source) over tokenised lines
    raw_lines: List[str] = row.get("raw_lines") or row.get("lines") or []
    per_line_labels: List[int] = row.get("label") or []
    per_line_types: List[str] = row.get("type") or []

    if not raw_lines:
        return None

    # ── Function-level label: 1 if ANY line is vulnerable ───────────────────
    try:
        function_label = int(any(int(l) for l in per_line_labels if l is not None))
    except (ValueError, TypeError):
        function_label = 0

    # ── Dominant vulnerability type ──────────────────────────────────────────
    non_empty_types = [str(t).strip() for t in per_line_types if t and str(t).strip() != "0"]
    if non_empty_types:
        vuln_type = Counter(non_empty_types).most_common(1)[0][0]
    else:
        vuln_type = "unknown"

    # ── Reconstruct source code ──────────────────────────────────────────────
    source_code = "\n".join(str(l) for l in raw_lines).strip()
    if not source_code:
        return None

    sample_id = f"vudenc::{vuln_type}::{index:06d}::{uuid.uuid4().hex[:6]}"

    # VUDENC carries no repository / commit metadata, so the provenance
    # fields are emitted empty for schema consistency with the CVEfixes
    # manifest (convert_cvefixes.py) — they are simply never populated.
    return {
        "sample_id": sample_id,
        "file_path": "inline_snippet.py",   # synthetic; source_code is embedded below
        "function_name": _infer_function_name(source_code),
        "start_line": 1,
        "end_line": max(len(raw_lines), 1),
        "label": function_label,
        "vulnerability_type": vuln_type,
        "source_code": source_code,
        "fixed_code": None,
        "repo_url": None,
        "commit_id": None,
        "target_commit": None,
        "cve_id": None,
        "cvss_score": 0.0,
        "diff_with_context": "",
    }


def _infer_function_name(code: str) -> str:
    """Best-effort: extract the function name from the first `def` line."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            parts = stripped.split("(")[0].split()
            if len(parts) >= 2:
                return parts[-1]
    return "unknown_function"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_vudenc(
    output_path: str,
    local_dir: Optional[str] = None,
    limit: Optional[int] = None,
    split: str = "train",
    dedup: bool = True,
    min_code_lines: int = 2,
) -> None:
    """Converts VUDENC into a CEVuD benchmark manifest.

    Samples are grouped by vulnerability type (e.g. 'sql', 'xss') to form
    logical "projects" that the DatasetSplitter can cleanly assign to
    train/val/test without leakage.

    Args:
        output_path: Path to write the benchmark_manifest JSON.
        local_dir: If set, read from a local VUDENC clone instead of HF.
        limit: Maximum number of samples to include. None = all.
        split: HuggingFace split to use ('train' or 'test').
        dedup: When True (default), drop duplicate snippets and any snippet
            that collides (identical text) with a snippet of the OPPOSITE
            label — those are hard contradictions the classifier cannot learn.
        min_code_lines: Drop a snippet with fewer than this many lines of real
            code signal (comments/docstrings/version assignments do not count).
    """
    row_stream = (
        _stream_local(local_dir) if local_dir else _stream_hf(split=split)
    )

    # Group samples by vuln_type (used as "project" for split isolation)
    samples_by_project: Dict[str, List[Dict[str, Any]]] = {}
    total_added = total_skipped = 0
    skipped_dup = skipped_contradiction = skipped_short = 0
    # normalized source_code -> label, for duplicate / contradiction detection
    seen_norm: Dict[str, int] = {}

    for idx, row in enumerate(row_stream):
        sample = _row_to_sample(row, idx)
        if sample is None:
            total_skipped += 1
            continue

        # ── Minimum code-signal filter ───────────────────────────────────────
        # A snippet with no real code (e.g. a lone ``__version__ = '3.7'``)
        # carries no learnable signal and would add label noise.
        if code_signal_line_count(sample["source_code"]) < min_code_lines:
            skipped_short += 1
            continue

        # ── Duplicate / contradiction filter ─────────────────────────────────
        # Identical text with different labels is a hard contradiction; identical
        # text with the same label is just a redundant duplicate. Drop both.
        norm = normalize_code(sample["source_code"])
        if dedup and norm in seen_norm:
            if seen_norm[norm] == sample["label"]:
                skipped_dup += 1
            else:
                skipped_contradiction += 1
            continue
        if dedup:
            seen_norm[norm] = sample["label"]

        vuln_type = sample["vulnerability_type"]
        project_key = f"vudenc_{vuln_type}"

        if project_key not in samples_by_project:
            samples_by_project[project_key] = []
        samples_by_project[project_key].append(sample)
        total_added += 1

        if limit is not None and total_added >= limit:
            print(f"[*] Reached --limit {limit}. Stopping early.")
            break

    # ── Assemble the manifest ────────────────────────────────────────────────
    # Mirrors the CVEfixes manifest's per-project grouping, but each VUDENC
    # "project" is a vulnerability category (e.g. vudenc_sql) rather than a
    # git repository — VUDENC ships no repo/commit metadata to clone. The
    # evaluation harness recognises this `local_source` shape and scores the
    # embedded `source_code` directly (the same inline path CVEfixes uses as
    # its clone-failure fallback). This keeps VUDENC and CVEfixes fully
    # interchangeable downstream.
    manifest = [
        {
            "project": project_name,
            "local_source": {"root_path": "."},   # sentinel — source_code used directly
            "samples": samples,
        }
        for project_name, samples in sorted(samples_by_project.items())
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    vulnerable   = sum(s["label"] for p in manifest for s in p["samples"])
    safe         = total_added - vulnerable
    print(f"\n{'─'*60}")
    print(f"  Source        : {'HuggingFace ' + HF_DATASET_ID if not local_dir else local_dir}")
    print(f"  Samples added : {total_added:,}  ({vulnerable:,} vulnerable, {safe:,} safe)")
    print(f"  Skipped empty : {total_skipped:,}  (empty / malformed rows)")
    print(f"  Skipped dup   : {skipped_dup:,}  (duplicate snippets)")
    print(f"  Skipped contra: {skipped_contradiction:,}  (identical text, both labels)")
    print(f"  Skipped short : {skipped_short:,}  (< {min_code_lines} code-signal lines)")
    print(f"  Projects      : {len(manifest):,}  (one per vulnerability type)")
    print(f"  Role          : gate-study corpus for src/evaluation/")
    print(f"                 (classifier is trained/validated/evaluated on CVEfixes;")
    print(f"                  VUDENC is the held-out corpus for the gate study)")
    for entry in manifest:
        v = sum(s["label"] for s in entry["samples"])
        print(f"    · {entry['project']:<35} {len(entry['samples']):>5} samples  ({v} vuln)")
    print(f"  Output        : {output_path}")
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert VUDENC (DetectVul/Vudenc on HuggingFace, or a local "
            "GitHub clone) into the CEVuD benchmark manifest format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the output benchmark manifest JSON.",
    )
    p.add_argument(
        "--local-dir",
        dest="local_dir",
        default=None,
        help=(
            "Path to local VUDENC data directory (e.g. VulnerabilityDetection/data). "
            "If omitted, the script streams from HuggingFace."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to include. Default: all.",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="HuggingFace split to load. Default: train.",
    )
    p.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        default=True,
        help=(
            "Disable duplicate / contradiction filtering (keeps identical "
            "snippets and hard label contradictions)."
        ),
    )
    p.add_argument(
        "--min-code-lines",
        type=int,
        default=2,
        help=(
            "Drop a snippet with fewer than this many lines of real code "
            "signal. Default: 2."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_vudenc(
        output_path=args.output,
        local_dir=args.local_dir,
        limit=args.limit,
        split=args.split,
        dedup=args.dedup,
        min_code_lines=args.min_code_lines,
    )
