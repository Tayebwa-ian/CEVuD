"""
convert_vudenc.py
==================
Converts the VUDENC dataset (DetectVul/Vudenc on HuggingFace, or the
original LauraWartschinski/VulnerabilityDetection GitHub clone) into the
CEVuD benchmark manifest format.

ABOUT VUDENC
------------
VUDENC (Vulnerability Detection with Deep Learning on a Natural Codebase)
contains Python functions from real open-source projects, labeled at a
per-LINE level for seven vulnerability categories:
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
  1. Joins `raw_lines` back into a full function string (source_code).
  2. Derives a function-level label: 1 if ANY line is labeled 1, else 0.
  3. Derives the vulnerability type from the most-common type in `type`.
  4. Groups samples into logical "projects" by vulnerability category
     so that the DatasetSplitter can enforce project-level train/val/test
     splits (preventing data leakage across the boundary).

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

    return {
        "sample_id": sample_id,
        "file_path": "inline_snippet.py",
        "function_name": _infer_function_name(source_code),
        "start_line": 1,
        "end_line": max(len(raw_lines), 1),
        "label": function_label,
        "vulnerability_type": vuln_type,
        "source_code": source_code,
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
    """
    row_stream = (
        _stream_local(local_dir) if local_dir else _stream_hf(split=split)
    )

    # Group samples by vuln_type (used as "project" for split isolation)
    samples_by_project: Dict[str, List[Dict[str, Any]]] = {}
    total_added = total_skipped = 0

    for idx, row in enumerate(row_stream):
        sample = _row_to_sample(row, idx)
        if sample is None:
            total_skipped += 1
            continue

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
    print(f"  Samples skip  : {total_skipped:,}  (empty / malformed rows)")
    print(f"  Projects      : {len(manifest):,}  (one per vulnerability type)")
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
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_vudenc(
        output_path=args.output,
        local_dir=args.local_dir,
        limit=args.limit,
        split=args.split,
    )
