"""
convert_cvefixes.py
====================
Converts the CVEfixes dataset from Hugging Face (hitoshura25/cvefixes or
similar) into the CEVuD benchmark manifest format — completely free, no
database download required.

WHY HuggingFace instead of the raw SQLite?
-------------------------------------------
The full CVEfixes SQLite database is ~4 GB compressed. The HuggingFace
version is a pre-filtered, parquet-backed subset that streams row-by-row,
so you never hold the entire dataset in memory and never need to download
the raw database file. A Python-only filtered subset is ~300 MB total.

Dataset schema (parquet artifact of `hitoshura25/cvefixes`):
    - vulnerable_code     (str): The vulnerable code snippet
    - fixed_code          (str): The fixed code snippet (patched version)
    - repo_url            (str): GitHub repo URL (e.g., https://github.com/user/repo)
    - hash                (str): Git commit SHA where the fix was applied
                               (NOTE: the dataset stores the commit SHA here,
                               NOT in a field named `commit_id`)
    - cve_id              (str): CVE identifier (e.g., CVE-2021-1234)
    - cwe_id              (str): CWE identifier (e.g., CWE-79)
    - cvss2_base_score    (float): CVSS v2 score
    - cvss3_base_score    (float): CVSS v3 score
    - diff_with_context   (str): Unified diff with surrounding context
    - language            (str): Programming language (e.g., 'python')
    - commit_message      (str): Commit message describing the fix

Since the function bodies are pre-extracted, each sample embeds
`source_code` inline in the manifest rather than using a `git_source`
reference. The CEVuD evaluation engine handles this cleanly — when
`source_code` is present on a BenchmarkSample it is used directly,
skipping the git clone. The original `repo_url` and commit `hash` are
still captured per sample (as `commit_id`) so downstream stages can trace
each finding back to its exact upstream commit.

Usage:
    pip install datasets pyarrow

    # Preferred: load a local `save_to_disk` artifact (no network needed)
    python -c "from datasets import load_dataset; load_dataset('hitoshura25/cvefixes', split='train').save_to_disk('./cvefixes_dataset')"
    python src/scripts/convert_cvefixes.py \\
        --local-dir ./cvefixes_dataset \\
        --output benchmark_manifest_cvefixes.json \\
        --limit 5000

    # Or stream directly from the HuggingFace Hub (needs network):
    python src/scripts/convert_cvefixes.py \\
        --dataset hitoshura25/cvefixes \\
        --output benchmark_manifest_cvefixes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

# code_context lives in src/evaluation/ — it turns a CVEfixes diff into a
# real function anchor (start/end line) so the evaluation pipeline can
# clone the repo and read the complete vulnerable function + context.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
)
from code_context import parse_diff_anchors  # noqa: E402

# data_quality lives in src/ — shared noise/trivial-change filters that keep
# label noise (version bumps, docs, near-identical pre/post pairs) out of the
# manifest. Imported here so the converter can reject noisy rows at the source.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from data_quality import (  # noqa: E402
    _is_noise_file,
    code_signal_line_count,
    is_trivial_change,
    normalize_code,
)

# ---------------------------------------------------------------------------
# Robust import: give the user a clear error if `datasets` is not installed.
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, DownloadConfig, load_from_disk
except ImportError:
    print(
        "[ERROR] The `datasets` library is required to use this script.\n"
        "Install it with: pip install datasets\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Known HuggingFace dataset IDs that carry CVEfixes-style data.
# The first one that loads successfully is used.
# ---------------------------------------------------------------------------
FALLBACK_DATASET_IDS = [
    "hitoshura25/cvefixes",
    "dima806/fixedbugs",
]

# This map is no longer used since we're not using the safety field,
# but we keep it for backward compatibility in case of future changes.
# Maps the 'safety' field values found in HuggingFace CVEfixes datasets
# to binary labels (1 = vulnerable, 0 = safe).
SAFETY_LABEL_MAP: Dict[str, int] = {
    "vulnerable": 1,
    "unsafe":     1,
    "bad":        1,
    "safe":       0,
    "fixed":      0,
    "good":       0,
}


def _resolve_label(row: Dict[str, Any]) -> Optional[int]:
    """Extracts a binary label from a CVEfixes HuggingFace row.
    We use the presence of 'vulnerable_code' to determine vulnerability.
    """
    vulnerable_code = row.get("vulnerable_code")
    if vulnerable_code and isinstance(vulnerable_code, str) and vulnerable_code.strip():
        return 1  # Vulnerable
    return None  # Skip if no vulnerable_code


def _resolve_code(row: Dict[str, Any]) -> Optional[str]:
    """Extracts the vulnerable code snippet."""
    return row.get("vulnerable_code")


def _resolve_fixed_code(row: Dict[str, Any]) -> Optional[str]:
    """Extracts the fixed code snippet (for patch analysis)."""
    return row.get("fixed_code")


def _resolve_repo_url(row: Dict[str, Any]) -> str:
    """Extracts the repository URL for cloning."""
    return row.get("repo_url", "").strip() or "unknown_repo"


def _resolve_commit_id(row: Dict[str, Any]) -> str:
    """Extracts the commit SHA where the fix was applied.

    The CVEfixes parquet schema does NOT carry a field literally named
    `commit_id`; the fix-commit SHA is stored in the `hash` column. We try
    the likely names in order and fall back to `unknown_commit` only if none
    are present/non-empty.
    """
    for key in ("hash", "commit_id", "commit_hash", "commit_sha"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown_commit"


def _resolve_cve_id(row: Dict[str, Any]) -> str:
    """Extracts the CVE identifier."""
    return row.get("cve_id", "").strip() or "unknown_cve"


def _resolve_cvss_score(row: Dict[str, Any]) -> float:
    """Extracts CVSS score (prefer v3, fallback to v2)."""
    cvss3 = row.get("cvss3_base_score")
    cvss2 = row.get("cvss2_base_score")
    return float(cvss3) if cvss3 is not None else float(cvss2) if cvss2 is not None else 0.0


def _resolve_vuln_type(row: Dict[str, Any]) -> str:
    """Extracts the vulnerability category (CWE/type)."""
    return row.get("cwe_id", "unknown").strip()


def _resolve_diff_with_context(row: Dict[str, Any]) -> str:
    """Extracts the full diff with context (for patch analysis)."""
    return row.get("diff_with_context", "").strip() or ""


def _load_source(
    dataset_id: str,
    local_dir: Optional[str],
    split: str,
):
    """Resolves the CVEfixes rows to iterate over.

    Preference order:
      1. ``local_dir`` if it exists on disk (a ``save_to_disk`` artifact,
         e.g. ``./cvefixes_dataset``). This is the default and requires no
         network access.
      2. Otherwise stream from the HuggingFace Hub using ``dataset_id``.

    Returns an iterable of row dicts.
    """
    if local_dir and os.path.exists(local_dir):
        print(f"[*] Loading dataset from local disk: {local_dir}")
        ds = load_from_disk(local_dir)
        ds = ds.to_iterable_dataset()  # memory-safe streaming
        return ds

    if dataset_id:
        print(f"[*] Streaming '{dataset_id}' (split='{split}') from HuggingFace ...")
        print("[*] This streams row-by-row — no full download required.\n")
        return load_dataset(dataset_id, split=split, streaming=True)

    raise SystemExit(
        "[ERROR] No dataset available. Pass --local-dir pointing at a "
        "`save_to_disk` artifact, or --dataset with a HuggingFace ID and network access."
    )


def convert_cvefixes(
    dataset_id: str,
    output_path: str,
    limit: Optional[int] = None,
    split: str = "train",
    local_dir: Optional[str] = None,
    noise_filter: bool = True,
    trivial_filter: bool = True,
    min_code_lines: int = 2,
    dedup: bool = True,
) -> None:
    """Converts CVEfixes to a CEVuD benchmark manifest.

    Args:
        dataset_id: HuggingFace dataset repository ID (used only when no
            local dir is available).
        output_path: Path to write the resulting benchmark manifest JSON.
        limit: Maximum number of Python samples to include (None = all).
        split: HuggingFace split to load (usually 'train').
        local_dir: Path to a local ``save_to_disk`` artifact
            (e.g. ``./cvefixes_dataset``). Preferred over Hub streaming.
        noise_filter: When True (default), skip rows whose changed file is a
            docs/test/packaging/version file — these only yield version bumps
            and doc edits as (vuln, safe) pairs, which is pure label noise.
        trivial_filter: When True (default), skip (vuln, safe) pairs whose
            only difference is non-semantic (comments, docstrings, version
            assignments). Such pairs have no learnable vulnerability signal.
        min_code_lines: Drop a sample whose vulnerable snippet contains fewer
            than this many lines of real code signal (comments/docstrings/
            version assignments do not count). Prevents training on snippets
            that are, e.g., a single ``__version__ = '3.7'`` line.
        dedup: When True (default), drop a (vuln, safe) pair if its normalized
            text duplicates an already-emitted sample (redundancy) or collides
            with the OPPOSITE label (a hard contradiction the classifier cannot
            learn). Keeps the manifest compact and noise-free.
    """
    ds = _load_source(dataset_id, local_dir, split)

    samples_by_project: Dict[str, List[Dict[str, Any]]] = {}
    project_repo_urls: Dict[str, str] = {}
    total_seen = total_skipped = total_added = 0
    skipped_noise = skipped_trivial = skipped_short = skipped_no_safe = 0
    skipped_dup = skipped_contradiction = 0
    # normalized source_code -> label, to drop duplicate / contradictory pairs
    # across the whole manifest (redundancy + hard-label-noise control).
    seen_norm: Dict[str, int] = {}

    # Stream row-by-row — no memory explosion
    for row in ds:
        total_seen += 1

        # ── Filter to Python only ────────────────────────────────────────────
        lang = str(row.get("language") or row.get("lang") or "").strip().lower()
        if lang and lang not in ("python", "py"):
            total_skipped += 1
            continue

        # ── Extract mandatory fields ─────────────────────────────────────────
        code = _resolve_code(row)
        label = _resolve_label(row)

        if code is None or label is None:
            total_skipped += 1
            continue

        code = code.strip()
        if not code:
            total_skipped += 1
            continue

        # ── Resolve the real Python file + the vulnerable function anchor ──
        repo_url = _resolve_repo_url(row)
        if not repo_url or repo_url == "unknown_repo":
            total_skipped += 1
            continue

        # ── Resolve the changed Python file ──────────────────────────────────
        # Prefer the path parsed from the unified diff (authoritative — every
        # CVEfixes row carries a diff), falling back to the dataset's
        # ``file_paths`` column (some exports omit it). Without a resolvable
        # .py file there is nothing to clone, so skip.
        diff = _resolve_diff_with_context(row)
        anchors = parse_diff_anchors(diff) if diff else {}
        file_paths = row.get("file_paths") or []
        py_candidates = [f for f in anchors if str(f).endswith(".py")]
        py_candidates += [str(f) for f in file_paths if str(f).endswith(".py")]
        if not py_candidates:
            total_skipped += 1
            continue
        file_path = py_candidates[0]

        # ── Noise-file filter ────────────────────────────────────────────────
        # Docs/tests/packaging/version files produce pairs whose only diff is a
        # version string or doc edit — unlearnable, and they dominate some
        # repos (e.g. idna's package_data.py). Skip them up front.
        if noise_filter and _is_noise_file(file_path):
            skipped_noise += 1
            continue

        anchor = anchors.get(file_path)
        if anchor is None:
            # Could not locate the changed function in the diff; skip.
            total_skipped += 1
            continue

        # ── Minimum code-signal filter (vulnerable side) ─────────────────────
        # A snippet with no real code (e.g. a lone ``__version__ = '3.6'``)
        # carries no learnable signal and would just add label noise.
        if code_signal_line_count(code) < min_code_lines:
            skipped_short += 1
            continue

        project_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        commit_id = _resolve_commit_id(row)
        cve_id = _resolve_cve_id(row)
        cvss_score = _resolve_cvss_score(row)
        vuln_type = _resolve_vuln_type(row)
        fixed_code = _resolve_fixed_code(row)

        def _make_sample(s_code: str, s_label: int, s_start: int, s_end: int,
                          s_target_commit: str, s_function_name: str,
                          s_subtype: str) -> Dict[str, Any]:
            """Builds a single BenchmarkSample dict for one (pre- or post-fix)
            version of the changed function.

            ``s_subtype`` disambiguates *why* the sample carries its label
            for the safe-counterpart audit (see docs/SAFE_COUNTERPARTS.md):
              - "vulnerable" : the pre-fix (label=1) half of the pair.
              - "fixed"       : the post-fix (label=0) half — the "safe
                              counterpart" we are auditing for bundled edits.
            """
            return {
                # Unique id per emitted version so vulnerable + safe never
                # collide inside the same project.
                "sample_id": f"cvefixes::{project_name}::{uuid.uuid4().hex[:8]}",
                "file_path": file_path,            # real repo-relative path
                "function_name": s_function_name,
                "start_line": s_start,
                "end_line": s_end,
                "label": s_label,
                "vulnerability_type": vuln_type,
                "sample_subtype": s_subtype,
                "source_code": s_code,             # fallback if clone / git-show fails
                "fixed_code": fixed_code,
                "repo_url": repo_url,
                "commit_id": commit_id,           # fix-commit SHA
                "target_commit": s_target_commit, # exact commit to check out for this version
                "cve_id": cve_id,
                "cvss_score": cvss_score,
                "diff_with_context": diff,
            }

        if project_name not in samples_by_project:
            samples_by_project[project_name] = []
            project_repo_urls[project_name] = repo_url

        # ── SAFE sample (post-fix version) ───────────────────────────────────
        # Mirroring the pre-fix version with the POST-image anchor gives us a
        # label=0 companion for every vulnerable row — a naturally *balanced*
        # 1:1 dataset. We explicitly pin target_commit to the fix commit so the
        # extractor reads the CORRECT (patched) file for this sample.
        # ── Resolve the SAFE (post-fix) companion ────────────────────────────
        # The balanced 1:1 design needs BOTH a vulnerable and a safe sample. A
        # row whose fix left no post-fix code cannot form a pair, so skip it
        # rather than emitting an orphan vulnerable sample that skews the
        # class balance.
        safe_code = fixed_code.strip() if fixed_code and fixed_code.strip() else None
        if safe_code is None:
            skipped_no_safe += 1
            continue

        # ── Trivial / short / redundant / contradictory-pair filters ──────────
        # If the vulnerable and fixed snippets differ ONLY in non-semantic ways
        # (comments, docstrings, version bumps) the (vuln, safe) pair is pure
        # label noise. Drop the whole pair. Also drop when the safe side has no
        # learnable code signal, or when the pair duplicates / contradicts an
        # already-emitted sample (e.g. the same function fixed in several
        # commits, or identical text with both labels).
        if trivial_filter and is_trivial_change(code, safe_code):
            skipped_trivial += 1
            continue
        if code_signal_line_count(safe_code) < min_code_lines:
            skipped_short += 1
            continue
        if dedup:
            norm_v = normalize_code(code)
            norm_s = normalize_code(safe_code)
            drop = False
            for nrm, lab in ((norm_v, 1), (norm_s, 0)):
                if nrm in seen_norm:
                    if seen_norm[nrm] == lab:
                        skipped_dup += 1
                    else:
                        skipped_contradiction += 1
                    drop = True
                    break
            if drop:
                continue
            seen_norm[norm_v] = 1
            seen_norm[norm_s] = 0

        # ── VULNERABLE sample (pre-fix version) ───────────────────────────────
        # The pre-image anchor locates the vulnerable function in the PARENT of
        # the fix commit. We leave target_commit unset so the extractor checks
        # out the parent of `commit_id` (the historical default).
        vuln_start = int(anchor["pre_start"])
        vuln_end = int(anchor["pre_end"])
        vuln_func = _infer_function_name(code) or "unknown_function"
        samples_by_project[project_name].append(
            _make_sample(code, 1, vuln_start, vuln_end, None, vuln_func, "vulnerable")
        )
        total_added += 1

        if safe_code:
            safe_start = int(anchor["post_start"])
            safe_end = int(anchor["post_end"])
            safe_func = _infer_function_name(safe_code) or "unknown_function"
            samples_by_project[project_name].append(
                _make_sample(safe_code, 0, safe_start, safe_end, commit_id, safe_func, "fixed")
            )
            total_added += 1

        if limit is not None and total_added >= limit:
            print(f"[*] Reached --limit {limit}. Stopping early.")
            break

        if total_added % 100 == 0:
            print(f"[*] Processed {total_added} samples...")

    # ── Assemble the manifest ────────────────────────────────────────────────
    # One git_source PER REPO. The evaluation suite clones each repo once
    # (full/partial history so `git show <commit>` is reachable) and then,
    # per sample, checks out the *parent* of the fix commit to read the
    # REAL vulnerable function plus its imports and cross-file context.
    # Drop projects whose every sample was filtered out (noise / trivial /
    # no-safe / dedup) so we never emit an empty `samples` list — an empty
    # project fails `benchmark_manifest.load_manifest` validation.
    manifest = [
        {
            "project": project_name,
            "git_source": {
                "git_url": project_repo_urls[project_name],
                "ref": None,          # clone default branch; per-sample commit drives git-show
                "shallow": False,  # need history to reach arbitrary fix commits
            },
            "samples": samples,
        }
        for project_name, samples in samples_by_project.items()
        if samples
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"  Dataset       : {dataset_id}")
    print(f"  Rows scanned  : {total_seen:,}")
    print(f"  Rows skipped  : {total_skipped:,}  (non-Python / missing fields)")
    print(f"  Skipped noise : {skipped_noise:,}  (docs/tests/packaging/version files)")
    print(f"  Skipped short : {skipped_short:,}  (< {min_code_lines} code-signal lines)")
    print(f"  Skipped trivial: {skipped_trivial:,}  (vuln==safe up to comments/version)")
    print(f"  Skipped no-safe: {skipped_no_safe:,}  (no post-fix code -> no balanced pair)")
    print(f"  Skipped dup   : {skipped_dup:,}  (duplicate pairs)")
    print(f"  Skipped contra: {skipped_contradiction:,}  (identical text, both labels)")
    print(f"  Samples added : {total_added:,}")
    print(f"  Projects      : {len(manifest):,}")
    print(f"  Output        : {output_path}")
    print(f"{'─'*60}\n")


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
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert CVEfixes (via HuggingFace streaming API) into the "
            "CEVuD benchmark manifest format. No database download required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        default=FALLBACK_DATASET_IDS[0],
        help=(
            f"HuggingFace dataset ID to stream when no --local-dir is given. "
            f"Default: {FALLBACK_DATASET_IDS[0]}. Alternative: dima806/fixedbugs"
        ),
    )
    p.add_argument(
        "--local-dir",
        dest="local_dir",
        default="./cvefixes_dataset",
        help=(
            "Path to a local `save_to_disk` artifact (e.g. ./cvefixes_dataset). "
            "Preferred over Hub streaming; used automatically when it exists."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the benchmark manifest JSON.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Python samples to include. Default: all.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="HuggingFace dataset split to load. Default: train.",
    )
    p.add_argument(
        "--no-noise-filter",
        dest="noise_filter",
        action="store_false",
        default=True,
        help=(
            "Disable the docs/tests/packaging/version-file filter (keeps "
            "noisy rows such as version bumps in package_data.py)."
        ),
    )
    p.add_argument(
        "--no-trivial-filter",
        dest="trivial_filter",
        action="store_false",
        default=True,
        help=(
            "Disable the trivial-change filter (keeps (vuln, safe) pairs that "
            "differ only in comments/docstrings/version assignments)."
        ),
    )
    p.add_argument(
        "--min-code-lines",
        type=int,
        default=2,
        help=(
            "Drop a sample whose vulnerable/fixed snippet has fewer than this "
            "many lines of real code signal. Default: 2."
        ),
    )
    p.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        default=True,
        help=(
            "Disable duplicate / contradiction filtering (keeps identical "
            "pairs and hard label contradictions)."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_cvefixes(
        dataset_id=args.dataset,
        output_path=args.output,
        limit=args.limit,
        split=args.split,
        local_dir=args.local_dir,
        noise_filter=args.noise_filter,
        trivial_filter=args.trivial_filter,
        min_code_lines=args.min_code_lines,
    )
