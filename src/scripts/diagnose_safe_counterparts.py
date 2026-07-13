"""
diagnose_safe_counterparts.py
============================
Measures how trustworthy the CVEfixes "safe counterpart" actually is.

WHY THIS EXISTS
---------------
The CEVuD training set is built from CVEfixes as a *balanced 1:1 pair*:
the pre-fix function is labeled vulnerable (label=1, ``sample_subtype=
"vulnerable"``) and the post-fix function is labeled safe (label=0,
``sample_subtype="fixed"``). The implicit assumption is that the post-fix
function differs from its vulnerable twin *only* by the security patch.

That assumption does not always hold. A fix commit can bundle unrelated
edits — a refactor, a reformat, a feature added to the same function.
When it does, the ``label=0`` sample differs from its twin in ways that
have nothing to do with the vulnerability, so the classifier can learn the
*refactor* instead of the *vulnerability* (and, on a held-out corpus such
as VUDENC, fail to generalize). The post-fix function is also only a
*relative* negative: it is guaranteed not to carry *that one* CVE, but it
may still contain a *different* weakness.

This script quantifies the contamination **without cloning any repository**.
The CVEfixes manifest already embeds both halves inline
(``vulnerable.source_code`` and ``fixed_code`` / ``safe.source_code``), so we
can pair them up and measure, per pair:

  * ``changed_line_ratio`` — what share of the (combined) function lines
    actually changed between the vulnerable and post-fix versions.
  * ``is_trivial_change`` — the only differences are non-semantic
    (comments / docstrings / version strings). These are *clean* minimal
    fixes (the opposite problem from bundled edits).
  * ``is_substantial_change`` — the function was reworked across >50% of its
    lines. This is the *bundled-edit* red flag.

It also reports, for the fix-commit *breadth* (how many files a fix commit
touched), via an optional ``--clone-subset N`` mode that clones the first N
projects and runs ``git show --stat`` on each unique fix commit. That mode
needs network + git and is OFF by default.

OUTPUT
------
  * A human-readable summary printed to stdout.
  * A machine-readable report written to
    ``workspace_storage/evaluation_runs/safe_counterpart_diagnosis.json``:
    per-pair records plus aggregate buckets/histograms. This is the artifact
    to cite when writing up the safe-counterpart methodology.

USAGE
------
    # Inline-only analysis (no network) — run this first:
    python src/scripts/diagnose_safe_counterparts.py \
        --manifest benchmark_manifest_cvefixes.json

    # Also measure fix-commit breadth (clones the first 20 projects):
    python src/scripts/diagnose_safe_counterparts.py \
        --manifest benchmark_manifest_cvefixes.json --clone-subset 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
)

from data_quality import (  # noqa: E402
    changed_line_ratio,
    code_signal_line_count,
    is_substantial_change,
    is_trivial_change,
    normalize_code,
)
from benchmark_manifest import load_manifest  # noqa: E402
from repo_provider import clone_repo  # noqa: E402
from run_context import get_eval_dir  # noqa: E402


# Buckets for the changed_line_ratio histogram. Each bucket is [lo, hi).
_RATIO_BUCKETS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def _bucket(ratio: float) -> str:
    for lo, hi in _RATIO_BUCKETS:
        if lo <= ratio < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "[0.75,1.01)"


def _pair_metrics(vuln_code: str, fixed_code: str) -> Dict[str, Any]:
    """Compute the per-pair contamination metrics."""
    return {
        "changed_line_ratio": round(changed_line_ratio(vuln_code, fixed_code), 4),
        "is_trivial": bool(is_trivial_change(vuln_code, fixed_code)),
        "is_substantial": bool(is_substantial_change(vuln_code, fixed_code)),
        "vuln_code_lines": len([l for l in vuln_code.splitlines() if l.strip()]),
        "fixed_code_lines": len([l for l in fixed_code.splitlines() if l.strip()]),
        "vuln_signal_lines": code_signal_line_count(vuln_code),
        "fixed_signal_lines": code_signal_line_count(fixed_code),
    }


def _match_pairs(projects) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pairs each vulnerable sample with its post-fix twin (matched by the
    normalized fixed function text) and returns (records, counters)."""
    records: List[Dict[str, Any]] = []
    counters = Counter()
    for proj in projects:
        # Index the project's safe (post-fix) samples by normalized text.
        safe_by_norm: Dict[str, Any] = {}
        for s in proj.samples:
            if s.label == 0 and s.source_code:
                safe_by_norm[normalize_code(s.source_code)] = s

        for s in proj.samples:
            if s.label != 1 or not s.fixed_code:
                continue
            counters["vulnerable_total"] += 1
            twin = safe_by_norm.get(normalize_code(s.fixed_code))
            if twin is None:
                counters["unmatched_vulnerable"] += 1
                continue
            counters["matched_pairs"] += 1
            m = _pair_metrics(s.source_code or "", s.fixed_code)
            records.append(
                {
                    "project": proj.project,
                    "sample_id": s.sample_id,
                    "cve_id": s.cve_id,
                    "file_path": s.file_path,
                    "vulnerability_type": s.vulnerability_type,
                    **m,
                }
            )
    return records, dict(counters)


def _clone_commit_breadth(projects, subset: int) -> Dict[str, Any]:
    """OPTIONAL, NETWORK-HEAVY. For the first ``subset`` git-source projects,
    clone each repo and, for every unique fix commit, count how many files the
    commit touched (``git show --stat``). Returns aggregate breadth stats.

    A fix commit that touches many files is more likely to bundle unrelated
    edits — independent evidence that corroborates (or refutes) the inline
    changed_line_ratio signal.
    """
    stats: Dict[str, Any] = {
        "projects_scanned": 0,
        "fix_commits_seen": 0,
        "files_per_commit": [],
        "commits_touching_multiple_files": 0,
    }
    scanned = 0
    for proj in projects:
        if scanned >= subset:
            break
        if proj.git_source is None:
            continue
        dest = None
        try:
            dest = clone_repo(
                proj.git_source.git_url, ref=None, dest_dir=None, shallow=False
            )
        except Exception as exc:
            print(f"[!] clone failed for {proj.project}: {exc}")
            continue
        try:
            commits = sorted(
                {s.commit_id for s in proj.samples if s.commit_id and s.commit_id != "unknown_commit"}
            )
            for commit in commits:
                stats["fix_commits_seen"] += 1
                try:
                    r = subprocess.run(
                        ["git", "-C", dest, "show", "--stat", "--format=", commit],
                        capture_output=True, text=True, timeout=120,
                    )
                    n_files = sum(
                        1 for ln in r.stdout.splitlines()
                        if ln.strip().endswith((".py",)) and ("|" in ln or "\t" in ln)
                    )
                    stats["files_per_commit"].append(n_files)
                    if n_files > 1:
                        stats["commits_touching_multiple_files"] += 1
                except Exception:
                    pass
            stats["projects_scanned"] += 1
            scanned += 1
        finally:
            import shutil
            if dest and os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)
    if stats["files_per_commit"]:
        fps = stats["files_per_commit"]
        stats["files_per_commit_mean"] = round(sum(fps) / len(fps), 3)
        stats["files_per_commit_max"] = max(fps)
    return stats


def diagnose(manifest_path: str, clone_subset: Optional[int] = None) -> Dict[str, Any]:
    projects = load_manifest(manifest_path)
    records, counters = _match_pairs(projects)

    # Aggregate the ratio histogram + trivial/substantial tallies.
    hist: Dict[str, int] = {b: 0 for b in (_bucket(lo) for lo, _ in _RATIO_BUCKETS)}
    n_trivial = n_substantial = 0
    ratios = []
    for rec in records:
        hist[_bucket(rec["changed_line_ratio"])] += 1
        ratios.append(rec["changed_line_ratio"])
        if rec["is_trivial"]:
            n_trivial += 1
        if rec["is_substantial"]:
            n_substantial += 1

    safe_total = sum(1 for p in projects for s in p.samples if s.label == 0)
    report: Dict[str, Any] = {
        "manifest": manifest_path,
        "counters": {
            "vulnerable_total": counters.get("vulnerable_total", 0),
            "matched_pairs": counters.get("matched_pairs", 0),
            "unmatched_vulnerable": counters.get("unmatched_vulnerable", 0),
            "safe_total": safe_total,
        },
        "pair_quality": {
            "trivial_pairs": n_trivial,
            "trivial_pair_fraction": round(n_trivial / max(1, len(records)), 4),
            "substantial_pairs": n_substantial,
            "substantial_pair_fraction": round(n_substantial / max(1, len(records)), 4),
            "mean_changed_line_ratio": round(sum(ratios) / max(1, len(ratios)), 4),
            "median_changed_line_ratio": round(
                sorted(ratios)[len(ratios) // 2] if ratios else 0.0, 4
            ),
        },
        "changed_line_ratio_histogram": hist,
        "pairs": records,
    }

    if clone_subset:
        report["fix_commit_breadth"] = _clone_commit_breadth(projects, clone_subset)

    return report


def _print_summary(report: Dict[str, Any]) -> None:
    c = report["counters"]
    q = report["pair_quality"]
    print(f"\n{'='*64}")
    print(f"  Safe-counterpart diagnosis: {report['manifest']}")
    print(f"{'='*64}")
    print(f"  Vulnerable samples (label=1) : {c['vulnerable_total']:,}")
    print(f"  Safe samples (label=0)        : {c['safe_total']:,}")
    print(f"  Matched (vuln -> fixed) pairs : {c['matched_pairs']:,}")
    print(f"  Unmatched vulnerable samples   : {c['unmatched_vulnerable']:,}")
    print(f"  ---")
    print(f"  TRIVIAL pairs (clean minimal fix, no learnable diff): "
          f"{q['trivial_pairs']:,} ({q['trivial_pair_fraction']:.1%})")
    print(f"  SUBSTANTIAL pairs (>50% of lines reworked; bundled-edit "
          f"red flag): {q['substantial_pairs']:,} "
          f"({q['substantial_pair_fraction']:.1%})")
    print(f"  Mean changed-line ratio  : {q['mean_changed_line_ratio']}")
    print(f"  Median changed-line ratio: {q['median_changed_line_ratio']}")
    print(f"  --- changed_line_ratio histogram ---")
    for bucket, n in report["changed_line_ratio_histogram"].items():
        print(f"    {bucket:<14}: {n:,}")
    if "fix_commit_breadth" in report:
        b = report["fix_commit_breadth"]
        print(f"  --- fix-commit breadth (cloned {b['projects_scanned']} repos) ---")
        print(f"    fix commits scanned         : {b['fix_commits_seen']:,}")
        print(f"    multi-file fix commits      : {b['commits_touching_multiple_files']:,}")
        if "files_per_commit_mean" in b:
            print(f"    mean files / fix commit    : {b['files_per_commit_mean']}")
    print(f"{'='*64}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantify how trustworthy the CVEfixes post-fix 'safe' "
                    "counterpart is (bundled-edit contamination)."
    )
    p.add_argument("--manifest", required=True, help="Path to benchmark_manifest_cvefixes.json")
    p.add_argument("--output", default=None,
                    help="Where to write the JSON report (default: "
                         "<config paths.evaluations_subdir>/safe_counterpart_diagnosis.json "
                         "if --config is supplied, otherwise "
                         "workspace_storage/evaluation_runs/safe_counterpart_diagnosis.json)")
    p.add_argument("--config", default=None,
                    help="Path to config.json. When supplied, the default --output "
                         "is resolved via run_context.get_eval_dir so it lands in "
                         "the canonical evaluation subtree.")
    p.add_argument("--clone-subset", type=int, default=None,
                    help="Also measure fix-commit breadth by cloning the first N "
                         "git-source projects (network-heavy; off by default).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    report = diagnose(args.manifest, clone_subset=args.clone_subset)
    _print_summary(report)

    out = args.output
    if not out and args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out = os.path.join(get_eval_dir(".", cfg, "eval_diagnose"), "safe_counterpart_diagnosis.json")
        except Exception:
            out = None
    if not out:
        out = os.path.join(
            "workspace_storage", "evaluation_runs", "safe_counterpart_diagnosis.json"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Diagnosis report -> {out}")
