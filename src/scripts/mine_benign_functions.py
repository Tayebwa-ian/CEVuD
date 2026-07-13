"""
mine_benign_functions.py
============================
Mines *verified-benign* control functions from the same repositories that
CVEfixes draws its vulnerabilities from — so the training set has true
negatives, not just "the function after a fix commit".

THE PROBLEM THIS SOLVES
-------------------------
CVEfixes gives us, per fix, a (vulnerable, post-fix) function pair. The
post-fix function is used as the ``label=0`` ("safe") half. But (see
``docs/SAFE_COUNTERPARTS.md``):

  * the post-fix function is only a *relative* negative (it lacks *that* CVE,
    but may still contain a *different* weakness), and
  * the fix commit can bundle unrelated edits, so the post-fix function may
    differ from its twin for non-security reasons.

The classifier therefore never sees what *genuinely clean* Python code looks
like — it only sees "pre-fix vs post-fix". This hurts generalization to a
held-out corpus such as VUDENC.

THE FIX
-------
For every fix commit, we checkout that commit in the real repo and take
functions from the files the commit did **NOT** modify. Functions in
unmodified files are, by construction, untouched by any vulnerability fix, so
they are strong benign controls (``sample_subtype="benign_control"``,
``label=0``). We still apply the usual code-signal + dedup filters, and we
document the residual limitation (an unmodified function can itself contain a
*latent* vulnerability — a benign control is "not-known-bad", not provably
safe).

OUTPUT
------
A manifest in the SAME schema as ``benchmark_manifest_cvefixes.json``:
one ``git_source`` project per upstream repository, each carrying the mined
benign functions. Drop it into ``convert_vudenc.py --benign-manifest`` to
give the VUDENC gate study real safe samples, or into
``build-dataset --benign-manifest`` to enrich the training set.

USAGE
------
    python src/scripts/mine_benign_functions.py \
        --manifest benchmark_manifest_cvefixes.json \
        --output benign_controls_manifest.json \
        --samples-per-commit 3 --max-workers 4
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
)

from benchmark_manifest import load_manifest  # noqa: E402
from repo_provider import clone_repo  # noqa: E402
from code_context import collect_module_imports, build_context_snippet  # noqa: E402
from data_quality import code_signal_line_count, normalize_code  # noqa: E402


_GIT_LOCK = threading.Lock()


def _git(repo: str, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _modified_py_files(repo: str, commit: str) -> Set[str]:
    """Files (ending in .py) touched by ``commit``."""
    r = _git(repo, "show", "--name-only", "--format=", commit)
    out = (r.stdout or "").splitlines()
    return {ln.strip() for ln in out if ln.strip().endswith(".py")}


def _checkout(repo: str, commit: str) -> bool:
    """Checkout ``commit``; fetch full history first if needed. Returns success."""
    r = _git(repo, "checkout", "--quiet", commit)
    if r.returncode == 0:
        return True
    # History may be partial; unshallow then retry once.
    _git(repo, "fetch", "--unshallow")
    r = _git(repo, "checkout", "--quiet", commit)
    return r.returncode == 0


def _collect_benign_for_commit(
    repo: str,
    commit: str,
    samples_per_commit: int,
    min_code_lines: int,
    rng: random.Random,
    seen_norm: Set[str],
) -> List[Dict[str, Any]]:
    """Return up to ``samples_per_commit`` benign functions from files the
    commit did NOT modify."""
    if not _checkout(repo, commit):
        return []

    modified = _modified_py_files(repo, commit)
    ls = _git(repo, "ls-files", "*.py")
    if ls.returncode != 0:
        return []
    candidates: List[str] = [
        ln.strip() for ln in ls.stdout.splitlines() if ln.strip() and ln.strip() not in modified
    ]

    rng.shuffle(candidates)
    collected: List[Dict[str, Any]] = []
    for rel in candidates:
        if len(collected) >= samples_per_commit:
            break
        try:
            with open(os.path.join(repo, rel), "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            continue
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            continue
        imports = collect_module_imports(content)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            func_src = "\n".join(content.splitlines()[start - 1 : end])
            if code_signal_line_count(func_src) < min_code_lines:
                continue
            norm = normalize_code(func_src)
            if norm in seen_norm:
                continue  # dedup across commits / projects
            seen_norm.add(norm)
            snippet = build_context_snippet(
                content, (start, end), imports, {}
            )
            collected.append(
                {
                    "sample_id": f"benign::{uuid.uuid4().hex[:10]}",
                    "file_path": rel,
                    "function_name": node.name,
                    "start_line": start,
                    "end_line": end,
                    "label": 0,
                    "vulnerability_type": "benign",
                    "sample_subtype": "benign_control",
                    "source_code": snippet,
                    "fixed_code": None,
                    "repo_url": None,  # filled by caller (project-level)
                    "commit_id": commit,
                    "target_commit": commit,
                    "cve_id": None,
                    "cvss_score": 0.0,
                    "diff_with_context": "",
                }
            )
    return collected


def _process_project(
    project,
    samples_per_commit: int,
    max_commits: int,
    max_per_project: Optional[int],
    min_code_lines: int,
    rng: random.Random,
    seen_norm: Set[str],
) -> List[Dict[str, Any]]:
    if project.git_source is None:
        return []  # benign mining requires a real repo
    commits = sorted(
        {s.commit_id for s in project.samples if s.commit_id and s.commit_id != "unknown_commit"}
    )
    if not commits:
        return []
    commits = commits[:max_commits]

    dest = None
    out: List[Dict[str, Any]] = []
    try:
        dest = clone_repo(
            project.git_source.git_url, ref=None, dest_dir=None, shallow=False
        )
        for commit in commits:
            if max_per_project and len(out) >= max_per_project:
                break
            found = _collect_benign_for_commit(
                dest, commit, samples_per_commit, min_code_lines, rng, seen_norm
            )
            out.extend(found)
            if max_per_project:
                out = out[:max_per_project]
    except Exception as exc:
        print(f"[!] project {project.project}: {exc}")
    finally:
        if dest and os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
    # Stamp provenance at project level.
    for s in out:
        s["repo_url"] = project.git_source.git_url
    return out


def mine_benign(
    manifest_path: str,
    output_path: str,
    samples_per_commit: int = 3,
    max_commits: int = 50,
    max_per_project: Optional[int] = None,
    max_total: Optional[int] = None,
    max_workers: int = 4,
    min_code_lines: int = 2,
    seed: int = 42,
) -> None:
    projects = load_manifest(manifest_path)
    rng = random.Random(seed)
    seen_norm: Set[str] = set()
    benign_by_project: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_project, p, samples_per_commit, max_commits,
                max_per_project, min_code_lines, rng, seen_norm,
            ): p
            for p in projects
        }
        for fut in as_completed(futures):
            proj = futures[fut]
            samples = fut.result()
            if samples:
                benign_by_project[proj.project] = samples

    # Global cap.
    if max_total is not None:
        all_samples: List[Dict[str, Any]] = []
        for s in benign_by_project.values():
            all_samples.extend(s)
        rng.shuffle(all_samples)
        all_samples = all_samples[:max_total]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for s in all_samples:
            grouped.setdefault(s["repo_url"] or "unknown", []).append(s)
        benign_by_project = grouped

    manifest = [
        {
            "project": f"benign::{name}",
            "git_source": {
                "git_url": samples[0]["repo_url"],
                "ref": None,
                "shallow": False,
            },
            "samples": samples,
        }
        for name, samples in sorted(benign_by_project.items())
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(len(p["samples"]) for p in manifest)
    print(f"\n{'─'*60}")
    print(f"  Source          : {manifest_path}")
    print(f"  Benign projects : {len(manifest):,}")
    print(f"  Benign samples  : {total:,}  (sample_subtype=benign_control, label=0)")
    print(f"  Output          : {output_path}")
    print(f"{'─'*60}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mine verified-benign control functions from CVEfixes repos."
    )
    p.add_argument("--manifest", required=True, help="benchmark_manifest_cvefixes.json")
    p.add_argument("--output", default="benign_controls_manifest.json",
                    help="Output manifest path.")
    p.add_argument("--samples-per-commit", type=int, default=3,
                    help="Max benign functions mined per fix commit (default 3).")
    p.add_argument("--max-commits", type=int, default=50,
                    help="Max fix commits processed per repo (default 50).")
    p.add_argument("--max-per-project", type=int, default=None,
                    help="Max benign samples per repo (default: unlimited).")
    p.add_argument("--max-total", type=int, default=None,
                    help="Hard cap on total benign samples.")
    p.add_argument("--max-workers", type=int, default=4, help="Parallel clones.")
    p.add_argument("--min-code-lines", type=int, default=2,
                    help="Drop functions with fewer code-signal lines.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p


if __name__ == "__main__":
    a = _build_parser().parse_args()
    mine_benign(
        a.manifest, a.output, a.samples_per_commit, a.max_commits,
        a.max_per_project, a.max_total, a.max_workers, a.min_code_lines, a.seed,
    )
