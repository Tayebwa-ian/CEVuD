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
For every fix commit, we checkout that commit in the real repo and mine
functions to use as ``label=0`` safe controls, with two refinements that make
them *sharp* negatives (docs/SAFE_COUNTERPARTS.md):

  1. **Same-file siblings first.** We prefer functions that live in the *same
     files the fix touched* (minus the patched function itself). These siblings
     share the vulnerable function's imports, APIs, and coding style, so they
     teach the model the *sharpest* "safe vs vulnerable" boundary rather than a
     trivially-different negative. Functions from untouched files are mined next.
  2. **Near-duplicate guard.** Every candidate is compared (token similarity)
     against *all* vulnerable snippets in the same project; any candidate more
     than ``--similarity-threshold`` similar (default 0.75) is dropped. This
     automatically excludes the patched function's post-fix twin (the very
     near-duplicate that used to collapse training to P=0.5) and any other
     lightly-edited copy, so the safe class can never re-introduce the
     contradiction.

By construction a sibling / untouched-file function is not the fix under study,
so it is a strong benign control (``sample_subtype="benign_control"``,
``label=0``). We still apply the usual code-signal + dedup filters, and we
document the residual limitation (an unmodified function can itself contain a
*latent* vulnerability — a benign control is "not-known-bad", not provably
safe). The volume mined per project is bounded by ``--ratio`` (default 5×) times
the number of vulnerable samples in that project, so the final safe:vuln balance
is a moderate, learnable ratio rather than an unbounded flood.

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

 PERFORMANCE
 -----------
 Cloning is the dominant cost. The miner therefore does NOT check out a
 working tree: it performs a *targeted* clone that fetches ONLY the fix
 commits (no default-branch history, no working tree) and then reads file
 contents directly via ``git show <commit>:<path>`` (trees/blobs pulled on
 demand). This is ~6x lighter per repo than a full working-tree clone and
 avoids rewriting the tree for every fix commit. Crank ``--max-workers`` on a
 powerful host (e.g. 32-64) to parallelise across repos.
 """

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
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
from data_quality import (  # noqa: E402
    code_signal_line_count,
    normalize_code,
    tokenize_code,
    max_token_similarity,
)


_GIT_LOCK = threading.Lock()
# Guards the shared ``seen_norm`` dedup set across the worker threads
# (a plain ``set`` check-then-add is a TOCTOU race without it).
_STATE_LOCK = threading.Lock()


def _commit_seed(commit: str, seed: int) -> int:
    """Stable, per-commit RNG seed (deterministic across runs, unlike
    the salted built-in ``hash``). Avoids sharing one ``random.Random``
    across threads, which is not thread-safe."""
    digest = hashlib.md5(commit.encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) & 0x7FFFfffF


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


def _fetch_commit(repo: str, commit: str) -> None:
    """Make ``commit`` resolvable locally. A bare partial clone only carries
    the default-branch history; fix commits on other branches are fetched by
    SHA on demand (GitHub allows fetching arbitrary commits by SHA).
    """
    has = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True, timeout=120,
    )
    if has.returncode != 0:
        subprocess.run(
            ["git", "-C", repo, "fetch", "--quiet", "origin", commit],
            capture_output=True, timeout=120,
        )


def _read_file_at(repo: str, commit: str, path: str) -> Optional[str]:
    """Read ``path`` at ``commit`` WITHOUT checking out a working tree.

    Uses ``git show <commit>:<path>`` so the clone stays bare (no working
    tree to rewrite per commit, no per-commit blob re-fetch storm). Trees and
    blobs are fetched on demand by the promisor remote. If the commit object
    is not yet locally present (it is an ancestor of a non-default branch), we
    fetch it by SHA once and retry — GitHub permits fetching arbitrary commits
    by SHA on public repos.
    """
    r = subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{path}"],
        capture_output=True, timeout=120,
    )
    if r.returncode == 0:
        return r.stdout.decode("utf-8", "ignore")
    # Commit possibly not on the default branch history; fetch by SHA.
    subprocess.run(
        ["git", "-C", repo, "fetch", "--quiet", "origin", commit],
        capture_output=True, timeout=120,
    )
    r = subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{path}"],
        capture_output=True, timeout=120,
    )
    if r.returncode == 0:
        return r.stdout.decode("utf-8", "ignore")
    return None


def _clone_only_commits(url: str, commits: List[str], dest_dir: Optional[str] = None) -> str:
    """Fastest clone for benign mining: fetch ONLY the fix ``commits`` (no
    default-branch history, no working tree).

    A fresh ``git init`` + ``git fetch origin --filter=tree:0 <sha>...`` pulls
    exactly the requested commits and their on-demand trees/blobs. It does NOT
    walk or download the repository's full commit graph, which is what made the
    old ``git clone`` slow on history-heavy repos. ``_read_file_at`` then reads
    files via ``git show <commit>:<path>``.

    Falls back to a safe bare partial clone if direct SHA fetch is refused
    (e.g. a server that disallows fetching by SHA).
    """
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="cevud_repo_")
    else:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)
    subprocess.run(["git", "init", "--quiet", dest_dir], check=True, capture_output=True, timeout=60)
    subprocess.run(
        ["git", "-C", dest_dir, "remote", "add", "origin", url],
        check=True, capture_output=True, timeout=60,
    )
    r = subprocess.run(
        ["git", "-C", dest_dir, "fetch", "origin", "--filter=tree:0",
         "--no-tags", *commits],
        capture_output=True, timeout=300,
    )
    if r.returncode == 0 and commits:
        return dest_dir
    # Fallback: full bare partial clone (commits+history, no working tree).
    shutil.rmtree(dest_dir, ignore_errors=True)
    return clone_repo(url, ref=None, dest_dir=dest_dir, shallow=False,
                      no_checkout=True, blob_filter="tree:0")


def _collect_benign_for_commit(
    repo: str,
    commit: str,
    samples_per_commit: int,
    min_code_lines: int,
    seed: int,
    seen_norm: Set[str],
    state_lock: threading.Lock,
    vuln_token_lists: List[List[str]],
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    """Return up to ``samples_per_commit`` benign functions for one fix commit.

    Candidate ordering: functions from the files the commit *modified* (the
    vulnerable function's *siblings* — sharpest hard negatives) are considered
    first, then functions from untouched files. Every candidate is passed
    through the near-duplicate guard: it is dropped when its token similarity to
    ANY vulnerable snippet in the project exceeds ``similarity_threshold`` (this
    is what excludes the patched function's post-fix twin and any lightly-edited
    copy).

    Thread-safety: ``seen_norm`` is a shared ``set`` across workers, so
    the check-then-add is guarded by ``state_lock``. The per-call RNG
    is derived deterministically from ``(seed, commit)`` (via a stable
    MD5 hash) so each commit gets a reproducible shuffle without sharing a
    single ``random.Random`` across threads (which is not thread-safe).
    """
    # ── No working-tree checkout ─────────────────────────────────────────────
    # The clone is bare (--no-checkout). We list the repo's .py files at this
    # commit via `git ls-tree` and read each candidate via `git show
    # <commit>:<path>`. This avoids (a) materialising the default-branch working
    # tree at clone time and (b) rewriting the whole working tree for every fix
    # commit — the two costs that made the old miner O(repos × history) slow.
    _fetch_commit(repo, commit)
    modified = _modified_py_files(repo, commit)
    # NOTE: a `git ls-tree` pathspec like `*.py` does NOT recurse (git glob
    # semantics), so we list every path at the commit and filter in Python —
    # robust for nested packages and cheap (names only, no blobs).
    ls = _git(repo, "ls-tree", "-r", "--name-only", commit)
    if ls.returncode != 0:
        return []
    all_py = [ln.strip() for ln in ls.stdout.splitlines() if ln.strip().endswith(".py")]

    rng = random.Random(_commit_seed(commit, seed))
    # Siblings (files the fix touched) FIRST, then untouched files. Each group
    # is shuffled independently so ordering is reproducible but not biased by the
    # repo's directory listing order.
    sibling_files = [f for f in all_py if f in modified]
    other_files = [f for f in all_py if f not in modified]
    rng.shuffle(sibling_files)
    rng.shuffle(other_files)
    candidates: List[str] = sibling_files + other_files

    collected: List[Dict[str, Any]] = []
    for rel in candidates:
        if len(collected) >= samples_per_commit:
            break
        is_sibling = rel in modified
        content = _read_file_at(repo, commit, rel)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            continue
        imports = collect_module_imports(content)
        for node in ast.walk(tree):
            if len(collected) >= samples_per_commit:
                break
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            func_src = "\n".join(content.splitlines()[start - 1 : end])
            if code_signal_line_count(func_src) < min_code_lines:
                continue
            # ── Near-duplicate guard ────────────────────────────────────────
            # Drop any candidate that is too similar to a vulnerable function in
            # this project. For sibling files this reliably removes the patched
            # function's post-fix twin (the near-duplicate that broke training).
            if vuln_token_lists and max_token_similarity(
                func_src, vuln_token_lists
            ) > similarity_threshold:
                continue
            norm = normalize_code(func_src)
            with state_lock:
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
                    # "benign_control" = mined from an untouched file;
                    # "benign_sibling" = mined from a file the fix touched
                    # (same imports/style as the vuln — a sharper negative).
                    # Both are label=0 verified-benign controls; the subtype
                    # only records provenance for auditing.
                    "sample_subtype": "benign_sibling" if is_sibling else "benign_control",
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
    seed: int,
    seen_norm: Set[str],
    state_lock: threading.Lock,
    ratio: Optional[float],
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    if project.git_source is None:
        return []  # benign mining requires a real repo
    commits = sorted(
        {s.commit_id for s in project.samples if s.commit_id and s.commit_id != "unknown_commit"}
    )
    if not commits:
        return []
    commits = commits[:max_commits]

    # ── Near-duplicate guard reference set ───────────────────────────────────
    # Tokenize every vulnerable snippet in this project ONCE; each candidate
    # benign function is checked against this set so it can never be a
    # near-duplicate of a known vulnerability (see _collect_benign_for_commit).
    vuln_token_lists: List[List[str]] = []
    n_vuln = 0
    for s in project.samples:
        if s.label == 1:
            n_vuln += 1
            if s.source_code:
                toks = tokenize_code(s.source_code)
                if toks:
                    vuln_token_lists.append(toks)

    # ── Per-project volume target ────────────────────────────────────────────
    # Aim for ``ratio`` × (#vulnerable) benign controls so the final
    # safe:vuln balance is a moderate, learnable ratio (default 5×), while
    # still honouring an explicit ``max_per_project`` ceiling when given.
    effective_cap = max_per_project
    if ratio and n_vuln > 0:
        target = int(math.ceil(ratio * n_vuln))
        effective_cap = target if max_per_project is None else min(max_per_project, target)

    dest = None
    out: List[Dict[str, Any]] = []
    try:
        # ── Minimal clone: fetch ONLY the fix commits ───────────────────────────
        # No default-branch history, no working tree. Trees/blobs are pulled on
        # demand by ``_read_file_at`` via ``git show <commit>:<path>``. This is
        # the core speedup over the old full working-tree clone (which also
        # rewrote the whole tree for every fix commit).
        dest = _clone_only_commits(project.git_source.git_url, commits)
        for commit in commits:
            if effective_cap and len(out) >= effective_cap:
                break
            found = _collect_benign_for_commit(
                dest, commit, samples_per_commit, min_code_lines,
                seed, seen_norm, state_lock,
                vuln_token_lists, similarity_threshold,
            )
            out.extend(found)
            if effective_cap:
                out = out[:effective_cap]
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
    ratio: Optional[float] = 5.0,
    similarity_threshold: float = 0.75,
) -> None:
    projects = load_manifest(manifest_path)
    seen_norm: Set[str] = set()
    state_lock = threading.Lock()
    benign_by_project: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_project, p, samples_per_commit, max_commits,
                max_per_project, min_code_lines, seed, seen_norm, state_lock,
                ratio, similarity_threshold,
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
        rng = random.Random(seed)
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
    n_sibling = sum(
        1 for p in manifest for s in p["samples"]
        if s.get("sample_subtype") == "benign_sibling"
    )
    print(f"\n{'─'*60}")
    print(f"  Source          : {manifest_path}")
    print(f"  Benign projects : {len(manifest):,}")
    print(f"  Benign samples  : {total:,}  (label=0)")
    print(f"    · siblings     : {n_sibling:,}  (from fix-touched files — sharp negatives)")
    print(f"    · untouched    : {total - n_sibling:,}  (from files the fix did not touch)")
    print(f"  Similarity guard: dropped candidates > {similarity_threshold:.2f} "
          f"token-similar to a vuln")
    if ratio:
        print(f"  Volume target   : {ratio:g}× vulnerable samples per project")
    print(f"  Output          : {output_path}")
    print(f"{'─'*60}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mine verified-benign control functions from CVEfixes repos."
    )
    p.add_argument("--manifest", required=True, help="benchmark_manifest_cvefixes.json")
    p.add_argument("--output", default="benign_controls_manifest.json",
                    help="Output manifest path.")
    p.add_argument("--samples-per-commit", type=int, default=10,
                    help="Max benign functions mined per fix commit (default 10; "
                         "raise for 'as many as possible').")
    p.add_argument("--max-commits", type=int, default=200,
                    help="Max fix commits processed per repo (default 200). "
                         "Pass a very large number (or edit to None) to process "
                         "EVERY fix commit — maximizes benign yield.")
    p.add_argument("--max-per-project", type=int, default=None,
                    help="Max benign samples per repo (default: unlimited).")
    p.add_argument("--max-total", type=int, default=None,
                    help="Hard cap on total benign samples.")
    p.add_argument("--max-workers", type=int, default=4, help="Parallel clones.")
    p.add_argument("--min-code-lines", type=int, default=2,
                    help="Drop functions with fewer code-signal lines.")
    p.add_argument("--ratio", type=float, default=5.0,
                    help="Target benign:vulnerable ratio per project (default 5×). "
                         "Set 0 to disable and rely on --max-per-project only.")
    p.add_argument("--similarity-threshold", type=float, default=0.75,
                    help="Drop a candidate benign function whose token similarity "
                         "to ANY vulnerable function in the project exceeds this "
                         "(default 0.75). This excludes near-duplicate post-fix "
                         "twins that would re-introduce label contradictions.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p


if __name__ == "__main__":
    a = _build_parser().parse_args()
    mine_benign(
        a.manifest, a.output, a.samples_per_commit, a.max_commits,
        a.max_per_project, a.max_total, a.max_workers, a.min_code_lines, a.seed,
        ratio=(a.ratio if a.ratio and a.ratio > 0 else None),
        similarity_threshold=a.similarity_threshold,
    )
