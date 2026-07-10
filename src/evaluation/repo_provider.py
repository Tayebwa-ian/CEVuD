"""
repo_provider.py
=================
Resolves a ProjectManifest's code onto local disk for analysis, and
guarantees cleanup afterwards.

Two cases:
    - GitRepoSource: clone the URL into a fresh temporary directory, resolve
      the exact commit SHA actually checked out, yield the local path, and
      delete the temporary directory on exit (success, failure, or exception
      — the `finally` block always runs).
    - LocalRepoSource: nothing to clone or delete; just yield the given path
      as-is.

In both cases, the caller receives a `RepoProvenance` record (project name,
git URL, requested ref, resolved commit SHA, clone timestamp) that survives
long after the temporary checkout is gone — this is what lets a results
report say "this came from repo X @ commit Y" without needing the clone to
still exist on disk.

Usage:
    with resolve_project_workspace(project_manifest) as (local_path, provenance):
        # local_path is guaranteed to exist and contain the project's code
        # for the duration of this block only.
        ...
    # local_path's temp directory (if any) has now been deleted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Tuple

from schema import ProjectManifest, RepoProvenance


class RepoCloneError(RuntimeError):
    """Raised when `git clone` or `git checkout` fails."""


import time

def _run_git(args, cwd=None, retries=3, timeout=300) -> subprocess.CompletedProcess:
    """Runs a git subcommand with retries, timeout, and exponential backoff.
    Raises RepoCloneError with full stderr on failure."""
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return result
            last_err = result.stderr
            
            # Fast fail for unrecoverable errors (e.g., repository not found or access denied)
            if "not found" in last_err.lower() or "fatal: could not read Username" in last_err:
                break
                
        except subprocess.TimeoutExpired as e:
            last_err = f"Timeout after {timeout}s: {e}"
        except Exception as e:
            last_err = str(e)

        if attempt < retries:
            backoff = 2 ** attempt
            print(f"[!] git {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RepoCloneError(
        f"git {' '.join(args)} failed after {attempt} attempts:\n{last_err}"
    )


def clone_repo(git_url: str, ref: str = None, dest_dir: str = None, shallow: bool = True) -> str:
    """Clones `git_url` into `dest_dir` (or a fresh temp dir if None), and
    checks out `ref` if given.

    Args:
        git_url: Clonable repository URL.
        ref: Branch, tag, or commit SHA to check out. If None, the default
            branch's HEAD is used.
        dest_dir: Target directory. Created via tempfile.mkdtemp() if omitted.
        shallow: If True and `ref` is not a specific commit, uses a shallow
            (--depth 1) clone for speed. Shallow clones are skipped when a
            specific ref is requested, since `git checkout <sha>` requires
            that commit's history to be present.

    Returns:
        str: Path to the cloned repository root.

    Raises:
        RepoCloneError: If cloning or checkout fails.
    """
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="cevud_repo_")

    clone_args = ["clone"]
    if shallow and ref is None:
        clone_args += ["--depth", "1"]
    clone_args += [git_url, dest_dir]

    print(f"[*] Cloning {git_url} into {dest_dir} ...")
    _run_git(clone_args)

    if ref is not None:
        # Full history may be required to reach an arbitrary ref/SHA; fetch
        # it explicitly rather than assuming the initial clone has it.
        try:
            _run_git(["checkout", ref], cwd=dest_dir)
        except RepoCloneError:
            print(f"[*] ref '{ref}' not found in initial clone; fetching full history ...")
            _run_git(["fetch", "--unshallow"], cwd=dest_dir)
            _run_git(["checkout", ref], cwd=dest_dir)

    return dest_dir


def resolve_commit_sha(repo_path: str) -> str:
    """Returns the full commit SHA currently checked out at `repo_path`."""
    result = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return result.stdout.strip()


@contextmanager
def resolve_project_workspace(project: ProjectManifest) -> Iterator[Tuple[str, RepoProvenance]]:
    """Context manager yielding (local_path, provenance) for a project's code.

    For a GitRepoSource project: clones to a temp dir, resolves the commit
    SHA, yields (temp_dir_path, provenance), and unconditionally deletes the
    temp directory on exit.

    For a LocalRepoSource project: yields (root_path, provenance) directly;
    nothing is cloned or deleted, since the caller doesn't own that path.

    Args:
        project: A validated ProjectManifest (see benchmark_manifest.py).

    Yields:
        Tuple[str, RepoProvenance]: Local filesystem path to the project's
            code (valid only within the `with` block for git sources), and
            a provenance record safe to persist indefinitely.
    """
    if project.local_source is not None:
        provenance = RepoProvenance(
            project=project.project,
            git_url=None,
            requested_ref=None,
            resolved_commit_sha=None,
            cloned_at_utc=None,
        )
        yield project.local_source.root_path, provenance
        return

    # git_source case
    git_source = project.git_source
    local_path = None
    try:
        local_path = clone_repo(git_source.git_url, ref=git_source.ref)
        resolved_sha = resolve_commit_sha(local_path)
        provenance = RepoProvenance(
            project=project.project,
            git_url=git_source.git_url,
            requested_ref=git_source.ref,
            resolved_commit_sha=resolved_sha,
            cloned_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[+] Resolved '{project.project}' to commit {resolved_sha[:12]} at {local_path}")
        yield local_path, provenance
    finally:
        if local_path is not None and os.path.exists(local_path):
            print(f"[*] Cleaning up temporary clone: {local_path}")
            shutil.rmtree(local_path, ignore_errors=True)
