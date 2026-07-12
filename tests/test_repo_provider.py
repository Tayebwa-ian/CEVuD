"""
Regression tests for src/evaluation/repo_provider.clone_repo.

The key case guards against the historical CI failure where a `git clone`
that failed partway (e.g. a network drop) left a non-empty destination
directory behind. On the next retry `git clone` then aborts with
"destination path already exists and is not an empty directory", turning a
transient failure into a fatal one. clone_repo must clean up any partial
clone before each retry so the attempt can actually succeed.
"""

import os
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src" / "evaluation"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from repo_provider import clone_repo, RepoCloneError  # noqa: E402


def _ok():
    cp = MagicMock()
    cp.returncode = 0
    cp.stderr = ""
    return cp


def _err(stderr):
    cp = MagicMock()
    cp.returncode = 1
    cp.stderr = stderr
    return cp


def _git_side_effect_that_mimics_real_clone(state):
    """Simulates `git` for clone_repo.

    - A clone into an existing, non-empty directory fails exactly like
      real git ("destination path already exists ...").
    - A clone into an absent directory emulates a transient network
      failure on the FIRST such call (but still eagerly creates the dir,
      as git does) and then succeeds on subsequent calls.

    With clone_repo's on-retry cleanup, attempt 1 fails, the partial
    dir is removed, and attempt 2 (dir now absent) succeeds. Without
    the cleanup, attempt 2 would hit the "already exists" path and
    raise after exhausting retries.
    """

    def _run(args, **kwargs):
        if args[1:2] == ["clone"]:
            dest = args[-1]
            if os.path.exists(dest) and os.listdir(dest):
                return _err(
                    f"fatal: destination path '{dest}' already exists "
                    f"and is not an empty directory."
                )
            os.makedirs(dest, exist_ok=True)
            state["clone_calls"] += 1
            if state["clone_calls"] == 1:
                # A real partial clone leaves a non-empty directory behind
                # (e.g. a half-written `.git`), which is exactly what
                # makes a naive retry abort with "already exists".
                Path(dest, ".git").mkdir(parents=True, exist_ok=True)
                return _err("fatal: the remote end hung up unexpectedly")
            return _ok()
        return _ok()

    return _run


def test_clone_recovers_from_partial_directory_on_retry(tmp_path):
    state = {"clone_calls": 0}
    dest = str(tmp_path / "repo")

    with patch(
        "repo_provider.subprocess.run",
        side_effect=_git_side_effect_that_mimics_real_clone(state),
    ), patch("repo_provider.time.sleep", return_value=None):
        resolved = clone_repo(
            "https://example.com/owner/repo", ref=None, dest_dir=dest
        )

    assert resolved == dest
    assert os.path.isdir(resolved)
    # One failed attempt + one successful retry.
    assert state["clone_calls"] == 2


def test_clone_with_existing_dest_dir_is_cleaned(tmp_path):
    state = {"clone_calls": 0}
    dest = str(tmp_path / "repo")
    # Pre-existing, non-empty directory (e.g. leftover from a prior run).
    os.makedirs(dest, exist_ok=True)
    Path(dest, "stale.txt").write_text("leftover")

    with patch(
        "repo_provider.subprocess.run",
        side_effect=_git_side_effect_that_mimics_real_clone(state),
    ), patch("repo_provider.time.sleep", return_value=None):
        resolved = clone_repo(
            "https://example.com/owner/repo", ref=None, dest_dir=dest
        )

    assert resolved == dest
    assert os.path.isdir(resolved)
    # Stale contents were removed before cloning.
    assert not Path(dest, "stale.txt").exists()
    # Recovered after at least one clone attempt.
    assert state["clone_calls"] >= 1


def test_clone_reports_unrecoverable_failure(tmp_path):
    state = {"clone_calls": 0}

    def _run(args, **kwargs):
        if args[1:2] == ["clone"]:
            dest = args[-1]
            # Emulate "repository not found": git creates no dir and fails.
            return _err("fatal: repository 'https://nope/nope' not found")
        return _ok()

    with patch("repo_provider.subprocess.run", side_effect=_run), patch(
        "repo_provider.time.sleep", return_value=None
    ):
        with pytest.raises(RepoCloneError):
            clone_repo("https://nope/nope", ref=None, dest_dir=str(tmp_path / "repo"))
