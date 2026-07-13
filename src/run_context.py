"""run_context.py — single source of truth for a pipeline run's identifier.

Why this exists
---------------
Stage 1 (Semgrep), Stage 2 (triage orchestrator), and Stage 3 (agent) must all
agree on the *same* artifact directory so the Stage-2 ledger written by one
process is found by the next. Previously each process derived the run id from
``GITHUB_RUN_ID`` / ``GITHUB_SHA`` and fell back to the literal
``"local-dev-run"``, which is (a) inconsistent across invocations and (b) not
unique. We now compute a single, stable run id:

1. An explicit ``CEVUD_RUN_ID`` env var. The CI workflows set this once and pass
   it to every stage, so all three stages share one id even though they run in
   separate ``docker run`` invocations.
2. Otherwise a persisted marker file under the workspace (so repeated
   ``docker run`` invocations that share the mounted volume reuse the SAME id
   for the same analysis run).
3. Otherwise a freshly generated unique id (timestamp + short uuid), which is
   persisted so later stages in the same workspace reuse it.

The ``GITHUB_RUN_ID`` / ``GITHUB_SHA`` env vars are still honoured as a
fallback so existing CI that exports them keeps working.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional


def normalize_run_id(run_id: str) -> str:
    """Coerce an arbitrary id into the canonical ``run_<id>`` form."""
    run_id = (run_id or "").strip()
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not run_id.startswith("run_"):
        run_id = f"run_{run_id}"
    return run_id


def _marker_path(workspace_path: str, workspace_root: str) -> Path:
    """Where the persisted run id lives for a workspace.

    ``workspace_root`` is resolved exactly the way ``config.json``'s
    ``paths.workspace_root`` is: absolute if it looks absolute, otherwise
    relative to ``workspace_path``.
    """
    ws = Path(workspace_path)
    root = Path(workspace_root) if os.path.isabs(workspace_root) else ws / workspace_root
    return root / ".cevud_run_id"


def resolve_run_id(
    workspace_path: str = ".",
    workspace_root: str = "workspace_storage",
    env_var: str = "CEVUD_RUN_ID",
) -> str:
    """Return a stable run id shared by every stage of a pipeline run.

    Resolution order: explicit env var (``CEVUD_RUN_ID``, then
    ``GITHUB_RUN_ID`` / ``GITHUB_SHA`` for backward compatibility) -> persisted
    marker file -> freshly generated unique id (which is then persisted).
    """
    explicit = (
        os.getenv(env_var)
        or os.getenv("GITHUB_RUN_ID")
        or os.getenv("GITHUB_SHA")
    )
    if explicit:
        return normalize_run_id(explicit)

    marker = _marker_path(workspace_path, workspace_root)
    try:
        if marker.exists():
            cached = marker.read_text(encoding="utf-8").strip()
            if cached:
                return cached if cached.startswith("run_") else f"run_{cached}"
    except OSError:
        pass

    run_id = normalize_run_id(f"{int(time.time())}_{uuid.uuid4().hex[:8]}")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(run_id, encoding="utf-8")
    except OSError:
        # If we cannot persist, the id is still unique for this process; later
        # stages that share this process tree would need the env var instead.
        pass
    return run_id
