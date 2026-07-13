"""run_context.py — single source of truth for a pipeline run's identifier
and for every path inside ``workspace_storage``.

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

In addition to the run id, this module owns **every** path inside the
``workspace_storage`` tree. ``config.json → paths`` is the single source of
truth for the directory names; the helpers below combine those names with the
resolved workspace root and run id so the rest of the codebase never hardcodes
``workspace_storage`` again.
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


# ---------------------------------------------------------------------------
# Path helpers — the single source of truth for the workspace_storage tree.
# Every subdirectory key lives in config.json → paths.  Absolute values are
# honoured as-is; relative values are resolved against ``workspace_path``.
# ---------------------------------------------------------------------------

def _resolve_ws_root(workspace_path: str, config: dict) -> str:
    """Return the absolute workspace storage root."""
    ws_root = config.get("paths", {}).get("workspace_root", "workspace_storage")
    if os.path.isabs(ws_root):
        return ws_root
    return os.path.abspath(os.path.join(workspace_path, ws_root))


def get_artifact_dir(workspace_path: str, config: dict, run_id: str) -> str:
    """Return ``<workspace_root>/<artifacts_subdir>/<run_id>``."""
    ws_root = _resolve_ws_root(workspace_path, config)
    artifacts_subdir = config.get("paths", {}).get("artifacts_subdir", "artifacts")
    return os.path.join(ws_root, artifacts_subdir, run_id)


def get_vector_db_dir(workspace_path: str, config: dict) -> str:
    """Return the directory used for the local SQLite vector store."""
    ws_root = _resolve_ws_root(workspace_path, config)
    vector_db_dir = config.get("paths", {}).get("vector_db_dir", "codebase_vectors")
    if os.path.isabs(vector_db_dir):
        return vector_db_dir
    return os.path.join(ws_root, vector_db_dir)


def get_model_cache_dir(workspace_path: str, config: dict) -> str:
    """Return the HuggingFace model cache directory."""
    ws_root = _resolve_ws_root(workspace_path, config)
    model_cache_dir = config.get("paths", {}).get("model_cache_dir", "model_cache")
    if os.path.isabs(model_cache_dir):
        return model_cache_dir
    return os.path.join(ws_root, model_cache_dir)


def get_eval_dir(workspace_path: str, config: dict, eval_id: str) -> str:
    """Return ``<workspace_root>/<evaluations_subdir>/<eval_id>``."""
    ws_root = _resolve_ws_root(workspace_path, config)
    evals_subdir = config.get("paths", {}).get("evaluations_subdir", "evaluation_runs")
    return os.path.join(ws_root, evals_subdir, eval_id)


def get_semgrep_output_path(workspace_path: str, config: dict, run_id: str) -> str:
    """Return the absolute path to the Semgrep JSON output for a specific run."""
    return os.path.join(get_artifact_dir(workspace_path, config, run_id),
                        config.get("paths", {}).get("semgrep_output", "semgrep_results.json"))


def get_triage_report_path(workspace_path: str, config: dict, run_id: str) -> str:
    """Return the absolute path to the Stage 2 triage ledger."""
    return os.path.join(get_artifact_dir(workspace_path, config, run_id),
                        config.get("paths", {}).get("triage_report", "stage1_2_triage.json"))


def get_remediation_dossier_path(workspace_path: str, config: dict, run_id: str) -> str:
    """Return the absolute path to the Stage 3 remediation dossier."""
    return os.path.join(get_artifact_dir(workspace_path, config, run_id),
                        "remediation_dossier.md")
