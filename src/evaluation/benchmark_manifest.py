"""
benchmark_manifest.py
======================
Loads and validates the multi-project labeled benchmark manifest: the file
that says "here are N projects, here is where to get each one's code, and
here are the ground-truth labeled functions inside each one."

This is intentionally decoupled from `dataset_ingest.py`'s benchmark format.
`dataset_ingest.py` seeds the RAG vector store (no ground-truth labels, no
notion of "safe" — every entry is just a code block to index for Stage 3
context retrieval). This module's manifest is for *evaluation*: every entry
carries a binary vulnerable/safe label, and covers real projects rather than
hand-written snippets.

Manifest JSON schema
---------------------
A list of project objects. Each project has EITHER a `git_source` OR a
`local_source`, plus a `samples` list.

    [
      {
        "project": "flask-example-app",
        "git_source": {
          "git_url": "https://github.com/some-org/flask-example-app.git",
          "ref": "a1b2c3d4"          // optional: branch, tag, or commit SHA
        },
        "samples": [
          {
            "sample_id": "flask-example-app::handle_upload::120",
            "file_path": "app/routes.py",
            "function_name": "handle_upload",
            "start_line": 120,
            "end_line": 145,
            "label": 1,
            "vulnerability_type": "path-traversal"
          },
          ...
        ]
      },
      {
        "project": "local-scratch-repo",
        "local_source": { "root_path": "/path/to/already/checked/out/repo" },
        "samples": [ ... ]
      }
    ]

Recommended sourcing strategy: populate this
manifest from a labeled vulnerability-fix corpus (e.g. CVEfixes /
PrimeVul-style commit datasets) filtered to Python — the pre-fix function
version becomes a label=1 sample, the post-fix version becomes a label=0
sample, and `git_source.ref` can point at the specific commit SHA so results
are exactly reproducible.

Alternatively, a self-contained corpus such as VUDENC (via
`src/scripts/convert_vudenc.py`) carries no repository metadata: its projects
use `local_source` and embed each function in `source_code`, which the
extraction pipeline scores directly (see `raw_score_extractor.py`'s inline
path). Both shapes share the schema below and are interchangeable downstream.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any

from schema import BenchmarkSample, ProjectManifest, GitRepoSource, LocalRepoSource


class ManifestValidationError(ValueError):
    """Raised when a benchmark manifest file is structurally invalid."""


def _parse_sample(raw: Dict[str, Any], project: str) -> BenchmarkSample:
    required = ("sample_id", "file_path", "function_name", "start_line", "end_line", "label")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ManifestValidationError(
            f"Project '{project}': sample missing required field(s) {missing}: {raw}"
        )
    if raw["label"] not in (0, 1):
        raise ManifestValidationError(
            f"Project '{project}': sample '{raw['sample_id']}' has non-binary label {raw['label']!r}"
        )
    if raw["start_line"] > raw["end_line"]:
        raise ManifestValidationError(
            f"Project '{project}': sample '{raw['sample_id']}' has start_line > end_line"
        )
    return BenchmarkSample(
        sample_id=raw["sample_id"],
        file_path=raw["file_path"],
        function_name=raw["function_name"],
        start_line=raw["start_line"],
        end_line=raw["end_line"],
        label=raw["label"],
        vulnerability_type=raw.get("vulnerability_type"),
        source_code=raw.get("source_code"),
        repo_url=raw.get("repo_url"),
        commit_id=raw.get("commit_id"),
        target_commit=raw.get("target_commit"),
        cve_id=raw.get("cve_id"),
        cvss_score=raw.get("cvss_score"),
        fixed_code=raw.get("fixed_code"),
        diff_with_context=raw.get("diff_with_context"),
        sample_subtype=raw.get("sample_subtype"),
    )


def _parse_project(raw: Dict[str, Any]) -> ProjectManifest:
    if "project" not in raw:
        raise ManifestValidationError(f"Project entry missing 'project' name: {raw}")
    project = raw["project"]

    has_git = "git_source" in raw and raw["git_source"] is not None
    has_local = "local_source" in raw and raw["local_source"] is not None
    if has_git == has_local:
        raise ManifestValidationError(
            f"Project '{project}' must specify exactly one of git_source / local_source."
        )

    git_source = None
    local_source = None
    if has_git:
        gs = raw["git_source"]
        if "git_url" not in gs:
            raise ManifestValidationError(f"Project '{project}': git_source missing 'git_url'.")
        git_source = GitRepoSource(git_url=gs["git_url"], ref=gs.get("ref"))
    else:
        ls = raw["local_source"]
        if "root_path" not in ls:
            raise ManifestValidationError(f"Project '{project}': local_source missing 'root_path'.")
        local_source = LocalRepoSource(root_path=ls["root_path"])

    raw_samples = raw.get("samples", [])
    if not raw_samples:
        raise ManifestValidationError(f"Project '{project}' has no samples.")
    samples = [_parse_sample(s, project) for s in raw_samples]

    # sample_id must be unique within a project (uniqueness across the whole
    # manifest is checked by load_manifest, once all projects are parsed).
    ids = [s.sample_id for s in samples]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ManifestValidationError(f"Project '{project}' has duplicate sample_id(s): {dupes}")

    return ProjectManifest(
        project=project,
        samples=samples,
        git_source=git_source,
        local_source=local_source,
    )


def load_manifest(path: str) -> List[ProjectManifest]:
    """Loads and validates a benchmark manifest file.

    Args:
        path: Path to the manifest JSON file (a top-level list of project objects).

    Returns:
        List[ProjectManifest]: One entry per project, fully validated.

    Raises:
        ManifestValidationError: If the manifest is structurally invalid,
            including duplicate sample_ids across the whole manifest.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    if not isinstance(raw_list, list):
        raise ManifestValidationError("Manifest root must be a JSON list of project objects.")

    projects = [_parse_project(p) for p in raw_list]

    all_ids = [s.sample_id for proj in projects for s in proj.samples]
    dupes = {i for i in all_ids if all_ids.count(i) > 1}
    if dupes:
        raise ManifestValidationError(f"Manifest has duplicate sample_id(s) across projects: {dupes}")

    return projects


def manifest_summary(projects: List[ProjectManifest]) -> Dict[str, Any]:
    """Produces a small human-readable summary (project count, sample counts,
    class balance) — useful to sanity-check a manifest before spending time
    on extraction, and to embed in the final report for transparency.
    """
    summary = {
        "num_projects": len(projects),
        "total_samples": sum(len(p.samples) for p in projects),
        "per_project": {},
    }
    for p in projects:
        pos = sum(1 for s in p.samples if s.label == 1)
        neg = len(p.samples) - pos
        summary["per_project"][p.project] = {
            "total": len(p.samples),
            "vulnerable": pos,
            "safe": neg,
            "source": "git" if p.git_source else "local",
        }
    return summary
