"""
raw_score_extractor.py
========================
The ONE module in this package that is allowed to be expensive: it runs
Semgrep once per project and the CodeSheriff SLM once (batched) per project,
and persists exactly two numbers per labeled sample — `severity_weight` and
`slm_score` — alongside the ground-truth label and repo provenance.

Every other module in this package (gate_strategies, grid_search,
sensitivity_analysis, linearity_check) reads ONLY the cached output of this
module. None of them re-run Semgrep or the classifier. This is what makes it
tractable to grid-search hundreds of (weight, threshold) combinations and
evaluate seven-plus baselines without seven-plus full pipeline runs.

Caching: results are written to a JSON file keyed by manifest path, so
re-running `run_comparative_evaluation.py` (e.g. to tweak a plot) does not
require re-cloning repos or re-scoring samples unless `--force-recompute`
is passed.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import List, Dict, Any

from schema import BenchmarkSample, ProjectManifest, RawScoreRecord
from repo_provider import resolve_project_workspace
from benchmark_manifest import load_manifest

import sys
# model_manager.py lives in src/, one level up from src/evaluation/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_manager import ModelManager  # noqa: E402


def _find_custom_rules_path() -> str:
    """Resolves semgrep_rules/custom_appsec_rules.yaml relative to the repo
    root (three levels up from this file: src/evaluation/ -> src/ -> root/).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "semgrep_rules", "custom_appsec_rules.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate custom_appsec_rules.yaml at expected path: {path}")
    return path


def _run_semgrep(project_root: str, output_path: str, custom_rules_path: str) -> Dict[str, Any]:
    """Runs the `semgrep` CLI (installed standalone, e.g. via pipx — see the
    project's Dockerfile) once over the whole project root and returns the
    parsed JSON results. A single run per project, regardless of how many
    labeled samples that project has.
    """
    cmd = [
        "semgrep",
        "--config", "p/python",
        "--config", custom_rules_path,
        "--no-git-ignore",
        "--json",
        "--output", output_path,
        project_root,
    ]
    print(f"[*] Running semgrep once over project root: {project_root}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # semgrep exits 1 when findings exist; that's not an error
        print(f"[!] semgrep exited with code {result.returncode}:\n{result.stderr}")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"results": []}


def _build_severity_index(
    semgrep_results: Dict[str, Any], project_root: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Indexes semgrep findings by file path (relative to project_root) for
    fast line-range lookups: {relative_file_path: [{"start": int, "end": int, "severity": str}, ...]}.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    for finding in semgrep_results.get("results", []):
        abs_path = finding.get("path", "")
        rel_path = os.path.relpath(abs_path, project_root) if os.path.isabs(abs_path) else abs_path
        rel_path = rel_path.replace("\\", "/")
        start = finding.get("start", {}).get("line", 0)
        end = finding.get("end", {}).get("line", start)
        severity = finding.get("extra", {}).get("severity", "INFO")
        index.setdefault(rel_path, []).append({"start": start, "end": end, "severity": severity})
    return index


def _match_severity(
    severity_index: Dict[str, List[Dict[str, Any]]],
    file_path: str,
    start_line: int,
    end_line: int,
    severity_map: Dict[str, float],
) -> str:
    """Returns the HIGHEST-weighted severity among semgrep findings whose
    line range overlaps [start_line, end_line] in `file_path`, or "NONE" if
    no finding overlaps (i.e. Semgrep did not flag this function at all).
    """
    candidates = severity_index.get(file_path.replace("\\", "/"), [])
    overlapping = [
        c for c in candidates
        if c["start"] <= end_line and c["end"] >= start_line
    ]
    if not overlapping:
        return "NONE"
    best = max(overlapping, key=lambda c: severity_map.get(c["severity"], 0.0))
    return best["severity"]


def _read_source_snippet(project_root: str, file_path: str, start_line: int, end_line: int) -> str:
    """Reads lines [start_line, end_line] (1-indexed, inclusive) from a file
    within the project root.
    """
    full_path = os.path.join(project_root, file_path)
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[start_line - 1:end_line])


class RawScoreExtractor:
    """Runs Stage 1 (Semgrep) and Stage 2's SLM (CodeSheriff) once per
    project across an entire benchmark manifest, and caches the resulting
    per-sample raw scores.

    Usage:
        extractor = RawScoreExtractor(config_path="config.json")
        records = extractor.extract("benchmark_manifest.json", cache_path="raw_scores.json")
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.severity_map: Dict[str, float] = self.config.get("semgrep_severity_map", {})
        self.severity_map.setdefault("NONE", 0.0)
        self.custom_rules_path = _find_custom_rules_path()
        # Lazily instantiate — avoids loading the ~500MB classifier model
        # until extraction actually needs it (e.g. when loading from cache).
        self._model_manager: ModelManager = None

    def _get_model_manager(self) -> ModelManager:
        if self._model_manager is None:
            self._model_manager = ModelManager()
        return self._model_manager

    def _extract_project(self, project: ProjectManifest) -> List[RawScoreRecord]:
        """Extracts raw scores for every sample in a single project. Clones
        the project's repo (if needed) for the duration of this call only —
        see repo_provider.resolve_project_workspace for cleanup guarantees.
        """
        with resolve_project_workspace(project) as (local_path, provenance):
            semgrep_output_path = os.path.join(local_path, "_eval_semgrep_results.json")
            semgrep_results = _run_semgrep(local_path, semgrep_output_path, self.custom_rules_path)
            severity_index = _build_severity_index(semgrep_results, local_path)

            # Resolve source code + severity for every sample BEFORE running
            # the SLM, so inference can be batched in one call per project.
            snippets: List[str] = []
            sample_meta: List[BenchmarkSample] = []
            severities: List[str] = []

            for sample in project.samples:
                code = sample.source_code or _read_source_snippet(
                    local_path, sample.file_path, sample.start_line, sample.end_line
                )
                severity = _match_severity(
                    severity_index, sample.file_path, sample.start_line, sample.end_line, self.severity_map
                )
                snippets.append(code)
                sample_meta.append(sample)
                severities.append(severity)

            print(f"[*] Running batched SLM inference on {len(snippets)} samples for '{project.project}' ...")
            slm_scores = self._get_model_manager().get_classifier_inference(snippets)

            records = []
            for sample, severity, slm_score in zip(sample_meta, severities, slm_scores):
                records.append(RawScoreRecord(
                    sample_id=sample.sample_id,
                    project=project.project,
                    file_path=sample.file_path,
                    function_name=sample.function_name,
                    label=sample.label,
                    severity=severity,
                    severity_weight=self.severity_map.get(severity, 0.0),
                    slm_score=round(float(slm_score), 4),
                    vulnerability_type=sample.vulnerability_type,
                    provenance=provenance,
                ))
            return records

    def extract(
        self, manifest_path: str, cache_path: str = None, force_recompute: bool = False
    ) -> List[RawScoreRecord]:
        """Extracts (or loads cached) raw scores for every sample across
        every project in the manifest.

        Args:
            manifest_path: Path to the benchmark manifest JSON (see
                benchmark_manifest.py for schema).
            cache_path: Where to read/write the cached raw_scores.json. If
                None, no caching is performed and extraction always re-runs.
            force_recompute: If True, ignores any existing cache and
                re-extracts everything (e.g. after updating the SLM model
                or the custom Semgrep rules).

        Returns:
            List[RawScoreRecord]: One record per labeled sample, across all projects.
        """
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            print(f"[+] Loading cached raw scores from: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return [RawScoreRecord.from_dict(d) for d in cached]

        projects = load_manifest(manifest_path)
        all_records: List[RawScoreRecord] = []
        for i, project in enumerate(projects, start=1):
            print(f"\n=== [{i}/{len(projects)}] Extracting raw scores for project: {project.project} ===")
            all_records.extend(self._extract_project(project))

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in all_records], f, indent=2)
            print(f"[+] Cached {len(all_records)} raw score records to: {cache_path}")

        return all_records
