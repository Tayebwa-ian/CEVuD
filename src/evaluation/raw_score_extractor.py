"""
raw_score_extractor.py
========================
The ONE module in this package that is allowed to be expensive: it runs
Semgrep once per project and the local SLM classifier once (batched) per project,
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
import re
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional

from code_context import (
    collect_module_imports,
    collect_cross_file_context,
    expand_to_function,
    build_context_snippet,
)
from code_chunks import chunk_code, aggregate_chunk_scores

from schema import BenchmarkSample, ProjectManifest, RawScoreRecord, RepoProvenance
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

    The Stage-1 (static taint) severity is a HARD input to the gate
    — if Semgrep never runs, every sample gets severity 0.0 and the
    whole comparative study is meaningless. We therefore FAIL FAST if the
    `semgrep` binary is not on PATH (rather than silently emitting
    empty results), so a misconfigured environment cannot produce a
    no-Semgrep report by accident.
    """
    import shutil as _shutil  # local import keeps the failure path dependency-free
    if _shutil.which("semgrep") is None:
        raise RuntimeError(
            "Semgrep is not installed / not on PATH. Stage-1 severity is a "
            "required input to the gate and CANNOT be skipped. Install it with "
            "`pip install semgrep` (or `pipx install semgrep`) and re-run. "
            "Do NOT use --cache to bypass this — --cache reuses a prior "
            "raw_scores_cache.json and will produce a report with no Semgrep signal."
        )
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
    """Runs Stage 1 (Semgrep) and Stage 2's local SLM classifier once per
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

    def _extract_project(self, project: ProjectManifest, force_inline: bool = False) -> List[RawScoreRecord]:
        """Extracts raw scores for every sample in a single project.

        Two workspace shapes are supported:

        * Real repo (git_source, or a local_source pointing at an actual
          checkout): the repo is scanned wholesale and severity is matched
          against each sample's real file_path / line range.

        * Inline-snippet manifest (e.g. converted CVEfixes / VUDENC, where
          every sample carries `source_code` but file_path is a synthetic
          "inline_snippet.py" that does not exist on disk): we materialize
          each snippet to its OWN .py file in a fresh temp dir and scan ONLY
          that temp dir. This is critical — otherwise we would scan the entire
          current working directory (the CEVuD repo itself) and never match
          the synthetic path, silently forcing every sample's severity to
          "NONE" and making the static stage contribute nothing.

        `force_inline` lets callers evaluate a *git_source* manifest in
        inline mode: every sample already embeds its `source_code`
        (and `fixed_code`), so cloning the real repo is unnecessary. This
        is the single biggest speedup — it skips every `git clone`,
        network round-trip, and temp-dir cleanup, while still scoring the
        exact code the converter extracted. It is the recommended mode for
        fast iteration and for air-gapped / offline CI.
        """
        if project.git_source is not None and force_inline:
            # No clone: synthesize a provenance record and score the embedded
            # snippets directly (see _extract_inline).
            provenance = RepoProvenance(
                project=project.project,
                git_url=project.git_source.git_url,
                requested_ref=project.git_source.ref,
                resolved_commit_sha=None,
                cloned_at_utc=None,
            )
            return self._extract_inline(project, None, provenance)
        if project.git_source is not None:
            with resolve_project_workspace(project) as (local_path, provenance):
                return self._extract_git_source(project, local_path, provenance)
        with resolve_project_workspace(project) as (local_path, provenance):
            return self._extract_inline(project, local_path, provenance)

    # ------------------------------------------------------------------
    # Git-source extraction: clone the REAL repo, check out the
    # vulnerable commit, and give every stage genuine code context.
    # ------------------------------------------------------------------

    @staticmethod
    def _git_show(repo_path: str, commit: str, rel_path: str) -> Optional[str]:
        """Reads ``rel_path`` at ``commit`` via ``git show <commit>:<rel_path>``."""
        try:
            r = subprocess.run(
                ["git", "-C", repo_path, "show", f"{commit}:{rel_path}"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0 and r.stdout != "":
                return r.stdout
        except Exception:
            pass
        return None

    @staticmethod
    def _parent_commit(repo_path: str, commit: Optional[str]) -> Optional[str]:
        """Returns the parent SHA of ``commit`` (the vulnerable version
        precedes the fix). None if it cannot be resolved.
        """
        if not commit or commit == "unknown_commit":
            return None
        try:
            r = subprocess.run(
                ["git", "-C", repo_path, "rev-parse", f"{commit}^"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        return None

    def _extract_inline(self, project, local_path, provenance) -> List[RawScoreRecord]:
        """Local-source / self-contained snippet manifest (e.g. VUDENC):
        materialize each ``source_code`` snippet and scan only it.
        """
        scan_root = tempfile.mkdtemp(prefix="cevud_inline_")
        try:
            snippet_rel: Dict[str, str] = {}
            for sample in project.samples:
                code = sample.source_code or ""
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sample.sample_id)
                with open(os.path.join(scan_root, f"{safe}.py"), "w", encoding="utf-8") as f:
                    f.write(code)
                snippet_rel[sample.sample_id] = f"{safe}.py"
            severity_index = _build_severity_index(
                _run_semgrep(scan_root, os.path.join(scan_root, "_eval_semgrep_results.json"),
                             self.custom_rules_path),
                scan_root,
            )
            snippets, meta, severities = [], [], []
            for sample in project.samples:
                code = sample.source_code or ""
                sev_path = snippet_rel.get(sample.sample_id, sample.file_path)
                sev_start, sev_end = 1, max(len(code.splitlines()), 1)
                severities.append(_match_severity(
                    severity_index, sev_path, sev_start, sev_end, self.severity_map))
                snippets.append(code)
                meta.append(sample)
            return self._finalize(project, meta, severities, snippets, provenance)
        finally:
            shutil.rmtree(scan_root, ignore_errors=True)

    def _extract_git_source(self, project, local_path, provenance) -> List[RawScoreRecord]:
        """CVEfixes-style git_source manifest.

        For each sample we clone the REAL repo, check out the PARENT of
        the fix commit (the vulnerable version), read the full file at
        that commit, AST-expand to the enclosing function, and pull in
        the module imports + best-effort cross-file context. Semgrep is
        run over the materialized REAL files (at their real relative
        paths, so cross-file taint works), and the SLM/LLM receive
        the complete function + imports + cross-file source.
        """
        scan_root = tempfile.mkdtemp(prefix="cevud_ctx_")
        try:
            snippets: List[str] = []
            meta: List[BenchmarkSample] = []
            sev_info: Dict[str, tuple] = {}

            for sample in project.samples:
                fp = sample.file_path
                # A sample may pin an exact commit to check out (e.g. a *safe*
                # post-fix sample targets the fix commit itself). Otherwise we
                # fall back to the historical default: the parent of the fix
                # commit, i.e. the vulnerable (pre-fix) version.
                vuln_commit = sample.target_commit or (
                    self._parent_commit(local_path, sample.commit_id) or sample.commit_id
                )
                content = self._git_show(local_path, vuln_commit, fp) if vuln_commit else None
                if content is None and sample.commit_id:
                    content = self._git_show(local_path, sample.commit_id, fp)
                use_inline = content is None
                if use_inline:
                    content = sample.source_code or ""

                # Materialize the full file at its REAL relative path.
                real_parts = fp.replace("\\", "/").split("/")
                fpath = os.path.join(scan_root, *real_parts)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

                if use_inline:
                    sev_path = os.path.relpath(fpath, scan_root).replace("\\", "/")
                    func_start, func_end = 1, max(len(content.splitlines()), 1)
                    slm_code = content
                else:
                    imports = collect_module_imports(content)
                    cross = collect_cross_file_context(
                        content, fp, local_path,
                        read_file=lambda rel: self._git_show(local_path, vuln_commit, rel),
                    )
                    for mrel, msrc in cross.items():
                        mparts = mrel.replace("\\", "/").split("/")
                        mpath = os.path.join(scan_root, *mparts)
                        os.makedirs(os.path.dirname(mpath), exist_ok=True)
                        with open(mpath, "w", encoding="utf-8") as f:
                            f.write(msrc)
                    func_start, func_end = expand_to_function(content, sample.start_line)
                    slm_code = build_context_snippet(
                        content, (func_start, func_end), imports, {}
                    )
                    sev_path = fp.replace("\\", "/")

                sev_info[sample.sample_id] = (sev_path, func_start, func_end)
                snippets.append(slm_code)
                meta.append(sample)

            severity_index = _build_severity_index(
                _run_semgrep(scan_root, os.path.join(scan_root, "_eval_semgrep_results.json"),
                             self.custom_rules_path),
                scan_root,
            )
            severities: List[str] = []
            for sample in meta:
                sev_path, fs, fe = sev_info[sample.sample_id]
                severities.append(_match_severity(
                    severity_index, sev_path, fs, fe, self.severity_map))
            return self._finalize(project, meta, severities, snippets, provenance)
        finally:
            shutil.rmtree(scan_root, ignore_errors=True)

    def _finalize(self, project, meta, severities, snippets, provenance) -> List[RawScoreRecord]:
        """Batched SLM inference + RawScoreRecord assembly (shared by both paths).

        Snippets are chunked into uniform windows (matching training and pipeline
        inference) before scoring, and per-chunk probabilities are aggregated
        (default: max) to produce a single score per sample.
        """
        print(f"[*] Running batched SLM inference on {len(snippets)} samples for '{project.project}' ...")
        slm_cfg = self.config.get("slm_inference", {})
        chunk_max_lines = slm_cfg.get("chunk_max_lines", 64)
        chunk_overlap = slm_cfg.get("chunk_overlap", 8)
        min_code_lines = slm_cfg.get("min_code_lines", 2)
        aggregation = slm_cfg.get("aggregation", "max")

        if snippets:
            per_snippet_chunks: List[list] = []
            flat_texts: List[str] = []
            for snippet in snippets:
                chunks = chunk_code(snippet or "", chunk_max_lines, chunk_overlap, min_code_lines)
                per_snippet_chunks.append(chunks)
                flat_texts.extend(c.text for c in chunks)

            flat_probs = self._get_model_manager().get_classifier_inference(flat_texts) if flat_texts else []

            slm_scores: List[float] = []
            cursor = 0
            for chunks in per_snippet_chunks:
                if not chunks:
                    slm_scores.append(0.0)
                    continue
                probs = flat_probs[cursor:cursor + len(chunks)]
                cursor += len(chunks)
                score = aggregate_chunk_scores([float(p) for p in probs], aggregation)
                slm_scores.append(round(float(score), 4))
        else:
            slm_scores = []

        records = []
        for sample, severity, slm_score in zip(meta, severities, slm_scores):
            records.append(RawScoreRecord(
                sample_id=sample.sample_id,
                project=project.project,
                file_path=sample.file_path,
                function_name=sample.function_name,
                label=sample.label,
                severity=severity,
                severity_weight=self.severity_map.get(severity, 0.0),
                slm_score=slm_score,
                vulnerability_type=sample.vulnerability_type,
                repo_url=sample.repo_url,
                commit_id=sample.commit_id,
                cve_id=sample.cve_id,
                cvss_score=sample.cvss_score,
                fixed_code=sample.fixed_code,
                diff_with_context=sample.diff_with_context,
                provenance=provenance,
            ))
        return records

    def extract(
        self, manifest_path: str, cache_path: str = None, force_recompute: bool = False,
        force_inline: bool = False,
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
            force_inline: If True, *git_source* projects are scored from
                their embedded `source_code` / `fixed_code` instead of
                cloning the real repo. This skips every `git clone` and is
                the recommended mode for fast iteration and offline CI (see
                `_extract_project`).

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
            all_records.extend(self._extract_project(project, force_inline=force_inline))

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in all_records], f, indent=2)
            print(f"[+] Cached {len(all_records)} raw score records to: {cache_path}")

        return all_records
