"""
schema.py
=========
Shared, dependency-free dataclasses used across the evaluation suite.

Keeping these in one module (rather than passing raw dicts between files) means
every downstream module — grid search, sensitivity analysis, the linearity
check, the final report writer — agrees on exactly what fields exist and what
they mean. This is the single source of truth for the evaluation data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Repo provenance (used so that once a cloned repo's temp directory is
# deleted, we can still say exactly which repo/commit produced a result).
# ---------------------------------------------------------------------------

@dataclass
class GitRepoSource:
    """A project whose code must be cloned from a remote git URL.

    Attributes:
        git_url: Clonable URL, e.g. "https://github.com/org/repo.git".
        ref: Branch, tag, or commit SHA to check out. If None, the remote's
            default branch HEAD is used.
        shallow: If True (default), clone with --depth 1. Set False when
            the evaluation needs to reach arbitrary historical commits via
            `git show <sha>` (e.g. CVEfixes, which checks out a
            specific fix commit's parent to read the vulnerable code).
    """
    git_url: str
    ref: Optional[str] = None
    shallow: bool = True


@dataclass
class LocalRepoSource:
    """A project whose code is already present on disk (no cloning needed).

    Attributes:
        root_path: Absolute or relative path to the repository root.
    """
    root_path: str


@dataclass
class RepoProvenance:
    """Frozen record of exactly what code was analyzed, kept even after the
    temporary clone directory has been deleted. This is what lets the final
    report say "these results came from repo X at commit Y" without needing
    the checkout to still exist on disk.
    """
    project: str
    git_url: Optional[str]
    requested_ref: Optional[str]
    resolved_commit_sha: Optional[str]
    cloned_at_utc: Optional[str]


# ---------------------------------------------------------------------------
# Benchmark manifest schema (input to raw score extraction)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """One ground-truth labeled function/snippet inside a project.

    Attributes:
        sample_id: Globally unique identifier (e.g. "{project}::{function_name}::{start_line}").
        file_path: Path to the source file, relative to the project root.
        function_name: Name of the function/method under evaluation.
        start_line: 1-indexed start line of the function within file_path.
        end_line: 1-indexed (inclusive) end line of the function.
        label: Ground truth: 1 = vulnerable, 0 = safe.
        vulnerability_type: Optional free-text category (e.g. "SQLi", "SSRF"),
            used only for per-category reporting; not used by any decision logic.
        source_code: Optional cached copy of the function source. If omitted,
            it is read from disk (file_path within the project root) at
            extraction time.
        repo_url: Optional upstream repository URL (e.g. the GitHub repo a
            CVEfixes row came from). Carried through for provenance/traceability.
        commit_id: Optional VCS commit SHA the sample is associated with
            (e.g. the fix commit in CVEfixes). Required by downstream stages
            that want to trace a finding back to its exact upstream commit.
        target_commit: Optional explicit commit SHA to check out when reading
            this sample's source from a cloned repo. For a *vulnerable* sample
            this is the parent of `commit_id` (the pre-fix version); for a
            *safe* (post-fix) sample it is `commit_id` itself. When omitted,
            the extractor derives the parent of `commit_id` (the historical
            default for vulnerable-only manifests).
        cve_id: Optional CVE identifier (e.g. "CVE-2021-1234").
        cvss_score: Optional CVSS score for severity ranking/comparison.
        fixed_code: Optional patched version of `source_code` (for patch
            analysis / post-fix vulnerability comparison).
        diff_with_context: Optional unified diff with surrounding context.
        sample_subtype: Optional semantic sub-label that disambiguates *why*
            a sample carries its `label`. One of:
              - "vulnerable"  : the pre-fix (label=1) half of a CVEfixes pair.
              - "fixed"       : the post-fix (label=0) half of a CVEfixes pair
                              (the "safe counterpart" we are auditing).
              - "benign_control" : a function that was NOT touched by any
                              vulnerability-fixing commit — a *verified* safe
                              sample mined from the same repositories
                              (see src/scripts/mine_benign_functions.py and
                              docs/SAFE_COUNTERPARTS.md).
              - "benign"      : a function labeled safe by a corpus's own
                              annotation (e.g. VUDENC's all-zero-line
                              functions) rather than by our fix-commit logic.
            Defaults to None for backwards-compatible manifests. Carried through
            to the enriched training JSONL so the safe-counterpart methodology
            is auditable end-to-end.
    """
    sample_id: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    label: int
    vulnerability_type: Optional[str] = None
    source_code: Optional[str] = None
    repo_url: Optional[str] = None
    commit_id: Optional[str] = None
    target_commit: Optional[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    fixed_code: Optional[str] = None
    diff_with_context: Optional[str] = None
    sample_subtype: Optional[str] = None


@dataclass
class ProjectManifest:
    """One project's worth of labeled samples, plus where to find its code.

    Exactly one of `git_source` / `local_source` must be set.
    """
    project: str
    samples: List[BenchmarkSample]
    git_source: Optional[GitRepoSource] = None
    local_source: Optional[LocalRepoSource] = None

    def __post_init__(self) -> None:
        if bool(self.git_source) == bool(self.local_source):
            raise ValueError(
                f"ProjectManifest '{self.project}' must set exactly one of "
                f"git_source or local_source (got git_source={self.git_source!r}, "
                f"local_source={self.local_source!r})."
            )


# ---------------------------------------------------------------------------
# Output of raw score extraction (input to every downstream analysis step)
# ---------------------------------------------------------------------------

@dataclass
class RawScoreRecord:
    """The two numbers everything else in this package is built on, plus
    enough metadata to trace a result back to its source repo/commit and to
    break metrics down per project.

    Attributes:
        sample_id: Matches BenchmarkSample.sample_id.
        project: Project name (matches ProjectManifest.project).
        file_path: Relative file path within the project.
        function_name: Function name.
        label: Ground truth (1 = vulnerable, 0 = safe).
        severity: Raw Semgrep severity string ("ERROR", "WARNING", "INFO", "NONE").
        severity_weight: Semgrep severity mapped to [0, 1] via config's
            `semgrep_severity_map` (0.0 if Semgrep produced no finding
            overlapping this sample's line range).
        slm_score: local SLM classifier's vulnerability probability, [0, 1].
        vulnerability_type: Carried through from BenchmarkSample, for
            per-category reporting only.
        provenance: Repo/commit this sample was extracted from.
        repo_url: Upstream repository URL, carried through from
            BenchmarkSample (None for git_source projects whose URL lives
            in `provenance` instead).
        commit_id: VCS commit SHA associated with the sample (e.g. the fix
            commit in CVEfixes). Carried through so downstream stages can
            trace a finding back to its exact upstream commit.
        cve_id: CVE identifier, carried through for traceability.
        cvss_score: CVSS score, carried through for severity ranking.
        fixed_code: Patched version of source_code, carried through.
        diff_with_context: Unified diff with context, carried through.
    """
    sample_id: str
    project: str
    file_path: str
    function_name: str
    label: int
    severity: str
    severity_weight: float
    slm_score: float
    vulnerability_type: Optional[str]
    provenance: RepoProvenance
    repo_url: Optional[str] = None
    commit_id: Optional[str] = None
    target_commit: Optional[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    fixed_code: Optional[str] = None
    diff_with_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RawScoreRecord":
        prov = RepoProvenance(**d["provenance"])
        d = {**d, "provenance": prov}
        return RawScoreRecord(**d)


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

SplitName = str  # one of "train", "validation", "test"

VALID_SPLITS = ("train", "validation", "test")
