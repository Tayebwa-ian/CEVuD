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
    """
    git_url: str
    ref: Optional[str] = None


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
    """
    sample_id: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    label: int
    vulnerability_type: Optional[str] = None
    source_code: Optional[str] = None


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
        slm_score: CodeSheriff SLM classifier's vulnerability probability, [0, 1].
        vulnerability_type: Carried through from BenchmarkSample, for
            per-category reporting only.
        provenance: Repo/commit this sample was extracted from.
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
