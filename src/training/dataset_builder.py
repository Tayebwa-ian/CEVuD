"""dataset_builder.py
=====================
Enriches benchmark_manifest_cvefixes.json samples with the FULL enclosing
function, module-level imports, and (optionally) cross-file context, then
writes stratified train / validation / test splits as JSONL files.

Design choices
--------------
* **Reuses existing codebase utilities** -- `code_context.expand_to_function`,
  `collect_module_imports`, and `build_context_snippet` ensure the training
  text is identical in format to what the production SLM sees during
  inference, eliminating train/inference skew.
* **Parallel cloning** -- a ThreadPoolExecutor clones repos concurrently,
  cutting wall-clock time on multi-core machines.
* **No clone persistence** -- each `--filter=blob:none` clone lives only for
  the duration of that project's enrichment and is deleted immediately
  afterwards, so no repository source is left on disk after the build.
* **Graceful degradation** -- if `git show` fails (rename, missing commit,
  network glitch), the builder falls back to the manifest's embedded
  `source_code` so the sample is not silently dropped.
* **Stratified splits** -- projects are assigned to splits to prevent
  data leakage; within splits, class balance is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Make evaluation utilities importable ───────────────────────────────────
_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, _SRC_DIR)

from code_context import (  # noqa: E402
    expand_to_function,
    collect_module_imports,
    build_context_snippet,
)
from benchmark_manifest import load_manifest  # noqa: E402
from repo_provider import clone_repo  # noqa: E402
from schema import BenchmarkSample, ProjectManifest  # noqa: E402
from data_quality import (  # noqa: E402
    code_signal_line_count,
    find_contradictions,
    normalize_code,
    tokenize_code,
    max_token_similarity,
)
from code_chunks import chunk_code  # noqa: E402


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class EnrichedSample:
    sample_id: str
    project: str
    text: str
    label: int
    vulnerability_type: str
    cwe: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    source_code_length: int
    context_length: int
    # Disambiguates *why* a sample carries its label (safe-counterpart
    # audit; see docs/SAFE_COUNTERPARTS.md). One of "vulnerable",
    # "fixed", "benign_control", "benign", or "unknown" (backwards-compat).
    sample_subtype: str = "unknown"
    # Chunking metadata (set when the sample is a uniform code window cut from
    # a larger function during build_dataset; -1 / 0 mean "whole function").
    chunk_index: int = -1
    chunk_start: int = 0
    chunk_end: int = 0
    # 1-based line offsets of the diff hunk (the changed / vulnerable lines)
    # WITHIN ``text``. 0/0 means "no hunk" (a safe/benign sample, or a
    # vulnerable sample whose file could not be read from git so line numbers
    # are unknown). Used to CENTER chunking on the sink instead of the function
    # start (see docs/SLM_CHUNKING.md), so a positive chunk always contains the
    # vulnerable statement.
    hunk_text_start: int = 0
    hunk_text_end: int = 0


# ── Low-level helpers ───────────────────────────────────────────────────────

_GIT_LOCK = threading.Lock()


def _git_show(repo_path: str, commit: str, rel_path: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "show", f"{commit}:{rel_path}"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    return None


def _get_parent_commit(repo_path: str, commit: str) -> Optional[str]:
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


def _clone_or_reuse(git_url: str, ref, cache_root: Path) -> str:
    """Clones ``git_url`` into a fresh, unique subdirectory of ``cache_root``.

    Each call gets its own directory (keyed by a URL hash *plus* a random
    suffix) so concurrent projects can never share — and therefore never
    delete — a checkout another thread is still reading from. The caller
    (``_process_project``) is responsible for deleting the directory once its
    samples have been enriched; clones are intentionally *not* reused or
    persisted across runs.
    """
    url_hash = hashlib.md5(git_url.encode()).hexdigest()[:12]
    dest = str(cache_root / f"{url_hash}_{uuid.uuid4().hex[:8]}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    clone_repo(git_url, ref=ref, dest_dir=dest, shallow=False)
    return dest


# ── Core enrichment ─────────────────────────────────────────────────────────

def _enrich_sample(
    repo_path: str,
    sample: BenchmarkSample,
    project_name: str,
    include_cross_file: bool = False,
) -> Optional[EnrichedSample]:
    if sample.label == 1:
        vuln_commit = _get_parent_commit(repo_path, sample.commit_id) or sample.commit_id
    else:
        vuln_commit = sample.target_commit or sample.commit_id

    content = _git_show(repo_path, vuln_commit, sample.file_path)
    fallback_used = False
    if content is None:
        content = sample.source_code or ""
        fallback_used = True
        if not content:
            return None

    imports = collect_module_imports(content)
    func_start, func_end = expand_to_function(content, sample.start_line)

    cross_file: Dict[str, str] = {}
    if include_cross_file and not fallback_used:
        try:
            from code_context import collect_cross_file_context
            cross_file = collect_cross_file_context(
                content,
                sample.file_path,
                repo_path,
                read_file=lambda rel: _git_show(repo_path, vuln_commit, rel),
                max_modules=3,
                max_lines_per_module=200,
            )
        except Exception:
            cross_file = {}

    text = build_context_snippet(content, (func_start, func_end), imports, cross_file)

    # ── Map the diff hunk (changed lines) into ``text`` line offsets ─────────
    # For a vulnerable sample, sample.start_line/end_line is the PRE-image
    # changed-line range read from the diff; ``content`` is the pre-fix file at
    # the parent commit, so those line numbers are valid here. We translate them
    # into 1-based line offsets inside the assembled snippet so the chunker can
    # keep the window that actually contains the sink. Only meaningful when we
    # read the real file (not the embedded-source fallback, whose line numbers
    # refer to the original file, not the snippet).
    hunk_text_start = hunk_text_end = 0
    if (
        sample.label == 1
        and not fallback_used
        and sample.start_line
        and sample.end_line
        and sample.start_line <= sample.end_line
    ):
        # Overlap of the hunk with the enclosing function (0-based within func).
        h_lo = max(func_start, int(sample.start_line))
        h_hi = min(func_end, int(sample.end_line))
        if h_lo <= h_hi:
            rel_start = h_lo - func_start
            rel_end = h_hi - func_start
            # Number of lines preceding the function body inside ``text``:
            # build_context_snippet prepends the imports block + one blank line.
            header_lines = (len(imports.splitlines()) + 1) if imports else 0
            n_text_lines = len(text.splitlines())
            hunk_text_start = min(header_lines + rel_start + 1, n_text_lines)
            hunk_text_end = min(header_lines + rel_end + 1, n_text_lines)

    return EnrichedSample(
        sample_id=sample.sample_id,
        project=project_name,
        text=text,
        label=sample.label,
        vulnerability_type=sample.vulnerability_type or "unknown",
        cwe=sample.vulnerability_type or "unknown",
        file_path=sample.file_path,
        function_name=sample.function_name,
        start_line=func_start,
        end_line=func_end,
        source_code_length=len(content.splitlines()),
        context_length=len(text.splitlines()),
        sample_subtype=sample.sample_subtype or "unknown",
        hunk_text_start=hunk_text_start,
        hunk_text_end=hunk_text_end,
    )


def _process_project(
    project: ProjectManifest,
    cache_root: Path,
    include_cross_file: bool = False,
) -> List[EnrichedSample]:
    # ── Benign-control fast path (no clone needed) ───────────────────────────
    # Benign projects (``benign::<repo>``) were already enriched by
    # mine_benign_functions.py — each sample's ``source_code`` is the final
    # imports+function snippet. Re-cloning the repo to re-read it would double
    # the build's clone count for no benefit (and the mined line numbers refer
    # to the original file, so re-expansion would be wrong). Use the embedded
    # snippet directly.
    if project.project.startswith("benign::") or project.local_source is not None:
        out: List[EnrichedSample] = []
        for sample in project.samples:
            text = sample.source_code or ""
            if not text:
                continue
            out.append(
                EnrichedSample(
                    sample_id=sample.sample_id,
                    project=project.project,
                    text=text,
                    label=sample.label,
                    vulnerability_type=sample.vulnerability_type or "benign",
                    cwe=sample.vulnerability_type or "benign",
                    file_path=sample.file_path,
                    function_name=sample.function_name,
                    start_line=sample.start_line,
                    end_line=sample.end_line,
                    source_code_length=len(text.splitlines()),
                    context_length=len(text.splitlines()),
                    sample_subtype=sample.sample_subtype or "benign_control",
                )
            )
        return out

    dest_dir = None
    try:
        dest_dir = _clone_or_reuse(
            project.git_source.git_url,
            project.git_source.ref,
            cache_root,
        )
        samples: List[EnrichedSample] = []
        for sample in project.samples:
            try:
                enriched = _enrich_sample(dest_dir, sample, project.project, include_cross_file)
                if enriched is not None:
                    samples.append(enriched)
            except Exception as exc:
                print(f"[!]  sample {sample.sample_id}: {exc}")
        return samples
    except Exception as exc:
        print(f"[!]  project {project.project}: {exc}")
        return []
    finally:
        # Delete the clone immediately after extracting the needed information.
        # The enriched samples already carry the full function, imports, and
        # (optionally) cross-file context, so the on-disk checkout is no longer
        # required and must not persist on disk.
        if dest_dir and os.path.exists(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)


# ── Public API ──────────────────────────────────────────────────────────────

def select_projects(
    projects: List[ProjectManifest],
    max_projects: Optional[int] = None,
) -> List[ProjectManifest]:
    if max_projects is None or max_projects >= len(projects):
        return projects

    from collections import defaultdict
    cwe_to_projects = defaultdict(set)
    project_list = list(projects)
    for p in project_list:
        for s in p.samples:
            cwe_to_projects[s.vulnerability_type or "unknown"].add(p.project)

    all_cwes = set(cwe_to_projects.keys())
    selected = set()
    remaining = set(all_cwes)

    project_list_sorted = sorted(
        project_list, key=lambda p: len(p.samples), reverse=True
    )
    for p in project_list_sorted:
        if len(selected) >= max_projects:
            break
        proj_cwes = {
            cwe for cwe, projs in cwe_to_projects.items() if p.project in projs
        }
        if remaining & proj_cwes:
            selected.add(p.project)
            remaining -= proj_cwes
        elif p.project not in selected:
            selected.add(p.project)

    project_map = {p.project: p for p in project_list}
    return [project_map[name] for name in selected if name in project_map]


def _split_key(project_name: str) -> str:
    """Collapse a benign-control project (``benign::<repo>``) onto its
    underlying repo so a repo's *vulnerable* samples and its *mined safe*
    samples are always assigned to the SAME split.

    Without this, ``benign::keylime`` and ``keylime`` are distinct project names
    and the by-project split could place a repo's safe code in train and its
    vulnerable code in test — repo-level leakage, and (at small scale) splits
    that contain only one class. Grouping by repo keeps every split
    project-disjoint *and* two-class. See docs/SAFE_COUNTERPARTS.md §6.
    """
    return project_name[len("benign::"):] if project_name.startswith("benign::") else project_name


def _allocate_class(
    keys: List[str],
    assigned: Dict[str, List[str]],
    targets: Dict[str, int],
    rng: "random.Random",
) -> None:
    """Place ``keys`` (all of one class) across splits so each split receives a
    count proportional to its target size (max-deficit greedy). Allocating the
    vulnerable pool and the safe-only pool *separately* guarantees every split
    gets a share of BOTH classes — so the largest split can no longer swallow
    the whole majority class and leave itself single-class (the previous bug,
    where 'train' ended up 100% vulnerable)."""
    if not keys:
        return
    rng.shuffle(keys)
    total_target = sum(targets.values())
    desired = {sp: targets[sp] * len(keys) / total_target for sp in targets}
    have = {sp: 0 for sp in targets}
    for k in keys:
        best = max(targets, key=lambda sp: desired[sp] - have[sp])
        assigned[best].append(k)
        have[best] += 1


def assign_splits(
    enriched: List[EnrichedSample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, str]:
    rng = __import__("random").Random(seed)
    # Split by REPO key (benign::X grouped with X), not raw project name, so a
    # repo's vulnerable + mined-safe samples never straddle the split boundary.
    keys = list({_split_key(s.project) for s in enriched})

    # Split each CLASS independently and proportionally. A repo key may be
    # "safe-only" if its only vulnerable sample was dropped as low-signal; by
    # allocating vulnerable repo-keys and safe-only repo-keys as two SEPARATE
    # pools (each spread across splits by target weight), every split receives
    # BOTH a vulnerable repo and a safe repo -> a guaranteed two-class split,
    # while the overall 60/20/20 (train/val/test) ratio is preserved.
    has_vuln: set = {_split_key(s.project) for s in enriched if s.label == 1}
    vuln_keys = [k for k in keys if k in has_vuln]
    safe_keys = [k for k in keys if k not in has_vuln]
    rng.shuffle(vuln_keys)
    rng.shuffle(safe_keys)

    n = len(keys)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)
    n_train = n - n_test - n_val
    targets = {"train": n_train, "validation": n_val, "test": n_test}
    order = ["train", "validation", "test"]

    assigned: Dict[str, List[str]] = {sp: [] for sp in order}
    _allocate_class(vuln_keys, assigned, targets, rng)
    _allocate_class(safe_keys, assigned, targets, rng)

    assignment: Dict[str, str] = {}
    for s in enriched:
        k = _split_key(s.project)
        for sp, ks in assigned.items():
            if k in ks:
                assignment[s.sample_id] = sp
                break
    return assignment


def _cap_samples(
    samples: List[EnrichedSample],
    max_per_class: Optional[int] = None,
    max_per_cwe: Optional[int] = None,
    max_total: Optional[int] = None,
    seed: int = 42,
) -> List[EnrichedSample]:
    rng = __import__("random").Random(seed)
    capped = list(samples)

    if max_total is not None and len(capped) > max_total:
        rng.shuffle(capped)
        capped = capped[:max_total]

    if max_per_cwe is not None:
        by_cwe: Dict[str, List[EnrichedSample]] = {}
        for s in capped:
            by_cwe.setdefault(s.cwe, []).append(s)
        capped = []
        for cwe, group in by_cwe.items():
            rng.shuffle(group)
            capped.extend(group[:max_per_cwe])

    if max_per_class is not None:
        by_label: Dict[int, List[EnrichedSample]] = {}
        for s in capped:
            by_label.setdefault(s.label, []).append(s)
        capped = []
        for label, group in by_label.items():
            rng.shuffle(group)
            capped.extend(group[:max_per_class])

    return capped


def save_jsonl(samples: List[EnrichedSample], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    return texts, labels


def _drop_contradictory(
    enriched: List[EnrichedSample],
    keep_contradictory: bool,
    min_code_lines: int,
) -> "tuple[List[EnrichedSample], int, int]":
    """Defensive data-quality pass over already-enriched samples.

    Returns ``(kept, n_contradiction, n_low_signal)``.

    * ``n_contradiction``: samples whose ``text`` is byte-identical (ignoring
      whitespace) to a sample of the OPPOSITE label — a hard contradiction the
      classifier cannot learn. Always dropped unless ``keep_contradictory``.
    * ``n_low_signal``: samples whose enriched ``text`` carries fewer than
      ``min_code_lines`` lines of real code signal (comments/docstrings/version
      assignments don't count) — e.g. a snippet that expanded to a lone
      ``__version__ = '3.7'`` line.
    """
    if keep_contradictory:
        contradictions: set = set()
    else:
        contradictions = set(
            find_contradictions((s.text, s.label) for s in enriched)
        )

    kept: List[EnrichedSample] = []
    n_contra = n_short = 0
    for s in enriched:
        norm = normalize_code(s.text)
        if norm in contradictions:
            n_contra += 1
            continue
        if code_signal_line_count(s.text) < min_code_lines:
            n_short += 1
            continue
        kept.append(s)
    return kept, n_contra, n_short


def _drop_near_duplicate_safe(
    chunked: List[EnrichedSample], threshold: float
) -> "tuple[List[EnrichedSample], int]":
    """Drop every ``label=0`` chunk that is more than ``threshold`` token-similar
    to some ``label=1`` chunk **in the same project**.

    This is the near-duplicate complement of the exact-contradiction pass: it
    stops a lightly-edited copy of a vulnerable function (e.g. a post-fix twin
    that survived the miner, or a sibling that differs by one line) from
    entering the safe class, which would otherwise re-create the "near-identical
    text, opposite label" collapse. The comparison is scoped per project (the
    only place a genuine twin can occur) and uses the length short-circuits in
    ``data_quality.max_token_similarity`` to stay tractable.
    """
    from collections import defaultdict

    vuln_tokens_by_project: Dict[str, List[List[str]]] = defaultdict(list)
    for c in chunked:
        if c.label == 1:
            toks = tokenize_code(c.text)
            if toks:
                vuln_tokens_by_project[c.project].append(toks)

    kept: List[EnrichedSample] = []
    dropped = 0
    for c in chunked:
        if c.label == 0:
            refs = vuln_tokens_by_project.get(c.project)
            if refs and max_token_similarity(c.text, refs) > threshold:
                dropped += 1
                continue
        kept.append(c)
    return kept, dropped


def build_dataset(
    manifest_path: str,
    output_dir: str = "training_data",
    max_projects: Optional[int] = None,
    max_workers: int = 4,
    include_cross_file: bool = False,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_seed: int = 42,
    max_samples_per_class: Optional[int] = None,
    max_samples_per_cwe: Optional[int] = None,
    max_total: Optional[int] = None,
    sample_cap_seed: int = 42,
    keep_contradictory: bool = False,
    min_code_lines: int = 2,
    # ── Chunking ─────────────────────────────────────────────────────────────
    # When ``chunk_data`` is True (default), every enriched function is cut into
    # uniform code windows of ``chunk_max_lines`` (with ``chunk_overlap``) so the
    # classifier is trained on inputs that fit its 512-token context — matching
    # how the Stage-2 gate scores code at inference. Each chunk inherits the
    # function-level label. Chunks with < ``chunk_min_code_lines`` of real code
    # signal are dropped, and (vuln, safe) chunk pairs with identical normalized
    # text are removed as hard contradictions.
    chunk_data: bool = True,
    chunk_max_lines: int = 64,
    chunk_overlap: int = 8,
    chunk_min_code_lines: int = 2,
    # ── Hunk-centering + near-duplicate guard (safe-counterpart remedy) ───────
    # ``hunk_centered``: for a vulnerable sample, keep only the chunk(s) that
    # overlap the diff hunk (the changed / vulnerable lines) so the positive
    # signal always contains the sink. ``near_dup_threshold``: drop any safe
    # chunk that is more than this token-similar to a vulnerable chunk in the
    # same project (0 or >=1.0 disables the guard). See docs/SAFE_COUNTERPARTS.md.
    hunk_centered: bool = True,
    near_dup_threshold: float = 0.75,
    # ── Verified-benign controls (safe-counterpart remedy) ───────────────────
    # When ``benign_manifest_path`` points at a manifest produced by
    # ``src/scripts/mine_benign_functions.py``, its projects are merged into the
    # training pool *before* project selection / splitting. Every benign control is
    # a ``label=0`` function that was NOT touched by any vulnerability-fixing
    # commit, so it is a genuine safe sample (``sample_subtype="benign_control"``)
    # rather than "the function after a fix". Benign projects keep their own
    # project identity, so the project-level split still prevents leakage, and they
    # are counted as label=0 in the usual few-shot caps. See docs/SAFE_COUNTERPARTS.md.
    benign_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds an enriched, split training dataset from the benchmark manifest.

    Returns a summary dict with counts, CWE coverage, and split assignments.
    """
    cache_root = Path(".training_cache") / "clones"
    os.makedirs(str(cache_root), exist_ok=True)

    projects = load_manifest(manifest_path)
    benign_projects: List = []
    if benign_manifest_path:
        try:
            benign_projects = load_manifest(benign_manifest_path)
            print(f"[*] Loaded {len(benign_projects)} benign-control project(s) "
                  f"from {benign_manifest_path}")
        except Exception as exc:
            print(f"[!] Could not load benign manifest '{benign_manifest_path}': {exc}")

    # Keep benign controls SEPARATE from project selection. `select_projects`
    # chooses by CWE coverage / `--max-projects`; benign controls are
    # tagged vulnerability_type="benign", which would give them almost no
    # coverage weight and let `--few-shot` silently DROP them — defeating
    # the whole safe-counterpart remedy. They always keep their own
    # project identity, so the project-level split still prevents leakage,
    # and they are always included in the training pool regardless of
    # how many primary (vulnerable) projects are selected.
    primary = [p for p in projects if not p.project.startswith("benign::")]
    benign_from_manifest = [p for p in projects if p.project.startswith("benign::")]
    benign_projects = benign_projects + benign_from_manifest
    selected_primary = select_projects(primary, max_projects)
    projects = selected_primary + benign_projects
    if benign_projects:
        print(f"[*] Always-include {len(benign_projects)} benign-control "
              f"project(s) (not subject to CWE-coverage selection).")
    print(f"[*] Building dataset from {len(projects)} projects ...")

    enriched: List[EnrichedSample] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_project, p, cache_root, include_cross_file): p
            for p in projects
        }
        for future in as_completed(futures):
            samples = future.result()
            enriched.extend(samples)

    if not enriched:
        raise RuntimeError("No samples were enriched. Check network/git access.")

    # ── Data-quality filter: drop hard contradictions + low-signal snippets ──
    # Even with clean converters, the full-function expansion in
    # `_enrich_sample` can still produce byte-identical (vuln, safe) pairs or
    # snippets that expanded to a single version/comment line. Drop them so the
    # classifier never trains on contradictory or signal-free text.
    before_filter = len(enriched)
    enriched, n_contra, n_short = _drop_contradictory(
        enriched, keep_contradictory, min_code_lines
    )
    dropped_total = before_filter - len(enriched)
    if dropped_total:
        print(
            f"[*] Data-quality filter: {dropped_total} dropped "
            f"({n_contra} contradictory, {n_short} low-signal). "
            f"{len(enriched)} remain."
        )

    # ── Chunking: train on uniform code windows, not whole functions ──────────
    # CodeBERT is capped at 512 tokens. Feeding whole functions silently
    # truncates the vulnerable code; feeding uniform chunks keeps every input
    # inside the context window and matches how the Stage-2 gate scores code at
    # inference time (no train/inference skew). See docs/SLM_CHUNKING.md.
    #
    # Hunk-centering (``hunk_centered``): for a VULNERABLE sample we keep only
    # the chunk(s) that overlap the diff hunk (the changed / vulnerable lines).
    # This fixes the "~50% of positive chunks contain no sink" problem — the old
    # code chunked from the function start and often labeled a sink-free window
    # as vulnerable. A vulnerable sample whose hunk maps to no surviving chunk
    # (e.g. the sink was a comment/blank line dropped by min_code_lines) is
    # dropped rather than kept as a mislabeled positive.
    n_sink_dropped = 0
    n_near_dup = 0
    if chunk_data and chunk_max_lines:
        chunked: List[EnrichedSample] = []
        for s in enriched:
            raw_chunks = chunk_code(
                s.text, chunk_max_lines, chunk_overlap, chunk_min_code_lines
            )
            if (
                hunk_centered
                and s.label == 1
                and s.hunk_text_start > 0
                and len(raw_chunks) > 1
            ):
                focused = [
                    c for c in raw_chunks
                    if c.start_line <= s.hunk_text_end and c.end_line >= s.hunk_text_start
                ]
                if not focused:
                    # The sink is not contained in any surviving chunk — drop
                    # this positive rather than emit a sink-free "vulnerable".
                    n_sink_dropped += 1
                    continue
                raw_chunks = focused
            for i, c in enumerate(raw_chunks):
                chunked.append(
                    EnrichedSample(
                        sample_id=f"{s.sample_id}::c{i}",
                        project=s.project,
                        text=c.text,
                        label=s.label,
                        vulnerability_type=s.vulnerability_type,
                        cwe=s.cwe,
                        file_path=s.file_path,
                        function_name=s.function_name,
                        start_line=s.start_line,
                        end_line=s.end_line,
                        source_code_length=len(c.text.splitlines()),
                        context_length=len(c.text.splitlines()),
                        sample_subtype=s.sample_subtype,
                        chunk_index=i,
                        chunk_start=c.start_line,
                        chunk_end=c.end_line,
                    )
                )
        if n_sink_dropped:
            print(f"[*] Hunk-centering: {n_sink_dropped} vulnerable samples dropped "
                  f"(sink not contained in any chunk).")
        # Chunk-level contradiction/dedup pass: a (vuln, safe) chunk pair with
        # identical normalized text is a hard contradiction — drop it.
        if not keep_contradictory:
            contradictions = set(find_contradictions((c.text, c.label) for c in chunked))
            kept = [c for c in chunked if normalize_code(c.text) not in contradictions]
            n_chunk_contra = len(chunked) - len(kept)
            if n_chunk_contra:
                print(f"[*] Chunk filter: {n_chunk_contra} contradictory chunks dropped.")
            chunked = kept
        # Near-duplicate guard: drop any SAFE chunk that is >near_dup_threshold
        # token-similar to a VULNERABLE chunk in the same project. This is the
        # near-duplicate complement of the exact-contradiction pass above — it
        # keeps a lightly-edited copy of a vulnerable function from sneaking into
        # the safe class and collapsing training to P=0.5.
        if not keep_contradictory and near_dup_threshold and near_dup_threshold < 1.0:
            chunked, n_near_dup = _drop_near_duplicate_safe(chunked, near_dup_threshold)
            if n_near_dup:
                print(f"[*] Near-duplicate guard: {n_near_dup} safe chunks dropped "
                      f"(> {near_dup_threshold:.2f} similar to a vulnerable chunk).")
        n_before_chunk = len(enriched)
        enriched = chunked
        print(
            f"[*] Chunking: {n_before_chunk} functions -> {len(enriched)} chunks "
            f"(max_lines={chunk_max_lines}, overlap={chunk_overlap})."
        )
    else:
        n_near_dup = 0

    # ── Few-shot caps (applied BEFORE split to keep class balance) ─────────────
    cap_before = len(enriched)
    enriched = _cap_samples(
        enriched,
        max_per_class=max_samples_per_class,
        max_per_cwe=max_samples_per_cwe,
        max_total=max_total,
        seed=sample_cap_seed,
    )
    cap_after = len(enriched)
    if cap_before != cap_after:
        print(f"[*] Capped samples: {cap_before} -> {cap_after}")

    assignment = assign_splits(enriched, val_fraction, test_fraction, split_seed)

    splits: Dict[str, List[EnrichedSample]] = {"train": [], "validation": [], "test": []}
    for s in enriched:
        split_name = assignment[s.sample_id]
        splits[split_name].append(s)

    # ── Single-class split guard ─────────────────────────────────────────────
    # A split with only one class makes ROC/PR undefined and lets the model
    # "win" by always predicting the majority — a silent, useless eval. Fail
    # loudly instead of producing a degenerate dataset. This happens if
    # ``benign::<repo>`` and ``<repo>`` were split apart (fixed by ``_split_key``
    # + the stratified seeding in assign_splits), OR if every vulnerable sample
    # in a split's repos was dropped upstream (e.g. low-signal filter). Either
    # way a single-class split must not be trained/evaluated on.
    for split_name, samples in splits.items():
        labels = {s.label for s in samples}
        if len(labels) < 2:
            present = "safe" if 0 in labels else "vulnerable"
            raise ValueError(
                f"Split '{split_name}' contains ONLY {present} samples "
                f"({len(samples)} total). Every split must be two-class for a "
                f"meaningful ROC/PR. This means either benign::<repo> and <repo> "
                f"were split apart (ensure assign_splits collapses the 'benign::' "
                f"prefix), or all vulnerable samples in this split's repos were "
                f"filtered out upstream (reduce low-signal dropping or add more "
                f"vulnerable repos)."
            )

    for split_name, samples in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        save_jsonl(samples, path)
        print(f"[+] {split_name:12s}: {len(samples):4d} samples -> {path}")

    from collections import Counter
    cwe_counts = Counter(s.cwe for s in enriched)
    label_counts = Counter(s.label for s in enriched)
    subtype_counts = Counter(s.sample_subtype for s in enriched)

    summary = {
        "total_samples": len(enriched),
        "projects_processed": len(projects),
        "label_distribution": {"vulnerable": label_counts.get(1, 0), "safe": label_counts.get(0, 0)},
        "sample_subtypes": dict(subtype_counts),
        "unique_cwe_types": len(cwe_counts),
        "cwe_coverage": dict(cwe_counts.most_common()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "avg_context_lines": round(
            sum(s.context_length for s in enriched) / len(enriched), 1
        ),
        "capped_from": cap_before,
        "capped_to": cap_after,
        "dropped_contradictory": n_contra,
        "dropped_low_signal": n_short,
        "dropped_total": dropped_total,
        "chunking": {
            "enabled": bool(chunk_data and chunk_max_lines),
            "max_lines": chunk_max_lines,
            "overlap": chunk_overlap,
            "min_code_lines": chunk_min_code_lines,
            "chunks_total": len(enriched),
            "hunk_centered": bool(chunk_data and chunk_max_lines and hunk_centered),
            "sink_dropped": n_sink_dropped,
            "near_dup_threshold": near_dup_threshold,
            "near_dup_dropped": n_near_dup,
        },
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Dataset summary -> {summary_path}")
    return summary
