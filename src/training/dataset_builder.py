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
* **Clone caching** -- a `.training_cache/clones/` directory persists
  `--filter=blob:none` clones across runs so only the first run pays the
  network cost.
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


def _get_cached_clone_dir(git_url: str, cache_root: Path) -> Optional[str]:
    url_hash = hashlib.md5(git_url.encode()).hexdigest()[:12]
    cache_path = cache_root / url_hash
    if cache_path.exists() and (cache_path / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(cache_path), "rev-parse", "--git-dir"],
                capture_output=True, check=True,
            )
            return str(cache_path)
        except subprocess.CalledProcessError:
            shutil.rmtree(cache_path, ignore_errors=True)
    return None


def _clone_or_reuse(git_url: str, ref, cache_root: Path) -> str:
    cached = _get_cached_clone_dir(git_url, cache_root)
    if cached is not None:
        return cached

    url_hash = hashlib.md5(git_url.encode()).hexdigest()[:12]
    dest = str(cache_root / url_hash)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

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
    )


def _process_project(
    project: ProjectManifest,
    cache_root: Path,
    include_cross_file: bool = False,
) -> List[EnrichedSample]:
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
        if dest_dir and os.path.exists(dest_dir):
            pass  # keep clone in cache; cleanup happens on cache eviction


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


def assign_splits(
    enriched: List[EnrichedSample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, str]:
    rng = __import__("random").Random(seed)
    projects = list({s.project for s in enriched})
    rng.shuffle(projects)

    n = len(projects)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)

    test_projects = set(projects[:n_test])
    val_projects = set(projects[n_test:n_test + n_val])
    train_projects = set(projects[n_test + n_val:])

    assignment: Dict[str, str] = {}
    for s in enriched:
        if s.project in test_projects:
            assignment[s.sample_id] = "test"
        elif s.project in val_projects:
            assignment[s.sample_id] = "validation"
        else:
            assignment[s.sample_id] = "train"
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
) -> Dict[str, Any]:
    """Builds an enriched, split training dataset from the benchmark manifest.

    Returns a summary dict with counts, CWE coverage, and split assignments.
    """
    cache_root = Path(".training_cache") / "clones"
    os.makedirs(str(cache_root), exist_ok=True)

    projects = load_manifest(manifest_path)
    projects = select_projects(projects, max_projects)
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

    for split_name, samples in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        save_jsonl(samples, path)
        print(f"[+] {split_name:12s}: {len(samples):4d} samples -> {path}")

    from collections import Counter
    cwe_counts = Counter(s.cwe for s in enriched)
    label_counts = Counter(s.label for s in enriched)

    summary = {
        "total_samples": len(enriched),
        "projects_processed": len(projects),
        "label_distribution": {"vulnerable": label_counts.get(1, 0), "safe": label_counts.get(0, 0)},
        "unique_cwe_types": len(cwe_counts),
        "cwe_coverage": dict(cwe_counts.most_common()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "avg_context_lines": round(
            sum(s.context_length for s in enriched) / len(enriched), 1
        ),
        "capped_from": cap_before,
        "capped_to": cap_after,
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Dataset summary -> {summary_path}")
    return summary
