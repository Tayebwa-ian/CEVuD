"""
dataset_splitter.py
====================
Produces a held-out validation split (used ONLY for grid search / logistic
regression fitting) and a held-out test split (used ONLY for final reported
numbers), so that "the results" and "the numbers used to choose the gate's
weights and threshold" are never the same data.

Two split strategies are supported:

    "by_project" (preferred whenever there are enough projects):
        Whole projects are assigned to train/validation/test. No project
        appears in more than one split. This is the stronger generalization
        test: it measures whether the gate (and its tuned weights) transfer
        to genuinely unseen codebases, not just unseen functions within
        already-seen codebases.

    "stratified" (fallback for small numbers of projects):
        Every project contributes samples to every split, stratified by
        label so each split keeps a similar vulnerable/safe ratio. Use this
        only when there are too few projects for "by_project" to leave a
        reasonable number of samples in validation/test.

Both are seeded for reproducibility.
"""

from __future__ import annotations

import random
from typing import List, Dict

from schema import RawScoreRecord, SplitName, VALID_SPLITS


def split_by_project(
    records: List[RawScoreRecord],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, SplitName]:
    """Assigns entire projects to train/validation/test.

    Args:
        records: All raw score records across all projects.
        val_frac: Target fraction of PROJECTS (not samples) assigned to validation.
        test_frac: Target fraction of PROJECTS assigned to test.
        seed: Random seed for reproducible project shuffling.

    Returns:
        Dict[str, SplitName]: sample_id -> "train" | "validation" | "test".

    Raises:
        ValueError: If there are fewer than 3 distinct projects (need at
            least one project per split).
    """
    projects = sorted({r.project for r in records})
    if len(projects) < 3:
        raise ValueError(
            f"split_by_project requires >= 3 distinct projects to guarantee "
            f"every split is non-empty; got {len(projects)}. Use "
            f"split_stratified instead for small project counts."
        )

    rng = random.Random(seed)
    shuffled = projects[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    # Guard against val+test consuming the entire project list.
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)

    test_projects = set(shuffled[:n_test])
    val_projects = set(shuffled[n_test:n_test + n_val])
    train_projects = set(shuffled[n_test + n_val:])

    assignment: Dict[str, SplitName] = {}
    for r in records:
        if r.project in test_projects:
            assignment[r.sample_id] = "test"
        elif r.project in val_projects:
            assignment[r.sample_id] = "validation"
        else:
            assignment[r.sample_id] = "train"

    print(
        f"[*] split_by_project: {len(train_projects)} train / {len(val_projects)} "
        f"validation / {len(test_projects)} test projects "
        f"(train={train_projects}, validation={val_projects}, test={test_projects})"
    )
    return assignment


def split_stratified(
    records: List[RawScoreRecord],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, SplitName]:
    """Assigns individual samples to splits, stratified by (project, label)
    so every split retains a similar class balance and every project appears
    in every split. Use only when too few projects exist for `split_by_project`.

    Note this is a WEAKER generalization test than `split_by_project`, since
    the model/gate may have "seen" (via tuning) other functions from the same
    project that end up in validation or test. Always prefer `split_by_project`
    when there are enough distinct projects; document which strategy was used
    in the final report (run_comparative_evaluation.py does this automatically).

    Args:
        records: All raw score records.
        val_frac: Target fraction of SAMPLES per (project, label) stratum assigned to validation.
        test_frac: Target fraction of SAMPLES per stratum assigned to test.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dict[str, SplitName]: sample_id -> "train" | "validation" | "test".
    """
    rng = random.Random(seed)
    strata: Dict[tuple, List[RawScoreRecord]] = {}
    for r in records:
        strata.setdefault((r.project, r.label), []).append(r)

    assignment: Dict[str, SplitName] = {}
    for key, group in strata.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        for i, r in enumerate(shuffled):
            if i < n_test:
                assignment[r.sample_id] = "test"
            elif i < n_test + n_val:
                assignment[r.sample_id] = "validation"
            else:
                assignment[r.sample_id] = "train"

    return assignment


def apply_split(
    records: List[RawScoreRecord], assignment: Dict[str, SplitName]
) -> Dict[SplitName, List[RawScoreRecord]]:
    """Partitions `records` into {"train": [...], "validation": [...], "test": [...]}
    according to a split assignment produced by either splitter above.
    """
    out: Dict[SplitName, List[RawScoreRecord]] = {s: [] for s in VALID_SPLITS}
    for r in records:
        split = assignment.get(r.sample_id)
        if split is None:
            raise KeyError(f"No split assignment found for sample_id={r.sample_id!r}")
        out[split].append(r)
    return out
