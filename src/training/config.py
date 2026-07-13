"""Training configuration and hyperparameters for the custom vulnerability classifier."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Centralized configuration for dataset building, training, and evaluation."""

    # ── Model ──────────────────────────────────────────────────────────────
    base_model: str = "microsoft/codebert-base"
    max_length: int = 512
    num_labels: int = 2

    # ── Training hyperparameters ───────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    # High default so training runs until early stopping (on val loss) halts it.
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    seed: int = 42
    gradient_accumulation_steps: int = 1
    # When True, freeze the CodeBERT backbone and train only the classifier
    # head — far more sample-efficient and stable for small datasets.
    freeze_backbone: bool = False
    # When True, allow training even if the dataset contains contradictory
    # (identical-text, opposite-label) samples. Off by default — such data
    # cannot be learned and wastes a full training run.
    allow_noisy_data: bool = False
    # Early stopping: halt once validation loss fails to improve for this many
    # consecutive evaluations. Best checkpoint (by val loss) is restored.
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # ── Data paths ─────────────────────────────────────────────────────────
    manifest_path: str = "benchmark_manifest_cvefixes.json"
    train_path: str = "training_data/train.jsonl"
    val_path: str = "training_data/validation.jsonl"
    test_path: str = "training_data/test.jsonl"
    output_dir: str = "training_output"

    # ── Dataset builder options ─────────────────────────────────────────────
    max_projects: int | None = None
    max_workers: int = 4
    include_cross_file: bool = False
    max_lines_per_context: int = 512
    max_samples_per_class: int | None = None
    max_samples_per_cwe: int | None = None
    max_total: int | None = None

    # ── Chunking (train on uniform code windows, not whole functions) ───────
    # Mirrors how the Stage-2 gate scores code at inference. Each chunk keeps
    # the function-level label. Set ``chunk_data=False`` to revert to
    # whole-function samples.
    chunk_data: bool = True
    chunk_max_lines: int = 64
    chunk_overlap: int = 8
    chunk_min_code_lines: int = 2

    # ── Verified-benign controls (safe-counterpart remedy) ────────────────
    # Path to a manifest produced by ``src/scripts/mine_benign_functions.py``.
    # When set, ``build-dataset`` merges genuine (label=0) functions that were
    # NOT touched by any vulnerability-fixing commit into the training pool, so
    # the classifier learns what *clean* code looks like (not just
    # pre-fix vs post-fix). See docs/SAFE_COUNTERPARTS.md.
    benign_manifest_path: str | None = None

    # ── Optional contrastive objective (safe-counterpart remedy, Step 2) ──
    # When enabled, a supervised-contrastive term is added on top of the
    # cross-entropy loss: a ``vulnerable`` function is pulled toward its
    # ``fixed`` twin and pushed from ``benign_control`` functions. This uses
    # the (vulnerable, fixed) pair as a *contrastive* signal instead of a
    # hard label=0 target, which is more robust to the noise a post-fix
    # function can carry. OFF by default — the standard CE objective is the
    # recommended/reported training setup; contrastive is for experiments.
    contrastive: bool = False
    contrastive_lambda: float = 0.1
    contrastive_temperature: float = 0.1

    # ── Split ratios ────────────────────────────────────────────────────────
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int = 42

    # Set by run_dir on first access; persisted for the lifetime of the
    # process so train -> evaluate (run-all) agree on the same output dir.
    _run_dir: Path | None = field(default=None, init=False, repr=False)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_dir = Path(self.output_dir) / f"run_{ts}"
        return self._run_dir

    @property
    def latest_dir(self) -> Path:
        """Stable path (a symlink maintained by the trainer) to the most
        recent run, so `evaluate` and deployment can locate the model without
        knowing the timestamped run directory."""
        return Path(self.output_dir) / "latest"
