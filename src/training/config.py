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
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    seed: int = 42
    gradient_accumulation_steps: int = 1

    # ── Data paths ─────────────────────────────────────────────────────────
    manifest_path: str = "benchmark_manifest_cvefixes.json"
    train_path: str = "training_data/train.jsonl"
    val_path: str = "training_data/val.jsonl"
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
