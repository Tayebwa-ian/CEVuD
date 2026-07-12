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

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def run_dir(self) -> Path:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.output_dir) / f"run_{ts}"
