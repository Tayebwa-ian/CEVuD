"""cli.py
========
Command-line interface for the custom vulnerability classifier training
pipeline.

Usage:
    python -m training.cli build-dataset
    python -m training.cli train
    python -m training.cli evaluate
    python -m training.cli run-all

All commands read from src/training/config.py and write artifacts under
training_output/.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from training.config import TrainingConfig  # noqa: E402
from training.dataset_builder import build_dataset  # noqa: E402
from training.trainer import train  # noqa: E402
from training.evaluator import evaluate  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.cli",
        description="CEVuD custom vulnerability classifier — dataset, training, and evaluation pipeline.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── build-dataset ───────────────────────────────────────────────────────
    bd = sub.add_parser("build-dataset", help="Enrich manifest and create train/val/test splits")
    bd.add_argument("--manifest", default=None, help="Path to benchmark_manifest_*.json")
    bd.add_argument("--output-dir", default=None, help="Output directory for JSONL splits")
    bd.add_argument("--max-projects", type=int, default=None, help="Max projects to process (None = all)")
    bd.add_argument("--max-workers", type=int, default=4, help="Parallel clone workers")
    bd.add_argument("--cross-file", action="store_true", help="Include cross-file context (slower)")
    bd.add_argument("--few-shot", action="store_true", help="Build a small balanced dataset (20 projects, 50 samples/class)")
    bd.add_argument("--max-samples-per-class", type=int, default=None, help="Max vulnerable + max safe samples")
    bd.add_argument("--max-samples-per-cwe", type=int, default=None, help="Max samples per CWE type")
    bd.add_argument("--max-total", type=int, default=None, help="Hard cap on total samples")
    bd.add_argument("--sample-cap-seed", type=int, default=42, help="Seed for sample capping shuffle")
    bd.add_argument("--keep-contradictory", action="store_true",
                    help="Keep (vuln, safe) pairs with identical text (NOT recommended).")
    bd.add_argument("--min-code-lines", type=int, default=2,
                    help="Drop enriched samples with fewer code-signal lines (default 2).")
    bd.add_argument("--no-chunk", action="store_true",
                    help="Disable chunking; train on whole functions instead of uniform windows.")
    bd.add_argument("--chunk-max-lines", type=int, default=64,
                    help="Max lines per training chunk (default 64).")
    bd.add_argument("--chunk-overlap", type=int, default=8,
                    help="Overlapping lines between consecutive chunks (default 8).")
    bd.add_argument("--chunk-min-code-lines", type=int, default=2,
                    help="Drop chunks with fewer code-signal lines (default 2).")
    bd.add_argument("--no-hunk-centering", action="store_true",
                    help="Disable hunk-centering; keep all chunks of a vulnerable "
                         "function instead of only the one(s) containing the sink.")
    bd.add_argument("--near-dup-threshold", type=float, default=0.75,
                    help="Drop safe chunks > this token-similar to a vulnerable "
                         "chunk in the same project (default 0.75; >=1.0 disables).")
    bd.add_argument("--contrastive", action="store_true",
                    help="Add a supervised-contrastive term (vulnerable vs fixed vs "
                         "benign_control). Experimental; see docs/SAFE_COUNTERPARTS.md.")
    bd.add_argument("--contrastive-lambda", type=float, default=0.1,
                    help="Weight of the contrastive term (default 0.1).")
    bd.add_argument("--contrastive-temperature", type=float, default=0.1,
                    help="Temperature of the contrastive term (default 0.1).")
    bd.add_argument("--benign-manifest", dest="benign_manifest", default=None,
                    help="Path to a benign-control manifest (mine_benign_functions.py).")

    # ── train ───────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fine-tune CodeBERT on the training split")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze CodeBERT, train only the classifier head (sample-efficient).")
    tr.add_argument("--early-stopping-patience", type=int, default=None,
                    help="Stop after N epochs without val-loss improvement (default 3).")
    tr.add_argument("--early-stopping-threshold", type=float, default=None,
                    help="Min val-loss improvement to count as progress (default 0.0).")
    tr.add_argument("--allow-noisy-data", action="store_true",
                    help="Train even if the dataset contains contradictory samples.")
    tr.add_argument("--focal-loss", action="store_true",
                    help="Use Focal Loss instead of cross-entropy (helps with class imbalance).")
    tr.add_argument("--focal-gamma", type=float, default=None,
                    help="Focal loss focusing parameter (default 2.0).")
    tr.add_argument("--focal-alpha", type=float, default=None,
                    help="Focal loss weight for vulnerable class (default 0.25).")

    # ── evaluate ────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Evaluate a trained model on the test split")
    ev.add_argument("--model-path", default=None, help="Path to trained model directory")
    ev.add_argument("--test-path", default=None, help="Path to test.jsonl")
    ev.add_argument("--output-dir", default=None, help="Where to save metrics and plots")

    # ── run-all ─────────────────────────────────────────────────────────────
    ra = sub.add_parser("run-all", help="Build dataset, train, and evaluate in one shot")
    ra.add_argument("--manifest", default=None)
    ra.add_argument("--max-projects", type=int, default=None)
    ra.add_argument("--max-workers", type=int, default=4)
    ra.add_argument("--cross-file", action="store_true")
    ra.add_argument("--few-shot", action="store_true")
    ra.add_argument("--max-samples-per-class", type=int, default=None)
    ra.add_argument("--max-samples-per-cwe", type=int, default=None)
    ra.add_argument("--max-total", type=int, default=None)
    ra.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze CodeBERT, train only the classifier head (sample-efficient).")
    ra.add_argument("--early-stopping-patience", type=int, default=None,
                    help="Stop after N epochs without val-loss improvement (default 3).")
    ra.add_argument("--early-stopping-threshold", type=float, default=None,
                    help="Min val-loss improvement to count as progress (default 0.0).")
    ra.add_argument("--keep-contradictory", action="store_true",
                    help="Keep (vuln, safe) pairs with identical text (NOT recommended).")
    ra.add_argument("--min-code-lines", type=int, default=2,
                    help="Drop enriched samples with fewer code-signal lines (default 2).")
    ra.add_argument("--no-chunk", action="store_true",
                    help="Disable chunking; train on whole functions instead of uniform windows.")
    ra.add_argument("--chunk-max-lines", type=int, default=64,
                    help="Max lines per training chunk (default 64).")
    ra.add_argument("--chunk-overlap", type=int, default=8,
                    help="Overlapping lines between consecutive chunks (default 8).")
    ra.add_argument("--chunk-min-code-lines", type=int, default=2,
                    help="Drop chunks with fewer code-signal lines (default 2).")
    ra.add_argument("--no-hunk-centering", action="store_true",
                    help="Disable hunk-centering (keep all chunks of a vulnerable function).")
    ra.add_argument("--near-dup-threshold", type=float, default=0.75,
                    help="Drop safe chunks > this token-similar to a vulnerable chunk (default 0.75).")
    ra.add_argument("--benign-manifest", dest="benign_manifest", default=None,
                    help="Path to a benign-control manifest (mine_benign_functions.py).")
    ra.add_argument("--contrastive", action="store_true",
                    help="Add a supervised-contrastive term (experimental).")
    ra.add_argument("--contrastive-lambda", type=float, default=0.1,
                    help="Weight of the contrastive term (default 0.1).")
    ra.add_argument("--contrastive-temperature", type=float, default=0.1,
                    help="Temperature of the contrastive term (default 0.1).")
    ra.add_argument("--allow-noisy-data", action="store_true",
                    help="Train even if the dataset contains contradictory samples.")
    ra.add_argument("--focal-loss", action="store_true",
                    help="Use Focal Loss instead of cross-entropy (helps with class imbalance).")
    ra.add_argument("--focal-gamma", type=float, default=None,
                    help="Focal loss focusing parameter (default 2.0).")
    ra.add_argument("--focal-alpha", type=float, default=None,
                    help="Focal loss weight for vulnerable class (default 0.25).")

    return p


def cmd_build_dataset(args, cfg: TrainingConfig) -> None:
    if args.few_shot:
        max_projects = args.max_projects if args.max_projects is not None else 20
        max_samples_per_class = 50
        max_total = 500
    else:
        max_projects = args.max_projects if args.max_projects is not None else cfg.max_projects
        max_samples_per_class = args.max_samples_per_class
        max_total = args.max_total

    build_dataset(
        manifest_path=args.manifest or cfg.manifest_path,
        output_dir=args.output_dir or "training_data",
        max_projects=max_projects,
        max_workers=args.max_workers or cfg.max_workers,
        include_cross_file=args.cross_file or cfg.include_cross_file,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_cwe=args.max_samples_per_cwe,
        max_total=max_total,
        sample_cap_seed=args.sample_cap_seed,
        keep_contradictory=args.keep_contradictory,
        min_code_lines=args.min_code_lines,
        chunk_data=not args.no_chunk,
        chunk_max_lines=args.chunk_max_lines,
        chunk_overlap=args.chunk_overlap,
        chunk_min_code_lines=args.chunk_min_code_lines,
        hunk_centered=not getattr(args, "no_hunk_centering", False),
        near_dup_threshold=getattr(args, "near_dup_threshold", 0.75),
        benign_manifest_path=args.benign_manifest or cfg.benign_manifest_path,
    )


def cmd_train(args, cfg: TrainingConfig) -> None:
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    cfg.freeze_backbone = args.freeze_backbone
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_threshold is not None:
        cfg.early_stopping_threshold = args.early_stopping_threshold
    cfg.allow_noisy_data = args.allow_noisy_data
    if getattr(args, "contrastive", False):
        cfg.contrastive = True
    if getattr(args, "contrastive_lambda", None) is not None:
        cfg.contrastive_lambda = args.contrastive_lambda
    if getattr(args, "contrastive_temperature", None) is not None:
        cfg.contrastive_temperature = args.contrastive_temperature
    if getattr(args, "focal_loss", False):
        cfg.use_focal_loss = True
    if getattr(args, "focal_gamma", None) is not None:
        cfg.focal_loss_gamma = args.focal_gamma
    if getattr(args, "focal_alpha", None) is not None:
        cfg.focal_loss_alpha = args.focal_alpha
    train(cfg)


def cmd_evaluate(args, cfg: TrainingConfig) -> None:
    model_path = args.model_path or str(cfg.latest_dir / "model")
    test_path = args.test_path or cfg.test_path
    out_dir = args.output_dir or str(cfg.latest_dir / "eval")

    best_ckpt = ""
    trainer_state_path = os.path.join(model_path, "checkpoint-*", "trainer_state.json")
    import glob
    state_files = sorted(glob.glob(trainer_state_path))
    if state_files:
        try:
            with open(state_files[-1], "r", encoding="utf-8") as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint", "")
            if best_ckpt and not os.path.isabs(best_ckpt):
                best_ckpt = os.path.join(os.path.dirname(model_path), best_ckpt)
        except Exception:
            pass

    evaluate(
        model_path=model_path,
        test_path=test_path,
        output_dir=out_dir,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        best_checkpoint_path=best_ckpt,
    )


def cmd_run_all(args, cfg: TrainingConfig) -> None:
    cmd_build_dataset(args, cfg)
    cmd_train(args, cfg)
    cmd_evaluate(args, cfg)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = TrainingConfig()

    dispatch = {
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "run-all": cmd_run_all,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
