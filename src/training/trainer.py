"""trainer.py
=============
Fine-tunes `microsoft/codebert-base` on the enriched vulnerability dataset
using HuggingFace `Trainer`. Produces a single-label softmax classifier whose
output probability P(vulnerable) ∈ [0, 1] is directly comparable to the
existing SLM scores in `RawScoreRecord.slm_score`.

The resulting model directory can be dropped into `config.json` as the new
`models.classifier_model` and used by `ModelManager` with zero code changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from training.config import TrainingConfig  # noqa: E402
from training.dataset_builder import load_jsonl  # noqa: E402


# ── PyTorch Dataset ─────────────────────────────────────────────────────────

class VulnerabilityDataset(Dataset):
    """Tokenized JSONL dataset for vulnerability classification."""

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.texts, self.labels = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, list):
        logits = np.array(logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = float((preds == labels).mean())

    try:
        roc_auc = float(roc_auc_score(labels, probs[:, 1]))
        pr_auc = float(average_precision_score(labels, probs[:, 1]))
    except Exception:
        roc_auc = pr_auc = 0.0

    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
    }


# ── Main training loop ──────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    device = cfg.device
    use_cpu = device.type == "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    model.to(device)

    train_ds = VulnerabilityDataset(cfg.train_path, tokenizer, cfg.max_length)
    val_ds = VulnerabilityDataset(cfg.val_path, tokenizer, cfg.max_length)

    run_dir = cfg.run_dir
    model_dir = run_dir / "model"
    os.makedirs(str(model_dir), exist_ok=True)

    args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=cfg.seed,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        report_to="none",
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()

    # ── Persist artifacts ────────────────────────────────────────────────────
    tokenizer.save_pretrained(str(model_dir))
    trainer.save_model(str(model_dir))

    # Keep a stable `latest` symlink so `evaluate` / deployment can find the
    # freshest model without knowing the timestamped run directory.
    latest_link = Path(cfg.output_dir) / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        try:
            latest_link.unlink()
        except OSError:
            pass
    try:
        rel_target = os.path.relpath(run_dir, latest_link.parent)
        latest_link.symlink_to(rel_target, target_is_directory=True)
    except OSError:
        pass

    summary = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "base_model": cfg.base_model,
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "model_dir": str(model_dir),
    }

    summary_path = run_dir / "training_summary.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[+] Training complete. Best model -> {model_dir}")
    print(f"[+] Summary -> {summary_path}")
    return summary
