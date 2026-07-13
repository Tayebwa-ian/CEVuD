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

import collections
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
from data_quality import find_contradictions  # noqa: E402


# ── PyTorch Dataset ─────────────────────────────────────────────────────────

class VulnerabilityDataset(Dataset):
    """Tokenized JSONL dataset for vulnerability classification.

    Also reads the optional ``sample_subtype`` field so the (optional)
    contrastive training mode can reason about *why* a sample is safe
    (vulnerable / fixed / benign_control) — see docs/SAFE_COUNTERPARTS.md.
    """

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.texts, self.labels = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subtypes: List[str] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.subtypes.append(obj.get("sample_subtype", "unknown"))
        except Exception:
            self.subtypes = ["unknown"] * len(self.texts)

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


# ── Loss / Trainer ────────────────────────────────────────────────────────────

def _supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Supervised contrastive loss over [CLS] embeddings.

    Positives are same-label samples in the batch; negatives are different-label
    samples. Returns the mean per-anchor ``-log(sum(pos)/sum(all≠self))``,
    skipping anchors that have no same-label partner (avoids ``log(0)``).

    This is the *optional* second term of the safe-counterpart training
    objective (docs/SAFE_COUNTERPARTS.md, Step 2): it teaches the encoder
    that a ``vulnerable`` function should sit near its ``fixed`` twin and far
    from a ``benign_control`` function — using the post-fix pair as a
    *contrastive* signal rather than as a hard ``label=0`` target, which is
    more robust to the noise the post-fix function can carry. Disabled by
    default (``contrastive=False``); the standard CE objective is unchanged.
    """
    device = embeddings.device
    emb = torch.nn.functional.normalize(embeddings, dim=-1)
    n = emb.size(0)
    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim = torch.matmul(emb, emb.T) / max(temperature, 1e-4)
    labels = labels.view(-1)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    all_mask = ~eye

    exp_sim = torch.exp(sim)
    pos = (exp_sim * pos_mask).sum(dim=1)
    denom = (exp_sim * all_mask).sum(dim=1).clamp(min=1e-8)

    has_pos = pos_mask.sum(dim=1) > 0
    per_anchor = -torch.log((pos / denom).clamp(min=1e-8) + 1e-8)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return per_anchor[has_pos].mean()


class WeightedTrainer(Trainer):
    """`Trainer` variant that applies per-class weights to the cross-entropy
    loss and optionally trains only the classifier head (frozen backbone).

    Vulnerability datasets are typically class-imbalanced; weighting the loss
    prevents the model from collapsing toward the majority class and gives the
    minority (vulnerable) class a stronger gradient signal.

    When ``contrastive=True`` an optional supervised-contrastive term is added
    on top of the cross-entropy loss (weighted by ``contrastive_lambda``), so
    the post-fix / benign-control pairing is exploited as a *contrastive*
    signal. The standard CE objective is used unchanged when the flag is off.
    """

    def __init__(
        self, *args, class_weights=None, freeze_backbone=False,
        contrastive: bool = False, contrastive_lambda: float = 0.1,
        contrastive_temperature: float = 0.1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.freeze_backbone = freeze_backbone
        self.contrastive = contrastive
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        if self.contrastive:
            # Second forward pass to obtain [CLS] embeddings for the
            # supervised-contrastive term. This adds one extra encoder pass
            # per batch (only when the contrastive objective is enabled).
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states
            cls_emb = hidden[-1][:, 0]  # (batch, hidden)
            supcon = _supervised_contrastive_loss(
                cls_emb, labels, self.contrastive_temperature
            )
            loss = loss + self.contrastive_lambda * supcon

        return (loss, outputs) if return_outputs else loss


def _compute_class_weights(labels, num_labels: int) -> torch.Tensor:
    counts = collections.Counter(labels)
    total = max(1, sum(counts.values()))
    # Inverse-frequency weighting normalised so the average weight is 1.0.
    weights = [
        total / (num_labels * max(1, counts.get(i, 0)))
        for i in range(num_labels)
    ]
    return torch.tensor(weights, dtype=torch.float)


def _check_training_data_quality(train_ds: "VulnerabilityDataset", allow_noisy: bool) -> None:
    """Pre-flight guard: count texts that appear with BOTH labels (hard
    contradictions). Training on such data is hopeless — the loss plateaus at
    ``ln(2)`` and ROC-AUC stays at ~0.5. Refuse by default so a full run isn't
    wasted; pass ``allow_noisy`` to override (e.g. when debugging the filter)."""
    contradictions = find_contradictions(zip(train_ds.texts, train_ds.labels))
    n = len(contradictions)
    total = len(train_ds)
    if n == 0:
        print("[*] Data-quality pre-flight: no contradictory (identical-text, "
              "opposite-label) samples found.")
        return
    frac = (n / total) if total else 0.0
    print(
        f"[!] Data-quality pre-flight: {n} normalized texts ({frac:.1%} of "
        f"training samples) appear with BOTH labels (hard contradictions)."
    )
    if frac >= 0.05 and not allow_noisy:
        raise RuntimeError(
            "Refusing to train on a dataset with >=5% contradictory samples.\n"
            "  -> Rebuild the manifest with the noise/trivial filters in\n"
            "     src/scripts/convert_cvefixes.py / convert_vudenc.py, then run\n"
            "     `build-dataset` again. Or pass --allow-noisy-data to override."
        )
    print("[!] Training will likely fail to learn (loss ~ln(2), ROC-AUC ~0.5). "
          "Rebuild the data, or pass --allow-noisy-data to proceed anyway.")

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

    # Data-quality pre-flight: refuse to waste a run on contradictory data.
    _check_training_data_quality(train_ds, cfg.allow_noisy_data)

    # Per-class weights from the training distribution to counter imbalance.
    class_weights = _compute_class_weights(train_ds.labels, cfg.num_labels)
    print(f"[*] Class weights (vuln=1 vs safe=0): {class_weights.tolist()}")

    # Frozen-backbone mode: only the classifier head is trained on top of
    # frozen CodeBERT embeddings. Far more sample-efficient and stable for
    # small datasets, and much faster to train.
    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
        print("[*] Frozen backbone — training only the classifier head.")

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
        # Select / restore the best checkpoint by validation loss, and stop
        # training once val loss stops improving (see EarlyStoppingCallback).
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.seed,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        report_to="none",
        use_cpu=use_cpu,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        freeze_backbone=cfg.freeze_backbone,
        contrastive=cfg.contrastive,
        contrastive_lambda=cfg.contrastive_lambda,
        contrastive_temperature=cfg.contrastive_temperature,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        ],
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
