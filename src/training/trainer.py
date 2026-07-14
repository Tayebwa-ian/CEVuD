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
from typing import Dict, Any, List

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
from data_quality import find_contradictions, count_cross_label_near_duplicates  # noqa: E402


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

def _describe_el(x) -> str:
    """Short human-readable description of a predictions element."""
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    return type(x).__name__


def _collect_2d(preds, out: list) -> None:
    """Recursively gather every 2D (b, num_labels) numeric array hidden inside
    the Trainer's `predictions`, no matter how deeply it is nested (the Trainer
    sometimes returns ``(batch0_array, (batch1, batch2, ...))`` or per-batch
    tuples). 3D tensors are pooled to 2D; 1D/0D leaves are dropped (they are
    scalars, not logits)."""
    if preds is None:
        return
    if isinstance(preds, (list, tuple)):
        for a in preds:
            _collect_2d(a, out)
        return
    # Tensor or ndarray.
    if hasattr(preds, "detach"):  # torch.Tensor
        arr = preds.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(preds, dtype=np.float32)
        except Exception:
            return
    if arr.ndim == 0:
        return
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:  # (b, seq, c) -> pool over the sequence axis
        arr = arr.mean(axis=1)
    if arr.ndim == 2:
        out.append(arr.astype(np.float32, copy=False))


def _extract_logits(preds):
    """Coerce the Trainer's `eval_pred.predictions` into a single 2D float32
    array of shape (N, num_labels). Flattens arbitrarily nested batch outputs
    and keeps only the columns that match the binary-classifier contract
    (width 2); stray wider/narrower outputs (e.g. hidden states or loss
    scalars) are ignored so they can never corrupt or crash the metric."""
    leaves = []
    _collect_2d(preds, leaves)
    if not leaves:
        raise ValueError("compute_metrics: no 2D logits found in predictions")

    widths = [p.shape[-1] for p in leaves]
    # Prefer the binary-classifier width (2); otherwise the most common width.
    if 2 in widths:
        target = 2
    else:
        from collections import Counter as _Counter
        target = _Counter(widths).most_common(1)[0][0]
    kept = [p for p in leaves if p.shape[-1] == target]
    if not kept:  # defensive fallback
        kept = leaves
    return np.concatenate(kept, axis=0)


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    labels = np.asarray(labels)

    # One-time diagnostic: reveal exactly what the Trainer handed us, so a
    # packaging mismatch (e.g. a ragged/extra output) is visible instead of a
    # cryptic crash.
    if not getattr(compute_metrics, "_logged_shape", False):
        compute_metrics._logged_shape = True
        try:
            _t = type(logits).__name__
            if isinstance(logits, (list, tuple)):
                _info = [_describe_el(x) for x in logits[:3]]
            else:
                _info = _describe_el(logits)
            print(f"[metrics] predictions type={_t} sample={_info}")
        except Exception:
            pass

    logits = _extract_logits(logits)
    if not getattr(compute_metrics, "_logged_shape2", False):
        compute_metrics._logged_shape2 = True
        print(f"[metrics] recovered logits shape={logits.shape} (expect (N, 2))")

    logits = _extract_logits(logits)
    probs = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
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


class FocalLoss(torch.nn.Module):
    """Focal Loss for multi-class classification.

    Addresses class imbalance by down-weighting easy examples and focusing
    training on hard negatives. Standard cross-entropy is recovered when
    ``gamma=0``.

    Args:
        gamma: Focusing parameter. Higher values increase the focus on hard
            examples (default 2.0).
        alpha: Weight for the vulnerable class (label=1). The safe class
            weight is ``1 - alpha`` (default 0.25).
        reduction: Reduction method — ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )
        focal = alpha_t * (1 - pt).pow(self.gamma) * ce
        if self.reduction == "sum":
            return focal.sum()
        return focal.mean()


class WeightedTrainer(Trainer):
    """`Trainer` variant that applies per-class weights to the cross-entropy
    loss, optionally replaces it with Focal Loss, and optionally trains only
    the classifier head (frozen backbone).

    Vulnerability datasets are typically class-imbalanced; weighting the loss
    prevents the model from collapsing toward the majority class and gives the
    minority (vulnerable) class a stronger gradient signal.

    When ``use_focal_loss=True`` the standard cross-entropy is replaced with
    ``FocalLoss(gamma, alpha)``, which down-weights easy negatives and forces
    the model to focus on hard positives. See docs/MODEL_TRAINING.md §Loss.

    When ``contrastive=True`` an optional supervised-contrastive term is added
    on top of the cross-entropy loss (weighted by ``contrastive_lambda``), so
    the post-fix / benign-control pairing is exploited as a *contrastive*
    signal. The standard CE objective is used unchanged when the flag is off.
    """

    def __init__(
        self, *args, class_weights=None, freeze_backbone=False,
        contrastive: bool = False, contrastive_lambda: float = 0.1,
        contrastive_temperature: float = 0.1,
        use_focal_loss: bool = False, focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 0.25, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.freeze_backbone = freeze_backbone
        self.contrastive = contrastive
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        if self.use_focal_loss:
            loss_fct = FocalLoss(
                gamma=self.focal_loss_gamma,
                alpha=self.focal_loss_alpha,
            )
        elif self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        if self.contrastive:
            # Reuse the SAME forward (with grad + hidden states) so the
            # supervised-contrastive term back-propagates into the encoder.
            # NOTE: the forward above must request output_hidden_states and
            # must NOT be under torch.no_grad(), or the contrastive term
            # would carry no gradient and silently fail to train.
            hidden = outputs.hidden_states
            if hidden is None:
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


def _check_training_data_quality(
    train_ds: "VulnerabilityDataset", allow_noisy: bool, near_dup_threshold: float = 0.75
) -> None:
    """Pre-flight guard: refuse data that cannot be learned.

    Two failure modes are checked, both of which pin the loss at ``ln(2)`` and
    ROC-AUC at ~0.5:

    1. **Hard contradictions** — a normalized text that appears with BOTH
       labels (identical input, opposite ground truth).
    2. **Near-duplicate contradictions** — a ``safe`` sample that is
       >``near_dup_threshold`` token-similar to a ``vulnerable`` sample. This
       is the dominant failure of the old "safe = the post-fix twin" design
       (median twin similarity ~0.94): the text is not byte-identical, so (1)
       misses it, but it is close enough that the classifier still collapses.
       Catching it here is what makes the guard meaningful for the
       safe-counterpart methodology (see docs/SAFE_COUNTERPARTS.md).

    Refuse by default when either signal exceeds 5% of the training set; pass
    ``allow_noisy`` to override (e.g. when debugging the filters)."""
    total = len(train_ds)
    contradictions = find_contradictions(zip(train_ds.texts, train_ds.labels))
    n = len(contradictions)
    n_near = count_cross_label_near_duplicates(
        zip(train_ds.texts, train_ds.labels), threshold=near_dup_threshold
    )
    frac = (n / total) if total else 0.0
    frac_near = (n_near / total) if total else 0.0

    if n == 0 and n_near == 0:
        print("[*] Data-quality pre-flight: no contradictory or near-duplicate "
              "(safe≈vulnerable) samples found.")
        return

    if n:
        print(
            f"[!] Data-quality pre-flight: {n} normalized texts ({frac:.1%} of "
            f"training samples) appear with BOTH labels (hard contradictions)."
        )
    if n_near:
        print(
            f"[!] Data-quality pre-flight: {n_near} safe samples ({frac_near:.1%} "
            f"of training) are >90% token-similar to a vulnerable sample "
            f"(near-duplicate contradictions)."
        )

    worst = max(frac, frac_near)
    if worst >= 0.05 and not allow_noisy:
        raise RuntimeError(
            "Refusing to train on a dataset with >=5% contradictory / "
            "near-duplicate samples.\n"
            "  -> The 'safe' class is likely still the post-fix twin of the\n"
            "     vulnerable class. Regenerate the manifest vulnerable-only\n"
            "     (src/scripts/convert_cvefixes.py) and mine a genuine safe\n"
            "     class (src/scripts/mine_benign_functions.py), then rebuild\n"
            "     with `build-dataset`. See docs/SAFE_COUNTERPARTS.md.\n"
            "     Or pass --allow-noisy-data to override."
        )
    print("[!] Training may fail to learn (loss ~ln(2), ROC-AUC ~0.5). "
          "Rebuild the data, or pass --allow-noisy-data to proceed anyway.")

# ── Main training loop ──────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    # ── Use all available CPU cores ───────────────────────────────────────────
    # PyTorch defaults to a single intra-op thread unless told otherwise; on a
    # many-core machine this leaves most cores idle. Pin the intra-op thread
    # pool (and the BLAS/OMP pools) to the full core count so both the forward/
    # backward passes and DataLoader workers saturate the CPU.
    cpu_count = max(1, os.cpu_count() or 1)
    torch.set_num_threads(cpu_count)
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, str(cpu_count))
    # Avoid the tokenizers fork/thread warning noise under multiple workers.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Heuristic: a few data-prep workers (capped) keep the GPU/CPU fed without
    # oversubscribing cores; the heavy compute uses the intra-op thread pool.
    dataloader_workers = min(cpu_count, 8)

    device = cfg.device
    use_cpu = device.type == "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    # Fix the head as a *binary* vulnerability classifier and persist an
    # explicit label map. P(vulnerable) is the softmax probability of class 1,
    # independent of the underlying CWE/vulnerability type. The label names are
    # chosen so downstream consumers (ModelManager) unambiguously detect a
    # single-label softmax model and gate on the vulnerable class.
    if cfg.num_labels == 2:
        model.config.id2label = {0: "safe", 1: "vulnerable"}
        model.config.label2id = {"safe": 0, "vulnerable": 1}
        model.config.problem_type = "single_label_classification"
    model.to(device)

    train_ds = VulnerabilityDataset(cfg.train_path, tokenizer, cfg.max_length)
    val_ds = VulnerabilityDataset(cfg.val_path, tokenizer, cfg.max_length)

    # Data-quality pre-flight: refuse to waste a run on contradictory data.
    _check_training_data_quality(train_ds, cfg.allow_noisy_data, cfg.near_dup_threshold)

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

    # Resolve warmup in steps (warmup_ratio is deprecated in transformers >=5.2).
    _steps_per_epoch = max(1, len(train_ds) // max(1, cfg.batch_size))
    _total_steps = _steps_per_epoch * max(1, cfg.num_epochs)
    _warmup_steps = max(1, int(cfg.warmup_ratio * _total_steps))

    args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=_warmup_steps,
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
        dataloader_num_workers=dataloader_workers,
        dataloader_prefetch_factor=4 if dataloader_workers > 0 else None,
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
        use_focal_loss=cfg.use_focal_loss,
        focal_loss_gamma=cfg.focal_loss_gamma,
        focal_loss_alpha=cfg.focal_loss_alpha,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        ],
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    # Final eval for the summary. Guarded: a metrics hiccup must NEVER block
    # persisting the trained model — that would waste the whole long run.
    try:
        eval_metrics = trainer.evaluate()
    except Exception as exc:
        print(f"[!] Final evaluation failed ({exc}); continuing to save the model.")
        eval_metrics = {}

    # ── Persist artifacts (must succeed even after a long, expensive run) ──
    os.makedirs(str(model_dir), exist_ok=True)
    try:
        trainer.save_model(str(model_dir))
    except Exception as exc:
        # Fall back to saving the in-memory model directly so we never lose
        # the trained weights (e.g. output_dir collisions, permission quirks).
        print(f"[!] trainer.save_model failed ({exc}); saving model directly.")
        model.save_pretrained(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except Exception as exc:
        print(f"[!] tokenizer.save_pretrained failed ({exc}).")
    # Re-persist the config so the label map / num_labels survive intact
    # (id2label/label2id/problem_type), even if a partial save occurred.
    try:
        model.config.save_pretrained(str(model_dir))
    except Exception as exc:
        print(f"[!] config.save_pretrained failed ({exc}).")

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
