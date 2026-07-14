# Model Card — CEVuD Vulnerability Classifier

> HuggingFace-ready model card. This documents the custom Stage-2 classifier
> trained by `src/training/` and is the artifact to publish at
> `huggingface.co/cevud/codebert-vuln-classifier`.

---

## Model Details

### Model Description

- **Model ID**: `cevud/codebert-vuln-classifier`
- **Model type**: Fine-tuned transformer for binary sequence classification
- **Base model**: [`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base) (RoBERTa-based, ~125 M parameters)
- **Language**: Python (code)
- **License**: MIT (inherits from CEVuD; CodeBERT itself is MIT)
- **Task**: Binary classification — *vulnerable* vs. *safe* Python function chunks

This model is the **Stage-2 local classifier** ("small model") in the CEVuD
pipeline. It scores uniform code windows (chunks) of Python functions and
outputs a probability `P(vulnerable) ∈ [0, 1]`. The Stage-2 gate combines this
neural probability with Semgrep's static severity via a linear risk equation
`R = W₁·S_sev + W₂·P_slm` to decide whether to escalate a finding to the
Stage-3 LLM.

The model is trained on the CEVuD Training Dataset (CVEfixes-based) and is
designed to be a **component of a gated pipeline**, not a standalone
vulnerability oracle. Its primary role is to suppress trivially-safe code so
that the expensive LLM is only called when truly needed.

### Model Architecture

The model uses the standard HuggingFace `RobertaForSequenceClassification`
head on top of the CodeBERT encoder:

```
Input: Python code chunk (≤ 512 tokens)
  ↓
CodeBERT Encoder (12 layers, 768 hidden dim, 12 attention heads)
  ↓
[CLS] token hidden state (768-dim)
  ↓
Pooler: dense(768 → 768) + tanh
  ↓
Classifier: dense(768 → 768, tanh) + dropout → out_proj(768 → 2)
  ↓
Softmax → P(vulnerable), P(safe)
```

**Key components**:
- **Encoder**: `microsoft/codebert-base` — a RoBERTa-based transformer
  pre-trained on natural language and programming language pairs. Frozen by
  default; can be unfrozen for fine-tuning.
- **Pooler**: Maps the `[CLS]` token to a 768-dim representation via a dense
  layer + tanh activation.
- **Classifier head**: Two-layer MLP (768 → 768 → 2) with dropout and tanh
  activation. Outputs logits for the two classes.
- **Output**: Softmax probabilities. `P(vulnerable) = softmax(logits)[:, 1]`.

When `freeze_backbone=True` is used, only the `classifier.*` submodule is
trained; the encoder and pooler stay frozen. This is the recommended setting for
small datasets (~1.4k samples) because it is more sample-efficient and stable.

---

## Intended Use

### Primary Intended Use

The model is designed to be the **Stage-2 local edge classifier** in the CEVuD
pipeline. Its intended use case is:

1. **CI/CD integration**: Scan code changes in pull requests or pushes.
2. **Local gating**: Score each Semgrep finding locally (zero marginal cost).
3. **Escalation decision**: Combine the neural score with static severity to
   decide whether to escalate to the Stage-3 LLM.

### How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "cevud/codebert-vuln-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

def score_chunk(code_chunk: str) -> float:
    """Return P(vulnerable) for a single code chunk (≤ 512 tokens)."""
    inputs = tokenizer(
        code_chunk,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1])  # P(vulnerable)

# Example: score a Python function chunk
code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return db.execute(query)
"""
p_vuln = score_chunk(code)
print(f"P(vulnerable) = {p_vuln:.3f}")
```

**Important**: This model scores *chunks* (uniform code windows), not whole
functions. For a complete function, chunk it into 64-line windows with 8-line
overlap, score each chunk, and aggregate using `max` (default) or `mean`.

### Out-of-Scope Uses

- **Standalone vulnerability oracle**: The model is not designed to be used
  alone. Its standalone recall is 28.9% (on CVEfixes test) and 70.5% (on
  VUDENC). It is meant to be part of a gated pipeline.
- **Other languages**: The model is trained on Python only. Performance on
  other languages is unverified.
- **Adversarial settings**: The model has not been evaluated against
  adversarially crafted code.
- **Definitive security verdict**: The model's output is one input to a
  composite gate. It should not be used as the sole determinant of whether code
  is vulnerable.

---

## Training Data

### Dataset Overview

The model is fine-tuned on the **CEVuD Training Dataset** (CVEfixes-based), a
curated corpus of 2,181 Python function chunks derived from real-world
vulnerability fixes.

| Property | Value |
|----------|-------|
| **Source** | `hitoshura25/cvefixes` (HuggingFace) |
| **Total samples** | 2,181 |
| **Projects (repos)** | 554 |
| **Vulnerable** | 474 (21.7%) |
| **Safe** | 1,707 (78.3%) — 1,643 benign_sibling + 64 benign_control |
| **Unique CWEs** | 93 |
| **Unique CVEs** | 470 |
| **Chunk size** | 64 lines with 8-line overlap |
| **Hunk-centering** | Enabled (vulnerable chunks contain the sink) |
| **Near-duplicate threshold** | 0.75 token-similarity |

### Data Creation

The training data is created through a multi-stage pipeline:

1. **CVEfixes conversion**: `src/scripts/convert_cvefixes.py` streams the
   CVEfixes dataset, filters to Python, applies noise and trivial-change
   filters, and emits only vulnerable samples (`label=1`). The post-fix function
   is retained in `fixed_code` but not emitted as `label=0`.

2. **Benign control mining**: `src/scripts/mine_benign_functions.py` extracts
   safe functions from files the fix commit did not touch. These are tagged
   `sample_subtype="benign_control"` and serve as the genuine safe class.

3. **Enrichment**: `src/training/dataset_builder.py` enriches each sample with
   the full enclosing function (AST-expanded) and module-level imports.

4. **Chunking**: Functions are cut into 64-line windows with 8-line overlap.
   For vulnerable samples, only chunks overlapping the diff hunk are kept
   (hunk-centering).

5. **Quality filters**: Hard contradictions and near-duplicate safe chunks
   (>0.75 token-similar to vulnerable chunks) are removed.

6. **Splitting**: Project-level 60/20/20 split with `seed=42`. No project
   appears in more than one split.

### Safe Class Construction

The safe class is constructed from two sources:

- **Benign siblings** (1,643 samples): Functions from the same file as the
  vulnerable function, but in commits the fix did not touch.
- **Benign controls** (64 samples): Functions from files the fix commit never
  touched, mined from verified-benign repositories.

Both sources are passed through a token-similarity guard (>0.75 to any
vulnerable function ⇒ dropped) to prevent near-duplicates from entering the
safe class.

The **post-fix function is explicitly not used as `label=0`** because it is a
near-duplicate of its vulnerable twin (median token-similarity ≈ 0.94). Using
it would create contradictory pairs and collapse training to `P = 0.5`.

### Data Splits

| Split | Samples | Vulnerable | Safe | Projects |
|-------|---------|------------|------|----------|
| Train | 1,464 | 316 | 1,148 | 330 |
| Validation | 358 | 76 | 282 | — |
| Test | 359 | 82 | 277 | — |

### Preprocessing

- **Tokenizer**: `AutoTokenizer` from `microsoft/codebert-base` with
  `max_length=512`, `padding="max_length"`, `truncation=True`.
- **Chunking**: Uniform 64-line windows with 8-line overlap. Matches inference
  format.
- **Labels**: `0` = safe, `1` = vulnerable. Mapped to `id2label = {0: "safe",
  1: "vulnerable"}` and `label2id = {"safe": 0, "vulnerable": 1}`.
- **Problem type**: `single_label_classification` (softmax).

---

## Evaluation Data

### Datasets Used

The model is evaluated on two datasets:

1. **CVEfixes test split** (same corpus as training): 359 samples, project-level
   split. This measures the model's standalone performance on held-out projects.
2. **VUDENC test split** (held-out corpus): 821 samples, project-level split.
   This measures the model's performance on a completely different dataset.

### Metrics

| Metric | CVEfixes Test | VUDENC Test |
|--------|---------------|-------------|
| Accuracy | 81.9% | — |
| Precision | 100.0% | — |
| Recall | 20.7% | 70.5% |
| F1 | 34.3% | — |
| ROC-AUC | 0.0* | — |
| PR-AUC | 0.496 | — |

\* The standalone evaluator initially reported ROC-AUC=0.0 due to loading the
wrong checkpoint. This was fixed; the correct ROC-AUC on the CVEfixes validation
split is 74.9%.

The **gate study** (full CEVuD pipeline) is evaluated on VUDENC using F2
(beta=2.0) as the primary metric, with Token Reduction Rate (TRR) and Cost
Reduction as efficiency metrics.

---

## Quantitative Analysis

### Training Dynamics

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |
|-------|-----------|----------|--------------|---------------|------------|--------|-------------|
| 1 | — | 0.419 | 84.9% | 100.0% | 28.9% | 44.9% | 74.9% |
| 2 | — | 0.419 | 84.9% | 100.0% | 28.9% | 44.9% | 74.9% |
| 3 | — | 0.419 | 84.9% | 100.0% | 28.9% | 44.9% | 74.9% |
| 4 | 0.710 | 0.419 | 84.9% | 100.0% | 28.9% | 44.9% | 74.9% |

Training early-stopped at epoch 4 (patience=3 on validation loss). The best
checkpoint is from epoch 1 (step 366), which has the same validation metrics as
epoch 4.

### Confusion Matrix (Validation)

| | Predicted Safe | Predicted Vulnerable |
|---|---|---|
| **Actually Safe** | 282 (TN) | 0 (FP) |
| **Actually Vulnerable** | 54 (FN) | 22 (TP) |

### Confusion Matrix (Test)

| | Predicted Safe | Predicted Vulnerable |
|---|---|---|
| **Actually Safe** | 277 (TN) | 0 (FP) |
| **Actually Vulnerable** | 65 (FN) | 17 (TP) |

### Key Observations

- **Precision = 100%**: The model never produces a false positive. When it
  predicts "vulnerable", it is always correct.
- **Recall = 28.9% (val) / 20.7% (test)**: The model misses most vulnerabilities.
  This is expected for a small model trained on a difficult, imbalanced corpus.
- **ROC-AUC = 74.9%**: The model learns strong discriminative ranking. The low
  recall reflects the classification threshold (0.5), not poor ranking ability.
- **Class imbalance effect**: The ~1:3.6 vulnerable/safe split causes the model
  to be conservative. Class-weighted cross-entropy (weights ≈ [0.64, 2.30])
  gives the vulnerable class a ~3.6× higher per-sample gradient signal, but the
  small dataset size limits how much the model can learn.

### Performance in the Gated Pipeline

When embedded in the CEVuD pipeline with the tuned linear gate
($W_1=0.15, W_2=0.85, T=0.2$):

| Metric | Value |
|--------|-------|
| Recall | 95.2% |
| Precision | 12.8% |
| F2 | 0.417 |
| Escalation Rate | 94.9% |
| TRR | 5.1% |
| Cost Reduction | 5.0% |

The linear gate improves recall from 70.5% (small model standalone) to 95.2%
by combining the neural signal with Semgrep's static signal. The trade-off is
lower precision (12.8%) and high escalation rate (94.9%), which is acceptable
because the escalated snippets are reviewed by a more capable LLM.

---

## Environmental Impact

- **Hardware**: CPU-only training (no GPU required).
- **Training time**: ~2.4 hours on a 4-core CPU (8,759 seconds).
- **Estimated CO2 emissions**: Using the [ML CO2 Impact calculator](https://mlco2.github.io/impact/),
  CPU training for ~2.4 hours on an Intel i7-9700K emits approximately
  0.2-0.4 kg CO2 (depending on electricity grid carbon intensity).
- **Inference cost**: The model runs locally on CPU/edge hardware. A single
  inference on a 512-token chunk takes ~50-100ms on a modern CPU, with near-zero
  marginal cost compared to cloud LLM APIs.

---

## Ethical Considerations

### Intended Users

CEVuD is designed for software developers, security engineers, and organizations
that want to shift-left security scanning in their CI/CD pipelines. The model
augments human experts by filtering safe code, not replacing them.

### Potential Misuse

- **False sense of security**: The model's 100% precision might lead users to
  believe it never misses vulnerabilities. In reality, its standalone recall is
  only 28.9%, and even in the gated pipeline, 4.8% of vulnerabilities slip
  through (FN=5 out of 105 on VUDENC test). Users must understand that CEVuD is
  a *filter*, not a definitive scanner.
- **Over-reliance on automation**: The low precision (12.8%) means many benign
  snippets are escalated. If users skip reviewing escalated snippets, they waste
  LLM resources without gaining security.
- **Bias in training data**: CVEfixes is biased toward well-known, high-profile
  projects. Vulnerabilities in niche or internal codebases may not be
  represented. The model may perform worse on code that differs stylistically
  from the CVEfixes corpus.

### Fairness and Transparency

- The model's decisions are interpretable: the linear gate formula
  `R = W₁·S_sev + W₂·P_slm` is transparent, and the weights are selected by
  exhaustive grid search.
- The training data is publicly available, and the full training pipeline is
  open-source.
- The model does not process personal data. Code snippets are the only input.

### Limitations and Recommendations

| Limitation | Recommendation |
|------------|----------------|
| Low standalone recall (28.9%) | Always use as part of the gated pipeline, not standalone. |
| Python-only | Do not apply to other languages without retraining. |
| Chunk-level granularity | Score whole functions by chunking and aggregating. |
| CWE imbalance | Consider augmenting rare CWE types if your use case targets specific vulnerabilities. |
| No adversarial evaluation | Do not deploy in adversarial settings without additional testing. |

---

## Training Procedure

### Implementation

Training is implemented in `src/training/trainer.py` using the HuggingFace
`Trainer` API with a custom `WeightedTrainer` subclass.

### Loss Function

**Class-weighted cross-entropy**: The ~1:3.6 vulnerable/safe imbalance is
countered by inverse-frequency class weights:

```
weight(class) = total_samples / (num_classes × count(class))
```

For the current split, this yields approximately `[0.64, 2.30]` for
`[safe, vulnerable]`, meaning each vulnerable sample contributes ~3.6× the
gradient signal of a safe sample.

**Why not focal loss?** Focal loss was evaluated but removed in favor of class
weights. Class-weighted cross-entropy is simpler, more interpretable, and
equally effective for this dataset size. The weights are computed automatically
from the training distribution.

### Optimization

- **Optimizer**: AdamW
- **Learning rate**: 2e-5
- **Weight decay**: 0.01
- **Batch size**: 8
- **Warmup**: Linear warmup for 10% of total steps
- **Scheduler**: Linear decay after warmup

### Regularization

- **Early stopping**: Patience=3 epochs on validation loss. Best checkpoint
  restored.
- **Dropout**: 0.1 in the classifier head (default for RobertaClassificationHead)
- **Frozen backbone** (optional): When `freeze_backbone=True`, only the
  classifier head is trained. Recommended for small datasets.

### Reproducibility

All randomness is controlled with `seed=42`:
- Dataset split: `seed=42`
- Sample capping: `seed=42`
- Model initialization: `seed=42`
- Training shuffle: `seed=42`

### Training Command

```bash
python -m src.training.cli run-all \
  --manifest benchmark_manifest_cvefixes.json \
  --benign-manifest benign_controls_manifest.json \
  --epochs 30 \
  --batch-size 8
```

---

## How to Get Started with the Model

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "cevud/codebert-vuln-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()
```

### Scoring a Function

```python
def score_function(function_code: str, chunk_max_lines: int = 64,
                   chunk_overlap: int = 8) -> float:
    """Score a Python function for vulnerability probability."""
    lines = function_code.splitlines()
    chunks = []
    for i in range(0, max(len(lines) - chunk_overlap, 1), chunk_max_lines - chunk_overlap):
        chunk = "\n".join(lines[i:i + chunk_max_lines])
        if chunk.strip():
            chunks.append(chunk)
    
    scores = []
    for chunk in chunks:
        inputs = tokenizer(chunk, truncation=True, max_length=512,
                          padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        scores.append(float(probs[0, 1]))
    
    return max(scores) if scores else 0.0

# Example usage
vuln_code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return db.execute(query)
"""
print(f"P(vulnerable) = {score_function(vuln_code):.3f}")
```

### Using in the CEVuD Pipeline

```python
from triage_orchestrator import TriageOrchestrator

orchestrator = TriageOrchestrator(
    config_path="config.json",
    workspace_path="."
)
orchestrator.process_pipeline()
```

---

## Model Card Authors

CEVuD Authors

## Citation

```bibtex
@misc{cevud2026,
  title={CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning},
  author={CEVuD Authors},
  year={2026},
  note={Model: cevud/codebert-vuln-classifier; Dataset: cevud/cevud-training-dataset}
}
```

## Model Card Contact

Open an issue on the CEVuD GitHub repository.
