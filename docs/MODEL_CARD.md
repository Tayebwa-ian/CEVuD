# Model Card — CEVuD Vulnerability Classifier

> HuggingFace-ready model card. This documents the custom Stage-2 classifier
> trained by `src/training/` and is the artifact to publish at
> `huggingface.co/<org>/cevud-codebert-vuln-classifier`.

## Model summary

| Field | Value |
|---|---|
| Model ID (proposed) | `cevud/codebert-vuln-classifier` |
| Base model | `microsoft/codebert-base` (RoBERTa, ~125 M params) |
| Task | Binary sequence classification — *vulnerable vs. safe* Python function |
| Labels | `0` = safe, `1` = vulnerable |
| Input | Function body + module imports, cut into **uniform code chunks** (≤ 512 tokens) at both train and inference time (see `docs/SLM_CHUNKING.md`). Cross-file context is *not* fed to this model — it is attached only when a finding escalates to the Stage-3 LLM. |
| Output | Per-chunk `P(vulnerable) = softmax(logits)[:, 1]` ∈ [0, 1]; the Stage-2 gate aggregates chunk scores (default `max`) into the function-level `P_slm`. |
| Language | Python only |
| License | MIT (inherits from CEVuD; confirm CodeBERT's MIT license) |

## Architecture

CEVuD does **not** use a hand-written head — it loads the standard HuggingFace
`RobertaForSequenceClassification` head on top of the CodeBERT encoder
(`AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base",
num_labels=2)` in `src/training/trainer.py`). CodeBERT is a RoBERTa
architecture, so the head is the standard `RobertaClassificationHead`:

1. **Pooler** — takes the `[CLS]` token's last hidden state (768-dim) and
   applies `pooler.dense` (768 → 768) followed by `tanh`.
2. **Classifier** — `classifier.dense` (768 → 768, `tanh`) → dropout →
   `classifier.out_proj` (768 → 2) produces the raw logits for the two classes.
3. A softmax over the two logits yields `P(vulnerable)`; the model's
   `P_slm` score used by the Stage-2 gate is `probs[:, 1]`.

Both `pooler` and `classifier` weights are part of the published checkpoint.
When `freeze_backbone=True` is used for sample-efficient training, only the
`classifier.*` submodule is updated and the encoder + pooler stay frozen.

## Training data

The classifier is fine-tuned on **CVEfixes**
(`hitoshura25/cvefixes`, converted via `src/scripts/convert_cvefixes.py` →
`benchmark_manifest_cvefixes.json`), which provides the **vulnerable**
(`label = 1`) class — the pre-fix function, anchored to the parent of the fix
commit. The post-fix function is **not** used as the safe class (it is a
near-duplicate of its vulnerable twin, median token-similarity ≈ 0.94, which
collapses training to `P = 0.5`). The **safe** (`label = 0`) class is instead
mined by `src/scripts/mine_benign_functions.py` — same-file *sibling* functions
of the vulnerable function plus functions from files the fix never touched, each
passed through a token-similarity guard (>0.75 to any vulnerable function ⇒
dropped) so no near-duplicate can enter the safe class. See
`docs/SAFE_COUNTERPARTS.md`.

Function-level context is built by `src/training/dataset_builder.py` using
`code_context`: the enclosing function (AST-expanded), its module imports, and
best-effort cross-file context. Splits are **stratified by project** (repo) so
no project appears in more than one split.

The manifest is generated with **noise filtering enabled by default**
(`src/scripts/convert_cvefixes.py`): rows whose (vulnerable, safe) pair differs
only in comments / docstrings / version assignments, plus snippets with no real
code signal (e.g. a lone `__version__ = '3.7'`), are dropped at conversion
time. This is what makes the task learnable — an earlier unfiltered run
plateaued at `loss ≈ 0.693` / `roc_auc ≈ 0.5`. See `docs/DATA_QUALITY.md`.

> **Safe counterpart — important caveat.** The post-fix function is only a
> *relative* negative (it lacks *that one* CVE, but may still contain a
> *different* weakness) and a fix commit can bundle unrelated edits, so the
> `label=0` sample sometimes differs from its twin for non-security reasons.
> The recommended hardening is to **also inject verified-benign controls**
> (`src/scripts/mine_benign_functions.py` → `build-dataset
> --benign-manifest`): functions mined from files the fix commit did **not**
> modify, i.e. demonstrably not part of any vulnerability fix. These are
> merged into the training pool as `label=0`, `sample_subtype="benign_control"`.
> An optional **contrastive** training mode (`--contrastive`, OFF by default)
> additionally uses the (vulnerable, fixed) pair as a contrastive signal
> rather than a hard `label=0` target. Full methodology + measurements:
> `docs/SAFE_COUNTERPARTS.md`. State in the paper which regime the reported
> model used (standard CE vs CE + benign controls vs + contrastive).

Within CVEfixes the same split provides the classifier's own validation
(early stopping / best-checkpoint selection) and test evaluation
(accuracy / F1 / ROC-AUC). VUDENC is reserved for the **gate study** — the
comparative evaluation of the full CEVuD pipeline, not for tuning or measuring
this model in isolation.

## Training procedure

Implemented in `src/training/trainer.py`:

- **Loss**: class-weighted cross-entropy (inverse-frequency weights from the
  training distribution) to counter vulnerability-class imbalance.
- **Early stopping**: `EarlyStoppingCallback` halts training once validation
  loss fails to improve for `early_stopping_patience` (default 3) epochs; the
  best checkpoint (lowest `eval_loss`) is restored.
- **Best checkpoint selection**: by validation loss, not accuracy/F1.
- **Optional frozen backbone**: `--freeze-backbone` trains only the classifier
  head on frozen CodeBERT embeddings — more sample-efficient and stable for
  small datasets.

### Default hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `base_model` | `microsoft/codebert-base` | RoBERTa encoder |
| `num_labels` | `2` | vulnerable / safe |
| `max_length` | `512` | tokenizer truncation |
| `batch_size` | `8` | per-device |
| `learning_rate` | `2e-5` | AdamW |
| `num_epochs` | `20` | high ceiling; early stopping ends sooner |
| `warmup_ratio` | `0.1` | |
| `weight_decay` | `0.01` | |
| `freeze_backbone` | `False` | set `True` for head-only training |
| `early_stopping_patience` | `3` | stops on val-loss plateau |
| `early_stopping_threshold` | `0.0` | min val-loss drop to count |

## Intended use

- **Primary**: Stage-2 ("Smart Gate") local edge classifier in the CEVuD
  pipeline. Its `P_slm` is combined with Semgrep severity via the linear gate
  `R = W₁·S_sev + W₂·P_slm` to decide LLM escalation.
- **Standalone**: a zero-marginal-cost Python vulnerability scorer for PR /
  CI triage.

### How to load

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("cevud/codebert-vuln-classifier")
model = AutoModelForSequenceClassification.from_pretrained("cevud/codebert-vuln-classifier")
model.eval()

text = "def sql_query(user):\n    return conn.execute(\"SELECT * FROM users WHERE name = '\" + user + \"'\")"
inputs = tok(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)
p_vuln = float(probs[0, 1])   # P(vulnerable)
```

## Limitations

- **Python-only.** Trained and evaluated on Python corpora (CVEfixes, VUDENC).
- **Chunk-level granularity.** Scores uniform code windows (function body + imports) and aggregates them (default `max`); it localizes *suspicious chunks* for the Stage-3 LLM rather than scoring the whole function in one truncated pass.
- **Distribution shift.** Optimized for real-world Python vulnerabilities from
  CVEfixes; performance on other languages / DSLs is unverified.
- **Not a definitive oracle.** `P_slm` is one input to a composite gate; it is
  designed to suppress trivially-safe code, not to replace human/LLM review.
 - **Evaluation corpus.** The classifier's own accuracy / F1 / ROC-AUC are
   measured on its CVEfixes test split (project-level, so no leakage). The
   end-to-end gate-study metrics are measured on **VUDENC**; absolute numbers
   depend on the VUDENC class balance (see `DATASET_CARD.md`).

## Two-corpora workflow

| Role | Dataset | Converter | Manifest |
|---|---|---|---|
| Classifier train / validation / test | CVEfixes | `convert_cvefixes.py` | `benchmark_manifest_cvefixes.json` |
| Gate study (full pipeline) | VUDENC | `convert_vudenc.py` | `benchmark_manifest_vudenc.json` |

CVEfixes develops the classifier (train + validate + its own test eval);
VUDENC is the independently curated corpus for the comparative gate study.
