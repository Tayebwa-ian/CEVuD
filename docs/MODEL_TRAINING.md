# Model Training — Design Decisions & Results

> Focused reference for the custom Stage-2 classifier. Pipeline documentation
> lives in `USAGE.md` and `INDEX.md`.

## 1. Model Architecture

| Field | Value |
|---|---|
| Base | `microsoft/codebert-base` (RoBERTa, ~125 M params) |
| Head | Standard `RobertaClassificationHead` via `AutoModelForSequenceClassification` |
| Task | Binary single-label softmax: `P(vulnerable) ∈ [0, 1]` |
| Labels | `0 = safe`, `1 = vulnerable` |
| Pooler | `[CLS]` → `dense(768→768)` → `tanh` |
| Classifier | `dense(768→768, tanh)` → dropout → `out_proj(768→2)` |

**Decision:** CEVuD does **not** use a hand-written head. Loading
`AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)`
preserves the full CodeBERT encoder + pooler + classifier stack, so the trained
checkpoint is a drop-in replacement for the Hub model with zero code changes in
`ModelManager`.

## 2. Training Data

### Source
CVEfixes (`hitoshura25/cvefixes`) converted via `src/scripts/convert_cvefixes.py`
→ `benchmark_manifest_cvefixes.json`. The converter applies noise filtering
(rows whose (vuln, safe) pair differs only in comments / docstrings / version
assignments are dropped), which is what makes the task learnable — an earlier
unfiltered run plateaued at `loss ≈ 0.693` / `roc_auc ≈ 0.5`.

### Safe-class strategy
The CVEfixes post-fix function is **not** used as `label=0` (median
token-similarity to its vulnerable twin ≈ 0.94, which collapses training to
`P = 0.5`). The safe class is built from:

1. **Verified-benign controls** (`src/scripts/mine_benign_functions.py`):
   same-file sibling functions and functions from files the fix commit never
   touched, passed through a token-similarity guard (>0.75 to any vulnerable
   function ⇒ dropped).
2. **Optional contrastive mode** (`--contrastive`, OFF by default): the
   (vulnerable, fixed) pair is used as a contrastive signal rather than a hard
   `label=0` target, which is more robust to post-fix noise.

### Context format
Function body + module imports, cut into **uniform code chunks** (≤ 512 tokens)
at both train and inference time. Cross-file context is **not** fed to the SLM —
it is attached only when a finding escalates to the Stage-3 LLM.

### Splits
Stratified by project (repo) so no project appears in more than one split.
A `_split_key` collapses `benign::<repo>` → `<repo>` so a repo's vulnerable and
mined-safe samples always land in the same split.

## 3. Current Run Results

### Dataset
| Split | Samples | Vulnerable | Safe |
|---|---|---|---|
| Train | 1,464 | 316 | 1,148 |
| Validation | 358 | 76 | 282 |
| Test | 359 | 82 | 277 |

### Best checkpoint (validation, epoch 1 / step 366)
| Metric | Value |
|---|---|
| Eval loss | 0.4191 |
| Accuracy | 0.8492 |
| Precision | 1.0 |
| Recall | 0.2895 |
| F1 | 0.449 |
| ROC AUC | 0.7485 |
| PR AUC | 0.6187 |
| Confusion | TN=282, FP=0, FN=54, TP=22 |

### Final model (test, epoch 4)
| Metric | Value |
|---|---|
| Accuracy | 0.8189 |
| Precision | 1.0 |
| Recall | 0.2073 |
| F1 | 0.3434 |
| ROC AUC | 0.0 |
| PR AUC | 0.4964 |
| Confusion | TN=277, FP=0, FN=65, TP=17 |

### Failure modes observed
1. **Majority-class collapse** — Precision=1.0, recall≈0.2. The model learned
   to predict almost everything as `safe`. The class weights from
   `_compute_class_weights` (~[0.64, 2.30]) were too weak to overcome the
   imbalance on this small dataset.
2. **Test ROC AUC = 0.0** — Bug fixed in this revision. The standalone
   `evaluator.py` was loading `latest/model` (epoch 4) instead of the best
   checkpoint (`checkpoint-366`, epoch 1). The model degenerated after the best
   epoch; evaluation now resolves `best_model_checkpoint` from
   `trainer_state.json`.
3. **Near-duplicate threshold mismatch** — `config.py` set
   `near_dup_threshold=0.75`, but the trainer's pre-flight guard hardcoded
   `0.90`, letting noisy safe counterparts slip through. Fixed to read the
   config value.

## 4. Design Decisions

### 4.1 Frozen backbone (`--freeze-backbone`)
Freezes the CodeBERT encoder + pooler; trains only the `classifier.*` head.
Far more sample-efficient and stable for small datasets (~1.4k chunks), and
cuts training time by 5–10×. Recommended default for this data scale.

### 4.2 Class weights
CEVuD uses inverse-frequency class weights to counter the vulnerable/safe
imbalance. `_compute_class_weights` assigns each class a weight of
``total / (num_labels * count)``, so the minority (vulnerable) class gets a
higher weight. For the current ~1 : 3.6 vulnerable/safe split this yields
roughly ``[0.64, 2.30]`` — the vulnerable class receives about 3.6× the
per-sample gradient signal of the safe class. These weights are passed to
`torch.nn.CrossEntropyLoss(weight=...)` inside `WeightedTrainer`.

### 4.3 Supervised contrastive (`--contrastive`)
Adds a contrastive term on top of CE: a `vulnerable` function is pulled toward
its `fixed` twin and pushed from `benign_control` functions. Uses the
post-fix pair as a contrastive signal instead of a hard `label=0` target,
which is more robust to non-security noise in fix commits. OFF by default.

### 4.4 Hunk-centering
For vulnerable samples, only chunks overlapping the diff hunk are kept. This
fixes the "~50% of positive chunks contain no sink" problem — the old code
chunked from the function start and often labeled a sink-free window as
vulnerable.

### 4.5 Near-duplicate guard
After chunking, any `label=0` chunk >0.75 token-similar to a `label=1` chunk
in the same project is dropped. This prevents a lightly-edited copy of a
vulnerable function from entering the safe class and collapsing training to
`P=0.5`.

### 4.6 Near-duplicate guard
After chunking, any `label=0` chunk >0.75 token-similar to a `label=1` chunk
in the same project is dropped. This prevents a lightly-edited copy of a
vulnerable function from entering the safe class and collapsing training to
`P=0.5`.

## 5. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `base_model` | `microsoft/codebert-base` | 125 M params, 768 hidden, 12 layers |
| `max_length` | 512 tokens | Mirrors inference chunking |
| `batch_size` | 8 | Increase with `gradient_accumulation_steps` if memory allows |
| `learning_rate` | 2e-5 | Standard for CodeBERT fine-tuning |
| `weight_decay` | 0.01 | |
| `num_epochs` | 20 (early stop) | Early stopping patience=3 on val loss |
| `warmup_ratio` | 0.1 | |
| `freeze_backbone` | False | **Recommended True** for small datasets |
| `contrastive` | False | Experimental |
| `chunk_max_lines` | 64 | |
| `chunk_overlap` | 8 | |
| `near_dup_threshold` | 0.75 | Drop safe chunks > this similar to vuln chunks |

## 6. Deployment

After training, the model directory (`training_output/latest/model`) can be:

1. **Tested locally** — set `models.classifier_model` in `config.json` to
   `training_output/latest/model`.
2. **Baked into Docker** — `docker build --build-arg CLASSIFIER_MODEL=$(python -c "import json; print(json.load(open('config.json'))['models']['classifier_model']") ...`. The CI (`publish_image.yml`) does this automatically: it reads `config.json` and passes the value as the Docker build arg, so the `model_downloader` stage pre-caches the exact model the pipeline is configured to use.
3. **Uploaded to HuggingFace** — push the checkpoint to your Hub repo, then set
   `models.classifier_model` in `config.json` to that repo ID. The Docker
   build and CI workflows pick it up automatically on the next push.

### One-liner publish

```bash
python -m src.training.cli publish \
  --repo-id <your-hf-org>/<your-model-name> \
  --model-dir training_output/latest/model
```

### CI integration

1. Add a `HF_TOKEN` secret to your GitHub repo (Settings → Secrets → Actions).
2. Set `models.classifier_model` in `config.json` to your published Hub repo ID.
3. Push to `main`. The `publish_image.yml` workflow reads `config.json`, passes
   the model ID as the `CLASSIFIER_MODEL` Docker build arg, and pre-caches it.
4. The `security_pipeline.yml` workflow also reads `config.json` and pulls the
   model at runtime if it is not already in the image.

The `latest` symlink is maintained automatically by `trainer.py`, but the model
ID in `config.json` is the single source of truth for both CI and runtime.
