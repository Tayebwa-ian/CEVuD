# Training a Custom Vulnerability Classifier (Few-Shot)

## Overview

CEVuD's training pipeline fine-tunes a small CodeBERT-based classifier
(`microsoft/codebert-base`, ~125M params) on the **CVEfixes** benchmark
(`benchmark_manifest_cvefixes.json`, produced by `convert_cvefixes.py`), which
is used for the model's training, validation, and its own evaluation
(project-level splits prevent in-corpus leakage). The **VUDENC** corpus
(`benchmark_manifest_vudenc.json`, from `convert_vudenc.py`) is the dataset for
the **gate study** — the comparative evaluation of the full CEVuD pipeline run
by `src/evaluation/`. The pipeline supports two regimes:

* **Quick smoke test (`--few-shot`)** — a small, balanced, credible subset
  (~100 samples). Good for verifying the pipeline end-to-end, but **too small to
  learn anything useful** (a 125M-param model needs far more signal). Treat its
  metrics as a pipeline sanity check, not a usable model.
* **Real training** — build on a large slice of the full 1,538-sample manifest.
  This is what produces a model worth deploying.

The pipeline uses **class-weighted loss** to counter vulnerability-class
imbalance, and offers a **frozen-backbone mode** (`--freeze-backbone`) that trains
only the classifier head on frozen CodeBERT embeddings — far more sample-efficient
and stable when data is limited, and much faster to train.

The trained model produces `P(vulnerable) ∈ [0, 1]` and can be dropped into
`config.json` as the new Stage-2 local classifier with zero code changes.

## Model Architecture

The custom classifier is a **standard HuggingFace `RobertaForSequenceClassification`
head on top of `microsoft/codebert-base`** (a RoBERTa encoder, ~125 M params).
CEVuD does *not* use a hand-written head — `trainer.py` loads
`AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)`.

1. **Pooler** — takes the `[CLS]` token's last hidden state (768-dim) and
   applies `pooler.dense` (768 → 768) + `tanh`.
2. **Classifier** — `classifier.dense` (768 → 768, `tanh`) → dropout →
   `classifier.out_proj` (768 → 2) produces the two-class logits.

A softmax over the logits gives `P(vulnerable) = softmax(logits)[:, 1]`, which
is the `P_slm` score fed into the Stage-2 linear gate. The published checkpoint
contains both `pooler` and `classifier` weights. With `--freeze-backbone`, only
the `classifier.*` submodule is trained and the encoder + pooler stay frozen.

This is documented in full (with load snippet, hyperparameters, and limitations)
in **[`MODEL_CARD.md`](MODEL_CARD.md)** — that file is
the HuggingFace-ready model card.

## Datasets: model training vs. gate study

The two corpora play distinct roles in the pipeline:

| Role | Dataset | Converter | Manifest |
|---|---|---|---|
| Classifier training / validation / evaluation | CVEfixes | `convert_cvefixes.py` | `benchmark_manifest_cvefixes.json` |
| Gate study (comparative evaluation of the full pipeline) | VUDENC | `convert_vudenc.py` | `benchmark_manifest_vudenc.json` |

- CVEfixes develops the small model end-to-end: it is used for training,
  validation (early-stopping / best-checkpoint selection), and the model's own
  evaluation. Splits are stratified by project, so no project appears in more
  than one split (prevents in-corpus leakage).
- VUDENC is the corpus for the gate study: `src/evaluation/` runs the
  comparative evaluation of Semgrep + the CVEfixes-trained classifier + the
  gating strategies against VUDENC's real-world functions.

Both manifests share the same schema (see
`DATASET_CARD.md`), so the training and evaluation harnesses are
interchangeable. VUDENC ships no repository/commit metadata, so its manifest
uses `local_source` + embedded `source_code`; CVEfixes uses `git_source`
(real repos are cloned during evaluation). See `convert_vudenc.py` for how
VUDENC's per-line labels are collapsed to function-level labels.

## Directory Layout

```
src/training/
    __init__.py
    config.py            # Hyperparameters, paths, few-shot defaults
    dataset_builder.py   # Enriches manifest with full function context + caps
    trainer.py           # Fine-tunes CodeBERT
    evaluator.py         # Evaluation + plots
    cli.py               # Unified CLI

training_data/
    train.jsonl          # Training split (enriched samples)
    val.jsonl            # Validation split
    test.jsonl           # Held-out test split
    dataset_summary.json # Split stats, CWE coverage, cap info

training_output/
    run_YYYYMMDD_HHMMSS/
        model/           # Best fine-tuned model (pytorch_model.bin + config)
        training_summary.json
        eval/
            metrics.json
            confusion_matrix.png
            roc_curve.png
            pr_curve.png
            calibration.png
```

## Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `scikit-learn`, `matplotlib`,
`git` (for cloning repos during dataset construction).

## Step 1 — Build the Dataset

### Two regimes

* **Quick smoke test (`--few-shot`)** — ~100 samples to verify the pipeline.
  The model trained on this is **not** good enough to deploy; use it only to
  confirm the pipeline runs end-to-end.
* **Real training (recommended)** — build on a large slice of the full
  manifest. The more projects/samples you include, the better the model.
  Because clones are deleted immediately after enrichment, disk usage is just
  the JSONL files, so building large datasets is safe.

### The Few-Shot Strategy (smoke test only)

We do **not** augment the data. We use the existing `benchmark_manifest_cvefixes.json`
as-is and apply **strategic capping** to create a small, credible subset:

1. **Select a bounded number of projects** (`--max-projects`, default 20 for few-shot).
   The builder uses a greedy CWE-coverage algorithm: it picks projects that
   collectively cover the most unique CWE types first, so even 20 projects give
   broad vulnerability coverage.
2. **Cap samples per class** (`--max-samples-per-class`, default 50 for few-shot).
   This guarantees balanced vulnerable/safe representation regardless of the
   underlying distribution.
3. **Cap samples per CWE** (`--max-samples-per-cwe`, optional).
   Prevents any single CWE type from dominating the training signal.
4. **Hard cap on total samples** (`--max-total`, default 500 for few-shot).
   Ensures training stays fast on CPU.

All caps are applied **before** splitting, so class balance is preserved across
train/val/test. Splitting is done at the **project level** to prevent data leakage.

### CLI — Build

```bash
# SMOKE TEST (~100 samples, pipeline check only — do NOT deploy this model):
python -m src.training.cli build-dataset --few-shot --max-workers 8

# REAL TRAINING (recommended): build on a large slice of the manifest.
# No --few-shot -> uses all projects/samples (cap with --max-projects to bound time).
python -m src.training.cli build-dataset --max-workers 8

# Medium dataset, capped for faster iteration:
python -m src.training.cli build-dataset \
  --max-projects 100 \
  --max-samples-per-class 200 \
  --max-total 800 \
  --max-workers 8

# Include cross-file context (slower but more signal):
python -m src.training.cli build-dataset --cross-file --max-workers 8
```

### What the builder does

1. Loads `benchmark_manifest_cvefixes.json`.
2. For each selected project, clones the repo with `git clone --filter=blob:none`
   into `.training_cache/clones/`, extracts the needed context, then **deletes
   the clone immediately** so no repository source is left on disk.
3. For each sample:
   - Resolves the correct commit (parent of fix commit for vulnerable
     samples, fix commit itself for safe samples).
   - Reads the full file via `git show <commit>:<path>`.
   - Uses `code_context.expand_to_function` to find the smallest enclosing
     `def` / `async def`.
   - Uses `code_context.collect_module_imports` to prepend imports.
   - Optionally uses `code_context.collect_cross_file_context` for
     intra-repo module bodies (up to 3 modules, 200 lines each).
   - Assembles the final text via `build_context_snippet`.
4. Applies capping constraints (per-class, per-CWE, total).
5. Splits **by project** (60 % train / 20 % val / 20 % test) so no project
   leaks across splits.
6. Writes `training_data/train.jsonl`, `val.jsonl`, `test.jsonl`, and a
   `dataset_summary.json`.

### Capping example

If you run:

```bash
python -m src.training.cli build-dataset --few-shot
```

You might see:

```
[*] Building dataset from 20 projects ...
[+] train       :   60 samples -> training_data/train.jsonl
[+] validation  :   20 samples -> training_data/val.jsonl
[+] test        :   20 samples -> training_data/test.jsonl

[+] Dataset summary -> training_data/dataset_summary.json
```

And `dataset_summary.json`:

```json
{
  "total_samples": 100,
  "projects_processed": 20,
  "label_distribution": {"vulnerable": 50, "safe": 50},
  "unique_cwe_types": 45,
  "cwe_coverage": {"CWE-617": 6, "CWE-476": 4, ...},
  "split_sizes": {"train": 60, "validation": 20, "test": 20},
  "avg_context_lines": 62.1,
  "capped_from": 780,
  "capped_to": 100
}
```

The `--few-shot` defaults apply a **per-class cap of 50** (`--max-samples-per-class 50`),
so the resulting set is bounded to ~100 samples (50 vulnerable + 50 safe) regardless of
`--max-total 500` (which acts only as a hard safety ceiling). To build a larger few-shot
set, raise `--max-samples-per-class` and/or `--max-projects` (e.g. `--max-samples-per-class 250`
approaches a ~500-sample balanced set).

The `capped_from` / `capped_to` fields confirm the builder selected and trimmed
samples to hit your few-shot budget.

## Step 2 — Fine-Tune CodeBERT

### Getting a usable model

A 125M-param model cannot learn from ~40 training samples. For a model worth
deploying, **build on a large dataset (Step 1, no `--few-shot`)** and:

- **Train longer** — `num_epochs` defaults to 20; **early stopping** ends
  training automatically once validation loss stops improving (patience 3 by
  default), so you can set a high epoch ceiling without wasting compute.
- **Class-weighted loss** (always on) counteracts vulnerability-class imbalance.
- **Frozen-backbone (`--freeze-backbone`)** when data is still limited: trains
  only the classifier head on frozen CodeBERT embeddings. This is far more
  sample-efficient and stable than full fine-tuning on small data, and trains
  much faster (good for the constrained host). With a large dataset, full
  fine-tuning (`--freeze-backbone` off) is stronger.

```bash
# Real training on a large dataset (full fine-tune, early-stopped on val loss):
python -m src.training.cli train --epochs 20 --batch-size 8 --lr 2e-5

# Customise early stopping:
python -m src.training.cli train --epochs 40 --early-stopping-patience 5

# Sample-efficient alternative for smaller datasets:
python -m src.training.cli train --freeze-backbone --epochs 20 --batch-size 8 --lr 2e-5
```

### Hyperparameters

| Parameter                  | Default            | Description                            |
|----------------------------|--------------------|----------------------------------------|
| `base_model`               | `microsoft/codebert-base` | Base encoder (~125M params) |
| `max_length`               | 512                | Tokenizer truncation length            |
| `batch_size`               | 8                  | Per-device batch size (CPU-safe)       |
| `learning_rate`            | 2e-5               | AdamW learning rate                    |
| `num_epochs`               | 20                 | Max passes over training data (early stopping usually ends sooner) |
| `warmup_ratio`             | 0.1                | Linear warm-up proportion              |
| `gradient_accumulation`    | 1                  | Steps before optimizer update          |
| `weight_decay`             | 0.01               | L2 regularization                      |
| `freeze_backbone`          | `False`            | Train only the classifier head         |
| `early_stopping_patience`  | 3                  | Epochs w/o val-loss improvement before stopping |
| `early_stopping_threshold` | 0.0                | Min val-loss drop to count as progress |

The `Trainer` selects and restores the **best checkpoint by validation loss**
(`eval_loss`) and stops early when val loss plateaus, saving it to
`training_output/run_<timestamp>/model/`.

### Expected training time (few-shot)

On a **4-core CPU, 8 GB RAM** machine with a 350-sample few-shot dataset:

| Phase                      | Wall-clock |
|----------------------------|-----------|
| Dataset build (20 projects) | 10–30 min |
| Training (3 epochs, batch 8) | 15–40 min |
| Evaluation                 | < 2 min   |

On **CPU only**, batch size 8 is the practical maximum without OOM on 8 GB RAM.
Reduce `--batch-size` to 4 or 2 if memory is constrained.

## Step 3 — Evaluate

```bash
python -m src.training.cli evaluate
```

This loads the best checkpoint, runs inference on the held-out test split,
and writes:

| Output                          | Description                                  |
|---------------------------------|----------------------------------------------|
| `metrics.json`                  | Scalar metrics + confusion matrix            |
| `confusion_matrix.png`          | 2×2 heatmap                                  |
| `roc_curve.png`                 | ROC curve with AUC                           |
| `pr_curve.png`                  | Precision-recall curve with AUC              |
| `calibration.png`               | Reliability diagram                          |

### Metrics reported

- **Accuracy** — overall correctness
- **Precision** — P(predicted vulnerable | truly vulnerable)
- **Recall** — P(found | truly vulnerable)
- **F1** — harmonic mean of precision and recall
- **F2** — recall-weighted F-beta (security triage favours recall)
- **ROC-AUC** — discrimination ability
- **PR-AUC** — robust to class imbalance
- **Confusion matrix** — TP / FP / FN / TN counts

## Step 4 — Deploy the Custom Model

### Option A: Update `config.json` for local runs

```json
{
  "models": {
    "classifier_model": "training_output/latest/model",
    "embedding_model": "microsoft/codebert-base"
  }
}
```

> `ModelManager` auto-detects softmax vs multi-label scoring from the model
> config, so no code changes are needed.
>
> The trainer automatically maintains a `training_output/latest` symlink that
> points at the most recent timestamped run directory, so this path always
> resolves to the freshest model. `evaluate` also defaults to this path.

### Option B: Bake into the Docker image

```bash
# After training completes, note the model directory:
MODEL_PATH=$(ls -td training_output/run_*/model | head -1)

docker build \
  --build-arg CUSTOM_MODEL_PATH="$MODEL_PATH" \
  -t cevud:custom-model .
```

The Dockerfile stage:

```dockerfile
ARG CUSTOM_MODEL_PATH=""
RUN if [ -n "$CUSTOM_MODEL_PATH" ]; then \
        mkdir -p /app/custom_model && cp -r $CUSTOM_MODEL_PATH/* /app/custom_model/; \
    fi
```

At runtime, `config.json` is patched automatically to point the classifier at
`/app/custom_model`.

## Publishing to Hugging Face

Both the trained model and the two benchmark manifests are intended for
publication on the HuggingFace Hub (model + datasets), which also backs the
research paper. The repository already ships ready-to-publish cards:

- **Model card** — [`MODEL_CARD.md`](MODEL_CARD.md)
  (architecture, training data/procedure, hyperparameters, limitations, load
  snippet).
- **Dataset cards** — [`DATASET_CARD.md`](DATASET_CARD.md)
  (CVEfixes training set + VUDENC evaluation set, shared schema).

### Publish the model

```bash
# From the trained run directory (training_output/latest/model or run_*/model):
python -m huggingface_hub huggingface-cli upload cevud/codebert-vuln-classifier ./model
# then add MODEL_CARD.md as the repo README (rename to README.md on upload)
```

The checkpoint contains the full `RobertaForSequenceClassification` weights
(pooler + classifier), so it loads with
`AutoModelForSequenceClassification.from_pretrained("cevud/codebert-vuln-classifier")`.

### Publish the datasets

```bash
python -m huggingface_hub huggingface-cli upload cevud/cvefixes-benchmark benchmark_manifest_cvefixes.json
python -m huggingface_hub huggingface-cli upload cevud/vudenc-benchmark   benchmark_manifest_vudenc.json
```

> Before publishing, confirm the upstream CVEfixes and VUDENC licenses permit
> redistribution, and strip or document any provenance fields you do not wish
> to share.

## Dataset Construction Methodology

### Source data

`benchmark_manifest_cvefixes.json` — labeled Python function samples from
real-world projects, produced by `src/scripts/convert_cvefixes.py` from the
CVEfixes HuggingFace dataset. Each sample carries:

| Field               | Description                                       |
|---------------------|---------------------------------------------------|
| `sample_id`         | Globally unique identifier                        |
| `project`           | Repository slug                                   |
| `file_path`         | Path inside the repo                              |
| `start_line`        | 1-indexed vulnerable hunk start                   |
| `end_line`          | 1-indexed vulnerable hunk end                     |
| `label`             | 1 = vulnerable, 0 = safe (post-fix)               |
| `vulnerability_type`| CWE identifier (116 unique types)                 |
| `source_code`       | Raw snippet from the diff                         |
| `commit_id`         | Fix-commit SHA                                    |
| `target_commit`     | Explicit commit to read (null → use parent)       |
| `repo_url`          | GitHub HTTPS URL                                  |
| `diff_with_context` | Unified diff with surrounding context             |

### Enrichment pipeline

1. **Clone** — `git clone --filter=blob:none <repo_url>` into a temp dir.
   Only commit history is fetched; file blobs are retrieved on demand.
2. **Commit resolution** — for `label=1`, read `parent(commit_id)` (the
   pre-fix vulnerable version); for `label=0`, read `target_commit` (the
   fixed version).
3. **File retrieval** — `git show <commit>:<file_path>` to get the complete
   source file at the relevant commit.
4. **Function expansion** — `ast.parse` + `expand_to_function` locates the
   smallest enclosing `def` / `async def` around `start_line`.  Falls back
   to a ±3 / +8 line window for module-level code.
5. **Import collection** — `collect_module_imports` extracts all
   `import` / `from ... import` statements at module scope.
6. **Cross-file context** (optional) — `collect_cross_file_context` resolves
   up to 3 intra-repo imports and fetches their source (up to 200 lines each)
   via `git show`.
7. **Snippet assembly** — `build_context_snippet` concatenates:
   ```
   <imports>

   <enclosing function source>

   # ---- cross-file context ----
   # <mod_a.py>
   <source of mod_a>
   ```

### Few-shot capping

After enrichment, samples are capped to enforce the few-shot budget:

- **Per-class cap** — limits vulnerable and safe samples independently (e.g. 50 each).
- **Per-CWE cap** — limits any single CWE type (e.g. 10 samples) to prevent
  dominance by over-represented vulnerability classes.
- **Total cap** — hard limit on total samples (e.g. 500).

Capping is applied **before** splitting so the class and CWE balance is
preserved across all splits.

### Stratified splitting

Projects are shuffled (seed=42) and assigned to splits:

```
test      = first 20 % of projects
validation= next  20 % of projects
train     = remaining 60 %
```

This ensures no project appears in more than one split, preventing data
leakage and providing a realistic generalization measurement.

### Balanced coverage

The CVEFixes manifest is naturally balanced (1:1 vulnerable/safe). The greedy
project selection ensures broad CWE coverage even with small `--max-projects`
values.

## Reproducibility

All randomness is controlled:

| Source of randomness | Seed |
|----------------------|------|
| Dataset split        | 42   |
| Sample capping       | 42   |
| Model init + dropout | 42   |
| Training shuffle     | 42   |

To reproduce a full training run:

```bash
python -m src.training.cli run-all \
  --max-workers 4 \
  --epochs 5 \
  --batch-size 8
```

## Troubleshooting

**`git clone` fails for a repo (404 / auth / rate-limit)**
The builder logs the error and skips that project.  Re-run after fixing
network access; each run re-clones the needed repos (clones are not
persisted between runs, so a prior failure leaves nothing to clean up).

**OOM during training**
Reduce `--batch-size` to 4 or 2.  The trainer uses `use_cpu=True` by
default; GPU is supported automatically if `torch.cuda.is_available()`.

**Low ROC-AUC / PR-AUC**
Check `dataset_summary.json` to ensure enough samples per CWE type are present.
Increase `--max-samples-per-class` or `--max-projects` to add more signal.

**ModelManager does not pick up the new model**
Ensure `config.json` `models.classifier_model` points to the directory
containing `pytorch_model.bin` and `config.json`.  Restart the process
after editing the config.
