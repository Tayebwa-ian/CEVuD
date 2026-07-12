# Training a Custom Vulnerability Classifier (Few-Shot)

## Overview

CEVuD's training pipeline uses a **few-shot, no-augmentation** approach to fine-tune a
small CodeBERT-based classifier on the existing CVEFixes benchmark. Instead of
training on the full 1,538-sample dataset, we select a **small, balanced, credible
subset** — preserving the original, real-world ground truth — and optimize the
pipeline for fast iteration and reproducibility.

The result is a custom model (`microsoft/codebert-base`, ~125M params) that
produces `P(vulnerable) ∈ [0, 1]` and can be dropped into `config.json` as the new
Stage-2 local classifier with zero code changes.

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

## Step 1 — Build a Small, Balanced Dataset

### The Few-Shot Strategy

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

### CLI — Quick Few-Shot Build

```bash
# Recommended starting point: 20 projects, 50 vuln + 50 safe (~100 samples total)
python -m training.cli build-dataset --few-shot --max-workers 8

# Customize the caps:
python -m training.cli build-dataset \
  --max-projects 30 \
  --max-samples-per-class 100 \
  --max-total 1000 \
  --max-workers 8

# Include cross-file context (slower but more signal):
python -m training.cli build-dataset --few-shot --cross-file --max-workers 8
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
python -m training.cli build-dataset --few-shot
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

### Recommended few-shot hyperparameters

Few-shot learning with CodeBERT (125M params) works best with:
- **Fewer epochs** (2–3) to avoid overfitting on small data
- **Smaller batch size** (4–8 on CPU) for stable gradients
- **Lower learning rate** (1e-5 to 2e-5) to preserve pre-trained knowledge
- **Higher warmup** (10–20%) because small batches have noisy gradients

```bash
python -m training.cli train --epochs 3 --batch-size 8 --lr 2e-5
```

### Hyperparameters

| Parameter                  | Default (few-shot) | Description                            |
|----------------------------|--------------------|----------------------------------------|
| `base_model`               | `microsoft/codebert-base` | Base encoder (~125M params) |
| `max_length`               | 512                | Tokenizer truncation length            |
| `batch_size`               | 8                  | Per-device batch size (CPU-safe)       |
| `learning_rate`            | 2e-5               | AdamW learning rate                    |
| `num_epochs`               | 3                  | Full passes over training data         |
| `warmup_ratio`             | 0.1                | Linear warm-up proportion              |
| `gradient_accumulation`    | 1                  | Steps before optimizer update          |
| `weight_decay`             | 0.01               | L2 regularization                      |

The `Trainer` saves the best checkpoint (by validation F1) to
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
python -m training.cli evaluate
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

To reproduce a specific few-shot run:

```bash
python -m training.cli run-all \
  --few-shot \
  --max-workers 4 \
  --epochs 3 \
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
