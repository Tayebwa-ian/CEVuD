# Usage Guide

## Prerequisites

- Python 3.10 or newer
- Semgrep installed in the environment
- Access to the local model assets or a working Hugging Face cache
- Optional API credentials for Stage 3 if you want LLM-based synthesis

Install the Python dependencies:

```bash
pip install -r requirements.txt
pip install semgrep
```

## What you can run

CEVuD is a set of independent, composable commands. Pick the part(s) you need:

| Part | Command | See |
|---|---|---|
| Seed the local RAG vector store | `python src/dataset_ingest.py --mode benchmark\|repo` | [Local execution → Step 1](#1-seed-the-local-vector-store) |
| Run the static (SAST) scan | `semgrep --config …` | [Local execution → Step 2](#2-run-the-static-analysis-stage) |
| Run the triage / gating stage | `python src/triage_orchestrator.py …` | [Local execution → Step 3](#3-run-the-triage-stage) |
| Run the remediation agent | `python src/agent.py …` | [Local execution → Step 4](#4-run-the-remediation-agent) |
| **Train a custom classifier** | `python -m training.cli train …` | [Training the custom classifier](#training-the-custom-classifier) |
| Benchmark / evaluate the gate | `python src/evaluation/run_comparative_evaluation.py …` | [Running the comparative evaluation](#running-the-comparative-evaluation) |
| Run the pipeline end-to-end (eval harness) | `python src/evaluate_pipeline.py …` | `src/evaluate_pipeline.py` |
| Run the test suite | `python -m pytest tests/` | [Running the tests](#run-the-unit-tests) |

The first four are the **production pipeline** (Stages 1–3); the rest are
**developer/experiment commands** for building, measuring, and benchmarking the
system. Every command is also runnable inside the Docker image (see
[Docker usage](#docker-usage)).

## Local models (Stage 2 edge classifier)

The zero-cost local gate uses a small vulnerability classifier loaded and
cached by `src/model_manager.py`. The default is
`jayansh21/codesheriff-bug-classifier` (125M params, fine-tuned on
`microsoft/codebert-base`, the default small model). It is a single-label (softmax) 5-class model
whose **Security Vulnerability** class gives the threat probability `P_slm`.

To use a different local model, set `models.classifier_model` in
`config.json`. Both single-label (softmax) and multi-label (sigmoid) heads
are auto-detected from the model's `id2label` mapping, so no code change is
needed to swap classifiers. The embedding model used only for RAG retrieval
(`models.embedding_model`, default `microsoft/codebert-base`) is separate
from the classifier and can be changed independently.

## Local execution

### Storage layout — single source of truth

All on-disk paths inside `workspace_storage/` are defined in `config.json → paths`
and computed by the helpers in `src/run_context.py`. The production pipeline
(Stages 1–3) and the CI workflows all derive paths from the same source, so you
should never need to hardcode `workspace_storage/...` anywhere.

| Key | Default | Meaning |
|---|---|---|
| `paths.workspace_root` | `workspace_storage` | Root of the runtime artifact tree |
| `paths.artifacts_subdir` | `artifacts` | Per-run Stage 2/3 outputs (`artifacts/<run_id>/`) |
| `paths.evaluations_subdir` | `evaluation_runs` | Comparative evaluation outputs |
| `paths.semgrep_output` | `semgrep_results.json` | Semgrep JSON filename (scoped to the run artifact dir) |
| `paths.triage_report` | `stage1_2_triage.json` | Stage 2 ledger filename |
| `paths.vector_db_dir` | `codebase_vectors` | SQLite RAG store |
| `paths.model_cache_dir` | `model_cache` | HuggingFace cache |

`src/run_context.py` exports helpers (`get_artifact_dir`, `get_vector_db_dir`,
`get_model_cache_dir`, `get_eval_dir`, `get_semgrep_output_path`,
`get_triage_report_path`, `get_remediation_dossier_path`) that combine these
keys with the resolved workspace root and run id. Use them instead of manual
path concatenation.

### 1. Seed the local vector store

Use benchmark mode to populate the SQLite store with the bundled gold-standard examples:

```bash
python src/dataset_ingest.py --mode benchmark --file tests/data/gold_standard.json
```

Use repository mode to index a target repository:

```bash
python src/dataset_ingest.py --mode repo --path /path/to/target/code
```

### 2. Run the static analysis stage

Run Semgrep against the target repository:

```bash
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --no-git-ignore --exclude tests --exclude workspace_storage --json --output /path/to/target/semgrep_results.json /path/to/target
```

The output file should then be available at the path you specified.

### 3. Run the triage stage

```bash
python src/triage_orchestrator.py --workspace /path/to/target --config config.json --exclude-dirs "tests,workspace_storage,src"
```

This writes `stage1_2_triage.json` into the target workspace's artifact directory.

### 4. Run the remediation agent

If Stage 2 escalated any findings, run the agent:

```bash
export OPENAI_API_KEY=your-key
python src/agent.py --workspace /path/to/target --config config.json
```

The agent writes a consolidated `remediation_dossier.md` when the gate decides it should run.

## Docker usage

The repository also ships with a Dockerfile that builds a runtime image with the Python environment, Semgrep, and the model assets. The CI workflows mount the target workspace into the container and run the same scripts as above.

## GitHub Actions

The repository includes two workflows:

- [.github/workflows/security_pipeline.yml](.github/workflows/security_pipeline.yml): scans the current repository on push or pull request.
- [.github/workflows/reusable_pipeline.yml](.github/workflows/reusable_pipeline.yml): can be called from another repository to scan a target codebase.

To enable Stage 3 in CI, provide the relevant secrets for your provider, for example:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `UNIPASSAU_API_KEY`

### RAG context in CI (fast mode)

Neither scanner workflow runs `src/dataset_ingest.py`, so
`workspace_storage/codebase_vectors` is left empty and
`vector_store.get_explicit_flow_context()` returns `[]`. This is
**intentional**: CI stays fast and zero-cost (no embedding/ingest pass),
at the expense of the cross-file "caller/callee" context that
Stage 3 would otherwise attach to each finding. The advertised
cross-file RAG context is simply **absent in CI**; it is fully
available in local runs after you seed the store (see
`python src/dataset_ingest.py --mode repo --path /path/to/target` above).

To opt into RAG inside CI, add a **Stage-0** ingest step before
Stage 1 that indexes the mounted workspace, e.g.:

```yaml
- name: Stage 0 (Optional) Seed RAG vector store
  run: |
    docker run --rm -v ${{ github.workspace }}:/workspace \
      ${{ steps.image.outputs.image }}:latest \
      python src/dataset_ingest.py --mode repo --path /workspace --config /app/config.json
```

Because ingest populates the store from the workspace, the Stage-3
dossier will then include cross-file context. For
`reusable_pipeline.yml` use the `config-path` input to point at the
in-image `config.json` if it lives elsewhere.

After the workflow completes, inspect the uploaded artifact under `workspace_storage/artifacts/run_<sha>/`.

## Training the custom classifier

CEVuD can fine-tune its own local edge classifier (CodeBERT) on a benchmark
dataset. The training pipeline lives in `src/training/` and exposes four
sub-commands:

```bash
python -m src.training.cli build-dataset ...   # 1. build train/val/test splits
python -m src.training.cli train ...           # 2. fine-tune CodeBERT
python -m src.training.cli evaluate ...        # 3. measure the trained model
python -m src.training.cli run-all ...         # 1 + 2 + 3 in one shot
```

### Step 1 — Build the dataset

```bash
# Few-shot preset: 20 projects, 50 samples/class, ~500 total
python -m src.training.cli build-dataset --few-shot --max-workers 8

# Custom caps:
python -m src.training.cli build-dataset \
  --max-projects 30 \
  --max-samples-per-class 100 \
  --max-total 1000 \
  --max-workers 8

# With cross-file context (slower):
python -m src.training.cli build-dataset --few-shot --cross-file --max-workers 8
```

### Step 2 — Train (with custom parameters)

Every flag overrides the default in `src/training/config.py`. **`--epochs` is
only a ceiling** — training always stops early when the validation loss stops
improving (see `--early-stopping-patience`), so you can safely set a high epoch
count; the run ends as soon as it plateaus rather than wasting compute.

```bash
# Minimal:
python -m src.training.cli train --epochs 20

# Full custom example:
python -m src.training.cli train \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-5 \
  --freeze-backbone \
  --early-stopping-patience 5 \
  --early-stopping-threshold 0.001
```

`train` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `20` | Max training passes over the data; early stopping ends sooner |
| `--batch-size` | `8` | Per-device batch size |
| `--lr` | `2e-5` | AdamW learning rate |
| `--freeze-backbone` | off | Freeze CodeBERT, train only the classifier head (sample-efficient) |
| `--early-stopping-patience` | `3` | Epochs without validation-loss improvement before halting |
| `--early-stopping-threshold` | `0.0` | Minimum val-loss drop that counts as progress |
| `--allow-noisy-data` | off | Train despite contradictory samples (not recommended) |

Artifacts are written to `training_output/<run_timestamp>/` (with a stable
`latest` symlink); `training_summary.json` records the run's metrics and the
best checkpoint.

### Step 3 — Evaluate the model

```bash
# Uses the most recent model via the `latest` symlink:
python -m src.training.cli evaluate

# Or point at a specific model / test set:
python -m src.training.cli evaluate \
  --model-path training_output/latest/model \
  --test-path src.training_data/test.jsonl \
  --output-dir training_output/latest/eval
```

Writes `metrics.json` plus confusion-matrix, ROC, PR, and calibration plots.
Reported model metrics: accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC.

### One-shot

```bash
python -m src.training.cli run-all --few-shot --epochs 20 --freeze-backbone
```

### Deploy

Set `models.classifier_model` in `config.json` to the trained model directory
(e.g. `training_output/latest/model`) and rebuild the Docker image if needed.

See `TRAINING.md` for the full methodology, reproducibility checklist, and
troubleshooting.

## Training and evaluation

For full-dataset model training, gate weight search, and comparative evaluation,
see **[`docs/TRAINING.md`](TRAINING.md)**. That guide covers:

- Building the CVEfixes training corpus (`build-dataset`)
- Training the Stage-2 classifier (`train`, `evaluate`, `run-all`)
- Running the gate study on VUDENC (`run_comparative_evaluation.py`)
- Applying tuned weights back to `config.json`

## Tests

```bash
python -m pytest tests/            # fast unit tests
python -m pytest tests/ --run-e2e  # also runs the live pipeline (needs semgrep)
``` |