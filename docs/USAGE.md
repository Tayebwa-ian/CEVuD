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
`microsoft/codebert-base`). It is a single-label (softmax) 5-class model
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

## Full-dataset workflow (CVEfixes → model, VUDENC → gate weights)

CEVuD uses a deliberate **two-corpus split**: the small classifier is built and
measured entirely on **CVEfixes**, while the **gate weights** are found and the
full pipeline is evaluated on **VUDENC**. The two never touch, so the model's
recall/F1 (CVEfixes test) and the gate's TRR / Cost-reduction / recall (VUDENC
test) are independent, leakage-free measurements.

> **Important:** to use the *entire* CVEfixes dataset, pass **no caps** —
> `--few-shot`, `--limit`, `--max-projects`, `--max-samples-per-class`, and
> `--max-total` must all be omitted. The builder uses the whole manifest
> whenever those are absent (`src/training/dataset_builder.py`).

### Part A — Train, validate, and evaluate the model on the full CVEfixes

CVEfixes is the model's default source (`config.json` →
`training.manifest_path = benchmark_manifest_cvefixes.json`). The build step
writes project-stratified **train / validation / test** splits at a **60 / 20 /
20** ratio.

```bash
# 1. Convert the FULL CVEfixes (no --limit). Prefers a local
#    save_to_disk artifact; otherwise streams from HuggingFace.
python src/scripts/convert_cvefixes.py --output benchmark_manifest_cvefixes.json
#    (optional, faster/offline) build the local artifact first:
#    python -c "from datasets import load_dataset; load_dataset('hitoshura25/cvefixes', split='train').save_to_disk('./cvefixes_dataset')"

# 2. Build the splits — NO caps => the entire dataset is used.
python -m training.cli build-dataset

# 3. Train. CVEfixes train fits the model; CVEfixes validation drives
#    early stopping (val-loss plateau), so a high --epochs ceiling is fine.
python -m training.cli train --epochs 30 --batch-size 16 --lr 3e-5

# 4. Evaluate the model on the CVEfixes test split.
python -m training.cli evaluate
```

This is the model's **in-domain** measurement (accuracy / precision / recall /
F1 / F2 / ROC-AUC / PR-AUC). Building the full dataset clones every project at
full history to read the real function + context, so it is slow and
network-heavy; do **not** use `--cache` for reported model numbers.

### Part B — Find the gate weights and evaluate the full pipeline on VUDENC

VUDENC is reserved for the comparative gate study. The runner grid-searches
`(weight_static, weight_slm, escalation_threshold)` **on VUDENC's validation
split only** (never the test split) to maximize F2, then reports on the VUDENC
test split.

```bash
# 5. Convert VUDENC.
python src/scripts/convert_vudenc.py --output benchmark_manifest_vudenc.json --split train

# 6. Grid-search the gate weights + evaluate the whole pipeline on VUDENC.
python src/evaluation/run_comparative_evaluation.py \
  --manifest benchmark_manifest_vudenc.json --config config.json
```

The tuned weights are in `comparative_report.json` → `grid_search.best`
(`weight_static`, `weight_slm`, `escalation_threshold`). Apply them to
`config.json` so the production pipeline uses the VUDENC-tuned gate:

```json
"gate_parameters": {
  "weight_static": "<best.weight_static>",
  "weight_slm": "<best.weight_slm>",
  "escalation_threshold": "<best.escalation_threshold>",
  "slm_override_threshold": 0.90
}
```

### End-to-end summary

| Corpus | Role | Command | Produces |
|---|---|---|---|
| CVEfixes | Model train / val / test | `build-dataset` (no caps) → `train` → `evaluate` | fine-tuned classifier + model metrics |
| VUDENC | Gate weight search + pipeline eval | `run_comparative_evaluation.py` | `weight_static` / `weight_slm` / `escalation_threshold` + TRR / Cost-reduction / recall |

Run the unit tests:

```bash
python -m pytest tests/
```

The fast unit suite has no external dependencies. The live end-to-end pipeline
tests (`tests/test_pipeline.py`) are gated behind the `--run-e2e` flag and also
require `semgrep` to be installed:

```bash
python -m pytest tests/ --run-e2e
```

If `--run-e2e` is passed but `semgrep` is not on `PATH`, those tests are skipped
rather than failing.

## Benchmarking

The evaluation suite validates the cost-effectiveness and accuracy of the CEVuD gating logic against thousands of real-world Python vulnerabilities.

---

### Dataset Acquisition — No Database Download Required

CEVuD uses two publicly available vulnerability datasets, both accessible **for free via the HuggingFace streaming API**. You do not need to download any large database file.

#### Option A: VUDENC (Recommended starting point)

VUDENC contains ~15,000 real Python functions labeled at line level across seven vulnerability categories (SQLi, XSS, Command Injection, XSRF, RCE, Path Disclosure, Open Redirect).

**Step 1 — Install the HuggingFace `datasets` library:**
```bash
pip install datasets
```

**Step 2 — Run the VUDENC converter:**
```bash
python src/scripts/convert_vudenc.py \
    --output benchmark_manifest_vudenc.json \
    --split train
```

The script streams data from HuggingFace row-by-row (no bulk download), converts each sample to the CEVuD manifest format, and groups them into seven "projects" (one per vulnerability type) for clean train/val/test splitting.

To also include the test split:
```bash
python src/scripts/convert_vudenc.py --output benchmark_manifest_vudenc_test.json --split test
```

**Alternative — Use a local GitHub clone instead of HuggingFace:**
```bash
git clone --depth 1 https://github.com/LauraWartschinski/VulnerabilityDetection
python src/scripts/convert_vudenc.py \
    --local-dir VulnerabilityDetection/data \
    --output benchmark_manifest_vudenc.json
```

---

#### Option B: CVEfixes (Larger scale, Python-only subset)

CVEfixes links public CVEs to the exact open-source commits that introduced and patched them. This gives us rich real-world vulnerability provenance.

> **Balanced output (1:1 vulnerable / safe).** Each CVEfixes row carries both a
> `vulnerable_code` (pre-fix) and a `fixed_code` (post-fix) snippet. The converter
> emits **two** samples per row: the pre-fix function as a `label=1` (vulnerable)
> sample and the post-fix function as a `label=0` (safe) sample, anchored to the
> respective pre- / post-image diff line ranges. Treating the commit **before** the
> fix as vulnerable and the commit **after** the fix as safe therefore produces a
> naturally balanced dataset with no synthetic oversampling or resampling.

**Step 1 — Install the HuggingFace `datasets` library (if not already installed):**
```bash
pip install datasets
```

**Step 2 — Run the CVEfixes converter:**
```bash
python src/scripts/convert_cvefixes.py \
    --output benchmark_manifest_cvefixes.json
```

To limit the number of samples for rapid iteration (e.g. first 5,000):
```bash
python src/scripts/convert_cvefixes.py \
    --output benchmark_manifest_cvefixes.json \
    --limit 5000
```

To use an alternative HuggingFace dataset ID:
```bash
python src/scripts/convert_cvefixes.py \
    --dataset dima806/fixedbugs \
    --output benchmark_manifest_cvefixes.json
```

---

### Running the Comparative Evaluation

Once you have a manifest, run the full evaluation suite. The script performs grid search on the validation split, evaluates all baselines, and generates output artifacts.

```bash
python src/evaluation/run_comparative_evaluation.py \
    --manifest benchmark_manifest_vudenc.json \
    --config config.json
```

To skip re-running Semgrep/SLM on data you have already processed:
```bash
python src/evaluation/run_comparative_evaluation.py \
    --manifest benchmark_manifest_vudenc.json \
    --config config.json \
    --cache workspace_storage/evaluations/raw_scores_cache.json
```

---

#### Speeding up the evaluation (and what is actually expensive)
The **grid search is not the expensive step.** It is a pure sweep over the
*cached* `severity_weight` / `slm_score` arrays — a 21×21 grid is
evaluated in milliseconds, with zero Semgrep or model calls. The real cost
is Step 1 (`RawScoreExtractor.extract`), which runs Semgrep once per
project and the SLM, and — for CVEfixes — **clones the entire repo**
for every project. Levers, cheapest first:

* **`--cache <path>`** — reuse a `raw_scores_cache.json`. Grid search,
  every baseline, and the linearity/override ablations all read this cache;
  only `--force-recompute` re-extracts. This is the #1 win for
  iteration (skips the clone + Semgrep + SLM entirely).
  **Caveat — never use `--cache` for the *reported* gate-study
  numbers:** it reuses cached `severity_weight` / `slm_score`, i.e. it
  **skips Semgrep AND the SLM**. The `run_comparative_evaluation.py`
  runner now prints a loud warning when `--cache` is used, so a
  no-Semgrep report cannot be produced by accident. Stage-1
  Semgrep severity is a hard, non-skippable input to the gate;
  `raw_score_extractor._run_semgrep` also **fails fast** if the
  `semgrep` binary is missing, rather than silently scoring every
  sample at severity 0.0.
* **`--inline`** — score `git_source` projects from their embedded
  `source_code` / `fixed_code` instead of cloning the real repo. Removes
  every `git clone` (and the network round-trip), and is the only mode
  that runs fully offline / air-gapped. **Semgrep still runs** — it just
  scans the isolated materialized snippets instead of the whole repo
  root, so it is a weaker (less representative) Stage-1 signal. Use
  the default (git-clone) path for the reported study.
* **`--weight-step` / `--threshold-step`** — coarsen the grid
  (e.g. `0.1` → an 11×11 grid) for quick iteration. The default
  `0.05` (21×21) is already cheap; coarsening mostly shrinks the
  heatmap resolution, not wall-clock time.
* **`--limit` at convert time** — fewer projects / samples.

#### Cost KPIs in the report
The comparative report surfaces **TRR** (Token Reduction Rate) and
**Cost reduction** (a.k.a. CSR / `cost_savings_ratio`) alongside
`escalation_rate`. They are *deliberately distinct* metrics, not duplicates:

- **TRR** `= 1 − escalation_rate` — the share of samples (and, under CEVuD's
  uniform per-snippet token assumption, the share of tokens) that never reach
  the LLM. A raw *volume* metric.
- **Cost reduction** `= TRR × (1 − r)` where `r = c_gate / c_llm` (default
  `0.02`) — the *monetary* saving vs the Always-LLM baseline. It sits slightly
  *below* TRR because the gated pipeline still runs a cheap local scan (Semgrep
  + edge SLM) on every snippet. As `r → 0` the two converge.

Full formulas and justifications (incl. the cost-model derivation) are in
`docs/METRICS.md`. Recall is centered via the `F2` selection metric
(beta=2 weights recall twice as heavily as precision).

### Evaluation Output

All results are persisted in a timestamped directory under `workspace_storage/evaluations/comparative_eval_<timestamp>/`:

| File | Description |
|---|---|
| `comparative_report.md` | Human-readable full report with all metrics and baseline comparisons |
| `comparative_report.json` | Machine-readable equivalent for programmatic processing |
| `raw_scores_cache.json` | Cached Semgrep/SLM scores — reuse this to re-run without re-scanning |
| `gate_sensitivity_heatmap.png` | 2D heatmap proving optimal grid-searched weight selection |
| `gate_threshold_sensitivity.png` | Line plot showing F2 score across escalation threshold values |
| `gate_weight_sensitivity.png` | Line plot showing how recall and precision shift with `weight_static` |