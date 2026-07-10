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

## Local execution

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

After the workflow completes, inspect the uploaded artifact under `workspace_storage/artifacts/run_<sha>/`.

## Testing

Run the unit tests:

```bash
python -m pytest tests/
```

## Benchmarking

The evaluation suite tests the cost-effectiveness and accuracy of the CEVuD gating logic against real-world vulnerability datasets.

### 1. Prepare the Benchmark Manifest
First, convert a raw vulnerability dataset (like CVEfixes or VUDENC) into the CEVuD manifest format using the provided scripts:

For CVEfixes (SQLite DB):
```bash
python src/scripts/convert_cvefixes.py --db /path/to/cvefixes.db --output benchmark_manifest_cvefixes.json
```

For VUDENC (JSON):
```bash
python src/scripts/convert_vudenc.py --input /path/to/vudenc.json --output benchmark_manifest_vudenc.json
```

### 2. Run the Comparative Evaluation
Execute the evaluation suite against your newly generated manifest. The script dynamically clones repositories at their exact historical commits, extracts the raw scores (Semgrep & SLM), splits the dataset, tunes the gate parameters, and evaluates the baselines.

```bash
python src/evaluation/run_comparative_evaluation.py --manifest benchmark_manifest_cvefixes.json --config config.json
```

### 3. Review the Results
All evaluation artifacts are permanently persisted in a timestamped directory (typically under `workspace_storage/evaluations/comparative_eval_<timestamp>/`). This folder contains:
- `comparative_report.md`: A detailed Markdown report of metrics, baseline comparisons, and dataset splits.
- `comparative_report.json`: The machine-readable equivalent.
- `gate_sensitivity_heatmap.png` & `gate_threshold_sensitivity.png`: Visual graphs proving the optimal tuning parameters.
- `raw_scores_cache.json`: The extracted scores, so you can safely re-run the evaluation (`--cache`) without re-cloning repositories.