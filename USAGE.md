# CEVuD Usage Guide: Local & Remote Execution

This document provides technical instructions for running the Cost-Effective Vulnerability Detection (CEVuD) pipeline.

## 1. Local Development Setup

### Prerequisites
- **Python:** 3.10 or higher (Optimized for 3.14.6)
- **Static Analysis:** [Semgrep OSS](https://semgrep.dev/docs/getting-started/) installed (`pip install semgrep`)
- **Hardware:** CPU-based inference is supported for Stage 2 (CodeBERT).

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
If you intend to run Stage 3 (Deep Synthesis), set your LLM API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Step-by-Step Execution

#### A. Ingestion (Seeding Context)
Before running the pipeline, you must populate the `LocalVectorStore`.

*   **Benchmark Mode:** Seed the DB and deploy code samples for manual testing.
    ```bash
    python src/dataset_ingest.py --mode benchmark --file tests/data/gold_standard.json
    ```
    *This creates a `vulnerability_samples/` directory containing the test code.*

*   **Repository Mode:** Index an entire local codebase for RAG context.
    ```bash
    python src/dataset_ingest.py --mode repo --path /path/to/your/code
    ```

#### B. Stage 1: Static Analysis
Scan the target codebase to generate a findings report (run this command inside the target repo or point it there).
```bash
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --exclude tests --json --output /path/to/your/code/semgrep_results.json /path/to/your/code
```

#### C. Stage 2: Triage & Mathematical Gating
Compute the Risk Score ($R$) using the CodeBERT SLM relative to the target codebase.
```bash
python src/triage_orchestrator.py --workspace /path/to/your/code --exclude-dirs "tests,venv,workspace_storage"
```
*Check `stage1_2_triage.json` within the target's `workspace_storage/artifacts/run_<ID>/` directory to see if the escalation threshold was met.*

#### D. Stage 3: Deep Synthesis
If the gating logic triggers escalation, run the Reasoning Agent on the target workspace:
```bash
python src/agent.py --workspace /path/to/your/code
```
The final report will be stored in `/path/to/your/code/workspace_storage/artifacts/run_<ID>/remediation_dossier.md`.

---

## 2. Remote Execution (GitHub Actions)

### Option A: Reusable Workflow (Scanning Third-Party Repositories)
To scan a completely different codebase in another repository, you can call the reusable workflow directly from your GitHub Actions YAML file.

Create `.github/workflows/security_scan.yml` in your target repository:
```yaml
name: CEVuD Vulnerability Gate

on:
  pull_request:
    branches: [ main, master ]

jobs:
  triage-scan:
    uses: Tayebwa-ian/CEVuD/.github/workflows/reusable_pipeline.yml@main
    with:
      exclude-dirs: "tests,venv,workspace_storage"
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Option B: Local PR Scanning on this Pipeline Repository
The pipeline repository itself runs automated scans on its own code changes using `.github/workflows/security_pipeline.yml`.

### Configuration (Secrets)
To enable Stage 3 analysis in CI, you must add the following GitHub Actions Secrets:
- `OPENAI_API_KEY`: Required for GPT-4o synthesis.
- `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY`: (Optional) if using alternative providers.

### Artifacts & Logs
After the workflow completes:
1.  Go to the **Actions** tab in your repository.
2.  Select the specific workflow run.
3.  Download the `security-triage-dossier-<SHA>` artifact.
    - `stage1_2_triage.json`: View the SLM probability and risk scores.
    - `remediation_dossier.md`: View the AI-generated fix (if escalated).

---

## 3. Running Unit Tests
To run the automated pytest suite that validates AST extraction, gating calculations, and vector store relational lookups, execute:
```bash
python -m pytest tests/
```

---

## 4. Benchmarking & Evaluation
To measure the **Recall** and **Token Reduction Rate (TRR)** without running a full pipeline:
1. Seed the benchmark data:
    ```bash
    python src/dataset_ingest.py --mode benchmark --file tests/data/gold_standard.json
    ```
2. Run the evaluator:
   ```bash
   python src/evaluate_pipeline.py
   ```