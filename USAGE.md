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
    python src/dataset_ingest.py --mode benchmark --file src/data/gold_standard.json
    ```
    *This creates a `vulnerability_samples/` directory containing the test code.*

*   **Repository Mode:** Index an entire local codebase for RAG context.
    ```bash
    python src/dataset_ingest.py --mode repo --path ./path/to/your/code
    ```

#### B. Stage 1: Static Analysis
Scan the materialized samples (or your own code) to generate a findings report.
```bash
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --exclude src --exclude workspace_storage --json --output semgrep_results.json vulnerability_samples/
```

#### C. Stage 2: Triage & Mathematical Gating
Compute the Risk Score ($R$) using the CodeBERT SLM.
```bash
python src/triage_orchestrator.py
```
*Check `stage1_2_triage.json` to see if the escalation threshold was met.*

#### D. Stage 3: Deep Synthesis
If the gating logic triggers escalation, run the Reasoning Agent:
```bash
python src/agent.py
```
The final report will be stored in `workspace_storage/artifacts/run_<ID>/remediation_dossier.md`.

---

## 2. Remote Execution (GitHub Actions)

The pipeline is fully automated via `.github/workflows/security_pipeline.yml`.

### Triggers
- **Pull Requests:** Automatically scans the `git diff` against the `main` or `master` branches.

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

## 3. Benchmarking & Evaluation
To measure the **Recall** and **Token Reduction Rate (TRR)** without running a full pipeline:
1. Seed the benchmark data (Step A above).
2. Run the evaluator:
   ```bash
   python src/evaluate_pipeline.py
   ```