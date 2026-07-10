# CEVuD: Cost-Effective Vulnerability Detection

CEVuD is a Python-based security triage pipeline designed to integrate into large-scale CI/CD environments. It combines static analysis (Semgrep), local semantic machine learning (CodeBERT), and optional frontier LLM-based remediation synthesis.

The system is designed for **cost-aware code review workflows**. Running a frontier LLM (like GPT-4 or Claude) on every single file change is prohibitively expensive and slow. CEVuD solves this by using a zero-marginal-cost edge compute layer (a "gate") to filter out false positives and low-risk code, escalating only the highest-risk findings to the LLM.

---

## 🏗️ Architecture Overview

The pipeline operates in three distinct stages:

1. **Stage 1: Static Taint Analysis**
   - Runs Semgrep with standard Python rules plus proprietary custom taint rules.
   - Traces untrusted data from sources to dangerous sinks.
   - Outputs a fast, deterministic JSON report of potential vulnerabilities and their severity (`ERROR`, `WARNING`, `INFO`).

2. **Stage 2: Local Triage & Gating (The "Smart Gate")**
   - Parses the target workspace into an Abstract Syntax Tree (AST) to extract pristine function blocks corresponding to Semgrep findings.
   - Uses a local CodeBERT sequence classifier (`jayansh21/codesheriff-bug-classifier`) to output a probabilistic threat score (`P_slm`).
   - Calculates a combined risk score: `R = (W_1 * S_sev) + (W_2 * P_slm)`.
   - **Escalates** the finding to Stage 3 if `R` exceeds a configurable threshold, or if critical static/semantic override thresholds are triggered.

3. **Stage 3: Remediation Synthesis**
   - Only executed for findings that breach the Stage 2 gate.
   - An autonomous LLM agent receives the flagged code.
   - Uses a local SQLite vector store to perform Retrieval-Augmented Generation (RAG) for cross-file context mapping.
   - Writes a highly structured, developer-ready `remediation_dossier.md` containing root cause analysis, exploit proof-of-concepts, and a secure code patch.

---

## 📁 Repository Layout

### Core Execution
- `src/triage_orchestrator.py`: Orchestrates Stage 2. Scores snippets and writes the triage ledger (`stage1_2_triage.json`).
- `src/agent.py`: Runs Stage 3. Uses the triage ledger and vector store to generate the remediation dossier.
- `src/dataset_ingest.py`: Parses codebases/manifests and indexes them into the SQLite vector store.
- `src/diff_parser.py`: Analyzes Git diffs for PR-based incremental scanning.

### Core Support & Models
- `src/model_manager.py`: Centralizes HuggingFace model loading and local inference (CodeSheriff and embeddings).
- `src/vector_store.py`: Manages SQLite storage for AST-parsed function blocks and their dense vectors.
- `src/llm_factory.py`: Interface wrapper for calling external frontier LLMs.

### Evaluation Framework
- `src/evaluation/run_comparative_evaluation.py`: The master evaluation suite. Used to prove the cost-to-safety tradeoff across thousands of real-world commits.
- `src/evaluation/grid_search.py`: Optimizes gating weights without human bias.
- `src/evaluation/repo_provider.py`: Dynamically fetches specific historical git commits for testing and cleans them up.
- `src/scripts/convert_cvefixes.py` & `convert_vudenc.py`: Dataset parsers to transform raw vulnerability databases into CEVuD manifest format.

---

## 🚀 Quick Start & Usage

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install semgrep
```

### 2. Standard Codebase Scan
To scan a target codebase located at `/path/to/target`:

**A. Vector Indexing**
```bash
python src/dataset_ingest.py --mode repo --path /path/to/target
```

**B. Static Scan (Stage 1)**
```bash
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml \
  --no-git-ignore --json --output /path/to/target/semgrep_results.json /path/to/target
```

**C. Local Gating (Stage 2)**
```bash
python src/triage_orchestrator.py --workspace /path/to/target --config config.json
```
*(This produces `stage1_2_triage.json` inside the target's artifact directory)*

**D. Remediation Agent (Stage 3)**
```bash
export OPENAI_API_KEY=your-api-key
python src/agent.py --workspace /path/to/target --config config.json
```

### 3. Evaluating the Model (Benchmarking)
If you want to prove the model's token reduction rate on real datasets:
```bash
# Convert a dataset
python src/scripts/convert_cvefixes.py --db cvefixes.db --output benchmark_manifest.json

# Run evaluation
python src/evaluation/run_comparative_evaluation.py --manifest benchmark_manifest.json --config config.json
```
*(Results and sensitivity plots are output to `workspace_storage/evaluations/`)*

---

## ⚙️ Configuration
The default gating thresholds are in `config.json`:
- `weight_static`: 0.4
- `weight_slm`: 0.6
- `escalation_threshold`: 0.52
- `slm_override_threshold`: 0.90
