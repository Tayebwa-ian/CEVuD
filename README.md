# CEVuD: Cost-Effective Vulnerability Detection

CEVuD is a multi-stage security orchestration pipeline designed to identify vulnerabilities in Python codebases. It balances depth and cost by using a "Gated Logic" approach—only escalating findings to high-reasoning LLMs when local analysis confirms a high probability of risk.

## 🚀 Pipeline Workflow

1.  **Stage 1: Static Taint Analysis (Semgrep)**
    *   Scans the `git diff` for data flow vulnerabilities.
    *   Traces untrusted **Sources** (API inputs, CLI args) to dangerous **Sinks** (SQL, Shell, Filesystem).
    *   Assigns a severity weight ($S_{sev}$).

2.  **Stage 2: Semantic Gating (Local SLM)**
    *   Modified functions are extracted via AST.
    *   A local Small Language Model (SLM) evaluates the threat probability ($P_{slm}$).
    *   **Risk Formula:** $R = (0.4 \cdot S_{sev}) + (0.6 \cdot P_{slm})$.
    *   If $R \ge 0.65$, the workflow escalates to Stage 3.

3.  **Stage 3: Deep Synthesis (DeepAgent)**
    *   A frontier LLM performs task decomposition to trace data lineage.
    *   Queries the **Local Vector Store** (SQLite) for cross-file context.
    *   Generates a `remediation_dossier.md` with PoC steps and patches.

## 🔍 Detection Capabilities
- **Injection:** SQLi, Command Injection (RCE), and SSRF.
- **Data Safety:** Unsafe Deserialization (Pickle/YAML).
- **Web Security:** Reflected XSS and Path Traversal.
- **Secrets:** Hardcoded credentials and API tokens.

## 🛠️ Tech Stack

- **Core:** Python 3.14.6
- **Static Engine:** Semgrep OSS (Taint Mode)
- **ML Models:** CodeBERT (Classification & Embeddings)
- **Context Store:** SQLite with binary vector blobs
- **Reasoning:** `deepagents` (Task-driven LLM orchestration)

## 📂 Project Structure

- `src/diff_parser.py`: AST-based code extraction from git diffs.
- `src/triage_orchestrator.py`: Mathematical gating logic and SLM inference.
- `src/vector_store.py`: Local semantic index for cross-file context.
- `src/agent.py`: The final stage reasoning engine.
- `src/evaluate_pipeline.py`: Benchmark runner for measuring pipeline accuracy.

## ⚙️ Quick Start

1. **Setup (Benchmark):** `python src/dataset_ingest.py --mode benchmark --file gold_standard.json`
2. **Setup (Full Repo):** `python src/dataset_ingest.py --mode repo --path ./my_project`
2. **Benchmark:** `python src/evaluate_pipeline.py` to verify detection Recall/Precision.
4. **Run Pipeline:**
   ```bash
   python src/triage_orchestrator.py
   python src/agent.py
   ```
3. **CI/CD:** Use `.github/workflows/security_pipeline.yml` for automated PR scanning.
