# CEVuD: Cost-Effective Vulnerability Detection

CEVuD is a multi-stage security orchestration pipeline designed to identify vulnerabilities in Python codebases. It balances depth and cost by using a "Gated Logic" approach—only escalating findings to high-reasoning LLMs when local analysis confirms a high probability of risk.

## 🚀 Pipeline Workflow

1.  **Stage 1: Static Taint Analysis (Semgrep)**
    *   Scans the `git diff` using **standard Registry rules** and **custom Taint rules**.
    *   Traces untrusted **Sources** (API inputs, CLI args) to dangerous **Sinks** (SQL, Shell, Filesystem).
    *   Assigns a severity weight ($S_{sev}$).

2.  **Stage 2: Semantic Gating (Local SLM)**
    *   Modified functions are extracted via AST (including a fail-safe scan if Semgrep is silent).
    *   A local CodeBERT model evaluates the threat probability ($P_{slm}$).
    *   **Risk Formula:** $R = (0.3 \cdot S_{sev}) + (0.7 \cdot P_{slm})$.
    *   If $R \ge 0.55$, the workflow escalates to Stage 3.

3.  **Stage 3: Deep Synthesis (DeepAgent)**
    *   A frontier LLM performs task decomposition to trace data lineage.
    *   Queries the **Local Vector Store** (SQLite) for cross-file context for *all* escalated findings.
    *   Generates a *single, consolidated* `remediation_dossier.md` with PoC steps and patches for all high-risk items.

## 🔍 Detection Capabilities
- **Injection:** SQLi, Command Injection (RCE), and SSRF.
- **Data Safety:** Unsafe Deserialization (Pickle/YAML) and Cryptographic Failures (Weak Hashing).
- **Web Security:** Reflected XSS and Path Traversal.
- **Access Control:** Insecure Direct Object Reference (IDOR).
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
- `src/evaluate_pipeline.py`: Dynamic benchmark runner that executes live Semgrep scans on ledger snippets to verify real-world accuracy.

## ⚙️ Quick Start

1. **Setup (Benchmark):** `python src/dataset_ingest.py --mode benchmark --file src/data/gold_standard.json` (Creates `vulnerability_samples/`)
2. **Setup (Full Repo):** `python src/dataset_ingest.py --mode repo --path ./my_project`
2. **Benchmark:** `python src/evaluate_pipeline.py` to verify detection Recall/Precision.
4. **Run Pipeline:**
   ```bash
   semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --exclude src --json --output semgrep_results.json vulnerability_samples/
   python src/triage_orchestrator.py
   python src/agent.py
   ```
3. **CI/CD:** Use `.github/workflows/security_pipeline.yml` for automated PR scanning.

## 🛠️ Roadmap
- [ ] **CLI Packaging:** Transition to a `pip`-installable package with a `cevud` entry point.
- [ ] **Multi-File Taint:** Enhance SLM fallback to trace variables across multiple function imports.
