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
    *   **Risk Formula:** $R = (0.4 \cdot S_{sev}) + (0.6 \cdot P_{slm})$.
    *   If $R \ge 0.52$, the workflow escalates to Stage 3.

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

- `src/diff_parser.py`: Decoupled AST-based code extraction from git diffs with custom exclusions.
- `src/triage_orchestrator.py`: Mathematical gating logic accepting dynamic target workspaces.
- `src/vector_store.py`: Persistent database client supporting dynamic workspace lookup and embeddings.
- `src/agent.py`: Advanced agent runtime executing relative to targeted workspaces.
- `src/evaluate_pipeline.py`: Benchmark runner.
- `tests/`: Suite containing unit tests for AST parser, database store, and gating calculations.

## ⚙️ Quick Start

### 1. Run Unit Tests
To verify all pipeline logic works correctly:
```bash
python -m pytest tests/
```

### 2. Scanning a Target Repository
You can run the pipeline on an external codebase:
```bash
# 1. Generate static analysis findings in the target repo
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --json --output /path/to/target/semgrep_results.json /path/to/target

# 2. Run Stage 2 Triage with custom exclusions and target workspace
python src/triage_orchestrator.py --workspace /path/to/target --exclude-dirs "tests,workspace_storage"

# 3. Run Stage 3 Deep Synthesis Agent
python src/agent.py --workspace /path/to/target
```

### 3. CI/CD Integration (Option A: Reusable Workflow)
Other codebases can scan their PRs using this central pipeline via the reusable workflow:
```yaml
jobs:
  scan:
    uses: Tayebwa-ian/CEVuD/.github/workflows/reusable_pipeline.yml@main
    with:
      exclude-dirs: "tests,workspace_storage"
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 🛠️ Roadmap
- [ ] **CLI Packaging:** Transition to a `pip`-installable package with a `cevud` entry point.
- [ ] **Multi-File Taint:** Enhance SLM fallback to trace variables across multiple function imports.
