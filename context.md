# Project Context: CEVuD

**AI ASSISTANT GUIDE**: If you are an AI reading this, this document contains the core architectural philosophies, state management rules, and development guidelines for CEVuD. Reference this heavily when modifying the codebase.

## 🎯 Purpose and Philosophy

CEVuD (Cost-Effective Vulnerability Detection) is designed around a single undeniable reality in modern AppSec: **Frontier LLMs are too expensive to run on every line of code.**

Our philosophy is **Gated Reasoning**. We do not trust static analysis (too many false positives). We do not trust local SLMs (too small to reason about complex multi-file logic). But when mathematically combined via a linear formula, they serve as a highly accurate, zero-cost gate. 

If you are asked to modify the pipeline, **do not break the staging boundaries**.
- **Stage 1 (Semgrep)** should NEVER talk to an LLM.
- **Stage 2 (Local SLM)** should NEVER talk to the network. It must run locally on the edge.
- **Stage 3 (LLM)** should NEVER process raw files. It should only process the structured snippets output by Stage 2.

---

## 🧠 State Management & Data Flow

CEVuD operates in stateless batches, coordinated by the file system.

1. **The Vector Store (`workspace_storage/codebase_vectors`)**
   - Managed by `src/vector_store.py` backed by SQLite.
   - Contains AST-parsed function definitions and their dense CodeBERT embeddings.
   - **AI Rule**: Do not attempt to query the vector store via generic SQL. Always use `VectorStore` class methods for RAG retrievals.

2. **The Triage Ledger (`stage1_2_triage.json`)**
   - The contract between Stage 2 and Stage 3. 
   - Contains `evaluated_file`, `code_snippet`, risk `metrics`, and a global `gate_decision`.
   - **AI Rule**: If you modify the gating math in `src/evaluation/gate_strategies.py`, ensure the JSON structure output by `src/triage_orchestrator.py` accurately reflects those metric changes.

3. **The Remediation Dossier (`remediation_dossier.md`)**
   - The final, human-readable output of Stage 3. It must always include: Vulnerability Analysis, Source/Sink Lineage, Exploit PoC, and a Remediation Patch.

---

## 📊 The Evaluation Suite (`src/evaluation/`)

The evaluation logic is entirely decoupled from the runtime execution logic. This is intentional.

- `run_comparative_evaluation.py`: The master script. It pulls benchmark manifests, extracts raw scores using `raw_score_extractor.py`, splits the data (train/val/test), and tests our gating logic against baselines.
- `repo_provider.py`: Contains robust `git` operations (with retries and exponential backoff) to dynamically fetch thousands of real-world commits on the fly, process them, and delete them to save disk space.
- `grid_search.py`: Tunes the weights. **AI Rule**: Never hardcode gating weights based on test data. Weights must be dynamically derived via grid search on the validation split.

### Why do we use CVEfixes and VUDENC?
We transitioned to these datasets (converted via `src/scripts/`) because testing on 24 internal cases causes overfitting. These datasets provide thousands of historical Python commits, ensuring the linear gate handles diverse, real-world coding anomalies.

---

## ⚖️ Configuration Contract
All core thresholds are stored in `config.json`. 
- `weight_static` + `weight_slm` MUST equal 1.0.
- `escalation_threshold` (typically 0.52) defines the baseline for sending code to Stage 3.
- `static_override_value` (1.0) and `slm_override_threshold` (0.90) act as safety nets. If either is breached, the risk score is bypassed and escalation is forced.

**AI ASSISTANT INSTRUCTION**: If a user asks you to "tune" the pipeline, do not manually alter `config.json` blindly. Instead, run the `run_comparative_evaluation.py` suite over a benchmark dataset, read the resulting `comparative_report.json` and sensitivity plots, and propose the data-backed weights to the user.
