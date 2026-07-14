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
   - Contains AST-parsed function definitions and their dense CodeBERT (`microsoft/codebert-base`) embeddings, used for RAG retrieval only.
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

## 🏋️ Custom Classifier Training (`src/training/`)

The `src/training/` package provides a **few-shot, no-augmentation** pipeline to
fine-tune a custom CodeBERT vulnerability classifier on the existing CVEFixes
benchmark. It reuses the same enrichment utilities as the production pipeline,
so train/inference context is identical.

- `dataset_builder.py`: Clones repos, expands samples to full function blocks with
  imports, applies few-shot capping (per-class, per-CWE, total), and splits by
  project. Uses `.training_cache/clones/` to persist clones across runs.
- `trainer.py`: Fine-tunes `microsoft/codebert-base` with HuggingFace `Trainer`,
  saving the best checkpoint by validation F1.
- `evaluator.py`: Computes accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC,
  and generates calibration/ROC/PR/confusion plots.
- `cli.py`: Unified CLI (`build-dataset`, `train`, `evaluate`, `run-all`).

**AI Rule**: The training pipeline never augments data. It only selects and caps
the existing real-world samples. If a model cannot learn from the capped dataset,
increase the caps (more projects or more samples per class), not synthetic data.

## 📁 Workspace Storage Layout

`workspace_storage/` is the single on-disk tree for all runtime artifacts. Its
layout is defined **once** in `config.json → paths` and materialized by the
helpers in `src/run_context.py`. No Python module or CI workflow hardcodes a
`workspace_storage/...` path directly.

| Directory (relative to `workspace_root`) | Config key | Used for |
|---|---|---|
| `artifacts/<run_id>/` | `artifacts_subdir` | Stage 2 triage ledger, Stage 3 remediation dossier, Semgrep JSON per run |
| `evaluation_runs/<eval_id>/` | `evaluations_subdir` | Comparative evaluation outputs (reports, plots, caches) |
| `codebase_vectors/` | `vector_db_dir` | Local SQLite RAG vector store |
| `model_cache/` | `model_cache_dir` | HuggingFace model weights cache |

**AI Rule**: If you need to compute any of these paths, import the matching
helper from `src/run_context.py` (`get_artifact_dir`, `get_vector_db_dir`,
`get_model_cache_dir`, `get_eval_dir`, `get_semgrep_output_path`,
`get_triage_report_path`, `get_remediation_dossier_path`). Never concatenate
`workspace_storage` manually.

---

## ⚖️ Configuration Contract
All core thresholds are stored in `config.json`. 
- `weight_static` + `weight_slm` MUST equal 1.0.
- `escalation_threshold` (0.2 for the tuned gate) defines the baseline for sending code to Stage 3.
- The static/SLM override was evaluated as an ablation and removed from the production pipeline because it had zero measurable effect on any metric (precision, recall, F2). The evaluation suite still tests it for reproducibility.

**AI ASSISTANT INSTRUCTION**: If a user asks you to "tune" the pipeline, do not manually alter `config.json` blindly. Instead, run the `run_comparative_evaluation.py` suite over a benchmark dataset, read the resulting `comparative_report.json` and sensitivity plots, and propose the data-backed weights to the user.

**The Local SLM (Stage 2 classifier).** The edge gate is whatever is set in `config.json → models.classifier_model` (currently `Denash/codebert-vuln-classifier`, fine-tuned from `microsoft/codebert-base`, 125M params). It is a single-label (softmax) 5-class model; `ModelManager` auto-detects this and gates on its **Security Vulnerability** class, so the gate receives a single `[0, 1]` threat probability (`P_slm`). To swap in a different classifier, change `models.classifier_model` in `config.json` — both single-label (softmax) and multi-label (sigmoid) heads are supported automatically.
