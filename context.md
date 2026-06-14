# Context File: LLM-Guided Two-Stage Vulnerability Detection Pipeline

## 1. Project Purpose
An automated, cost-efficient security gate for Python source code running inside a public CI/CD workspace environment. It intercepts code modifications, uses lightweight AST rules and a fast local machine learning model to compute an exploitability probability score, and escalates to a frontier LLM only when a specific mathematical risk boundary is breached.

## 2 Tech Stack & Runtimes
- **Runtime Environment:** Python 3.14.6 (Isolated Docker Platform layer).
- **Stage 1 (Static Layer):** Semgrep OSS Engine running structural rule sets.
- **Stage 2 (Semantic Layer):** Local CodeBERT embedding tokenization and classification heads compiled via `onnxruntime` for zero-cost CPU processing.
- **Stage 3 (Synthesis Layer):** An LLM-agnostic LangChain engine integrated with `deepagents` for complex task decomposition.
- **Context Storage:** Local file-based SQLite relational databases storing serialized 768-dimensional float blocks.

## 3. Granularity & Slicing Strategy
To avoid losing short, critical software flaws within large source blocks, data is processed at the **logical function/method level** rather than the entire file level.

1. **AST Separation:** Python source changes are split into separate function blocks via Python's native `ast` module.
2. **Granular Embeddings:** Each method is vectorized independently and logged along with its file metadata (`file_path`, `function_name`).
3. **Targeted Context Queries:** When code is flagged, the system looks up semantic matches at the precise function level, keeping the input context compact and clean.

## 4. Systematic Run Storage Architecture
Every execution generates versioned, trackable records stored under a run-specific commit hash key format:

workspace_storage/
└── artifacts/
└── run_SHA256_TIMESTAMP/
├── stage1_2_triage.json       # Structural metrics and evaluation scores
└── remediation_dossier.md     # In-depth mitigation report (if escalated)

## 5. Core Component Contracts

### Data Pre-Processing & Diff Parsing Strategy
- Parse the input `git diff` payload to extract modified filenames and raw line modifications.
- Scan for updated structural identifiers (such as function definitions or classes) using the native `ast` module.
- Use the local vector store to pull external code segments that call or import those specific modified functions, building a complete data flow context package.

### Local Vector Store Schema Definition
The SQLite context database must adhere strictly to the following relational structure:
```sql
CREATE TABLE IF NOT EXISTS codebase_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    function_name TEXT NOT NULL,
    source_code TEXT NOT NULL,
    embedding_blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_file_lookup ON codebase_embeddings(file_path);
```

### Mathematical Gating Logic Formula
The pipeline computes an overall Risk Metric Score ($R$) for every structural match using the following weighted gating formula:

$$R = (W_1 \cdot S_{\text{sev}}) + (W_2 \cdot P_{\text{slm}})$$

Where:
- $S_{\text{sev}}$ is the Semgrep severity score (`INFO` = 0.3, `WARNING` = 0.7, `ERROR` = 1.0).
- $P_{\text{slm}}$ is the local ONNX model's classification probability output.
- $W_1 = 0.4$ and $W_2 = 0.6$ represent the default operational weights.
- If $R \ge 0.65$, trigger Stage 3. Otherwise, log the result and terminate cleanly to save tokens.


## 6. System Execution Manifest & File Map
- `/config.json`: Master orchestration configurations, weights, and model parameters.
- `/requirements.txt`: Pinned version requirements optimized for Python 3.14.6 execution loops.
- `/Dockerfile`: Clean multi-stage build blueprint that keeps deployment container profiles small.
- `/.github/workflows/security_pipeline.yml`: The automation engine file that runs the pipeline, parses changes, and handles artifact retention.
- `/src/diff_parser.py`: AST extraction component that finds modified functions in git patches.
- `/src/triage_orchestrator.py`: The evaluation stage that combines Semgrep data with local model checks and evaluates the gating logic.
- `/src/vector_store.py`: Local database client managing SQLite vector profiles.
- `/src/llm_factory.py`: The model mapping manager that handles agnostic provider switching.
- `/src/agent.py`: Advanced agent runtime using `deepagents` task structures to write the final remediation report.
