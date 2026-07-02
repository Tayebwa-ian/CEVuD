# CEVuD: Cost-Effective Vulnerability Detection

CEVuD is a Python-based security triage pipeline that combines static analysis, local semantic scoring, and optional LLM-based remediation synthesis. The repository is designed for cost-aware code review workflows where only the highest-risk findings are escalated to a frontier model.

## What the pipeline does

1. Stage 1: static analysis with Semgrep
   - Runs Semgrep with the default Python ruleset plus the repository's custom taint rules.
   - Produces a JSON findings file that records file locations and severity labels.

2. Stage 2: local triage and gating
   - Extracts complete function-level code blocks from the target workspace using Python AST parsing.
   - Runs a local CodeBERT-based classifier to estimate the probability that a snippet is a security vulnerability.
   - Combines Semgrep severity and SLM probability with the configured weighted formula:
     - $R = (W_1 \cdot S_{sev}) + (W_2 \cdot P_{slm})$
   - Escalates when the score meets the threshold, when a critical static severity is found, or when the SLM score crosses the override boundary.

3. Stage 3: optional remediation synthesis
   - Reads the Stage 2 triage report and, if escalation is triggered, passes the flagged findings to a DeepAgent-style LLM workflow.
   - Uses a local vector store for cross-file context and writes a consolidated remediation dossier.

## Repository layout

- [src/triage_orchestrator.py](src/triage_orchestrator.py): orchestrates Stage 2, extracts code snippets, and writes the triage ledger.
- [src/agent.py](src/agent.py): runs the Stage 3 reasoning loop and writes the remediation report.
- [src/dataset_ingest.py](src/dataset_ingest.py): seeds the vector store from benchmark data or a repository crawl.
- [src/evaluate_pipeline.py](src/evaluate_pipeline.py): runs the benchmark harness against the gold-standard cases.
- [src/model_manager.py](src/model_manager.py): centralizes loading of the local classifier and embedding model.
- [src/vector_store.py](src/vector_store.py): stores function-level code blocks and embeddings in SQLite.
- [src/llm_factory.py](src/llm_factory.py): provides the LLM provider abstraction used by the agent.
- [tests/](tests/): regression tests for parsing, gating, ingest, vector store access, and the end-to-end pipeline.
- [.github/workflows/](.github/workflows/): CI workflows for the repository-local pipeline and the reusable external-repo workflow.
- [Dockerfile](Dockerfile): builds a runtime image with Semgrep, application code, and the local model assets.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install semgrep
```

### 2. Seed the local context store

For benchmark mode:

```bash
python src/dataset_ingest.py --mode benchmark --file tests/data/gold_standard.json
```

For repository mode:

```bash
python src/dataset_ingest.py --mode repo --path /path/to/target/code
```

### 3. Run Stage 1: static scan

```bash
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml --no-git-ignore --exclude tests --exclude workspace_storage --json --output /path/to/target/semgrep_results.json /path/to/target
```

### 4. Run Stage 2: local triage and gating

```bash
python src/triage_orchestrator.py --workspace /path/to/target --config config.json --exclude-dirs "tests,workspace_storage,src"
```

This writes a file named `stage1_2_triage.json` under the target workspace's artifact directory.

### 5. Run Stage 3: remediation synthesis

```bash
export OPENAI_API_KEY=your-key
python src/agent.py --workspace /path/to/target --config config.json
```

The agent writes a consolidated `remediation_dossier.md` when Stage 2 decides that escalation is required.

## Configuration

The default gate settings are defined in [config.json](config.json):

- Static weight: `0.4`
- SLM weight: `0.6`
- Escalation threshold: `0.52`
- SLM override threshold: `0.90`

The runtime expects a workspace that contains the output files from the static scan and a writable artifact directory under `workspace_storage/artifacts/`.

## Evaluation and tests

Run the unit tests:

```bash
python -m pytest tests/
```

## CI/CD usage

The repository includes reusable GitHub Actions workflows for local PR scanning and external-repo scanning. The workflows run the same three stages in Docker and upload the generated artefacts under `workspace_storage/artifacts/run_<sha>/`.
