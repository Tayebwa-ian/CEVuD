# Project Context: CEVuD

## Purpose

CEVuD is a three-stage pipeline for Python code security triage. It is intended for environments where full LLM review is too expensive to run on every finding, so the system uses a local classifier to decide which cases justify deeper synthesis.

## Current implementation

The repository currently implements the following flow:

1. Stage 1 collects static findings with Semgrep.
2. Stage 2 extracts function-level source windows from the target workspace and scores them with a local CodeBERT classifier.
3. Stage 3 optionally uses an LLM-backed reasoning agent to produce a remediation dossier for escalated findings.

The implementation is driven by the scripts in [src/](src/) and the workflows in [.github/workflows/](.github/workflows/).

## Runtime contract

The main execution contract is:

- The target workspace must contain a Semgrep JSON result file or be scanned before Stage 2 runs.
- Stage 2 reads the Semgrep findings, extracts the enclosing function body, and writes a triage JSON report.
- Stage 3 reads that triage report and only generates a remediation dossier when escalation was triggered.

## Key configuration values

The current defaults are stored in [config.json](config.json):

- Static weight: `0.4`
- SLM weight: `0.6`
- Escalation threshold: `0.52`
- SLM override threshold: `0.90`
- Artifact directory: `workspace_storage/artifacts`
- Vector database directory: `workspace_storage/codebase_vectors`

## Data flow

- [src/dataset_ingest.py](src/dataset_ingest.py) ingests benchmark cases or repository functions into the SQLite-backed vector store.
- [src/triage_orchestrator.py](src/triage_orchestrator.py) uses the vector store and the local model manager to score snippets and produce the gate decision.
- [src/agent.py](src/agent.py) uses the triage report, the vector store, and an LLM factory to create the markdown remediation dossier.
- [src/evaluate_pipeline.py](src/evaluate_pipeline.py) runs the benchmark harness against the gold-standard dataset and persists evaluation artifacts.

## Output artifacts

The pipeline writes machine-readable and human-readable outputs under the target workspace's artifact directories:

- `semgrep_results.json`: raw Stage 1 findings
- `stage1_2_triage.json`: Stage 2 gate decisions and per-finding metrics
- `remediation_dossier.md`: consolidated Stage 3 output
- `evaluation_runs/<timestamp>/`: benchmark metrics and charts from the evaluator

## Operational notes

- The system is intentionally conservative about LLM usage. Stage 3 should only run when the gate says it is necessary.
- The local model and embeddings are cached under the workspace's model cache path to avoid repeated downloads.
- The repository workflows use Docker and mount the target workspace into the container so the same code path can run locally or in CI.
