# CEVuD: Cost-Effective Vulnerability Detection

CEVuD is a Python-based security triage pipeline for large-scale CI/CD
environments. It combines static analysis (Semgrep), a local semantic model (a
small vulnerability classifier), and an optional frontier LLM for remediation
synthesis — gated so the expensive LLM only runs on the highest-risk findings.

> **This file is the navigation hub.** All detailed documentation lives in
> [`docs/`](docs/). Start with [`docs/INDEX.md`](docs/INDEX.md) for the full map,
> and [`docs/design.md`](docs/design.md) / [`docs/context.md`](docs/context.md)
> for architecture and philosophy.

---

## Architecture (summary)

The pipeline runs in three stages:

1. **Stage 1 — Static taint analysis.** Semgrep traces untrusted data to
   dangerous sinks and emits a fast, deterministic JSON report
   (`ERROR` / `WARNING` / `INFO`).
2. **Stage 2 — Local triage & gating (the "Smart Gate").** For each Semgrep
   finding, the genuine function body (+ module imports) is cut into **uniform
   code chunks** and scored by a local CodeBERT classifier. The per-chunk
   probabilities are aggregated into `P_slm`, combined with Semgrep severity via
   `R = W₁·S_sev + W₂·P_slm`, and escalated only when `R` crosses a threshold.
   The classifier is trained on CVEfixes (`src/training/`); the default is
    `Denash/codebert-vuln-classifier` (the default small model), or set `models.classifier_model` in
   `config.json` to a custom model (see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)).
3. **Stage 3 — Remediation synthesis.** Only for escalated findings: an LLM
   agent receives the **suspicious code chunks** and **cross-file context**
   (callers/callees) gathered by the gate, and writes a `remediation_dossier.md`.

### Why chunks, and why cross-context only at escalation?
CodeBERT is capped at 512 tokens; feeding whole functions silently truncates
the vulnerable code. Training and scoring on uniform chunks keeps every input
inside the window and removes train/inference skew. Cross-file context is *not*
fed to the small model (it can't use it well and it bloats the input); instead
it is attached only when a finding escalates, so the LLM reasons over the real
evidence. Full rationale and the research we borrowed from:
[`docs/SLM_CHUNKING.md`](docs/SLM_CHUNKING.md).

---

## Repository layout (high level)

| Area | Path |
|---|---|
| Stage 2 orchestration | `src/triage_orchestrator.py` |
| SLM inference + chunking | `src/model_manager.py`, `src/code_chunks.py` |
| Stage 3 agent | `src/agent.py` |
| Classifier training | `src/training/` |
| Dataset converters | `src/scripts/convert_cvefixes.py`, `src/scripts/convert_vudenc.py` |
| Shared heuristics | `src/data_quality.py`, `src/code_chunks.py` |
| Gate study / evaluation | `src/evaluation/` |

See [`docs/INDEX.md`](docs/INDEX.md) for the complete module map.

---

## Quick start

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install semgrep
```

### 2. Scan a codebase
```bash
# A. Index for RAG
python src/dataset_ingest.py --mode repo --path /path/to/target

# B. Static scan (Stage 1)
semgrep --config p/python --config ./semgrep_rules/custom_appsec_rules.yaml \
  --no-git-ignore --json --output /path/to/target/semgrep_results.json /path/to/target

# C. Local gating (Stage 2) — chunks scored, cross-context attached on escalation
python src/triage_orchestrator.py --workspace /path/to/target --config config.json

# D. Remediation (Stage 3, escalated findings only)
export OPENAI_API_KEY=your-api-key
python src/agent.py --workspace /path/to/target --config config.json
```
Operational details: [`docs/USAGE.md`](docs/USAGE.md).

### 3. Train & evaluate the classifier
CEVuD uses two corpora: **CVEfixes** trains the Stage-2 classifier; **VUDENC**
is the held-out corpus for the gate study.
```bash
# Training corpus (CVEfixes) — noise/trivial filters + chunking ON by default
python src/scripts/convert_cvefixes.py --local-dir ./cvefixes_dataset \
    --output benchmark_manifest_cvefixes.json
python -m src.training.cli build-dataset --manifest benchmark_manifest_cvefixes.json
python -m src.training.cli train --epochs 20 --batch-size 8 --lr 2e-5

# Evaluation corpus (VUDENC) — held out from training
python src/scripts/convert_vudenc.py --output benchmark_manifest_vudenc.json
python src/evaluation/run_comparative_evaluation.py \
    --manifest benchmark_manifest_vudenc.json --config config.json
```
Full walkthrough: [`docs/TRAINING.md`](docs/TRAINING.md). Dataset schema:
[`docs/DATASET_CARD.md`](docs/DATASET_CARD.md). If a training run plateaus at
`loss≈0.693` / `roc_auc≈0.5`, read [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md)
and **delete `training_data/` + `training_output/`** before rebuilding.

---

## Configuration
Key gating thresholds live in `config.json`: `weight_static` (0.15),
`weight_slm` (0.85), `escalation_threshold` (0.2). The new chunking/aggregation behaviour is controlled by the
`slm_inference` block (`chunk_max_lines`, `chunk_overlap`, `aggregation`,
`top_chunks_for_llm`) and the training `chunk` block.

## Tests
```bash
python -m pytest tests/            # fast unit tests
python -m pytest tests/ --run-e2e  # also runs the live pipeline (needs semgrep)
```
