# CEVuD Documentation Index

This directory is the single home for all CEVuD documentation. `README.md` at
the repository root is the high-level navigation hub and links back here.

## How to navigate

1. **New here?** Start with the root `README.md` (architecture + quick start).
2. **Want the full design rationale?** Read `docs/design.md` and `docs/context.md`.
3. **Training or evaluating the classifier?** Read `docs/TRAINING.md`, then the
   dataset/model cards and the data-quality guide.
4. **Operating the pipeline?** Read `docs/USAGE.md`.
5. **Understanding the latest design change (chunked SLM + cross-context)?**
   Read `docs/SLM_CHUNKING.md`.

## Document map

| Document | What it covers |
|---|---|
| `README.md` (root) | High-level architecture, repo layout, quick start, and this navigation hub's pointer. |
| `docs/INDEX.md` | This file — the map of all documentation. |
| `docs/design.md` | End-to-end data/training/eval pipeline design, module responsibilities, diagrams. |
| `docs/context.md` | AI-assistant guide: architectural philosophy, stage boundaries, state-management rules. |
| `docs/TRAINING.md` | Step-by-step guide to building datasets, training, evaluating, and publishing the classifier. |
| `docs/USAGE.md` | Operational prerequisites and commands for running the pipeline. |
| `docs/DATA_QUALITY.md` | Why the first training run failed to learn, the noise/trivial/contradiction filters, and the regenerate procedure. |
| `docs/DATASET_CARD.md` | Schema and provenance of the CVEfixes (training) and VUDENC (gate-study) manifests. |
| `docs/MODEL_CARD.md` | The custom Stage-2 CodeBERT classifier: architecture, training data, procedure, limitations. |
| `docs/SLM_CHUNKING.md` | **New.** Research on vulnerability-detection approaches we borrow from, plus the chunked-SLM + cross-context-argumentation design. |
| `docs/METRICS.md` | **New.** Canonical metric definitions, formulas, and justifications: recall, F1/F2, TRR, Cost reduction (and supporting confusion-matrix / precision / specificity / escalation-rate metrics) for both the model and the pipeline. |
| `docs/SAFE_COUNTERPARTS.md` | **New.** The "safe counterpart" problem & fix: why the post-fix function is a *noisy/relative* negative, how we measure the contamination, and the three-step remedy (diagnose → verified-benign controls → optional contrastive). Authoritative methodology for the paper. |
| `docs/research_outline.md` | Research write-up / paper outline referencing the classifier and corpora. |

## Where the code lives (quick reference)

| Concern | Path |
|---|---|
| Stage 1 static scan | `semgrep` (external), results consumed by `src/triage_orchestrator.py` |
| Stage 2 gating (SLM) | `src/triage_orchestrator.py`, `src/model_manager.py`, `src/code_chunks.py` |
| Stage 3 remediation (LLM) | `src/agent.py`, `src/llm_factory.py`, `src/vector_store.py` |
| Classifier training | `src/training/` (`dataset_builder.py`, `trainer.py`, `cli.py`, `config.py`) |
| Dataset converters | `src/scripts/convert_cvefixes.py`, `src/scripts/convert_vudenc.py` |
| Shared heuristics | `src/data_quality.py` (noise/trivial/contradiction), `src/code_chunks.py` (chunking) |
| Evaluation / gate study | `src/evaluation/` (`run_comparative_evaluation.py`, `gate_strategies.py`) |
