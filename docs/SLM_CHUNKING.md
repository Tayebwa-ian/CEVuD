# Small-Model Chunking & Cross-Context Argumentation

> Design note for the Stage-2 classifier change: instead of feeding whole
> functions to the small model, we feed **uniform code chunks** the model can
> actually predict on, and we apply **cross-file context argumentation only when
> a finding escalates to the Stage-3 LLM**.

---

## 1. TL;DR

* **Training:** `src/training/dataset_builder.py` now cuts every enriched
  function into uniform, line-windowed chunks (`src/code_chunks.py`). Each chunk
  inherits the function-level label. The classifier is therefore trained on
  inputs that fit CodeBERT's 512-token window — no silent truncation of the
  vulnerable code.
* **Inference:** `src/triage_orchestrator.py` + `src/model_manager.py` score each
  function by chunking it, running the SLM on every chunk, and aggregating
  (`max` by default: a function is vulnerable if *any* chunk is).
* **Cross-context:** cross-file context (callers / callees from the vector
  store) is **not** fed to the small model. It is attached to a finding **only
  when it escalates**, so the Stage-3 LLM reasons over the suspicious chunks
  plus the real call-graph evidence.

This removes the train/inference skew (same chunking both places) and matches
how state-of-the-art vulnerability localizers work (see §2).

---

## 2. What others do — and what we borrow

We surveyed current approaches for small / encoder models. The short version:
whole-function fed to a 512-token encoder is both a *truncation* problem and a
*localization* problem, and the field has converged on **windowed / chunked /
line-level** inputs plus **augmentation** and **contrastive** training.

| Approach | Core idea | What CEVuD borrows |
|---|---|---|
| **VulDeePecker** (Li et al., 2018) | Represent a program as *code gadgets* — semantically related lines assembled via data/control-flow slicing — rather than the whole function. | The "cut the function into focused windows" instinct. We use simple **uniform line windows** instead of flow-slices (cheaper, no compiler needed, and we still keep the function-level label for the binary task). |
| **LineVul** (Fu & Tantithamthavorn, 2022) | Transformer that produces **line-level** vulnerability scores via attention over the token sequence; a function is vulnerable if any line is. | **Chunk-level scoring + `max` aggregation** is our pragmatic stand-in for line-level localization without a custom attention-head retraining. It directly drives *which* lines we hand to the LLM. |
| **Sliding-window for encoders** (Zhang et al., *Evaluating LLMs for Line-Level Vuln Localization*, 2024) | Explicitly addresses the 512-token cap: *"the truncated sequences of code cause the model to miss vulnerabilities… sliding window processing yields up to 29.7% F1 improvement."* | Our `chunk_code(max_lines, overlap)` sliding window with overlap so a vuln near a boundary is never split from its context. |
| **Semantic-preserving augmentation** (Qi et al., 2024, arXiv:2410.00249) | Natural program transformations (variable rename, dead-branch insert, etc.) that preserve vulnerability semantics; +8.7–10.1% acc / +15.5–23.6% F1 on CodeBERT. | **Recommendation (not yet implemented):** add a `--augment` path in `convert_cvefixes.py` to multiply the small CVEfixes set with safe, realistic transforms. Low risk, high upside for a 1.5k-sample corpus. |
| **Supervised / hierarchical contrastive learning** (Wang et al., *SCL-CVD*, Computers & Security 2024; EMNLP 2024 contrastive CWE work) | Pull same-class / same-CWE representations together, push others apart; `max-pooling` to exceed the length limit. | **Recommendation (future):** when we move to multi-class CWE detection, add a SupCon term and chunk-level `max-pooling`. For the current binary gate, contrastive pre-training is optional. |
| **Synthetic vulnerability injection (SVA / VGX)** | Inject vulnerability patterns into benign code to balance classes. | **Caution:** the literature shows naive injection often *breaks semantics* and hurts real-world F1 (SARD even degrades). We prefer semantic-preserving augmentation over raw injection. |
| **Graph models** (Devign, ReGVD, IVDetect) | Use AST/CFG/DFG structure. | Out of scope for the small edge model (too heavy / needs parsers). Our cross-file context at Stage 3 is the lightweight substitute for graph reasoning. |

**Net borrow list (priority order):**
1. ✅ **Uniform chunking + overlap** (done) — directly fixes truncation + gives localization.
2. ✅ **Cross-context only at escalation** (done) — keeps the SLM fast and avoids feeding it context it can't use.
3. 🔲 **Semantic-preserving augmentation** — next highest ROI for the tiny training set.
4. 🔲 **Contrastive / multi-class CWE head** — only when we extend beyond binary.

---

## 3. Design

### 3.1 The chunking module — `src/code_chunks.py`

* `chunk_code(code, max_lines=64, overlap=8, min_code_lines=2)`
  * One snippet that already fits `max_lines` → a single chunk.
  * Otherwise a sliding window of `max_lines` with `overlap` overlapping lines.
  * Chunks with fewer than `min_code_lines` *real* code-signal lines
    (comments / docstrings / `__version__` assignments don't count — reuse
    `data_quality.code_signal_line_count`) are dropped.
* `aggregate_chunk_scores(scores, method="max"|"mean")` — reduces per-chunk
  probabilities to one function score. `max` is the default and the right
  semantic: one dangerous statement makes the whole function unsafe.

### 3.2 Training — `src/training/dataset_builder.py`

After the noise/contradiction filter, `build_dataset` optionally expands each
enriched function into chunks (off with `--no-chunk`):

* Each chunk becomes its own `EnrichedSample` with `text = chunk.text` and the
  **same** `label` as the parent function (standard function-level-label /
  chunk-level-input setup).
* Chunk-level metadata (`chunk_index`, `chunk_start`, `chunk_end`) is recorded.
* A second **contradiction pass** drops (vuln, safe) chunk pairs with identical
  normalized text.
* Project-level stratified splits and few-shot caps still apply (now at the
  chunk granularity), so class balance is preserved.

The trainer (`src/training/trainer.py`) needs **no change** — it just sees more,
smaller, in-window samples.

CLI: `build-dataset` gains `--no-chunk`, `--chunk-max-lines`, `--chunk-overlap`,
`--chunk-min-code-lines` (also wired into `run-all`). Defaults match
`config.json → training.chunk`.

### 3.3 Inference — `src/triage_orchestrator.py` + `src/model_manager.py`

`TriageOrchestrator.process_pipeline` now:

1. Resolves the real file, extracts the enclosing function span via AST.
2. Builds the SLM input with `build_context_snippet` — **function + module
   imports only** (identical format to training, so no skew). Cross-file context
   is *not* baked in.
3. Calls `ModelManager.get_classifier_chunk_scores(...)` which chunks each
   function, scores every chunk in one batched call, and aggregates.
4. Feeds the aggregated `P_slm` into the existing `linear_weighted_gate`.
5. **On escalation**, attaches to the finding:
   * `suspicious_chunks` — the top-`K` chunks by SLM probability (the evidence),
   * `cross_file_context` — the callers/callees gathered from the vector store.

`src/agent.py` (Stage 3) consumes these two fields and renders them into the
remediation prompt, so the LLM performs **cross-context argumentation** over
the flagged windows and their call-graph, rather than reasoning about the whole
function in isolation.

Config (`config.json → slm_inference`): `chunk_max_lines`, `chunk_overlap`,
`min_code_lines`, `aggregation` (`max`/`mean`), `top_chunks_for_llm` (how many
suspicious chunks to hand the LLM).

### 3.4 Interaction with data quality

Chunking composes with the noise filters in `docs/DATA_QUALITY.md`:

* The converters drop trivial / noise / contradictory (vuln, safe) **functions**
  up front.
* `_drop_contradictory` + the new chunk-level contradiction pass catch any
  near-identical (vuln, safe) **windows** that survive (e.g. a version-bump line
  sitting inside an otherwise-identical function).
* `min_code_lines` at both the function and chunk level keeps signal-free text
  out of training.

---

## 4. Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| `chunk_max_lines` | `config.json → slm_inference` / `--chunk-max-lines` | Window size. 64 ≈ ~400–600 tokens (safe under 512). Raise for bigger functions, lower if precision drops. |
| `chunk_overlap` | `slm_inference` / `--chunk-overlap` | Overlap between windows. 8 keeps a vuln near a boundary attached to context. |
| `min_code_lines` | `slm_inference` / `--chunk-min-code-lines` | Drop pure-comment/docstring chunks. |
| `aggregation` | `slm_inference` | `max` (default, sensitive) vs `mean` (smoother). |
| `top_chunks_for_llm` | `slm_inference` | How many suspicious windows the LLM receives. |
| `--no-chunk` | `build-dataset` / `run-all` | Revert training to whole functions (for A/B comparison). |

## 5. Verification checklist

After regenerating with chunking on:

* `build-dataset` prints `Chunking: N functions -> M chunks` and a
  `chunk filter: X contradictory chunks dropped`.
* `dataset_summary.json → chunking` records `enabled / max_lines / chunks_total`.
* Training loss should fall below `0.69` within the first epochs (chunking does
  not by itself fix label noise — pair it with the filters in
  `docs/DATA_QUALITY.md`).
* At inference, escalated findings in `stage1_2_triage.json` carry
  `suspicious_chunks` and `cross_file_context`; the Stage-3
  `remediation_dossier.md` cites them.
