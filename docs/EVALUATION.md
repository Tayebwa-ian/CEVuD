# CEVuD Comparative Evaluation — Gating Weight Search & Results

> Canonical reference for the comparative evaluation study that selects the
> linear gate's `weight_static` and `escalation_threshold`, justifies the
> linear form against non-linear alternatives, and measures every baseline
> on a held-out test split.

---

## 1. What This Framework Does

The comparative evaluation answers five questions:

1. **Which weights are best?** Grid-search `(weight_static, escalation_threshold)` on validation, maximizing Fβ.
2. **Does the static/SLM override help?** Ablation: tuned gate with vs without override on test.
3. **Is linearity justified?** Logistic regression fit on validation, compared to tuned linear gate on test.
4. **How does CEVuD compare to baselines?** Seven strategies evaluated on the same test split.
5. **Are results stable across projects?** Per-project metric breakdown reveals spread / overfitting to one project.

Everything is driven by a single command. No manual steps, no notebook required.

---

## 2. One-Command Execution

```bash
python src/evaluation/run_comparative_evaluation.py \
  --manifest benchmark_manifest_cvefixes.json \
  --config config.json
```

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--manifest` | (required) | Path to benchmark manifest JSON |
| `--config` | `config.json` | Master config with gate parameters, paths, severity map |
| `--cache` | `None` | Path to `raw_scores_cache.json`; reuses Semgrep + SLM scores |
| `--force-recompute` | `False` | Ignore cache; re-extract everything |
| `--inline` | `False` | Score embedded `source_code` instead of cloning repos |
| `--weight-step` | `0.05` | Grid step for `weight_static` (0.05 → 21×21 grid) |
| `--threshold-step` | `0.05` | Grid step for `escalation_threshold` |

### Output Directory

Results are written to a timestamped directory under `workspace_storage/evaluation_runs/`:

```
workspace_storage/evaluation_runs/
  comparative_eval_20260713_143022/
    raw_scores_cache.json          # Semgrep + SLM scores (reusable)
    comparative_report.json         # Machine-readable full results
    comparative_report.md           # Paper-ready human-readable report
    gate_sensitivity_heatmap.png    # 2D grid heatmap
    gate_threshold_sensitivity.png  # Metric vs threshold at best weight
    gate_weight_sensitivity.png     # Metric vs weight at best threshold
```

---

## 3. End-to-End Pipeline Steps

### Step 1: Raw Score Extraction

**What happens:** Semgrep runs once per project; the local SLM classifier runs once (batched) per project. Two numbers are produced per labeled sample:

- `severity_weight`: Semgrep severity mapped to `[0, 1]` via `config.json → semgrep_severity_map`
- `slm_score`: `P(vulnerable)` from the local classifier

**Input format to the SLM (evaluation, after the fix):**
- Module-level imports + enclosing function source (same as training and pipeline inference)
- Cross-file context: **excluded** (consistent with training's `include_cross_file=False` default)
- Chunked into 64-line sliding windows with 8-line overlap (same as training and pipeline)

**Caching:** Results are written to `raw_scores_cache.json`. Re-running with `--cache` skips Semgrep and the SLM, reusing the cached scores. **Do NOT use `--cache` for the paper's reported numbers** — it is only for fast iteration (re-grid, re-plot).

### Step 2: Leakage-Safe Split

**Strategies:**

| Strategy | When used | What it does |
|----------|-----------|--------------|
| `by_project` | ≥ 3 distinct projects | Whole projects assigned to train/validation/test. No project appears in more than one split. |
| `stratified` | < 3 projects | Individual samples assigned, stratified by (project, label). Every project in every split. |

**Default:** `by_project` (preferred — stronger generalization test).

**Split fractions:** `validation_fraction=0.2`, `test_fraction=0.2` (configurable in `config.json → evaluation`).

**Key invariant:** The test split is NEVER touched by grid search, logistic regression fitting, or any weight selection. It is used ONLY for final reported numbers.

### Step 3: Grid Search (Validation Only)

**What is swept:**
- `weight_static ∈ [0, 1]` (step `--weight-step`, default 0.05)
- `escalation_threshold ∈ [0, 1]` (step `--threshold-step`, default 0.05)
- `weight_slm = 1 - weight_static` (constrained to sum to 1, matching production formula)

**Grid size:** Default 21 × 21 = 441 combinations.

**Selection metric:** Fβ with β = 2.0 (`f2`), configurable via `config.json → evaluation.fbeta`. F2 weights recall twice as heavily as precision, reflecting the security-asymmetry: a missed vulnerability (FN) ships; an extra LLM call (FP) costs cents.

**Tie-breaking:** Among cells with equal F2, prefer higher recall, then lower escalation rate (cheaper gate among equally good ones).

**Override rule:** `override_enabled=False` for EVERY grid point. The static/SLM override is measured separately as an ablation on top of the already-tuned gate. This prevents confounding "which weights are best" with "does the override help".

### Step 4: Linearity Check

**What happens:**
1. Fit a 2-feature logistic regression on the **validation** split: `P(label=1 | severity_weight, slm_score)`
2. Compare the logistic gate vs the tuned linear gate on the **test** split
3. Produce a plain-English verdict

**Implementation:** Pure numpy batch gradient descent (no scikit-learn dependency). L2-regularized, 3000 epochs, learning rate 0.5.

**Verdict logic:**
- If `|Fβ(logistic) - Fβ(linear)| < 0.02`: "Linear gate preferred — monotonic, interpretable, no overfitting risk"
- If logistic outperforms by ≥ 0.02: "Non-linear boundary may be worth adopting"
- If linear outperforms: "Linear design supported outright"

### Step 5: Baseline Evaluation (Test Only)

Seven strategies are evaluated on the held-out test split:

| Strategy | Description | What it answers |
|----------|-------------|-----------------|
| `semgrep_only` | Escalate if Semgrep severity > threshold | "Does static analysis alone catch enough?" |
| `small_model_only` | Escalate if SLM `P(vuln)` > 0.5 | "Does the SLM alone catch enough?" |
| `always_llm` | Escalate everything | Upper bound on recall, upper bound on cost |
| `semgrep_or_small_model` | OR of the two single-signal rules | "Does a simple OR-gate beat learned weights?" |
| `cevud_tuned_no_override` | Linear gate with tuned weights, no override | Production CEVuD pipeline performance |
| `cevud_tuned_with_override` | Linear gate + tuned weights + override | Ablation: does the override add anything? |
| `logistic_regression` | Learned non-linear boundary | "Is linearity justified?" |

**Metrics reported for every strategy:**
- Precision, Recall, F1, Fβ
- Specificity, Accuracy
- Escalation rate (fraction sent to LLM)
- Token Reduction Rate (TRR = 1 - escalation_rate)
- Cost reduction (monetary saving vs Always-LLM, accounting for local scan cost)

### Step 6: Override Ablation (Test Only)

Grid search tunes the linear gate with `override_enabled=False`. After best weights are selected:
- Re-enable override on top of tuned gate
- Compare with/without on test split
- Report per-metric delta table

**Provenance:** The override originated as an engineering safety rule (ERROR severity or SLM > 90% should never be silently suppressed). The ablation measures its empirical contribution.

### Step 7: Report Generation

Two files are produced:

**`comparative_report.json`** (machine-readable):
```json
{
  "generated_at": "...",
  "dataset_summary": {"num_projects": N, "total_samples": N, ...},
  "split_strategy": "by_project",
  "split_sizes": {"train": N, "validation": N, "test": N},
  "grid_search": {"best": {...}, "full_grid_size": 441},
  "linearity_check": {...},
  "override_ablation": {...},
  "strategy_results": {...}
}
```

**`comparative_report.md`** (paper-ready):
1. Dataset summary
2. Baseline comparison table (all 7 strategies)
3. Per-project breakdown (reveals spread / single-project dominance)
4. Gate tuning provenance (exact weights, grid size, selection metric)
5. Sensitivity plots (3 PNGs)
6. Linearity justification
7. Override provenance + ablation delta

---

## 4. How to Apply the Tuned Weights in Production

Once the best weights are determined:

### 1. Update `config.json`

```json
{
  "gate_parameters": {
    "weight_static": <best_weight_static>,
    "weight_slm": 1 - <best_weight_static>,
    "escalation_threshold": <best_threshold>
  }
}
```

### 2. Rebuild the Docker image

```bash
docker build -t ghcr.io/<owner>/<repo>/appsec-pipeline:latest .
```

### 3. Push and deploy

```bash
docker push ghcr.io/<owner>/<repo>/appsec-pipeline:latest
```

No Python code changes are needed. `TriageOrchestrator` reads `gate_parameters` from `config.json` at runtime via `ModelManager`.

---

## 5. Guardrails Against Data Leakage

| Guardrail | Where enforced |
|-----------|---------------|
| Grid search uses validation ONLY | `grid_search.py` — never receives test records |
| Logistic regression fit uses validation ONLY | `linearity_check.py` — `fit_logistic_regression(validation)` |
| Test split used ONLY for final reporting | `run_comparative_evaluation.py` steps 4–5 |
| Override ablation separate from weight selection | `override_enabled=False` in every grid cell |
| Reproducibility seeds | `config.json → evaluation.random_seed` and `training.seed` |
| Cache safety warning | `run_comparative_evaluation.py` prints explicit warning if `--cache` used |

---

## 6. Baseline Definitions and Equivalences

Two pairs of "requested" baselines collapse onto the same decision rule:

- **"Semgrep only"** and **"Semgrep + LLM without Stage 2"** are the SAME rule: without the SLM gate, every Semgrep finding goes straight to the LLM.
- **"SLM only"** and **"SLM + LLM without Stage 1"** are the SAME rule: without Semgrep pre-filtering, the SLM score alone decides escalation.

This is not a limitation — it is worth stating explicitly in the paper because it shows the Stage-1/Stage-2 staged architecture has no "obvious" unstudied configuration that might outperform it.

---

## 7. Metric Definitions

All formulas and justifications are in `docs/METRICS.md`. The key metrics:

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| F2 | `(1+β²)·P·R / (β²·P + R)` with β=2 | Primary selection metric; weights recall over precision |
| TRR | `1 - escalation_rate` | Volume efficiency: share of snippets kept from the LLM |
| Cost reduction | `TRR × (1 - r)` where `r = c_gate / c_llm` | Monetary saving vs Always-LLM; accounts for local scan cost |

---

## 8. What to Run and When

| Scenario | Command | Notes |
|----------|---------|-------|
| Full study (paper numbers) | `python src/evaluation/run_comparative_evaluation.py --manifest benchmark_manifest.json --config config.json` | No `--inline`, no `--cache` |
| Quick iteration | Add `--weight-step 0.1 --threshold-step 0.1` | 11×11 = 121 cells instead of 441 |
| Re-plot from cache | `--cache raw_scores_cache.json --force-recompute` | Skips Semgrep + SLM, re-runs grid/search/plots only |
| Offline / air-gapped | Add `--inline --cache ...` | Uses embedded `source_code`, no git clone |

---

## 9. Recent Changes

### Change 1: Evaluation SLM input now matches training/pipeline format

**Before:** `_extract_git_source` passed cross-file context to `build_context_snippet`, so the SLM received `<imports>\n\n<function>\n\n<cross-file modules>`. Training's default `include_cross_file=False` never saw this content.

**After:** `build_context_snippet(content, (func_start, func_end), imports, {})` — cross-file context is empty. The SLM receives exactly `<imports>\n\n<function>`, matching training and pipeline inference.

### Change 2: Evaluation SLM inference now applies chunking

**Before:** `_finalize` called `get_classifier_inference(snippets)` directly on the assembled (potentially long) snippets. Training and pipeline both chunk at 64 lines / 8 overlap.

**After:** `_finalize` calls `chunk_code()` on each snippet, flattens all chunks, runs batched inference, then aggregates per-chunk probabilities (default: `max`). This matches how the classifier is trained and how the pipeline scores code.

### New tests

`tests/test_evaluation/` contains 77 unit tests covering:
- `gate_strategies`: all 7 strategies, override conditions, edge cases
- `metrics`: confusion matrix, derived metrics, per-group breakdowns
- `grid_search`: grid sweep, best-entry selection, tie-breaking
- `dataset_splitter`: by-project and stratified splits, reproducibility
- `linearity_check`: logistic regression fit, linear-vs-logistic comparison
- `raw_score_extractor`: cross-file exclusion, chunking integration
