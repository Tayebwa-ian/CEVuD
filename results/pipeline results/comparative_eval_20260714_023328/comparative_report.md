# CEVuD Comparative Evaluation Report

Generated: 2026-07-14T02:57:35.917744

## 1. Dataset

- Projects: 12
- Total labeled samples: 7680
- Split strategy: `by_project` (see dataset_splitter.py for the by_project vs. stratified tradeoff)
- Split sizes: train=1863, validation=4996, test=821

| Project | Total | Vulnerable | Safe | Source |
|---|---|---|---|---|---|---|
| vudenc_AnnAssign' | 2 | 0 | 2 | local |
| vudenc_Assert' | 22 | 5 | 17 | local |
| vudenc_Assign' | 2952 | 850 | 2102 | local |
| vudenc_AsyncFunctionDef' | 18 | 1 | 17 | local |
| vudenc_AugAssign' | 19 | 3 | 16 | local |
| vudenc_Condition | 789 | 101 | 688 | local |
| vudenc_Expr' | 1255 | 295 | 960 | local |
| vudenc_For | 32 | 4 | 28 | local |
| vudenc_FunctionDef' | 2044 | 176 | 1868 | local |
| vudenc_Import' | 85 | 25 | 60 | local |
| vudenc_ImportFrom' | 197 | 98 | 99 | local |
| vudenc_Return' | 265 | 29 | 236 | local |

## 2. Baseline / Ablation Comparison (held-out TEST split)

Selection metric: `f2` (F-beta with beta weighting recall over precision — see metrics.py).

| Strategy | Precision | Recall | F1 | F2 | Escalation rate | TRR | Cost reduction |
|---|---|---|---|---|---|
| semgrep_only | 0.250 | 0.009 | 0.018 | 0.012 | 0.005 | 0.995 | 0.975 |
| small_model_only | 0.151 | 0.705 | 0.249 | 0.407 | 0.596 | 0.404 | 0.396 |
| always_llm | 0.128 | 1.000 | 0.227 | 0.423 | 1.000 | 0.000 | 0.000 |
| semgrep_or_small_model | 0.151 | 0.705 | 0.249 | 0.407 | 0.596 | 0.404 | 0.396 |
| cevud_production_defaults | 0.153 | 0.305 | 0.204 | 0.254 | 0.255 | 0.745 | 0.731 |
| cevud_tuned_with_override | 0.128 | 0.952 | 0.226 | 0.417 | 0.949 | 0.051 | 0.050 |
| cevud_tuned_no_override | 0.128 | 0.952 | 0.226 | 0.417 | 0.949 | 0.051 | 0.050 |
| logistic_regression | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.980 |

Escalation rate is the fraction of samples sent to the Stage 3 LLM — the pipeline's direct cost proxy. **TRR** (Token Reduction Rate, `token_reduction_rate`) = 1 - escalation_rate: the share of samples — and, under CEVuD's uniform per-snippet token assumption, the share of tokens — that never reach the LLM. **Cost reduction** (`cost_savings_ratio`) is the *monetary* saving vs the Always-LLM baseline and is deliberately distinct from TRR: the gated pipeline still runs a cheap local scan (Semgrep + edge SLM, ~2% of an LLM call) on every snippet, so Cost reduction = TRR × (1 − cost_ratio) and sits slightly *below* TRR. As the local scan cost → 0 (the paper's 'zero marginal-cost edge' idealisation) the two converge. See docs/METRICS.md for the full derivation and justifications. `always_llm` is the recall/cost upper bound (TRR=0, Cost reduction=0); `semgrep_only` and `codesheriff_only` are equivalent to the 'skip one stage' ablations requested in review (see gate_strategies.py docstring for why those pairs collapse to the same rule).

## 3. Per-Project Breakdown (CEVuD, tuned, with override)

| Project | Precision | Recall | F1 | F2 | N |
|---|---|---|---|---|---|
| vudenc_Condition | 0.128 | 0.960 | 0.226 | 0.418 | 789 |
| vudenc_For | 0.130 | 0.750 | 0.222 | 0.385 | 32 |

If performance varies sharply across projects here, that indicates the gate (or its tuned weights) does not generalize uniformly — report this spread explicitly rather than only the aggregate row above.

## 4. Gate Tuning and Sensitivity

The linear gate's `weight_static` and `escalation_threshold` were selected by grid search on the VALIDATION split only (never the test split), maximizing `f2`. Selected configuration: `weight_static=0.15`, `weight_slm=0.85`, `escalation_threshold=0.2`, achieving `f2=0.5937` on validation.

Sensitivity plots (saved alongside this report):
- `gate_sensitivity_heatmap.png` — full grid over (weight_static, threshold)
- `gate_threshold_sensitivity.png` — metric vs. threshold at the selected weight
- `gate_weight_sensitivity.png` — metric vs. weight at the selected threshold

## 5. Linearity Justification

A logistic regression gate (bias=-1.8615, weight_severity=0.1495, weight_slm=1.2181) was fit on the validation split and compared against the tuned linear gate on the test split.

The linear gate outperforms logistic regression by 0.4170 f2 on the test split. This supports the linear design outright.

## 6. Override Rule: Provenance and Ablation

**Provenance.** The static/SLM override (`static_override_value=1.0`, `slm_override_threshold=0.9`) originated as an engineering safety rule, not a value tuned on data: the concern was that a catastrophic static finding (Semgrep severity ERROR) or an extremely confident SLM prediction (>90%) could, in principle, still fall below the linear threshold if the other signal were low, silently suppressing escalation of a likely-real vulnerability. The override forces escalation in exactly those two cases, independent of the linear formula.

**Ablation (empirical).** Grid search above tuned the linear gate WITHOUT the override (`override_enabled=False` for every grid point — see grid_search.py), so weight/threshold selection cannot be confounded with the override's effect. The override was then re-enabled on top of the already-tuned gate and measured on the test split:

| Metric | Without override | With override | Delta |
|---|---|---|---|
| precision | 0.1284 | 0.1284 | +0.0000 |
| recall | 0.9524 | 0.9524 | +0.0000 |
| f2 | 0.4170 | 0.4170 | +0.0000 |
