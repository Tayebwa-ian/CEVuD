# CEVuD Metrics — Formulas & Justifications

> Canonical reference for every metric reported by CEVuD, at both the
> **model** level (the Stage-2 CodeBERT classifier) and the **pipeline** level
> (the full gated triage system). Every formula here is implemented in
> `src/evaluation/metrics.py` (pipeline / gate study), `src/evaluate_pipeline.py`
> (pipeline dashboard), and `src/training/evaluator.py` (model). If a number
> appears in a report, its definition is here.

---

## 1. The decision problem and the confusion matrix

CEVuD frames vulnerability triage as a **binary decision** per code snippet
(function-level chunk):

- **Positive class** (`y = 1`): the snippet is *vulnerable* and must be
  escalated to the Stage-3 LLM.
- **Negative class** (`y = 0`): the snippet is *safe* and should be filtered
  out by the local gate (no LLM call).

A gate/escalation policy produces a per-snippet decision `ŷ ∈ {0,1}`
(`True` = escalate). From the `N` labelled snippets we tabulate the four
confusion-matrix cells:

| | Ground truth: Vulnerable (1) | Ground truth: Safe (0) |
|---|---|---|
| **Predicted escalate (1)** | TP — true positive | FP — false positive |
| **Predicted filter (0)** | FN — false negative | TN — true negative |

```text
TP = Σ 1[ŷ=1 ∧ y=1]      FP = Σ 1[ŷ=1 ∧ y=0]
FN = Σ 1[ŷ=0 ∧ y=1]      TN = Σ 1[ŷ=0 ∧ y=0]
N  = TP + FP + FN + TN
```

**Why this matters for defensibility.** In a security pipeline the two error
types are *not* symmetric:

- A **false negative (FN)** lets a real vulnerability ship. This is the
  high-cost error.
- A **false positive (FP)** merely spends a few cents on an unnecessary LLM
  call. This is the low-cost error.

Every metric choice below is justified by this asymmetry: we deliberately
favour *recall* and tolerate *lower precision*, because missing a vulnerability
is materially worse than over-calling the LLM.

---

## 2. Detection metrics (model + pipeline)

All of the following are computed in `metrics.py:compute_metrics` and
`training/evaluator.py:compute_all_metrics`.

### 2.1 Precision

```text
Precision = TP / (TP + FP)
```

Fraction of escalated snippets that were *actually* vulnerable. Low precision =
the LLM is being bothered by safe code (cost, not risk).

### 2.2 Recall (Sensitivity, True Positive Rate)

```text
Recall = TP / (TP + FN)
```

Fraction of *real* vulnerabilities that were correctly escalated. **This is the
primary safety metric.** A missed vulnerability (FN) ships; recall is the
probability we do *not* miss it. CEVuD's whole premise is that the gate must
preserve the recall of an Always-LLM pipeline while cutting cost — so recall is
the metric we are *not allowed* to sacrifice.

### 2.3 Specificity (True Negative Rate, TNR)

```text
Specificity = TN / (TN + FP)
```

Fraction of *safe* snippets correctly filtered. This is the mirror image of
recall and is the direct driver of the cost metrics in §3 — every true negative
is one LLM call saved.

### 2.4 Accuracy

```text
Accuracy = (TP + TN) / N
```

Overall correct decisions. **Reported but de-emphasised**: on a realistic,
heavily safe-dominated dataset accuracy is dominated by the majority (safe)
class and is a poor measure of security value. We report it for completeness,
never as a tuning target.

### 2.5 F1 and F-beta

```text
F1   = 2 · Precision · Recall / (Precision + Recall)

Fβ   = (1 + β²) · Precision · Recall / (β² · Precision + Recall)
```

The harmonic mean of precision and recall. **F1 (β = 1)** weights them equally.
**F2 (β = 2)** weights recall *twice* as heavily as precision — chosen as the
gate-study's selection metric because of the FN/FP cost asymmetry in §1.

**Justification for F2 over F1 in the gate study.** In triage, a false negative
(a shipped vulnerability) is far costlier than a false positive (a wasted LLM
call). F2 encodes exactly this preference: under F2, improving recall is worth
~twice as much as the equivalent precision gain. `β` is a parameter (not
hardcoded) everywhere in `metrics.py`, taken from `config.json → evaluation.fbeta`
(default `2.0`), so the choice is reported and sensitivity-testable rather than
baked in silently.

---

## 3. Cost / efficiency metrics (pipeline)

These are what make CEVuD a *cost* paper, not just a detection paper. They are
produced by the gate study (`run_comparative_evaluation.py`) and the pipeline
dashboard (`evaluate_pipeline.py`).

### 3.1 Escalation rate

```text
Escalation rate = (TP + FP) / N = N_escalated / N
```

The fraction of snippets the gate sends to the Stage-3 LLM. This is the
pipeline's **direct cost proxy**: every escalated snippet triggers an LLM call.
Lower is cheaper. It is the denominator of both §3.2 and §3.3.

### 3.2 TRR — Token Reduction Rate

```text
TRR = 1 − Escalation rate = 1 − (TP + FP) / N
```

The fraction of snippets — and, under CEVuD's **uniform per-snippet token
assumption**, the fraction of *tokens* — that the gate keeps away from the LLM.

**Defensible proxy justification.** CEVuD operates on function-level chunks of
comparable size, so each escalated unit is approximately one uniform-cost LLM
work item. Under that assumption the share of *snippets* not escalated equals
the share of *tokens* not sent, so TRR is a faithful volume-efficiency measure.
**Extension note (honesty):** if real per-sample token counts ever become
available, replace `(TP+FP)/N` with `Σ tokens_escalated / Σ tokens_total`; the
formula is unchanged in spirit. TRR=0 for the `always_llm` baseline (nothing is
filtered) and TRR=1 for a perfect filter.

### 3.3 Cost reduction (a.k.a. CSR / cost_savings_ratio)

```text
Cost reduction = TRR × (1 − r),      where  r = c_gate / c_llm
```

The **monetary** saving versus the Always-LLM baseline. Unlike TRR (a token
*volume* metric), Cost reduction accounts for the fact that the gated pipeline
is **not free**: every *non-escalated* snippet still costs the cheap local scan
(Semgrep + edge SLM, unit cost `c_gate`), whereas an *escalated* snippet costs a
full LLM call (unit cost `c_llm`, with `c_llm ≫ c_gate`).

**Derivation.** Let `N` snippets, `N_esc` escalated (`r = c_gate / c_llm`):

```text
Cost_full   = N · c_llm                         (every snippet hits the LLM)
Cost_gated  = N_esc · c_llm + (N − N_esc) · c_gate

Cost reduction = 1 − Cost_gated / Cost_full
              = 1 − [ N_esc/N + (1 − N_esc/N) · r ]
              = (1 − N_esc/N) · (1 − r)
              = TRR · (1 − r)
```

With the default `r = 0.02` (the local scan costs ~2% of an LLM call), Cost
reduction sits slightly *below* TRR — a conservative, honest figure.

**Why Cost reduction is distinct from TRR (and must stay that way).** TRR
measures *volume* saved; Cost reduction measures *dollars* saved. They coincide
only under the paper's "zero marginal-cost edge" idealisation (`r → 0`), where
the local scan is treated as free. Keeping them separate lets a reviewer see
both the engineering-efficiency claim (TRR) and the economic claim (Cost
reduction) independently, and the gap between them is itself an honest
statement that the gate is not literally free. `r` is a single documented
constant; change it to reflect real pricing and both metrics update
consistently.

---

## 4. The gate-tuning selection metric

Grid search (`grid_search.py`) and the comparative report pick the linear gate's
`(weight_static, escalation_threshold)` to **maximise F2 on the validation
split only** (never the test split — see `run_comparative_evaluation.py`). F2 is
used as the selection metric precisely because of the recall-over-precision
preference argued in §2.5: we want the cheapest gate that still catches
vulnerabilities, and F2 penalises a gate that drops recall to save cost.

The grid search also reports `escalation_rate` alongside F2 so that, among
roughly-equal-F2 gates, the cheaper (lower escalation) one can be preferred — a
second, explicit cost-vs-recall tie-breaker.

---

## 5. Model-level metrics (Stage-2 classifier)

The classifier is evaluated **in isolation** on its held-out CVEfixes test split
(`training/evaluator.py`), independent of the gate study. Reported metrics:

| Metric | Definition | Why it is reported |
|---|---|---|
| Accuracy | §2.4 | completeness only, not a tuning target |
| Precision / Recall | §2.1 / §2.2 | standard classifier quality |
| F1 / F2 | §2.5 | F2 reflects the security cost asymmetry |
| ROC-AUC | area under TPR-vs-FPR curve | threshold-independent ranking quality |
| PR-AUC | area under precision-recall curve | robust to the safe-dominated class balance |
| Confusion matrix | §1 | full breakdown for follow-up analysis |

These are saved to `metrics.json` alongside the confusion-matrix, ROC, PR, and
calibration plots. Note ROC-AUC / PR-AUC use the **continuous** `P(vulnerable)`
probability, whereas F1 / F2 / the confusion matrix use the **argmax** class
decision — equivalent to a hard threshold of **0.5** on `P(vulnerable)`. So a
model can have high AUC but a modest F1 if its decision threshold is poorly
placed; the gate study consumes the probability, not the argmax, which is why
the pipeline's recall can exceed the standalone F1.

---

## 6. What "works as expected" means — checklist

- [x] **Training**: `num_epochs` is a configurable ceiling; `EarlyStoppingCallback`
      halts when validation F2 stops improving for `early_stopping_patience`
      epochs and the best (highest validation F2) checkpoint is restored. See
      `MODEL_CARD.md` and `TRAINING.md`.
- [x] **Model metrics**: accuracy / precision / recall / F1 / F2 / ROC-AUC /
      PR-AUC + confusion matrix, computed and plotted by `evaluator.py`.
- [x] **Pipeline metrics**: recall, F1, **TRR**, and **Cost reduction** are all
      computed by `metrics.py` / `evaluate_pipeline.py` and surfaced in both the
      comparative (`comparative_report.md`) and dashboard (`summary.json`)
      outputs.
- [x] **Distinctness**: TRR (volume) and Cost reduction (dollars) are now
      explicitly separate metrics via the `cost_ratio` model, so the report no
      longer shows two identical columns.
- [x] **Formulas + justifications**: this document.
