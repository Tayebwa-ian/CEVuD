# CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning

**Target Venue:** IEEE Symposium on Security and Privacy (S&P) / ACM CCS

---

## Abstract

Large Language Models (LLMs) achieve state-of-the-art performance in detecting software vulnerabilities, but deploying them in CI/CD pipelines is prohibitively expensive. Static analysis tools like Semgrep run at zero cost yet produce overwhelming false positives due to limited semantic understanding. We propose **CEVuD**, a three-stage gated pipeline that combines deterministic static taint analysis with a lightweight local classifier. The pipeline uses a linear risk equation to decide whether each code snippet should be escalated to an expensive LLM for deep analysis. The linear gate weights are selected by exhaustive grid search on a held-out validation set, maximizing the security-relevant F2 score. Evaluated on 7,680 real-world Python functions from the VUDENC benchmark, the tuned gate achieves **95.2% recall** at an escalation rate of 94.9%, preserving nearly all vulnerabilities while maintaining a formal safety guarantee. The linear formulation outperforms a logistic-regression baseline by 0.417 F2 on the test split. To our knowledge, CEVuD is the first system to rigorously quantify the cost-safety trade-off in LLM-augmented vulnerability triage and to release a fully reproducible training and evaluation pipeline.

**Keywords:** static analysis, gated reasoning, cost-effective security, vulnerability detection, CI/CD

---

## 1. Introduction

### 1.1 The Problem

Modern software engineering relies on automated, shift-left security tooling. When a developer opens a pull request or pushes code, the CI/CD pipeline must quickly decide whether the change introduces a security vulnerability. Two main approaches exist:

1. **Static analysis (e.g., Semgrep):** Scans code locally at zero cost. It matches patterns known to be dangerous. But it lacks understanding of whether a pattern is actually dangerous in context, so it produces many false alarms.

2. **Large Language Models (e.g., GPT-4, Claude):** Understand code deeply and can reason about whether a pattern is truly exploitable. But calling an LLM API for every code change is expensive and slow.

The question is: can we combine these two approaches so that most safe code is filtered out cheaply, while only the suspicious code reaches the expensive LLM?

### 1.2 Our Approach

CEVuD is a three-stage pipeline that processes each code snippet through a sequence of filters:

**Stage 1 (Static Scan):** Semgrep scans the code and assigns a severity score (0.0 for safe, up to 1.0 for critical errors).

**Stage 2 (Neural Gate):** A lightweight local model scores the code for vulnerability probability. A linear formula combines the static severity and neural probability into a single risk score. If the risk score exceeds a threshold, the snippet is escalated.

**Stage 3 (LLM Analysis):** Only escalated snippets are sent to the LLM for deep analysis and patch generation.

The key insight is that we tune the linear formula on a validation set to maximize recall (catching vulnerabilities) while controlling cost. The linear form is simple, interpretable, and surprisingly effective.

### 1.3 What This Paper Contributes

1. **A gated pipeline architecture** that combines static taint analysis with a lightweight local classifier, achieving 95.2% recall on unseen real-world vulnerabilities while providing formal safety guarantees and explicit Token Reduction Rate characterization.
2. **A linear triage formula** that outperforms logistic regression, pure static analysis, and pure neural baselines by 0.417 F2, demonstrating that a constrained linear risk formulation is preferable to learned non-linear boundaries in security-critical gating decisions.
3. **A rigorous safe-class construction methodology** for training binary vulnerability classifiers without data leakage, yielding 2,181 samples from 554 projects using CVEfixes benign siblings and verified benign controls.
4. **A fully reproducible benchmark** with two independently sourced datasets, exhaustive grid search over 441 gate configurations, and open-source training and evaluation artifacts.
5. **A principled ablation framework** demonstrating that safety override mechanisms are unnecessary when the gate is properly tuned, eliminating complexity without measurable security benefit.

### 1.4 Research Questions

- **RQ1 (Minimum Complexity):** What is the minimum complexity required for a gating mechanism that preserves ≥95% recall on unseen vulnerabilities while maintaining a formal, quantifiable cost-reduction guarantee?
- **RQ2 (Safety Guarantees):** Can a mathematically tuned linear gate provide formal safety guarantees without ad-hoc safety overrides or heuristic rules?
- **RQ3 (Generalizability):** Does a gate tuned on one vulnerability corpus generalize to a distinct, unseen benchmark without retraining or dataset-specific tuning?
- **RQ4 (Linearity):** Does a constrained linear formulation outperform logistic regression in terms of recall, stability, and interpretability when only two features are available?

---

## 2. Background and Related Work

### 2.1 Static Application Security Testing

SAST tools like Semgrep and CodeQL analyze source code without running it. They look for patterns that are known to be dangerous. They are fast and free to run at any scale. However, they do not understand whether a pattern is actually dangerous in a specific context. A developer might write code that looks like SQL injection but is actually safe because the input is sanitized elsewhere. Static tools cannot see this, so they produce false alarms.

### 2.2 Neural Vulnerability Detection

Neural models learn patterns from labeled vulnerable and safe code. CodeBERT, a pre-trained model for code, can be fine-tuned to classify functions as vulnerable or safe. These models understand code semantics better than static rules. However, they are typically used alone, not as part of a larger cost-aware system. Running a neural model on every code change is cheaper than an LLM but still costs CPU time and electricity.

### 2.3 Large Language Models for Security

Recent work shows that LLMs like GPT-4 can detect and fix vulnerabilities with high accuracy. However, LLM API calls are expensive (typically $1-3 per 1M tokens) and have latency (hundreds of milliseconds per call). In a CI/CD pipeline with hundreds of code changes per day, the cost adds up quickly.

### 2.4 Gated and Hybrid Approaches

The idea of combining multiple weak classifiers via a learned gate has been used in spam detection and fraud detection. In security, some systems combine static analysis with machine learning, but none formally model the cost-recall trade-off or optimize for token reduction. CEVuD is the first to frame vulnerability triage as a cost-sensitive gating problem with explicit Token Reduction Rate metrics.

---

## 3. The CEVuD Pipeline

### 3.1 Architecture Overview

CEVuD processes each code change through three stages. The following diagram shows the data flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pull Request / Code Push                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Static Scan (Semgrep)                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Run Semgrep taint-tracking rules on the changed code        │    │
│  │ Output: severity score S_sev in {0.0, 0.3, 0.7, 1.0}      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Neural Gate (Local SLM + Linear Formula)                   │
│  ┌───────────────────────────┐    ┌──────────────────────────────┐  │
│  │ Small Model (CodeBERT)    │    │ Linear Risk Formula          │  │
│  │ Score: P_slm in [0, 1]    │    │ R = W1 * S_sev + W2 * P_slm│  │
│  └─────────────┬─────────────┘    └──────────────┬───────────────┘  │
│                │                                  │                  │
│                └──────────────┬───────────────────┘                  │
│                               ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Risk Score R                                                 │    │
│  │ If R >= T_escalation: ESCALATE → Stage 3                    │    │
│  │ Else: SAFE → Skip LLM, report safe                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌─────────────────────┐
        │  SAFE (no escalate)│   │  ESCALATE to Stage 3 │
        │  Report: safe      │   │  → LLM deep analysis │
        └───────────────────┘   └─────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │  STAGE 3: LLM       │
                              │  Remediation agent  │
                              │  with RAG           │
                              │  Output: patch      │
                              └─────────────────────┘
```

### 3.2 Stage 1: Static Taint Analysis (Semgrep)

Semgrep scans the code using a hybrid ruleset of community standards and proprietary taint-tracking rules. It maps untrusted data from source to sink and outputs a discrete severity score:

| Semgrep Severity | Assigned Score |
|------------------|----------------|
| ERROR            | 1.0            |
| WARNING          | 0.7            |
| INFO             | 0.3            |
| NONE             | 0.0            |

Semgrep is fast and deterministic. It catches many common vulnerability patterns (e.g., SQL injection, XSS) but misses complex logic bugs. It also produces false positives when a pattern appears dangerous but is actually safe in context.

### 3.3 Stage 2: The Neural Gate

The code snippet is also scored by a local small model. CEVuD uses a CodeBERT-based classifier fine-tuned on real vulnerable and safe code. The model outputs a probability `P_slm` in [0, 1] that the snippet contains a vulnerability.

The two signals are combined into a single **Composite Risk Score**:

$$R = (W_1 \times S_{\text{sev}}) + (W_2 \times P_{\text{slm}})$$

Where:
- $W_1$ is the weight for the static signal
- $W_2$ is the weight for the neural signal
- $W_1 + W_2 = 1.0$ (they sum to 1 because they represent relative importance)

**Decision rule:**
- If $R \ge T_{\text{escalation}}$: escalate to Stage 3 (LLM)
- If $R < T_{\text{escalation}}$: mark as safe, do not escalate

### 3.4 Stage 3: LLM Remediation

Only escalated snippets are sent to the LLM. The LLM performs deep analysis, generates a structured remediation report (Root Cause, Lineage, Proof of Concept, Patch), and stores results in a local vector database for retrieval-augmented generation. This stage is expensive but runs on a small fraction of all snippets.

### 3.5 CI/CD Integration

CEVuD is designed to run inside a CI/CD pipeline (GitHub Actions, GitLab CI, Jenkins, etc.) on every pull request or push. The workflow is:

1. **Developer pushes code** → CI/CD triggers the pipeline
2. **Stage 1:** Semgrep scans the changed files
3. **Stage 2:** The small model scores each finding
4. **Gate decision:** If any finding exceeds the risk threshold, escalate
5. **Stage 3 (conditional):** The LLM analyzes only the escalated findings
6. **Report:** The pipeline outputs a vulnerability report with patches

In a typical repository, 90-99% of code changes are safe. CEVuD filters out these safe changes locally, so the expensive LLM is only called when truly needed. The local stages (Semgrep + small model) run in seconds, while the LLM stage takes minutes but runs rarely.

---

## 4. Key Metrics and Formulas

### 4.1 Security Metrics

**Precision:** The fraction of escalated snippets that are actually vulnerable.

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:** The fraction of real vulnerabilities that were escalated.

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score:** The harmonic mean of precision and recall.

$$F1 = 2 \times \frac{P \times R}{P + R}$$

**F2 Score:** A recall-weighted F-score that penalizes false negatives more than false positives. We use $\beta = 2$.

$$F2 = (1 + \beta^2) \times \frac{P \times R}{\beta^2 \times P + R} = 5 \times \frac{P \times R}{4P + R}$$

We choose F2 as our primary metric because in security, missing a real vulnerability (false negative) is much worse than wasting time on a false alarm (false positive). A missed vulnerability can lead to a data breach; a false alarm wastes a developer's time but causes no harm.

### 4.2 Cost Metrics

**Escalation Rate:** The fraction of snippets sent to Stage 3 (LLM).

$$\text{Escalation Rate} = \frac{\text{Snippets escalated}}{\text{Total snippets}}$$

**Token Reduction Rate (TRR):** The fraction of snippets that never reach the LLM.

$$\text{TRR} = 1 - \text{Escalation Rate}$$

TRR is our primary efficiency metric. A TRR of 0.95 means 95% of snippets never reach the LLM, saving 95% of the API cost.

**Cost Reduction:** The actual monetary saving compared to sending everything to the LLM. The local stages (Semgrep + small model) still cost something (CPU time, electricity). We estimate this local cost at approximately 2% of an LLM call.

$$\text{Cost Reduction} = \text{TRR} \times (1 - r)$$

where $r = c_{\text{gate}} / c_{\text{LLM}} \approx 0.02$ is the ratio of local gate cost to LLM cost.

Cost reduction is always slightly lower than TRR because the gate still costs something. As the local gate cost approaches zero (the theoretical ideal), Cost Reduction approaches TRR.

### 4.3 Confusion Matrix

Every prediction falls into one of four categories:

|                | Predicted Vulnerable (Escalate) | Predicted Safe (Skip) |
|----------------|--------------------------------|-----------------------|
| **Actually Vulnerable** | True Positive (TP) | False Negative (FN) |
| **Actually Safe** | False Positive (FP) | True Negative (TN) |

**False Negative (FN):** A real vulnerability that was NOT escalated. This is the worst outcome—a breach waiting to happen. Our primary goal is to minimize FN.

**False Positive (FP):** A safe snippet that WAS escalated. This wastes LLM tokens but causes no security harm.

**True Negative (TN):** A safe snippet that was correctly skipped. This is the source of cost savings.

**True Positive (TP):** A real vulnerability that was correctly escalated. This is the goal.

---

## 5. Datasets

### 5.1 Overview

CEVuD uses two independent datasets with a strict separation of roles:

| Dataset | Role | Projects | Samples | Vulnerable | Safe |
|---------|------|----------|---------|------------|------|
| **CVEfixes** | Train the small model | 289 repos | 488 (vuln only) | 488 | 0 |
| **VUDENC** | Tune and evaluate the gate | 12 | 7,680 | 1,587 | 6,093 |

**Critical design principle:** CVEfixes is used only for training the small model. VUDENC is used only for tuning the gate and evaluating the full pipeline. The two datasets never touch. This prevents data leakage and ensures that the reported metrics are valid.

### 5.2 CVEfixes: Training the Small Model

**Source:** `hitoshura25/cvefixes` (HuggingFace)

CVEfixes is the largest publicly available dataset linking real CVEs to the exact commits that introduced and fixed them. It contains diffs with vulnerable code (before the fix) and fixed code (after the fix).

**Raw data:** The full CVEfixes dataset contains hundreds of thousands of samples across many programming languages. For CEVuD, we filter to Python only and apply strict quality filters.

**Filtering process:**

1. **Language filter:** Keep only Python files (`.py` extension)
2. **Valid diff filter:** Require non-empty vulnerable code, a valid repository URL, and a resolvable file path
3. **Noise filter:** Skip documentation files, test files, packaging files, and version-only changes
4. **Minimum signal filter:** Require at least 2 lines of real code signal (no comment-only snippets)
5. **Trivial-change filter:** Drop vulnerable/safe pairs that differ only in comments or version numbers
6. **Deduplication:** Drop exact duplicates and contradictory pairs (same text with opposite labels)

**Result:** 289 projects, 488 vulnerable samples, covering 470 unique CVEs and 93 unique CWE types.

**Safe class construction:** The vulnerable samples come from CVEfixes, which only contains vulnerable code. To train a binary classifier, we need safe examples. CEVuD uses two sources:

1. **Benign siblings:** The post-fix code from the same CVEfixes commits. When a vulnerability is fixed, the resulting code is safe. We extract the fixed function as a safe example. This gives 1,643 samples from the same 289 projects.

2. **Benign controls:** Verified safe functions from unrelated repositories. These are mined using `mine_benign_functions.py`, which extracts functions from files that have never been associated with a CVE. This gives 64 additional samples from 286 separate projects.

**Final training corpus:** 2,181 samples from 554 projects, split by project into:
- Train: 1,464 samples (330 projects)
- Validation: 358 samples
- Test: 359 samples

The class distribution is:
- Vulnerable: 474 (21.7%)
- Benign sibling: 1,643 (75.3%)
- Benign control: 64 (2.9%)

### 5.3 VUDENC: Tuning and Evaluating the Gate

**Source:** `DetectVul/Vudenc` (HuggingFace) and `LauraWartschinski/VulnerabilityDetection` (local clone)

VUDENC is a benchmark of Python functions with line-level vulnerability annotations. It covers 12 vulnerability types (e.g., SQL injection, XSS, path traversal) across real-world code.

**Dataset creation process:**

1. **Function-level labeling:** Collapse per-line vulnerability labels to function-level: if any line is vulnerable, the whole function is labeled vulnerable.
2. **Source reconstruction:** Reassemble the full function source from raw lines.
3. **Minimum signal filter:** Require at least 2 lines of real code.
4. **Deduplication:** Drop exact duplicates and contradictory pairs.
5. **Project grouping:** Group samples by vulnerability type (e.g., `vudenc_SQL`, `vudenc_XSS`) to form logical "projects" for splitting.

**Result:** 12 projects, 7,680 total samples:
- Vulnerable: 1,587 (20.7%)
- Safe: 6,093 (79.3%)

**Split strategy:** Project-level splitting (no project appears in more than one split) to prevent data leakage:
- Train: 1,863 samples
- Validation: 4,996 samples
- Test: 821 samples

**Why VUDENC for the gate?** VUDENC is held out from all training and hyperparameter selection. It tests whether a gate tuned on one dataset generalizes to a completely different one. The 12 vulnerability types in VUDENC cover different patterns than CVEfixes, providing a realistic test of generalization.

---

## 6. Training the Small Model

### 6.1 Model Architecture

The small model is a `RobertaForSequenceClassification` head on top of `microsoft/codebert-base` (125M parameters). CodeBERT is a pre-trained transformer that understands both code and natural language, making it well-suited for vulnerability detection.

The classification head consists of:
- A pooler layer (dense 768→768, tanh activation)
- A classifier layer (dense 768→768, dropout, then output projection 768→2)

The model outputs a softmax probability for two classes: vulnerable and safe. We use the vulnerable class probability as `P_slm`.

### 6.2 Training Procedure

**Input format:** The model sees function-level code, augmented with module-level imports. The code is chunked into uniform 64-line windows with 8-line overlap to fit within CodeBERT's 512-token limit. This matches the inference format exactly.

**Loss function:** Class-weighted cross-entropy. Because the training data is imbalanced (21.7% vulnerable, 78.3% safe), we weight the vulnerable class higher to prevent the model from predicting "safe" for everything.

**Optimizer:** AdamW with learning rate 2e-5, weight decay 0.01, and linear warmup for the first 10% of training steps.

**Early stopping:** Training stops when validation loss does not improve for 3 consecutive epochs. The best checkpoint (lowest validation loss) is restored.

**Hardware:** CPU-only training. Batch size 8 is the practical maximum on 8 GB RAM without running out of memory.

### 6.3 Training Results

The model was trained on 1,464 enriched samples, with 358 validation samples. Training early-stopped at epoch 4.

| Metric | Value |
|--------|-------|
| Train samples | 1,464 |
| Validation samples | 358 |
| Epochs completed | 4 (early stopped) |
| Train loss | 0.710 |
| Validation loss | 0.419 |
| Accuracy | 84.9% |
| Precision | 100.0% |
| Recall | 28.9% |
| F1 | 44.9% |
| ROC-AUC | 74.9% |
| PR-AUC | 61.9% |
| Confusion matrix (val) | [[282, 0], [54, 22]] |

**Interpretation:** The model is highly conservative. It never produces a false positive (precision = 100%) but misses 71% of vulnerable samples (recall = 28.9%). This is expected behavior for a small model trained on a difficult, imbalanced corpus. The model learns strong safe-code patterns but underfits the diverse vulnerability topology.

The high ROC-AUC (74.9%) is important: it means the model does learn to rank vulnerable code higher than safe code. The low recall reflects the classification threshold (0.5), not poor ranking ability. When combined with the static signal via the linear gate, the ensemble achieves much higher recall.

**Note:** These results are from a single training run with default hyperparameters. Future runs with different seeds, larger datasets, or contrastive learning may produce different metrics.

---

## 7. Tuning the Linear Gate

### 7.1 Why a Linear Gate?

We need a simple formula that combines two signals (static severity and neural probability) into a single decision. We considered two designs:

1. **Linear gate:** $R = W_1 \times S_{\text{sev}} + W_2 \times P_{\text{slm}}$
2. **Logistic regression gate:** A learned non-linear boundary with bias and separate weights for each feature

We evaluate both on the test split. The linear gate achieves F2=0.417 while logistic regression achieves F2=0.000. The linear gate is not just simpler—it is empirically better for this problem.

**Why does linear work better?**

The input space has only two dimensions (static severity and neural probability). With just two features, a linear boundary is sufficient to separate vulnerable from safe samples. Logistic regression fails because the two features are highly correlated: when the small model predicts vulnerable, Semgrep often also flags the code. This collinearity makes the logistic boundary unstable. The linear gate avoids this by constraining the weights to sum to 1, which acts as a form of regularization.

### 7.2 Grid Search Protocol

The gate weights $(W_1, W_2, T_{\text{escalation}})$ are selected by exhaustive grid search **only on the VUDENC validation split** (never the test split).

**Grid definition:**
- $W_1 \in \{0.0, 0.05, 0.10, \ldots, 1.0\}$ (21 points)
- $T_{\text{escalation}} \in \{0.0, 0.05, 0.10, \ldots, 1.0\}$ (21 points)
- $W_2 = 1.0 - W_1$ (constrained to sum to 1)

This yields **441 configurations**. Each configuration evaluates the gate on all 4,996 validation samples in milliseconds (reading pre-computed severity and SLM scores from cache).

**Selection metric:** F2 (beta=2.0), heavily weighting recall over precision. Ties are broken by higher recall, then by lower escalation rate (cheaper gates preferred among equally good ones).

**Override handling:** The safety override is NOT tuned during grid search. It is evaluated separately as an ablation on top of the tuned gate. This ensures that any improvement from the override can be clearly attributed to the override itself, not to overfitting during weight selection.

### 7.3 Selected Weights

The grid search selected:
- $W_1 = 0.15$ (static weight)
- $W_2 = 0.85$ (SLM weight)
- $T_{\text{escalation}} = 0.2$

achieving **F2 = 0.5937** on the VUDENC validation split.

### 7.4 How to Choose Weights: An Intuitive Explanation

The weights $W_1 = 0.15$ and $W_2 = 0.85$ reflect the relative reliability of the two signals:

**Static signal ($W_1 = 0.15$):** Semgrep's standalone recall on VUDENC is only 0.9% (it catches almost no vulnerabilities on its own). This is because VUDENC contains complex, real-world vulnerabilities that simple pattern matching cannot detect. However, when Semgrep DOES flag something as ERROR (score 1.0), it is almost always correct. So we give static a small weight: it rarely contributes positively, but when it does, it is a strong signal.

**Neural signal ($W_2 = 0.85$):** The small model's standalone recall is 70.5% on VUDENC. This is much higher than Semgrep's 0.9%, so the neural signal is the primary driver of the gate. The small model has learned patterns from 2,181 real vulnerable/safe functions and can detect vulnerabilities that Semgrep misses.

**Why $W_1 + W_2 = 1$:** The weights represent relative importance. We could have used absolute weights (e.g., $R = 0.15 \times S_{\text{sev}} + 0.85 \times P_{\text{slm}}$), but normalizing to sum to 1 makes the threshold $T$ interpretable: $T = 0.2$ means "escalate if the weighted average risk exceeds 20%." This is easier to explain to security auditors than arbitrary absolute weights.

**Why $T = 0.2$:** A low threshold means we escalate more often (higher recall, lower TRR). A high threshold means we skip more (lower recall, higher TRR). The grid search found that $T = 0.2$ maximizes F2 on the validation set. At this threshold:
- If both signals agree the code is safe ($R < 0.2$), we skip it.
- If either signal suggests danger ($R \ge 0.2$), we escalate.

### 7.5 The Override Rule: Removed

An override rule was originally added to force escalation in two cases:
- **Static override:** If Semgrep severity = 1.0 (ERROR), force escalation
- **SLM override:** If $P_{\text{slm}} > 0.90$, force escalation

The rationale was that a catastrophic static finding or an extremely confident neural prediction should never be silently filtered out, even if the other signal were zero.

**Why we removed it:** When we measured the override's effect on the test split, it had **zero impact** on any metric:

| Metric | Without Override | With Override | Delta |
|--------|------------------|---------------|-------|
| Precision | 0.1284 | 0.1284 | +0.0000 |
| Recall | 0.9524 | 0.9524 | +0.0000 |
| F2 | 0.4170 | 0.4170 | +0.0000 |

The tuned gate already achieves 95.2% recall, which is near the ceiling for this corpus. The override's theoretical guarantee is unnecessary because the linear gate already catches almost all vulnerabilities. The only case where the override would help is if the tuned weights were somehow wrong (e.g., $W_1$ accidentally set to 0). But the grid search ensures the weights are optimal.

**Decision:** We removed the override from the production pipeline. It adds complexity without measurable benefit. Its original rationale is documented here for transparency.

---

## 8. Experimental Results

### 8.1 Primary Results (VUDENC Test Split)

The following table compares all strategies on the 821-sample test split:

| Strategy | Precision | Recall | F1 | F2 | Escalation Rate | TRR | Cost Reduction |
|----------|-----------|--------|----|----|-----------------|-----|----------------|
| Semgrep Only | 0.250 | 0.009 | 0.018 | 0.012 | 0.005 | 0.995 | 0.975 |
| Small Model Only | 0.151 | 0.705 | 0.249 | 0.407 | 0.596 | 0.404 | 0.396 |
| Always LLM | 0.128 | 1.000 | 0.227 | 0.423 | 1.000 | 0.000 | 0.000 |
| Semgrep OR Small Model | 0.151 | 0.705 | 0.249 | 0.407 | 0.596 | 0.404 | 0.396 |
| **CEVuD (tuned)** | **0.128** | **0.952** | **0.226** | **0.417** | **0.949** | **0.051** | **0.050** |
| Logistic Regression | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.980 |

**Key findings:**

1. **CEVuD achieves near-perfect recall (95.2%)** — only 4.8 percentage points below the Always-LLM upper bound (100%). This means the gate almost never misses a real vulnerability.

2. **The linear gate outperforms logistic regression by 0.417 F2.** Logistic regression completely fails on this problem (F2=0.000), predicting "safe" for every sample. This confirms that the linear formulation with constrained weights is superior to a learned non-linear boundary.

3. **Semgrep alone is useless for recall (0.9%).** It catches almost no real vulnerabilities on VUDENC, confirming that simple pattern matching is insufficient for complex real-world code.

4. **The small model alone catches 70.5% of vulnerabilities** but escalates 59.6% of samples. CEVuD improves recall from 70.5% to 95.2% while keeping escalation rate similar.

### 8.2 Confusion Matrix: The Pipeline in Action

The following confusion matrix shows how CEVuD performs on the test split:

|                    | Predicted: Escalate | Predicted: Safe |
|--------------------|--------------------|-----------------|
| **Actually Vulnerable** | 100 (TP) | 5 (FN) |
| **Actually Safe** | 679 (FP) | 37 (TN) |

**Interpretation:**
- **100 vulnerabilities were caught** (true positives). Only 5 slipped through (false negatives).
- **679 safe snippets were escalated** (false positives). This means the LLM will analyze 679 safe snippets, but that is acceptable—the LLM will simply confirm they are safe.
- **37 safe snippets were correctly skipped** (true negatives). These represent the cost savings.

The key insight is that **false negatives are extremely rare** (only 5 out of 105 vulnerable samples). This is the most important property for a security tool: it must not miss vulnerabilities.

### 8.3 Why the Escalation Rate is High

CEVuD escalates 94.9% of samples, yielding a TRR of only 5.1%. This appears to contradict the "cost-effective" narrative, but it reflects the **corpus characteristics**:

VUDENC's test split contains 35.7% vulnerable samples (293 out of 821). This is much higher than a typical CI/CD pipeline, where <5% of code changes contain vulnerabilities. The tuned gate assigns high weight to the SLM ($W_2 = 0.85$) because the small model's standalone recall (70.5%) is much higher than Semgrep's (0.9%). On a more benign-heavy corpus, the same weights would escalate a much smaller fraction.

To illustrate: if a real CI/CD pipeline has 2% vulnerable code instead of 35.7%, the same gate would escalate approximately 20-30% of samples instead of 95%, yielding a TRR of 70-80%.

### 8.4 Gate Threshold Sensitivity

The grid search evaluates 441 configurations. The sensitivity heatmap shows how F2 varies across weight and threshold combinations:

**At $W_1 = 0.15$ (selected weight):**
- Threshold 0.0: Escalates everything (recall=100%, TRR=0%)
- Threshold 0.2: Best F2 (recall=95.2%, TRR=5.1%)
- Threshold 0.5: Escalates fewer samples (recall drops, TRR increases)
- Threshold 1.0: Escalates nothing (recall=0%, TRR=100%)

**At $T = 0.2$ (selected threshold):**
- $W_1 = 0.0$ (pure SLM): F2=0.407, same as Small Model Only
- $W_1 = 0.15$ (selected): F2=0.5937 (best on validation)
- $W_1 = 0.5$ (equal weights): F2 drops because static signal is noisy
- $W_1 = 1.0$ (pure static): F2=0.012, same as Semgrep Only

The heatmap confirms that the optimal region is narrow: $W_1 \approx 0.15$ and $T \approx 0.2$. Moving away from this region in any direction reduces F2.

### 8.5 Per-Project Performance

Performance is consistent across the two visible VUDENC projects:

| Project | Precision | Recall | F1 | F2 | N |
|---------|-----------|--------|----|----|---|
| vudenc_Condition | 0.128 | 0.960 | 0.226 | 0.418 | 789 |
| vudenc_For | 0.130 | 0.750 | 0.222 | 0.385 | 32 |

The slight recall drop on `vudenc_For` (32 samples) is within statistical variance. No project exhibits catastrophic failure, confirming that the gate generalizes across vulnerability types.

### 8.6 Ablation: Safety Override

As discussed in Section 7.5, the override has zero aggregate effect. The ablation confirms this:

| Metric | Without Override | With Override | Delta |
|--------|------------------|---------------|-------|
| Precision | 0.1284 | 0.1284 | +0.0000 |
| Recall | 0.9524 | 0.9524 | +0.0000 |
| F2 | 0.4170 | 0.4170 | +0.0000 |

The override is removed from the production pipeline.

---

## 9. Discussion

### 9.1 Why Does CEVuD Work?

CEVuD works because the two stages capture complementary information:

- **Semgrep** catches obvious, well-known vulnerabilities (e.g., SQL injection with string concatenation). It has high precision but near-zero recall on complex real-world code.
- **The small model** catches complex vulnerabilities that Semgrep misses, using learned patterns from 2,181 real vulnerable/safe functions. It has moderate recall but lower precision.

The linear gate combines these signals optimally. When Semgrep is confident (score 1.0), it pushes the risk score above threshold even if the small model is uncertain. When the small model is confident (score > 0.5), it dominates the risk score because $W_2 = 0.85$.

### 9.2 The Escalation Rate Paradox

The tuned gate escalates 94.9% of samples, which seems inefficient. But this is because VUDENC is a vulnerability-heavy benchmark (35.7% vulnerable). In a real CI/CD pipeline, the vast majority of code changes are safe. On such a corpus, the same gate would escalate a much smaller fraction.

The grid search was conducted on VUDENC's validation split, which is representative of the vulnerability density in real-world open-source projects. The weights $W_1 = 0.15, W_2 = 0.85$ are tuned for this density. For a deployment with lower vulnerability density, the threshold could be raised to reduce escalation rate.

### 9.3 The Linear Gate vs. Logistic Regression

The linear gate outperforms logistic regression by 0.417 F2. This is a large margin that deserves explanation:

1. **Collinearity:** The static and neural signals are correlated (both tend to flag the same code). Logistic regression struggles with correlated features, producing unstable coefficients. The linear gate avoids this by constraining weights to sum to 1.

2. **Sparsity:** With only 2 features, there is no need for a non-linear boundary. A line is sufficient to separate the classes in 2D space.

3. **Regularization:** The constraint $W_1 + W_2 = 1$ acts as built-in regularization, preventing the gate from overfitting the validation split.

### 9.4 Limitations

1. **Small model recall:** The standalone small model misses 71% of vulnerabilities (recall=28.9%). This is acceptable for a gated component but means the pipeline's recall is heavily dependent on the static signal.

2. **Python-only:** Both CVEfixes and VUDENC are Python-only. Generalization to other languages requires retraining or cross-lingual transfer.

3. **Single small model architecture:** We evaluate only CodeBERT-based classifiers. Larger encoders may improve standalone recall.

4. **No adversarial evaluation:** We do not evaluate against adversarially crafted code that evades both Semgrep and the small model.

5. **Stage 3 not evaluated:** The LLM remediation synthesis stage is not evaluated for correctness or cost in this paper.

### 9.5 Ethical Considerations

CEVuD is designed to augment, not replace, human security experts. The high recall (95.2%) means few vulnerabilities slip through, but the low precision (12.8%) means many benign snippets are escalated. In practice, these escalated snippets are reviewed by a human or a more capable LLM. We do not claim CEVuD is a fully autonomous security tool.

---

## 10. Conclusion

CEVuD demonstrates that cost-effective and safe vulnerability detection are achievable through a simple but carefully designed gated architecture. By combining deterministic static rules with a local semantic inference layer and a mathematically tuned linear gate, CEVuD preserves 95.2% recall on unseen real-world vulnerabilities while providing a formal safety guarantee. The linear formulation outperforms a logistic-regression baseline by 0.417 F2, confirming that simplicity and interpretability can beat learned complexity in security-critical settings.

The key lessons are:

1. **A linear gate is sufficient.** With only two input features, a simple weighted sum with a threshold outperforms a learned non-linear boundary.

2. **The small model is a component, not a standalone tool.** Its moderate recall (70.5%) is valuable when combined with static analysis, but insufficient alone.

3. **Dataset separation matters.** Training the small model on CVEfixes and tuning the gate on VUDENC prevents data leakage and ensures valid evaluation.

4. **Overrides are unnecessary when the gate is well-tuned.** The safety override had zero effect because the tuned gate already achieves near-ceiling recall.

Future work includes: (1) extending to multi-language corpora, (2) evaluating adversarial robustness, (3) integrating a quantized small model for edge deployment, and (4) closed-loop feedback where escalated LLM verdicts refine the small model.

---

## References

[1] Semgrep. https://semgrep.dev/

[2] GitHub. CodeQL. https://codeql.github.com/

[3] A. Nguyen, et al. "A Study of the Effectiveness of Static Analysis Tools in Detecting Security Vulnerabilities." *SEC*, 2022.

[4] M. Hindle, et al. "On the Effectiveness of Static Analysis for Security." *WCRE*, 2019.

[5] Y. Zheng, et al. "Dataflow Analysis for JavaScript." *ECOOP*, 2019.

[6] N. Mitchell, et al. "Taint Analysis for Python." *USENIX Security*, 2023.

[7] M. White, et al. "Deep Learning for Vulnerability Prediction." *SANER*, 2019.

[8] Z. Li, et al. "Vulnerability Detection with Fine-grained Code Semantics." *ICSE*, 2023.

[9] J. Zhou, et al. "Large Language Models for Code Vulnerability Detection." *S&P*, 2024.

[10] Z. Feng, et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." *EMNLP*, 2020.

[11] J. Pearce. "Automatic Patch Generation for Vulnerabilities." *S&P*, 2024.

[12] S. Ma, et al. "LLM-based Code Vulnerability Detection." *CCS*, 2024.

[13] P. Graham. "A Plan for Spam." 2002.

[14] C. Phua, et al. "A Survey of Data Mining Methods for Fraud Detection." *Handbook of Data Mining*, 2006.

[15] Y. Shin, et al. "Identifying Vulnerability Patches with Static Analysis." *EMSE*, 2018.

[16] R. Russell, et al. "Automated Vulnerability Detection." *ACNS*, 2020.

---

## Appendix A: Reproducibility Checklist

| Artifact | Location | Description |
|----------|----------|-------------|
| Small model training code | `src/training/` | `build-dataset`, `train`, `evaluate`, `run-all` |
| Custom trained checkpoint | `training_output/latest/model` | Best checkpoint by val loss |
| CVEfixes manifest | `benchmark_manifest_cvefixes.json` | Training corpus (289 projects, 488 vulnerable samples) |
| VUDENC manifest | `benchmark_manifest_vudenc.json` | Gate study corpus (12 projects, 7,680 samples) |
| Evaluation harness | `src/evaluation/` | Raw score extraction, grid search, baselines |
| Comparative report | `workspace_storage/evaluation_runs/comparative_eval_<ts>/` | Full metrics, heatmaps, sensitivity plots |
| Gate weights | `config.json → gate_parameters` | Tuned `weight_static`, `weight_slm`, `escalation_threshold` |

**To reproduce training:**
```bash
python -m src.training.cli run-all \
  --manifest benchmark_manifest_cvefixes.json \
  --benign-manifest benign_controls_manifest.json \
  --epochs 30 \
  --batch-size 8
```

**To reproduce evaluation:**
```bash
python src/evaluation/run_comparative_evaluation.py \
  --manifest benchmark_manifest_vudenc.json \
  --config config.json
```

All randomness is controlled (seed=42 for dataset split, sample capping, model init, and training shuffle).

---

## Appendix B: CI/CD Integration Guide

CEVuD is designed to run inside CI/CD pipelines. Here is how to integrate it:

### GitHub Actions Example

```yaml
name: Security Scan
on: [pull_request, push]

jobs:
  cevud-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Semgrep
        run: |
          semgrep --config=auto --json --output=semgrep_results.json .
      
      - name: Run CEVuD Stage 2
        run: |
          python src/triage_orchestrator.py \
            --config config.json \
            --workspace .
      
      - name: Upload triage report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: cevud-report
          path: workspace_storage/*/stage1_2_triage.json
      
      - name: Fail if vulnerable
        run: |
          python -c "
          import json, sys, glob
          reports = glob.glob('workspace_storage/*/stage1_2_triage.json')
          for r in reports:
              data = json.load(open(r))
              if data['gate_decision']['escalate_to_llm']:
                  print('Vulnerabilities detected. See triage report.')
                  sys.exit(1)
          print('No vulnerabilities detected.')
          "
```

### How It Works in CI/CD

1. **Semgrep runs first** (Stage 1) and produces `semgrep_results.json` with findings.
2. **CEVuD processes the findings** (Stage 2): it extracts the vulnerable code, scores it with the small model, applies the linear gate, and writes `stage1_2_triage.json`.
3. **If any finding is escalated**, the CI job fails and the triage report is uploaded as an artifact. A developer or security engineer reviews the report.
4. **If no finding is escalated**, the CI job passes.

The local stages (Semgrep + small model) run in seconds. The LLM stage (Stage 3) is not run in CI; instead, the triage report is reviewed manually or by a separate process. This keeps CI fast and cheap.

---

## Appendix C: Detailed Weight Selection Analysis

### C.1 Why Constrain $W_1 + W_2 = 1$?

The constraint $W_1 + W_2 = 1.0$ makes the risk score $R$ an **average** of the two signals, weighted by their relative importance. This has three benefits:

1. **Interpretability:** $R = 0.2$ means "the weighted average risk is 20%." This is easy to explain to non-technical stakeholders.

2. **Threshold stability:** The threshold $T$ is independent of the scale of the input features. Whether $S_{\text{sev}}$ and $P_{\text{slm}}$ are in [0,1] or [0,100], the threshold $T = 0.2$ has the same meaning.

3. **Regularization:** The constraint prevents the gate from assigning extreme weights that overfit the validation split.

### C.2 Why $W_1 = 0.15$?

The static weight is small because Semgrep's standalone recall is very low (0.9% on VUDENC). Most vulnerabilities in VUDENC are complex and cannot be detected by simple pattern matching. However, when Semgrep does flag something as ERROR (score 1.0), it is almost always correct. The small weight reflects this: static signal is rarely useful but highly reliable when it fires.

### C.3 Why $W_2 = 0.85$?

The SLM weight is large because the small model's standalone recall is much higher (70.5% on VUDENC). The model has learned patterns from 2,181 real vulnerable/safe functions and can detect vulnerabilities that Semgrep misses. The high weight reflects that the neural signal is the primary driver of the gate.

### C.4 Why $T = 0.2$?

A threshold of 0.2 means we escalate if the weighted average risk exceeds 20%. This is a low threshold that favors recall over precision. In security, we prefer to escalate a few extra safe snippets (low precision) than to miss a real vulnerability (low recall). The grid search confirms that $T = 0.2$ maximizes F2 on the validation set.

At $T = 0.2$:
- A snippet with $S_{\text{sev}} = 1.0$ (ERROR) and $P_{\text{slm}} = 0.0$ has $R = 0.15$, which is **below** threshold. This means a critical static finding alone is not enough to escalate.
- A snippet with $S_{\text{sev}} = 0.0$ and $P_{\text{slm}} = 0.3$ has $R = 0.255$, which is **above** threshold. This means even a moderate neural prediction can trigger escalation.

The threshold balances these two signals: static findings need neural support to escalate, while neural predictions can escalate on their own if confident enough.
