# Research Outline: CEVuD (Cost-Effective Vulnerability Detection)

*Target Venue: IEEE Symposium on Security and Privacy (S&P) / ACM CCS*

---

## 1. Abstract
Large Language Models (LLMs) demonstrate state-of-the-art capability in detecting and remediating software vulnerabilities. However, their deployment in large-scale Continuous Integration/Continuous Deployment (CI/CD) pipelines is severely constrained by prohibitive financial costs, extreme token consumption, and API latency. Conversely, traditional Static Application Security Testing (SAST) tools scale effortlessly but suffer from notoriously high false-positive rates due to a lack of semantic understanding. This paper proposes **CEVuD**, a hybrid, multi-stage "gated reasoning" architecture that mathematically couples deterministic static taint analysis with a zero-marginal-cost edge compute layer (a local Small Language Model, or SLM). By filtering trivial code changes locally, CEVuD achieves a massive Token Reduction Rate (TRR) and slashes LLM API costs while preserving the recall of a pure LLM-based pipeline. Evaluated on thousands of real-world commits from the CVEfixes and VUDENC datasets, our empirically tuned linear gate demonstrates that enterprise-grade security and cost efficiency are not mutually exclusive.

---

## 2. Introduction

### 2.1 The Research Gap
Modern software engineering relies on automated, shift-left security tooling. Existing paradigms fall into two extremes:
1. **The Legacy Paradigm (SAST):** Tools like Semgrep or CodeQL process code locally at zero marginal cost. However, they rely on rigid syntactical rules, resulting in alert fatigue and high false-positive rates that waste developer hours.
2. **The Frontier Paradigm (LLMs):** Cloud-based LLMs possess the deep contextual reasoning required to map complex data flows and synthesize exact patches. However, evaluating every single code commit in a large enterprise repository via GPT-4 or Claude is financially unviable and frequently hits API rate limits.

**The Gap:** There is currently no mathematically robust orchestration framework that safely bridges these two paradigms. Existing pipelines either blindly send everything to the cloud (costly) or blindly trust static tools (noisy). 

### 2.2 Research Questions (RQs)
This paper seeks to answer the following:
* **RQ1 (Efficiency):** To what extent can a localized, mathematically weighted semantic gate reduce the token volume sent to frontier LLMs without sacrificing recall?
* **RQ2 (Safety):** How do we guarantee that high-risk edge cases (e.g., neural blindspots) do not bypass the local filter and compromise the system's safety?
* **RQ3 (Generalizability):** Does a static-neural hybrid gate tuned on a controlled validation set generalize to unseen, real-world historical commits spanning diverse vulnerability topologies?

---

## 3. Intuition Behind the Approach
The intuition driving CEVuD is **Gated Reasoning**. We theorize that most code changes in a pull request are either strictly safe (e.g., UI tweaks) or trivially malicious (e.g., obvious hardcoded credentials). 
* An LLM is a supercomputer; we should not use a supercomputer to calculate `2 + 2`. 
* By running a lightweight, edge-based sequence classifier (SLM) alongside a static analyzer, we can confidently suppress the vast majority of code changes locally for free.
* Only when a code snippet occupies a zone of high semantic ambiguity or triggers severe static alarms do we "open the gate" and spend financial capital to escalate the snippet to a frontier LLM.

---

## 4. Proposed Methodology (The CEVuD Architecture)

### 4.1 Granular AST Parsing
To ensure the local models evaluate code precisely as a developer wrote it, CEVuD parses the repository into an Abstract Syntax Tree (AST). It extracts pristine `FunctionDef` blocks corresponding to modified lines, maintaining exact structural integrity rather than relying on naive text truncation.

### 4.2 Stage 1: Deterministic Static Taint (Semgrep)
The pipeline first executes Semgrep utilizing a hybrid ruleset of community standards and proprietary taint-tracking rules. It maps untrusted data from source to sink, outputting a discrete severity score ($S_{\text{sev}} \in [0, 1]$).

### 4.3 Stage 2: Probabilistic Neural Gating (Local SLM)
The AST-sliced snippet is passed to a lightweight **CodeBERT-based sequence classifier** ($P_{\text{slm}} \in [0, 1]$) that runs at zero marginal cost on the edge. CEVuD ships with two interchangeable options, both loading via `AutoModelForSequenceClassification` with no code change:

* **Default (pretrained):** `jayansh21/codesheriff-bug-classifier`, a 125M-parameter classifier fine-tuned on `microsoft/codebert-base`.
* **Custom (this work):** a CodeBERT classifier **fine-tuned by CEVuD** on the CVEfixes corpus (`src/training/`). The architecture is the standard `RobertaForSequenceClassification` head on `microsoft/codebert-base` — a `pooler` (`dense` 768→768, `tanh`) followed by a `classifier` (`dense` 768→768 → `out_proj` 768→2); the vulnerable-class probability is `softmax(logits)[:, 1]`. Training uses class-weighted cross-entropy, project-level stratified splits, and early stopping on validation loss (see `MODEL_CARD.md`). This custom model is the artifact evaluated in Section 5.

### 4.4 The Linear Risk Equation
The decision to escalate to the cloud LLM is governed by a Continuous Composite Risk Score ($R$):
$$R = (W_1 \cdot S_{\text{sev}}) + (W_2 \cdot P_{\text{slm}})$$
Where $W_1$ and $W_2$ are empirically derived weights constrained to $W_1 + W_2 = 1.0$. If $R \ge T_{\text{escalation}}$, the gate opens.

### 4.5 Stage 3: LLM Remediation Synthesis
Escalated snippets trigger an autonomous task-decomposition agent. Utilizing a local SQLite vector store for Retrieval-Augmented Generation (RAG) to trace cross-file dependencies, the agent generates a structured `remediation_dossier.md` (Root Cause, Lineage, PoC, and Patch).

---

## 5. Experimental Setup

### 5.1 Databases & Ground Truth

To avoid the performance inflation inherent in small or synthetic benchmarks, CEVuD is built and evaluated against two established, publicly available vulnerability datasets. Critically, **neither dataset requires a bulk database download** — both are accessed via the HuggingFace streaming API, enabling row-by-row ingestion without holding gigabytes in memory.

**Two-corpora workflow.** CVEfixes develops the small Stage-2 classifier end-to-end: it is the corpus for the model's **training**, **validation** (early-stopping / best-checkpoint selection), and its own **evaluation** (`convert_cvefixes.py` → `benchmark_manifest_cvefixes.json`). VUDENC is then the corpus for the **gate study** — the comparative evaluation of the full CEVuD pipeline (Semgrep + the CVEfixes-trained classifier + the gating strategies) run by `src/evaluation/` (`convert_vudenc.py` → `benchmark_manifest_vudenc.json`). Within CVEfixes, train/validation/test are split by project so no project leaks across splits; VUDENC additionally provides an independently curated corpus for the gate study. Both manifests share an identical schema (see `DATASET_CARD.md`), so the training and evaluation harnesses are interchangeable.

#### Dataset A: VUDENC (`DetectVul/Vudenc`)
VUDENC (Vulnerability Detection with Deep Learning on a Natural Codebase) contains real-world Python functions labeled at a **per-line granularity** across seven vulnerability categories: SQL Injection, Cross-Site Scripting (XSS), Command Injection, XSRF, Remote Code Execution, Path Disclosure, and Open Redirect.

* **Source:** `DetectVul/Vudenc` on HuggingFace (~15,000 functions)
* **Schema:** `raw_lines` (List[str]), `label` (List[int], per-line), `type` (List[str], vulnerability category)
* **CEVuD Adaptation (evaluation set):** The `convert_vudenc.py` script joins raw lines back into complete function strings, collapses per-line labels into a single function-level label (1 if any line is flagged), and groups samples by vulnerability type as logical "projects" so the gate study's splits are leakage-free. VUDENC ships no repository or commit metadata, so each sample embeds its `source_code` inline (`local_source`) and the provenance fields (`repo_url`, `commit_id`, `cve_id`, `cvss_score`, `diff_with_context`, `fixed_code`) are emitted empty for schema consistency with the CVEfixes manifest.
* **Why this dataset:** It covers seven distinct vulnerability topologies, ensuring the linear gate is not tuned to a single vulnerability pattern. Because VUDENC is held out from classifier training, it is the corpus on which the gate's generalization (RQ3) is measured.

#### Dataset B: CVEfixes (`hitoshura25/cvefixes`)
CVEfixes links public Common Vulnerabilities and Exposures (CVEs) directly to the open-source commits that introduced and fixed them. It is the largest and most reproducible real-world vulnerability dataset available.

* **Source:** `hitoshura25/cvefixes` on HuggingFace (~50,000+ Python samples after language filtering)
* **Schema:** `vulnerable_code` (str), `fixed_code` (str), `hash` (str, fix-commit SHA), `repo_url` (str), `cve_id` (str), `cwe_id` (str), `language` (str), `cvss2/cvss3_base_score` (float), `diff_with_context` (str)
* **CEVuD Adaptation (training set):** The `convert_cvefixes.py` script streams data, filters to Python, captures the fix-commit SHA from the `hash` column (as `commit_id`), and organises each repository as a `git_source` project so the evaluation harness can clone it and read the real vulnerable function plus context. The function body is also embedded inline as a clone-failure fallback. This makes training fast and reproducible while preserving full provenance.
* **Balanced labels (1:1).** Each row already contains the pre-fix function (`vulnerable_code`) and the post-fix function (`fixed_code`). The converter emits **both** as a matched pair: the pre-fix snippet is labeled `1` (vulnerable) and anchored to the diff's PRE-image line range (read from the *parent* of the fix commit), while the post-fix snippet is labeled `0` (safe) and anchored to the POST-image line range (pinned to the fix commit via `target_commit`). Treating the commit *before* the fix as vulnerable and the commit *after* the fix as safe therefore yields a naturally **balanced** dataset with no synthetic resampling — important for an unbiased F2 / recall evaluation.
* **Alternative ID:** `dima806/fixedbugs` carries a similar schema and can be used as a drop-in alternative.
* **Why this dataset:** CVEfixes provides direct CVE-to-commit traceability, making it the gold standard for training a real-world Python vulnerability classifier. The sheer scale (thousands of diverse projects) prevents the custom model from overfitting to any single codebase style. It is the training corpus; generalization is then measured separately on VUDENC (Section 5.1, Dataset A).

### 5.2 Data Splitting Strategy

Data is split **by project** (not by sample) into Train (60%), Validation (20%), and Test (20%) partitions. The DatasetSplitter enforces that no project appears in more than one split. This prevents the model from learning project-specific patterns during validation-set tuning that would artificially inflate test-set performance.

### 5.3 Unbiased Parameter Tuning (Grid Search)

Parameters are not hand-picked. $W_1$, $W_2$, and $T_{\text{escalation}}$ are derived via an exhaustive 2D grid search **exclusively on the held-out validation split**, maximizing $F_{\beta}$ (with $\beta=2.0$ to heavily penalize false negatives). 

### 5.4 Heuristic Safety Override

To counteract neural blindspots, an override heuristic is applied: if $S_{\text{sev}} = 1.0$ (Critical Error) or $P_{\text{slm}} > 0.90$, escalation is forced regardless of $R$. The efficacy of this heuristic is proven via an ablation study on the test split.

---

## 6. Baselines & Comparative Evaluation
CEVuD is evaluated against the following strict baselines to quantify its value:
1. **Semgrep Only:** Escalates all static findings. (Baseline for legacy SAST).
2. **SLM (Local Classifier) Only:** Escalates purely on neural signals.
3. **Always Escalate (Always-LLM):** The theoretical upper bound for Recall, but the absolute worst-case scenario for financial cost.
4. **Semgrep OR SLM (OR-gate):** Naive combined trigger.
5. **Logistic Regression Gate (Linearity Check):** A non-linear boundary fit on the validation split. Evaluated solely to prove whether the hand-defined linear formula is mathematically optimal.

---

## 7. Results & Key Performance Indicators (KPIs)
The paper will report findings based on:
* **Token Reduction Rate (TRR) & Cost Savings Ratio (CSR):** The financial API savings relative to the `Always-LLM` baseline, directly mapped by the pipeline's `escalation_rate`.
* **Recall / F2 Score:** Proof that the TRR does not come at the expense of dropping critical security alerts. 
* **Ablation Delta:** The exact marginal improvement provided by the safety override heuristics compared to the naked mathematical gate.

---

## 8. Conclusion
CEVuD proves that the integration of frontier LLMs into enterprise CI/CD pipelines does not require unbounded financial budgets. By marrying deterministic static rules with local semantic inference, organizations can achieve a mathematically rigorous cost-to-safety tradeoff, successfully executing high-fidelity vulnerability remediation at scale.
