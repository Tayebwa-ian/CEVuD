# Safe Counterparts — Methodology & Ground Truth

> **Authoritative reference for the "safe counterpart" problem and its fix in
> CEVuD.** This document is the single source of truth for the safe-counterpart
> methodology described in the research paper. It supersedes the informal
> reasoning in the project chat and must be cited whenever the training data's
> "safe" class is discussed.

---

## 1. The problem we are solving

CEVuD's custom Stage-2 classifier is a **binary** `vulnerable (1) vs safe (0)`
function classifier. To train it we need, for every vulnerable example, a
corresponding *safe* example. The naive assumption is:

> *"The safe counterpart of a vulnerability is just the code **after** the fix
> commit."*

That assumption is **wrong in two distinct ways**, and both corrupt the
training signal.

### 1.1 CVEfixes *does* ship a safe counterpart — but it is noisy

`convert_cvefixes.py` does **not** use "the commit after the fix". It uses the
**post-fix function at the fix commit itself** (`fixed_code`), which is the
correct, minimally-corrupted choice. But the fix commit is still a real-world
commit, and real-world fix commits routinely **bundle unrelated edits**:

* the actual security one-liner (`execute(... %s, (q,))`), **plus**
* a refactor of the surrounding function,
* a reformatting pass,
* an unrelated feature added to the same function.

When that happens, the `label=0` ("fixed") sample differs from its `label=1`
("vulnerable") twin in ways that have **nothing to do with the
vulnerability**. The classifier can then learn *"the refactor"* instead of
*"the vulnerability"*, and — critically — **fail to generalize** to a held-out
corpus (VUDENC) that it never saw during training (RQ3).

There is a second, subtler issue: the post-fix function is only a **relative**
negative. It is guaranteed *not* to carry *that one* CVE, but it may still
contain a *different* weakness. So it is a weak label, not a verified-safe
one.

### 1.2 VUDENC has *no* safe counterpart at all

`convert_vudenc.py` mines functions from vulnerability-fixing commits and labels
them at the per-line level. It ships **no `fixed_code` and no commit
metadata**. Its `label=0` samples are simply functions whose *every line* was left
unlabeled by the corpus's own annotators — functions pulled out of a
vulnerability commit, **not** verified-benign code. Worse, because VUDENC is
near-positive-only, the **gate study** (which evaluates the full CEVuD pipeline
against VUDENC) can only measure *recall* — **precision is undefined** without
a genuine safe class.

> **Net statement.** The safe counterpart is not "missing" — it is
> *noisy and weak*. The fix is not to invent a new source of safe code blindly,
> but to (a) **measure** the noise, (b) **inject verified-benign
> negatives** mined from the same repositories, and (c) **use the post-fix pair
> as a contrastive signal** rather than as a hard `label=0` target.

---

## 2. The three-step remedy

```
                         ┌─────────────────────────────────────────────┐
                         │ STEP 0 — MEASURE (diagnose_safe_counterparts) │
                         │  How noisy is the post-fix "safe" class?       │
                         └───────────────────────┬─────────────────────┘
                                                 │ quantify contamination
                                                 ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ STEP 1 — VERIFIED-BENIGN NEGATIVES (mine_benign_functions)    │
        │  Mine functions UNTOUCHED by any fix commit -> true label=0.     │
        │  Fed into BOTH training (build-dataset) AND the VUDENC gate     │
        │  study (convert_vudenc --benign-manifest).                     │
        └────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │ STEP 2 — CONTRASTIVE USE OF THE FIXED PAIR (trainer, opt.)   │
        │  Keep (vuln, fixed) as a contrastive pair (vuln≈fixed ≺ benign)│
        │  instead of a hard 0/1 label, so fixed-function noise hurts    │
        │  less. OFF by default; standard CE remains the reported setup.   │
        └────────────────────────────────────────────────────────────────────┘
```

---

## 3. STEP 0 — `diagnose_safe_counterparts.py` (measure)

**Goal.** Quantify, *without cloning any repository*, how trustworthy the
post-fix "safe" class is.

**Why no clone is needed.** The CVEfixes manifest already embeds both halves
inline: every `label=1` sample carries `fixed_code` (the post-fix function),
and its matching `label=0` sample carries that same text as `source_code`. The
script matches each vulnerable sample to its post-fix twin by normalized text and
computes three metrics per pair (see `src/data_quality.py`):

| Metric | Definition | Meaning |
|---|---|---|
| `changed_line_ratio` | `difflib`-based fraction of (combined) lines that differ between vuln and fixed | Low = minimal clean fix; high = heavily reworked |
| `is_trivial_change` | the only differences are comments/docstrings/version strings | A *clean minimal* fix (the opposite of the bundled-edit problem) |
| `is_substantial_change` | `changed_line_ratio > 0.5` | **Red flag:** the function was reworked across >50% of its lines — a bundled refactor, not just the patch |

It also reports an aggregate histogram of `changed_line_ratio` and, for the
fix-commit **breadth** (how many files a fix commit touched — independent
evidence of bundling), an optional `--clone-subset N` mode that clones the
first N projects and runs `git show --stat` on each unique fix commit. That mode
needs network + git and is **off by default**.

**Run it first:**

```bash
python src/scripts/diagnose_safe_counterparts.py \
    --manifest benchmark_manifest_cvefixes.json
# optional, network-heavy: also measure fix-commit breadth
python src/scripts/diagnose_safe_counterparts.py \
    --manifest benchmark_manifest_cvefixes.json --clone-subset 20
```

**Outputs:**
* a stdout summary (matched pairs, trivial %, substantial % — the bundled-edit
  flag — mean/median changed-line ratio, histogram), and
* a machine-readable report at
  `workspace_storage/evaluation_runs/safe_counterpart_diagnosis.json`
  containing per-pair records plus the aggregates. **This JSON is the artifact
  to cite when writing up the contamination measurement.**

**Decision rule.** If `substantial_pair_fraction` is small (say <10%), the
post-fix "safe" class is acceptably clean and Step 1 is a *quality* upgrade
rather than a *correctness* requirement. If it is large, the classifier trained
on the raw pairs is at high risk of learning refactors — run Step 1 before any
training run that will be reported.

---

## 4. STEP 1 — `mine_benign_functions.py` (verified-benign negatives)

**Goal.** Give the classifier genuine safe examples: functions that are
demonstrably *not* part of any vulnerability fix.

**Method.** For every fix commit in the CVEfixes manifest, check out that commit
in the real repository and take functions from the files the commit did **NOT**
modify. By construction, a function in an *unmodified* file was untouched by
any security fix, so it is a **strong benign control**
(`label=0`, `sample_subtype="benign_control"`). We then:

* apply the usual `code_signal_line_count >= min_code_lines` filter (drop
  comment/version-only fragments),
* de-duplicate by normalized text (no repeated controls),
* randomly sample up to `--samples-per-commit` per commit (default 3) and
  `--max-per-project` / `--max-total` caps,
* stamp provenance (`repo_url`, `commit_id`, `target_commit`) on every sample.

The output is a manifest in the **same schema** as `benchmark_manifest_cvefixes.json`
(one `git_source` project per upstream repository). Because it reuses the real
repositories, it is a **network/git operation** — run it once to produce the
manifest, then reuse the manifest.

```bash
python src/scripts/mine_benign_functions.py \
    --manifest benchmark_manifest_cvefixes.json \
    --output benign_controls_manifest.json \
    --samples-per-commit 3 --max-workers 4
```

**Residual limitation (state it in the paper).** An unmodified function is
"*not known to be bad*", not *provably safe* — it could still contain a
latent, un-annotated vulnerability. This is the standard weakness of
*negative sampling* and is strictly better than the alternative (treating the
post-fix function as safe). We do **not** claim the benign controls are
provably vulnerability-free; we claim they are *unrelated to the fix under
study*, which is exactly the property needed to teach the model "what clean code
looks like" rather than "what code-after-a-fix looks like".

### 4.1 Feeding benign controls into TRAINING

`build-dataset` now accepts the mined manifest and merges its projects into the
training pool *before* project selection and splitting:

```bash
python -m src.training.cli build-dataset \
    --manifest benchmark_manifest_cvefixes.json \
    --benign-manifest benign_controls_manifest.json
```

* Benign controls keep their **own project identity**, so the project-level split
  still prevents leakage (no benign sample shares a project with a vulnerable
  sample from the same repo? — actually it *does* share the repo; see §6).
* They count as `label=0`, so the existing few-shot `max_per_class` caps
  automatically balance them against the vulnerable and post-fix samples.
* The data-quality contradiction pass still drops any benign control whose text
  collides with a vulnerable sample (a genuine hard contradiction).

### 4.2 Feeding benign controls into the VUDENC GATE STUDY

The VUDENC gate study is otherwise near-positive-only, so precision is
undefined. `convert_vudenc.py` gains a `--benign-manifest` flag that merges
the mined controls in as **offline** `local_source` projects (their
`source_code` is already embedded, so the evaluation harness scores them inline
— no extra cloning):

```bash
python src/scripts/convert_vudenc.py \
    --output benchmark_manifest_vudenc.json \
    --benign-manifest benign_controls_manifest.json
```

The merged samples are tagged `sample_subtype="benign_control"` and grouped
into a `vudenc_benign_controls` project. The gate study can now report
precision/recall/F2 honestly.

---

## 5. STEP 2 — optional contrastive use of the fixed pair (`trainer.py`)

**Goal.** Reduce the damage done *when* the post-fix function is noisy, by
stopping it from being a hard `label=0` target.

**Idea.** Instead of forcing `vulnerable=1` and `fixed=0` as two independent
rows, treat them as a **contrastive triplet**: pull the vulnerable function
toward its fixed twin and push *both* away from a benign control. Noise in the
fixed function then only weakly perturbs a similarity objective, not a
confident 0/1 decision.

**Implementation.** `trainer.py` gains an optional supervised-contrastive term
(`_supervised_contrastive_loss` over the `[CLS]` embeddings, positives =
same-label samples in the batch, negatives = different-label). It is combined
with the existing class-weighted cross-entropy:

```
loss = CE_loss + λ · SupCon_loss ,   λ = contrastive_lambda (default 0.1)
```

Enabled with:

```bash
python -m src.training.cli train --contrastive --contrastive-lambda 0.1
```

**Default = OFF.** The **reported / recommended** training setup remains the
standard class-weighted cross-entropy on `(vulnerable=1, fixed=0,
benign_control=0)`. The contrastive objective is provided for ablation
experiments and as the designed methodology; it is OFF in `config.json`
(`training.contrastive.enabled: false`). This keeps the default pipeline
bit-for-bit unchanged while making the contrastive design reproducible.

---

## 6. Schema addition: `sample_subtype`

Every `BenchmarkSample` (and the enriched training JSONL) now carries an
optional `sample_subtype` that disambiguates *why* a sample has its label.
This makes the whole safe-counterpart methodology **auditable end-to-end**.

| `sample_subtype` | Meaning | `label` | Produced by |
|---|---|---|---|
| `"vulnerable"` | pre-fix function of a CVEfixes pair | 1 | `convert_cvefixes.py` |
| `"fixed"` | post-fix function (the "safe counterpart" we audit) | 0 | `convert_cvefixes.py` |
| `"benign_control"` | function untouched by any fix commit (verified-benign) | 0 | `mine_benign_functions.py` |
| `"benign"` | function labeled safe by a corpus's own annotation (VUDENC all-zero lines) | 0 | `convert_vudenc.py` |
| `None` | legacy manifest (backwards-compatible) | — | — |

`build_dataset` preserves `sample_subtype` through enrichment **and** chunking,
and `dataset_summary.json` now reports a `sample_subtypes` breakdown.

> **Leakage note (important for the paper).** Benign controls are mined from
> the *same repositories* as the vulnerabilities. They are added as their own
> projects (`benign::<repo>`), and the split is **by project**, so a benign
> control and a vulnerable sample from the same repo land in the *same* split —
> which is **correct and conservative**: it means the model is never tested on
> a repo it trained on, and any repo-level style signal is shared rather than
> leaked across splits. (If instead you want strict repo-isolation between
> vuln and benign, merge them under a single project name; the default keeps
> them separate for transparency.)

---

## 7. Reproducibility checklist

```bash
# 0. (optional) diagnose the post-fix "safe" class — no network needed
python src/scripts/diagnose_safe_counterparts.py \
    --manifest benchmark_manifest_cvefixes.json

# 1. mine verified-benign controls (network/git; run once, reuse manifest)
python src/scripts/mine_benign_functions.py \
    --manifest benchmark_manifest_cvefixes.json \
    --output benign_controls_manifest.json

# 2. build training splits WITH benign controls
python -m src.training.cli build-dataset \
    --manifest benchmark_manifest_cvefixes.json \
    --benign-manifest benign_controls_manifest.json

# 3. train (standard CE by default; add --contrastive for the ablation)
python -m src.training.cli train --epochs 20 --batch-size 8 --lr 2e-5

# 4. give the VUDENC gate study a real safe class
python src/scripts/convert_vudenc.py \
    --output benchmark_manifest_vudenc.json \
    --benign-manifest benign_controls_manifest.json
python src/evaluation/run_comparative_evaluation.py \
    --manifest benchmark_manifest_vudenc.json --config config.json
```

---

## 8. Relationship to the other docs

* `docs/DATA_QUALITY.md` — the *intra-pair* noise filters (trivial-change,
  contradiction). The `changed_line_ratio` / `is_substantial_change` helpers
  added here are the *inter-pair / bundled-edit* complement.
* `docs/DATASET_CARD.md` — the manifest schema; `sample_subtype` is now part
  of that schema.
* `docs/TRAINING.md` — the training walkthrough; §4.1 above is the new
  `--benign-manifest` step.
* `docs/SLM_CHUNKING.md` — the contrastive idea was already listed there as a
  future borrowing (#3); it is now implemented (Step 2).
* `docs/MODEL_CARD.md` — the trained classifier's "safe" class now *may*
  include verified-benign controls; state this in the training-data section.
