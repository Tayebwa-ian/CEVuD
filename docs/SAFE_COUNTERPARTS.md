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

> **⚠️ Current default (updated).** After measuring that the post-fix twin is a
> *near-duplicate* of its vulnerable twin (median token-similarity ≈ 0.94, so
> 68% of pairs are >0.90 similar), the pipeline **no longer emits the post-fix
> function as a `label=0` sample at all**. `convert_cvefixes.py` is now
> **vulnerable-only by default** (`--emit-fixed-safe` restores the legacy
> balanced-pair manifest for A/B studies), and the genuine safe class is
> supplied entirely by the mined controls of Step 1. Three guards enforce that
> the near-duplicate contradiction can never re-enter the data:
> 1. **Miner similarity guard** — a mined candidate is dropped if it is
>    >0.75 token-similar to *any* vulnerable function in its project
>    (this is what excludes the post-fix twin automatically).
> 2. **Builder near-duplicate guard** — after chunking, any `label=0`
>    chunk >0.75 similar to a `label=1` chunk in the same project is dropped.
> 3. **Trainer pre-flight** — refuses to start when ≥5% of training samples
>    are hard *or* near-duplicate (>0.90) contradictions.
> The post-fix function is still retained on the vulnerable sample's
> `fixed_code` field for the optional contrastive objective (Step 2) and for the
> Step 0 audit.

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
in the real repository and mine functions to use as safe controls, with two
refinements that make them *sharp* negatives:

* **Same-file siblings first.** We prefer functions from the files the fix
  *touched* (excluding the patched function itself, which the similarity guard
  removes). These siblings share the vulnerable function's imports, APIs, and
  coding style, so they teach the sharpest "safe vs vulnerable" boundary. They
  are tagged `sample_subtype="benign_sibling"`. Functions from files the commit
  did **NOT** modify are mined next (`sample_subtype="benign_control"`); by
  construction those are untouched by any security fix.
* **Near-duplicate guard.** Every candidate is compared (token similarity)
  against *all* vulnerable snippets in the same project and dropped if it exceeds
  `--similarity-threshold` (default **0.75**). This automatically discards the
  patched function's post-fix twin and any other lightly-edited copy, so the
  safe class can never re-introduce the contradiction.

We then:

* apply the usual `code_signal_line_count >= min_code_lines` filter (drop
  comment/version-only fragments),
* de-duplicate by normalized text (no repeated controls),
* mine up to `--ratio` × (number of vulnerable samples in the project) controls
  per project (default **5×**), honouring `--samples-per-commit` /
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
    --samples-per-commit 8 --ratio 5 --similarity-threshold 0.75 --max-workers 6
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
* **Always-included (important).** Benign-control projects are named
  `benign::<repo>` and are pulled *out* of `select_projects` (which
  selects by CWE coverage / `--max-projects`). Otherwise a `--few-shot`
  run would silently drop them (their `vulnerability_type` is
  `"benign"`, which carries almost no CWE-coverage weight), defeating the
  entire remedy. They are therefore **always added to the pool** regardless
  of how many primary (vulnerable) projects are selected, while still
  keeping their own project identity for leakage safety.
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
| `"fixed"` | post-fix function (legacy `--emit-fixed-safe` only; OFF by default) | 0 | `convert_cvefixes.py` |
| `"benign_sibling"` | function from a file the fix touched, but not the patched function (sharp negative) | 0 | `mine_benign_functions.py` |
| `"benign_control"` | function untouched by any fix commit (verified-benign) | 0 | `mine_benign_functions.py` |
| `"benign"` | function labeled safe by a corpus's own annotation (VUDENC all-zero lines) | 0 | `convert_vudenc.py` |
| `None` | legacy manifest (backwards-compatible) | — | — |

`build_dataset` preserves `sample_subtype` through enrichment **and** chunking,
and `dataset_summary.json` now reports a `sample_subtypes` breakdown.

> **Leakage note (important for the paper).** Benign controls are mined from
> the *same repositories* as the vulnerabilities. They are added as their own
> projects (`benign::<repo>`), but `assign_splits` **collapses the `benign::`
> prefix** to the underlying repo name before splitting, so a benign control and
> a vulnerable sample from the same repo land in the *same* split — which is
> **correct and conservative**: it means the model is never tested on a repo it
> trained on, and any repo-level style signal is shared rather than leaked
> across splits. Every split is therefore both project-disjoint *and* two-class
> (it cannot end up single-class, which would break ROC/early-stopping). See
> `_split_key` in `src/training/dataset_builder.py`.

> **Validity threat — same-repo overlap (state it in the paper).** Because
> benign controls are mined from the *same* CVEfixes repositories as
> the vulnerabilities, the SLM's *training distribution* and the benign
> "safe" class partly overlap at the repo level. Two consequences:
> (i) in **training**, a benign control from repo X and a vulnerable
> sample from repo X share repo-level style — harmless for the classifier,
> but it means the model is not tested on a repo it never saw;
> (ii) in the **VUDENC gate study** (§4.2), the merged benign
> controls are CVEfixes-derived, so the "safe" precision there is
> measured *in-distribution* (same repos the SLM trained on), not on
> truly held-out code. To claim genuine out-of-distribution
> safe-precision, mine benign controls from a *disjoint* repository
> set (e.g. a separate corpus such as CodeSearchNet) and merge those
> instead. The shipped `mine_benign_functions.py` uses CVEfixes repos
> by default because that maximises reuse of already-cloned history;
> swapping the input manifest is a one-line change.

---

## 7b. Generating a sufficient, well-balanced dataset in a 2–4 h budget

The goal: train the model **well** (enough benign signal + enough
CVEfixes pairs) without the run blowing past a few hours. Three
levers, in order:

1. **Mine as many benign controls as possible (one-time, network).**
   The miner revisits every fix commit and pulls functions from the
   files the commit did NOT touch. To maximize yield, raise the
   per-commit and per-repo ceilings:
   ```bash
   python src/scripts/mine_benign_functions.py \
       --manifest benchmark_manifest_cvefixes.json \
       --output benign_controls_manifest.json \
       --samples-per-commit 20 --max-commits 100000 --max-workers 8
   ```
   (`--max-commits 100000` is "process every fix commit"; the
   dedup set keeps it from exploding with repeats). This step is
   **separate from training** — run it once and reuse the manifest.

2. **Bound the training set, don't let benign dominate.**
   Benign controls are `label=0`, so without a cap they can outnumber
   the vulnerable class and slow convergence. `build-dataset` caps via
   `--max-samples-per-class` / `--max-total`; the benign projects are
   **always included** (see §4.1) but still respect those caps:
   ```bash
   python -m src.training.cli build-dataset \
       --manifest benchmark_manifest_cvefixes.json \
       --benign-manifest benign_controls_manifest.json \
       --max-total 8000 --max-samples-per-class 3000
   ```
   This keeps the set large (enough to train well) yet bounded
   (so the CodeBERT run fits the 2–4 h window; chunking
   multiplies effective samples but each chunk is tiny/fast).

3. **Train with early stopping, not a huge fixed epoch count.**
   Keep the high ceiling but let early stopping halt once validation
   loss plateaus:
   ```bash
   python -m src.training.cli train --epochs 20 --batch-size 8 --lr 2e-5
   ```
   On a GPU this finishes in minutes; on CPU it lands in the
   2–4 h window. The `dataset_summary.json` `sample_subtypes`
   breakdown confirms how many `vulnerable` / `fixed` / `benign_control`
   rows actually fed the model.

> **Cloning + cleanup are guaranteed.** Every clone in this pipeline
> (`dataset_builder`, `mine_benign_functions`, `diagnose_safe_counterparts`,
> and the evaluation extractor via `resolve_project_workspace`) is
> deleted in a `finally` block immediately after the needed
> information is extracted. No repository source is left on disk after
> any of these steps. (Audit: `grep -n "rmtree" src/` — every
> `clone_repo` call site has a matching delete.)

> **Semgrep is a HARD, non-skippable Stage-1 input.** The reported
> gate study MUST run `run_comparative_evaluation.py` **without**
> `--cache` (which reuses a cached `raw_scores.json` and skips
> Semgrep **and** the SLM) and **without** `--inline` for the
> final numbers (inline still runs Semgrep but only over the
> isolated materialized snippets, not the real repo root). If `semgrep`
> is not installed, `raw_score_extractor._run_semgrep` now **fails
> fast** with a clear message instead of silently emitting severity
> 0.0 for every sample.

---

## 8. Reproducibility checklist

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
