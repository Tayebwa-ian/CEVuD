# Data Quality & Noise Filtering — CEVuD Training/Eval Corpora

## TL;DR — why training wasn't learning

The first CVEfixes training run produced a loss pinned at `ln(2) ≈ 0.693` and a
`roc_auc` of ~0.5 for the whole run. That is the signature of a model that is
**not learning anything**: it just oscillates between predicting "all
vulnerable" and "all safe".

Root cause: the (vulnerable, safe) pairs were frequently **near-identical text
with opposite labels**. Two concrete cases from the original `train.jsonl`:

* `idna/package_data.py`: `__version__ = '3.6'` (label 1) vs `__version__ = '3.7'`
  (label 0) — the *only* difference is a version string.
* `mistune.py`: the vulnerable and safe `text` were **byte-for-byte identical**
  (only a metadata field differed).

When the input is (near-)identical but the label flips, no classifier can fit
it — it is pure label noise. A large fraction of CVEfixes rows are exactly this
kind of trivial fix (version bumps, docstring edits, changelog tweaks inside a
large function).

## What changed (project-wide)

A new shared module `src/data_quality.py` implements conservative, dependency-free
heuristics, and every stage of the pipeline now rejects noisy samples:

| Stage | File | What it does |
|---|---|---|
| Source | `src/data_quality.py` | `is_trivial_change`, `_is_noise_file`, `code_signal_line_count`, `find_contradictions` |
| CVEfixes converter | `src/scripts/convert_cvefixes.py` | Skips noise files, trivial (vuln, safe) pairs, and low-signal snippets **at conversion time** |
| VUDENC converter | `src/scripts/convert_vudenc.py` | De-duplicates snippets and drops hard contradictions (identical text, both labels) |
| Dataset build | `src/training/dataset_builder.py` | Defensive pass drops contradictory / low-signal enriched samples after full-function expansion |
| Trainer | `src/training/trainer.py` | Pre-flight guard refuses to train when ≥5% of samples are contradictory (override with `--allow-noisy-data`) |

### The *inter-pair* complement: bundled-edit contamination

The filters above catch **intra-pair** noise (a single (vuln, safe) pair that
is near-identical or trivial). They do **not** catch **inter-pair** noise: a
fix commit that bundles an *unrelated* edit into the same function, so the
`label=0` ("fixed") sample differs from its twin for non-security reasons.

Two helpers in `src/data_quality.py` address this, and they are the basis of
the safe-counterpart methodology documented in `docs/SAFE_COUNTERPARTS.md`:

* **`changed_line_ratio(a, b)`** — the `difflib`-based fraction of
  (combined) lines that differ between two snippets (0.0–1.0).
* **`is_substantial_change(a, b, threshold=0.5)`** — True when the two
  snippets differ across more than `threshold` of their lines, i.e. the fix
  commit reworked the function rather than just patching it.

`src/scripts/diagnose_safe_counterparts.py` pairs each vulnerable sample with
its post-fix twin (matched by normalized text, **no cloning needed** — both
halves are embedded inline in the manifest) and reports the trivial vs
substantial split plus a `changed_line_ratio` histogram. Run it *before* any
training run you intend to report; if the substantial fraction is large, the
post-fix "safe" class is contaminated and you should inject verified-benign
controls (Step 1 of `docs/SAFE_COUNTERPARTS.md`) before training.

### The heuristics (all conservative — they never drop a real code change)

* **Noise files** (`_is_noise_file`): paths containing `docs/`, `tests/`,
  `setup.py`, `package_data`, `conf.py`, `version`, `changelog`, `readme`,
  `requirements`, etc., or non-Python extensions (`.md`, `.json`, `.toml`, …).
  These only yield version bumps / doc edits as pairs.
* **Code signal** (`_has_code_signal` / `code_signal_line_count`): a line counts
  as signal only if it is executable code — not a comment, docstring, or a bare
  version assignment (`version = '3.7'`). A snippet with `< --min-code-lines`
  such lines is dropped (e.g. a lone `__version__` line).
* **Trivial change** (`is_trivial_change`): True when the vulnerable and fixed
  snippets are identical, or the *only* lines that differ are
  comments/docstrings/version assignments. Such (vuln, safe) pairs carry no
  learnable signal.
* **Contradiction** (`find_contradictions`): any normalized `text` that appears
  with **both** labels is a hard contradiction and is dropped.

## The near-duplicate contradiction (the *real* root cause) and its fix

The filters above catch text that is **byte-identical** (after normalization)
with opposite labels. Measurement on the CVEfixes "safe = the post-fix twin"
design showed the dominant failure is subtler and far more common:

* the post-fix (safe) function is a **near-duplicate** of its vulnerable twin —
  median token-similarity **≈ 0.94**, with **68%** of pairs >0.90 and **88%**
  >0.80 similar. The two differ by only 1–2 lines.
* A classifier told "this is vulnerable" and "this near-identical code is safe"
  receives contradictory gradients and collapses to `P=0.5` (loss ≈ ln 2,
  ROC ≈ 0.5) — exactly the observed symptom, even though *no exact
  contradiction exists* so `find_contradictions` reports zero.

The remedy (implemented across the converter, miner, builder, and trainer;
authoritative write-up in `docs/SAFE_COUNTERPARTS.md`):

1. **Vulnerable-only converter.** `convert_cvefixes.py` no longer emits the
   post-fix twin as `label=0` (default). It emits only the pre-fix
   **vulnerable** function; `fixed_code` is retained on it for the optional
   contrastive objective / audit. `--emit-fixed-safe` restores the legacy
   balanced-pair manifest for A/B comparison.
2. **Mined safe class.** The `label=0` class comes from
   `src/scripts/mine_benign_functions.py`: same-file **siblings** of the
   vulnerable function (sharpest hard negatives) plus functions from untouched
   files, at a moderate `--ratio` (default 5×) of the vulnerable count.
3. **Token-similarity guards** (new helpers in `src/data_quality.py`:
   `token_similarity`, `max_token_similarity`, `count_cross_label_near_duplicates`):
   * the **miner** drops any candidate >0.75 token-similar to a vulnerable
     function in the project (this is what excludes the post-fix twin);
   * the **builder** (`dataset_builder.py`) runs a per-project near-duplicate
     pass after chunking, dropping any safe chunk >0.75 similar to a vulnerable
     chunk;
   * the **trainer** pre-flight (`_check_training_data_quality`) now refuses to
     train when ≥5% of samples are hard *or* near-duplicate (>0.90)
     contradictions, not just exact ones.
4. **Hunk-centered positives.** The builder keeps only the chunk(s) that overlap
   the diff hunk (the changed / vulnerable lines) for a vulnerable sample, so a
   positive chunk always contains the sink (fixing the "~50% of positive chunks
   contain no sink" problem). See `docs/SLM_CHUNKING.md`.

## How to regenerate clean benchmarks

You must regenerate the manifests **and** rebuild the training splits. The old
`training_data/` from the broken run should be deleted so stale data isn't reused.

```bash
# 0. (optional but recommended) remove the noisy run's artifacts
rm -rf training_data training_output

# 1. Training corpus — CVEfixes (noise/trivial filters are ON by default)
python -c "from datasets import load_dataset; load_dataset('hitoshura25/cvefixes', split='train').save_to_disk('./cvefixes_dataset')"
python src/scripts/convert_cvefixes.py --local-dir ./cvefixes_dataset \
    --output benchmark_manifest_cvefixes.json
#    flags (all defaults are the recommended clean setting):
#      --no-noise-filter      disable docs/tests/packaging/version filtering
#      --no-trivial-filter    keep (vuln, safe) pairs that differ only in comments/version
#      --min-code-lines N     drop snippets with < N code-signal lines (default 2)

# 2. Evaluation corpus — VUDENC (held out from training)
python src/scripts/convert_vudenc.py --output benchmark_manifest_vudenc.json
#    flags:
#      --no-dedup             keep duplicate / contradictory snippets
#      --min-code-lines N     drop snippets with < N code-signal lines (default 2)

# 3. Build the training splits (defensive contradiction/low-signal filter ON)
python -m src.training.cli build-dataset --manifest benchmark_manifest_cvefixes.json
#    flags:
#      --keep-contradictory   keep identical-text (vuln, safe) pairs (NOT recommended)
#      --min-code-lines N     drop enriched samples with < N code-signal lines (default 2)

# 4. Train — the pre-flight guard will refuse if the data is still noisy
python -m src.training.cli train --epochs 20 --batch-size 8 --lr 2e-5
#    flag:
#      --allow-noisy-data     train anyway even with contradictory samples

# 5. Gate study on the held-out VUDENC corpus
python src/evaluation/run_comparative_evaluation.py \
    --manifest benchmark_manifest_vudenc.json --config config.json
```

## What "good" looks like after the fix

After regenerating with the filters on, you should see in the converter output:

```
  Skipped noise : <n>  (docs/tests/packaging/version files)
  Skipped short : <n>  (< 2 code-signal lines)
  Skipped trivial: <n>  (vuln==safe up to comments/version)
```

and during training the loss should drop **below 0.69** within the first epoch
or two, with `eval_roc_auc` climbing above ~0.6 (not pinned at 0.5). If loss
still plateaus at ~0.69, re-run step 3/4 with `--allow-noisy-data` to confirm
the pre-flight guard's contradiction count, then tighten `--min-code-lines` or
inspect the remaining pairs.

## Tuning knobs

* `--min-code-lines`: raise to 3–4 to be stricter (drops more borderline
  snippets); lower to 1 only for a quick smoke test.
* `--no-trivial-filter` / `--no-noise-filter`: turn filters **off** only when you
  specifically want to measure how much noise a baseline introduces.
* `convert_cvefixes` keeps the **balanced 1:1 pair** design (one pre-fix +
  one post-fix sample per fix commit); it only drops pairs that are unlearnable,
  so class balance is preserved for the rows that remain.
