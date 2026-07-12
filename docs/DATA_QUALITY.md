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
