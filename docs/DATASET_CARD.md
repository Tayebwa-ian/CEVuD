# Dataset Cards — CEVuD Benchmark Manifests

> HuggingFace-ready dataset cards for the two corpora used by CEVuD. Both are
> produced by the converters in `src/scripts/` and consumed by
> `src/training/` (classifier development) and `src/evaluation/` (gate study).

| Role | Dataset | Converter | Manifest file |
|---|---|---|---|
| Classifier train / validation / test | CVEfixes | `convert_cvefixes.py` | `benchmark_manifest_cvefixes.json` |
| Gate study (full pipeline) | VUDENC | `convert_vudenc.py` | `benchmark_manifest_vudenc.json` |

CVEfixes develops the small Stage-2 classifier end-to-end (its train/val/test
splits are project-level, preventing in-corpus leakage). VUDENC is the
independently curated corpus for the comparative gate study — the evaluation
of Semgrep + the CVEfixes-trained classifier + the gating strategies. Both
manifests share the same schema, so the harnesses are interchangeable.

---

## `cevud/cvefixes-benchmark` (training)

- **Source**: CVEfixes (`hitoshura25/cvefixes` on HuggingFace), a curated slice
  of the CVEfixes SQLite database linking CVEs to the open-source commits that
  introduced and fixed them.
- **Language**: Python only (rows are filtered by `language == "python"`).
- **Construction**: each row yields a **balanced 1:1 pair** — the pre-fix
  function (`vulnerable_code`) as `label = 1`, the post-fix function
  (`fixed_code`) as `label = 0`. No synthetic resampling is needed.
- **Ground-truth fields per sample**: `repo_url`, fix-commit `commit_id`
  (from the `hash` column), `target_commit`, `cve_id`, `cwe_id` →
  `vulnerability_type`, `cvss_score`, `diff_with_context`, `source_code`,
  `fixed_code`, `file_path`, `start_line`/`end_line` (function anchors from the
  diff).
- **Project organisation**: one `git_source` project per repository; the
  evaluation harness clones each repo and reads the real vulnerable function +
  imports + cross-file context. Function bodies are also embedded inline as a
  clone-failure fallback.
- **Size**: ~1,500+ samples across 370+ projects (after Python filtering).
- **License**: follows the upstream CVEfixes data license; verify before
  redistribution.

---

## `cevud/vudenc-benchmark` (evaluation)

- **Source**: VUDENC (Vulnerability Detection with Deep Learning on a Natural
  Codebase; Wartschinski et al., *Information and Software Technology*, 2022),
  distributed as `DetectVul/Vudenc` on HuggingFace. Real-world Python functions
  mined from vulnerability-fixing commits.
- **Granularity**: labeled at the **per-line (statement) level** across seven
  vulnerability categories: SQL injection, XSS, Command injection, XSRF, Remote
  Code Execution, Path Disclosure, Open Redirect.
- **Construction**: `raw_lines` are joined into a full function; a
  **function-level label** is `1` if *any* line is labeled vulnerable, else `0`.
  The `vulnerability_type` is the dominant per-line type. Samples are grouped
  into logical "projects" by vulnerability category (`vudenc_sql`,
  `vudenc_xss`, …) for project-level split isolation.
- **Provenance fields**: VUDENC ships **no repository or commit metadata**, so
  `repo_url`, `commit_id`, `target_commit`, `cve_id`, `cvss_score`,
  `diff_with_context`, and `fixed_code` are emitted empty (schema consistency
  only). Each project uses a `local_source` entry and `source_code` is embedded
  inline; the evaluation harness materialises each snippet to its own file for
  scanning.
- **Class balance**: because VUDENC is mined from vulnerable commits, the
  function-level set can skew toward the positive (vulnerable) class. Check
  the converter's printed balance and, if needed, restrict/pre-sample for a
  balanced evaluation.
- **Genuine safe class (recommended for the gate study)**: VUDENC has no
  `fixed_code`, so its `label=0` samples are functions the corpus's *own*
  annotators left unlabeled — not verified-benign code, and the corpus is
  near-positive-only (precision is undefined). To give the gate study a real
  safe class, merge verified-benign controls produced by
  `src/scripts/mine_benign_functions.py` via
  `convert_vudenc.py --benign-manifest benign_controls_manifest.json`
  (they are added as `local_source` projects tagged
  `sample_subtype="benign_control"`). See `docs/SAFE_COUNTERPARTS.md`.
- **License**: follows the upstream VUDENC data license; verify before
  redistribution.

---

## Shared manifest schema

Both manifests are a JSON list of project objects. Each project has exactly
one of `git_source` (CVEfixes) or `local_source` (VUDENC) plus a `samples`
list. Every sample carries the same field set so the two are interchangeable
downstream:

```jsonc
{
  "sample_id":          "vudenc::sql::000042::a1b2c3",   // or "cvefixes::<repo>::<hash>"
  "file_path":          "inline_snippet.py",              // real repo-relative path for CVEfixes
  "function_name":      "execute_query",
  "start_line":         1,
  "end_line":           42,
  "label":              1,                                // 1 = vulnerable, 0 = safe
  "vulnerability_type": "CWE-89",                         // or "sql" for VUDENC
  "source_code":        "...",                            // embedded function source
  "fixed_code":         null,                             // post-fix code (CVEfixes only)
  "repo_url":           null,                             // VUDENC: null
  "commit_id":          null,                             // VUDENC: null
  "target_commit":      null,                             // VUDENC: null
  "cve_id":             null,                             // VUDENC: null
  "cvss_score":         0.0,                              // VUDENC: 0.0
  "diff_with_context":  ""                // VUDENC: ""
  "sample_subtype":     "vulnerable"     // "vulnerable" | "fixed" | "benign_control" | "benign" | null
}
```

> **`sample_subtype` (added for the safe-counterpart methodology).** Disambiguates *why* a sample carries its label so the "safe" class is auditable end-to-end. `"vulnerable"` = pre-fix half of a CVEfixes pair; `"fixed"` = post-fix half (the "safe counterpart" we audit for bundled edits); `"benign_control"` = a function mined from unmodified code (verified-benign, `label=0`); `"benign"` = a function labeled safe by a corpus's own annotation (VUDENC all-zero lines). See `docs/SAFE_COUNTERPARTS.md` — the authoritative reference for how the safe counterpart is constructed and measured.

### Data-quality / noise filtering (important)

A training run on an unfiltered manifest plateaus at `loss ≈ ln(2) ≈ 0.693` and
`roc_auc ≈ 0.5` — the model learns nothing. The cause is (vulnerable, safe)
pairs that are **near-identical text with opposite labels** (version bumps in
`package_data.py`, docstring edits, byte-identical functions). Both converters
therefore reject noise by default; see `docs/DATA_QUALITY.md` for the full
rationale and tuning knobs.

* **CVEfixes** (`convert_cvefixes.py`): drops noise files (`docs/`, `tests/`,
  `setup.py`, `package_data`, `conf.py`, `version`, `changelog`, `readme`, …),
  drops (vuln, safe) pairs whose only difference is comments/docstrings/version
  assignments, and drops snippets with `< --min-code-lines` (default 2) lines of
  real code signal. The balanced 1:1 pair design is preserved for rows that
  remain. Flags: `--no-noise-filter`, `--no-trivial-filter`, `--min-code-lines`.
* **VUDENC** (`convert_vudenc.py`): de-duplicates snippets and drops any
  snippet whose normalized text collides with a snippet of the opposite label
  (hard contradiction). Flag: `--no-dedup`, `--min-code-lines`.

The dataset builder (`src/training/dataset_builder.py`) and the trainer
(`src/training/trainer.py`) add defensive passes on top: the builder drops
contradictory / low-signal enriched samples, and the trainer refuses to start
when ≥5% of training samples are contradictory (override with
`--allow-noisy-data`). The builder additionally **chunks** every enriched
function into uniform code windows (`src/code_chunks.py`) so the classifier is
trained on inputs that fit CodeBERT's 512-token window; each chunk inherits the
function-level label and a second contradiction pass drops identical (vuln,
safe) windows. See `docs/SLM_CHUNKING.md`.

Reproduce both manifests with (filters on by default):

```bash
# Training corpus (CVEfixes) — preferred: local save_to_disk artifact
python -c "from datasets import load_dataset; load_dataset('hitoshura25/cvefixes', split='train').save_to_disk('./cvefixes_dataset')"
python src/scripts/convert_cvefixes.py --local-dir ./cvefixes_dataset --output benchmark_manifest_cvefixes.json

# Evaluation corpus (VUDENC)
python src/scripts/convert_vudenc.py --output benchmark_manifest_vudenc.json
```

> After regenerating the manifest, **delete the old `training_data/` and
> `training_output/`** and re-run `build-dataset` so stale noisy samples from a
> previous run are not reused.
