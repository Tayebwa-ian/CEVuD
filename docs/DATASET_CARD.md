# CEVuD Datasets

> **Note**: This file is the legacy combined dataset card. Two dedicated cards
> now exist: [`DATASET_CARD_CVEFIXES.md`](DATASET_CARD_CVEFIXES.md) and
> [`DATASET_CARD_VUDENC.md`](DATASET_CARD_VUDENC.md). Publish those at
> `huggingface.co/datasets/Denash/cevud-training-dataset` and
> `huggingface.co/datasets/Denash/cevud-pipeline-dataset` respectively.

---

## Dataset 1: CEVuD Training Dataset (CVEfixes-based)

### Dataset Summary

The CEVuD training dataset is a curated, function-level Python vulnerability
corpus derived from **CVEfixes** (`hitoshura25/cvefixes`), the largest publicly
available dataset linking real CVEs to the exact commits that introduced and
fixed them. It is designed to train a lightweight local classifier (the CEVuD
Stage-2 "small model") to distinguish vulnerable Python functions from safe ones.

| Property | Value |
|----------|-------|
| **Proposed HF ID** | `Denash/cevud-training-dataset` |
| **Source dataset** | `hitoshura25/cvefixes` (HuggingFace) |
| **Language** | Python |
| **Task** | Binary sequence classification (vulnerable vs. safe) |
| **Label scheme** | `0` = safe, `1` = vulnerable |
| **Total samples** | 2,181 |
| **Projects (repos)** | 554 |
| **Vulnerable samples** | 474 (21.7%) |
| **Safe samples** | 1,707 (78.3%) |
| **Unique CWE types** | 93 |
| **Unique CVEs** | 470 |
| **Average context lines** | 33.0 |
| **License** | Same as CVEfixes (check upstream); CEVuD wrapper is MIT |

### Dataset Creation

#### Source Data

CVEfixes is a parquet-backed HuggingFace dataset (~300 MB Python-only subset)
that streams row-by-row. Each row contains:
- `vulnerable_code`: the vulnerable code snippet (pre-fix)
- `fixed_code`: the patched code snippet (post-fix)
- `repo_url`: GitHub repository URL
- `hash`: commit SHA where the fix was applied
- `cve_id`: CVE identifier
- `cwe_id`: CWE identifier
- `cvss2_base_score` / `cvss3_base_score`: CVSS scores
- `diff_with_context`: unified diff with surrounding context
- `language`: programming language

#### Curation Process

The raw CVEfixes data is converted and filtered through a multi-stage pipeline
implemented in `src/scripts/convert_cvefixes.py` and
`src/training/dataset_builder.py`. The process is designed to produce a
*learnable* two-class dataset with no label noise and no data leakage.

**Stage 1: Language and validity filtering**

1. **Language filter**: Keep only Python rows (`language` = `python` or `py`).
2. **Valid diff filter**: Require non-empty `vulnerable_code`, a valid
   `repo_url`, and a resolvable `.py` file path from the diff or `file_paths`
   column.
3. **Noise filter**: Skip rows whose changed file is a documentation, test,
   packaging, or version-only file. These produce only version bumps or doc
   edits as (vulnerable, safe) pairs, which is pure label noise.

**Stage 2: Signal and trivial-change filtering**

4. **Minimum code signal**: Require at least 2 lines of real code signal
   (comments, docstrings, and version assignments do not count). This drops
   snippets like a lone `__version__ = '3.7'`.
5. **Trivial-change filter**: Drop (vulnerable, safe) pairs that differ only in
   non-semantic ways (comments, docstrings, version assignments). Such fixes
   carry no learnable vulnerability signal.

**Stage 3: Deduplication**

6. **Exact deduplication**: Drop rows whose normalized vulnerable snippet
   duplicates an already-emitted sample (redundancy).
7. **Contradiction removal**: Drop rows whose normalized text appears with both
   labels (identical input, opposite ground truth). These are hard contradictions
   the classifier cannot learn.

**Stage 4: Safe-class construction**

CVEfixes provides only vulnerable samples (`label=1`). A genuine safe class
(`label=0`) is constructed from two sources:

1. **Benign siblings** (1,643 samples): Functions from the same file as the
   vulnerable function, but in commits the fix did not touch. These are
   extracted by `src/scripts/mine_benign_functions.py` and tagged
   `sample_subtype="benign_sibling"`.
2. **Benign controls** (64 samples): Functions from files the fix commit never
   touched, mined from verified-benign repositories. Tagged
   `sample_subtype="benign_control"`.

The **post-fix function** is explicitly *not* used as `label=0`. It is a
near-duplicate of its vulnerable twin (median token-similarity ≈ 0.94), which
would create contradictory pairs and collapse training to `P = 0.5` (loss ≈
ln(2), ROC-AUC ≈ 0.5).

**Stage 5: Enrichment**

Each sample is enriched with:
- The full enclosing function (AST-expanded via `expand_to_function`)
- Module-level imports (via `collect_module_imports`)
- Optional cross-file context (via `collect_cross_file_context`)

The enriched text format matches exactly what the production Stage-2 gate sees
at inference time, eliminating train/inference skew.

**Stage 6: Chunking**

Enriched functions are cut into uniform 64-line windows with 8-line overlap,
matching the 512-token context limit of CodeBERT. For vulnerable samples, only
chunks overlapping the diff hunk (the changed/vulnerable lines) are kept
(hunk-centering). This ensures every positive training chunk contains the actual
vulnerability sink.

**Stage 7: Near-duplicate guard**

Any safe chunk that is >0.75 token-similar to a vulnerable chunk in the same
project is dropped. This prevents lightly-edited copies of vulnerable functions
from entering the safe class.

**Stage 8: Splitting**

Samples are split by **project** (repository) using a stratified procedure that
guarantees:
- No project appears in more than one split (prevents data leakage)
- Every split contains both vulnerable and safe samples (no single-class splits)
- The split ratio is 60% train / 20% validation / 20% test

The split uses `seed=42` for full reproducibility.

### Dataset Structure

The published dataset follows the HuggingFace `json` format with the following
fields:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | str | Unique identifier (e.g. `cvefixes::salt::2874d100`) |
| `project` | str | Repository name (e.g. `salt`) |
| `text` | str | Enriched code snippet (function + imports, chunked) |
| `label` | int | `0` = safe, `1` = vulnerable |
| `vulnerability_type` | str | CWE identifier (e.g. `CWE-534`) |
| `cwe` | str | CWE identifier (same as `vulnerability_type`) |
| `file_path` | str | Relative path in the repository |
| `function_name` | str | Enclosing function name |
| `start_line` | int | Function start line (1-based) |
| `end_line` | int | Function end line (1-based) |
| `source_code_length` | int | Lines in the original source file |
| `context_length` | int | Lines in the enriched snippet |
| `sample_subtype` | str | One of `vulnerable`, `benign_sibling`, `benign_control` |
| `chunk_index` | int | Chunk index within the function (-1 for whole functions) |
| `chunk_start` | int | Chunk start line |
| `chunk_end` | int | Chunk end line |
| `hunk_text_start` | int | 1-based line offset of diff hunk start within `text` |
| `hunk_text_end` | int | 1-based line offset of diff hunk end within `text` |

### Data Splits

| Split | Samples | Vulnerable | Safe | Projects |
|-------|---------|------------|------|----------|
| Train | 1,464 | 316 | 1,148 | 330 |
| Validation | 358 | 76 | 282 | — |
| Test | 359 | 82 | 277 | — |

### Key Statistics

- **Class imbalance**: ~1 : 3.6 vulnerable/safe
- **Unique CWEs**: 93 (covers a wide range of vulnerability types)
- **Top CWEs**: CWE-79 (XSS, 30 samples), CWE-22 (Path Traversal, 27), CWE-20
  (Improper Input Validation, 22), CWE-601 (URL Redirection, 17), CWE-918
  (SSRF, 16)
- **Chunking**: All samples are uniform 64-line windows with 8-line overlap
- **Hunk-centering**: Enabled (vulnerable chunks are centered on the sink)
- **Near-duplicate threshold**: 0.75 token-similarity

### Dataset Creation Decisions and Justifications

| Decision | Justification |
|----------|---------------|
| **Python-only** | CVEfixes is multi-language; Python was selected as the initial target because it is widely used, has clear syntax for AST parsing, and has sufficient CVE coverage. |
| **Emit only vulnerable samples from CVEfixes** | The post-fix function is a near-duplicate of the vulnerable twin (median similarity ≈ 0.94). Using it as `label=0` would create contradictory pairs and collapse training to `P = 0.5`. |
| **Benign siblings + benign controls as safe class** | Benign siblings come from the same file (same coding style, same imports) but are not part of the vulnerability fix. Benign controls come from completely unrelated files, providing genuine safe examples. Both are passed through a token-similarity guard to prevent near-duplicates. |
| **Hunk-centering for vulnerable chunks** | Without hunk-centering, ~50% of positive chunks contain no vulnerability sink. The model would be trained on sink-free windows labeled as vulnerable, which is a noisy signal. |
| **Near-duplicate guard (threshold 0.75)** | Prevents lightly-edited copies of vulnerable functions from entering the safe class. The threshold was chosen to catch near-duplicates while allowing legitimate safe variations. |
| **Project-level splitting** | Prevents data leakage. If a repository appears in both train and test, the model can memorize project-specific patterns rather than learning general vulnerability indicators. |
| **Class-weighted cross-entropy** | The ~1:3.6 vulnerable/safe imbalance would cause the model to predict "safe" for everything. Inverse-frequency weighting gives the minority class a stronger gradient signal (~3.6× higher). |
| **Uniform chunking (64 lines, 8 overlap)** | Matches the 512-token context limit of CodeBERT and the inference-time chunking strategy, ensuring train/inference parity. |

### Limitations

- **Python-only**: The dataset contains only Python functions. Generalization to
  other languages requires retraining or cross-lingual transfer.
- **CWE coverage**: While 93 unique CWEs are represented, the distribution is
  imbalanced. Some rare CWE types have very few samples, which may limit the
  model's ability to generalize to those specific vulnerability patterns.
- **Temporal bias**: CVEfixes samples span many years. Older CVEs may represent
  different coding patterns and vulnerability types than modern code.
- **Safe-class construction**: The safe class is constructed from sibling and
  control functions, not from the post-fix code. This is intentional but means
  the safe class does not include "almost fixed" code that still has the
  vulnerability removed. The benign controls are verified-safe but may have
  different stylistic properties than the vulnerable code.
- **Chunking granularity**: The model scores uniform code windows, not whole
  functions. A vulnerability that spans multiple chunks may be missed if no
  single chunk contains the complete vulnerable pattern.
- **No adversarial examples**: The dataset does not include adversarially
  crafted code designed to evade detection.

### Additional Information

**Dataset curators**: CEVuD authors. CVEfixes is maintained by the open-source
community (hitoshura25 on HuggingFace).

**License**: Check the CVEfixes dataset license on HuggingFace. The CEVuD
curation wrapper is released under MIT.

**Citation**:

```
@misc{cevud2026,
  title={CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning},
  author={CEVuD Authors},
  year={2026},
  note={Dataset: Denash/cevud-training-dataset}
}
```

**Point of contact**: Open an issue on the CEVuD GitHub repository.

---

## Dataset 2: CEVuD Pipeline Dataset (VUDENC-based)

### Dataset Summary

The CEVuD pipeline dataset is a curated, function-level Python vulnerability
corpus derived from **VUDENC** (`DetectVul/Vudenc` on HuggingFace), a benchmark
of real-world Python functions with line-level vulnerability annotations. It is
used exclusively to tune the CEVuD linear gate and evaluate the full three-stage
pipeline (Semgrep + small model + gating + LLM).

| Property | Value |
|----------|-------|
| **Proposed HF ID** | `Denash/cevud-pipeline-dataset` |
| **Source dataset** | `DetectVul/Vudenc` (HuggingFace) |
| **Original paper** | Wartschinski et al., "Vulnerability Detection with Deep Learning on a Natural Codebase", *Information & Software Technology*, 2022 |
| **Language** | Python |
| **Task** | Binary sequence classification (vulnerable vs. safe) at function level |
| **Label scheme** | `0` = safe, `1` = vulnerable (derived from per-line annotations) |
| **Total samples** | 7,680 |
| **Projects (vulnerability types)** | 12 |
| **Vulnerable samples** | 1,587 (20.7%) |
| **Safe samples** | 6,093 (79.3%) |
| **License** | Same as VUDENC (check upstream) |

### Dataset Creation

#### Source Data

VUDENC is a HuggingFace dataset (`DetectVul/Vudenc`) containing Python functions
with per-line vulnerability annotations. Each row contains:
- `raw_lines`: original source code lines
- `lines`: tokenized/normalized code lines
- `label`: per-line binary vulnerability label (1/0)
- `type`: per-line vulnerability category string

The seven vulnerability categories in VUDENC are:
1. SQL injection (SQLi)
2. Cross-Site Scripting (XSS)
3. Command injection
4. Cross-Site Request Forgery (XSRF)
5. Remote Code Execution (RCE)
6. Path Disclosure
7. Open Redirect

#### Curation Process

The raw VUDENC data is converted through a pipeline implemented in
`src/scripts/convert_vudenc.py`. The process adapts VUDENC's line-level
annotations to the function-level format used by the CEVuD pipeline.

**Stage 1: Function-level labeling**

1. **Label aggregation**: A function is labeled `vulnerable` (`1`) if *any* of
   its lines is labeled vulnerable. Otherwise, it is labeled `safe` (`0`). This
   is a conservative approach: a function with even one vulnerable line is
   treated as vulnerable.

2. **Vulnerability type derivation**: The dominant vulnerability type is
   extracted from the most-common per-line `type` string. This allows the
   dataset to be grouped by vulnerability category.

**Stage 2: Source reconstruction**

3. **Source code assembly**: The `raw_lines` are joined back into a full
   function string. This provides the complete source code for each sample.

**Stage 3: Minimum signal filter**

4. **Code signal filter**: Functions with fewer than 2 lines of real code
   signal are dropped. This removes empty functions, single-line stubs, and
   other non-informative samples.

**Stage 4: Deduplication**

5. **Exact deduplication**: Drop duplicate function strings.
6. **Contradiction removal**: Drop functions whose normalized text appears with
   both labels.

**Stage 5: Project grouping**

7. **Vulnerability-type grouping**: Samples are grouped by their dominant
   vulnerability type (e.g., `vudenc_sql`, `vudenc_xss`) to form logical
   "projects." This grouping is used for project-level splitting to prevent data
   leakage.

**Stage 6: Optional benign controls**

8. **Benign control merging**: Verified-benign control samples from
   `mine_benign_functions.py` can be merged in as `local_source` projects. This
   provides a genuine safe class for the gate study, enabling precision/recall
   computation.

**Stage 7: Splitting**

9. **Project-level splitting**: No vulnerability type appears in more than one
   split. The split ratio is 60% train / 20% validation / 20% test, with
   `seed=42`.

### Dataset Structure

The published dataset follows the HuggingFace `json` format with the following
fields:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | str | Unique identifier (e.g. `vudenc::sql::000001::abc123`) |
| `project` | str | Vulnerability type group (e.g. `vudenc_sql`) |
| `source_code` | str | Full function source code |
| `label` | int | `0` = safe, `1` = vulnerable |
| `vulnerability_type` | str | Dominant vulnerability type (e.g. `sql`) |
| `file_path` | str | Synthetic path (`inline_snippet.py`) |
| `function_name` | str | Inferred function name |
| `start_line` | int | Function start line (1-based) |
| `end_line` | int | Function end line (1-based) |
| `sample_subtype` | str | `vulnerable` or `benign` |

**Note**: VUDENC ships no repository or commit metadata, so provenance fields
(`repo_url`, `commit_id`, `target_commit`, `cve_id`, `cvss_score`,
`diff_with_context`, `fixed_code`) are present for schema consistency but are
empty.

### Data Splits

| Split | Samples | Vulnerable | Safe | Projects |
|-------|---------|------------|------|----------|
| Train | 1,863 | 374 | 1,489 | 7 |
| Validation | 4,996 | 1,108 | 3,888 | 2 |
| Test | 821 | 293 | 528 | 3 |

### Per-Project Breakdown

| Project | Total | Vulnerable | Safe | Source |
|---------|-------|------------|------|--------|
| vudenc_AnnAssign | 2 | 0 | 2 | VUDENC |
| vudenc_Assert | 22 | 5 | 17 | VUDENC |
| vudenc_Assign | 2,952 | 850 | 2,102 | VUDENC |
| vudenc_AsyncFunctionDef | 18 | 1 | 17 | VUDENC |
| vudenc_AugAssign | 19 | 3 | 16 | VUDENC |
| vudenc_Condition | 789 | 101 | 688 | VUDENC |
| vudenc_Expr | 1,255 | 295 | 960 | VUDENC |
| vudenc_For | 32 | 4 | 28 | VUDENC |
| vudenc_FunctionDef | 2,044 | 176 | 1,868 | VUDENC |
| vudenc_Import | 85 | 25 | 60 | VUDENC |
| vudenc_ImportFrom | 197 | 98 | 99 | VUDENC |
| vudenc_Return | 265 | 29 | 236 | VUDENC |

### Key Statistics

- **Class distribution**: ~1 : 4 vulnerable/safe overall
- **Vulnerability type diversity**: 7 categories (SQL injection, XSS, command
  injection, XSRF, RCE, path disclosure, open redirect)
- **Most common type**: `vudenc_Assign` (2,952 samples, 850 vulnerable)
- **Least common type**: `vudenc_AnnAssign` (2 samples, 0 vulnerable)
- **Function size**: Average ~20-40 lines per function (estimated from line counts)

### Dataset Creation Decisions and Justifications

| Decision | Justification |
|----------|---------------|
| **Function-level labeling** | VUDENC provides per-line annotations. Aggregating to function level (vulnerable if *any* line is vulnerable) is conservative and appropriate for a classifier that scores entire functions. |
| **Vulnerability-type grouping as "projects"** | VUDENC has no repository metadata. Grouping by vulnerability type creates logical "projects" that can be split across train/val/test without leakage. |
| **No repository metadata** | VUDENC is a curated corpus of functions, not a commit database. `repo_url`, `commit_id`, etc. are empty for schema consistency with the CVEfixes manifest. |
| **Safe class from VUDENC's own non-vulnerable functions** | Unlike CVEfixes, VUDENC contains genuine safe functions (all lines labeled 0). These serve as the `label=0` class. Optional benign controls can be merged for additional safe samples. |
| **Project-level splitting** | Prevents data leakage across vulnerability types. A model should not see examples of one vulnerability type in train and another in test if they share characteristics. |

### Limitations

- **Function-level granularity**: The dataset labels entire functions as
  vulnerable if any line is vulnerable. This may over-label functions that
  contain both vulnerable and safe code.
- **Python-only**: All samples are Python functions.
- **Limited vulnerability diversity**: Only 7 vulnerability types, with
  `vudenc_Assign` dominating (2,952 of 7,680 samples).
- **No repository context**: VUDENC provides only isolated functions, not the
  full repository. Cross-file dependencies are not captured.
- **Class imbalance**: The safe class dominates (79.3%), which may affect
  precision/recall trade-offs.
- **Temporal bias**: VUDENC samples were collected at a specific point in time
  and may not reflect current vulnerability patterns.
- **Label noise**: Per-line annotations may contain errors. A function labeled
  safe might contain an undetected vulnerability, and vice versa.

### Intended Use

This dataset is intended for:
- **Gate tuning**: Selecting the linear gate weights $(W_1, W_2,
  T_{\text{escalation}})$ that maximize F2 on the validation split.
- **Pipeline evaluation**: Measuring the full CEVuD pipeline's recall,
  precision, TRR, and cost reduction on unseen data.
- **Benchmarking**: Comparing CEVuD against baseline strategies (Semgrep only,
  small model only, logistic regression, etc.).

It is **not** intended for:
- Training the small model (use the CVEfixes-based training dataset instead)
- Standalone vulnerability detection (the small model trained on CVEfixes is
  better suited for that)
- Training on the test split (the test split is reserved for final evaluation)

### Additional Information

**Dataset curators**: CEVuD authors. VUDENC is maintained by
LauraWartschinski (original paper) and DetectVul (HuggingFace port).

**License**: Check the VUDENC dataset license on HuggingFace.

**Citation**:

```
@misc{cevud2026,
  title={CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning},
  author={CEVuD Authors},
  year={2026},
  note={Dataset: Denash/cevud-pipeline-dataset}
}

@article{wartschinski2022vudenc,
  title={Vulnerability Detection with Deep Learning on a Natural Codebase},
  author={Wartschinski, Laura and others},
  journal={Information and Software Technology},
  year={2022}
}
```

**Point of contact**: Open an issue on the CEVuD GitHub repository.

---

## Cross-Dataset Consistency

Both datasets share the same manifest schema and are designed to be
interchangeable in the CEVuD pipeline. The key design principle is:

- **CVEfixes-based dataset**: Used for training, validating, and testing the
  small model. Never used for gate tuning or pipeline evaluation.
- **VUDENC-based dataset**: Used for gate tuning and pipeline evaluation. Never
  used for training the small model.

This separation prevents data leakage and ensures that reported metrics are
valid and generalizable.
