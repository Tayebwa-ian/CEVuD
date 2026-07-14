---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - code
  - vulnerability
  - security
  - python
size_categories:
  - 1K<n<10K
---

# CEVuD Training Dataset (CVEfixes-based)

> HuggingFace-ready dataset card for the CVEfixes-based training corpus.
> Published at [`Denash/cevud-training-dataset`](https://huggingface.co/datasets/Denash/cevud-training-dataset).

This dataset trains the **CEVuD Stage-2 local classifier** ([`Denash/codebert-vuln-classifier`](https://huggingface.co/Denash/codebert-vuln-classifier)). It is a curated, function-level Python vulnerability corpus derived from [CVEfixes](https://huggingface.co/datasets/hitoshura25/cvefixes), the largest publicly available dataset linking real CVEs to the exact commits that fixed them.

---

## Dataset Summary

| Property | Value |
|----------|-------|
| **HF Dataset ID** | [`Denash/cevud-training-dataset`](https://huggingface.co/datasets/Denash/cevud-training-dataset) |
| **Source dataset** | [`hitoshura25/cvefixes`](https://huggingface.co/datasets/hitoshura25/cvefixes) |
| **Language** | Python |
| **Task** | Binary sequence classification (vulnerable vs. safe) |
| **Label scheme** | `0` = safe, `1` = vulnerable |
| **Total samples** | 2,181 |
| **Projects (repos)** | 575 |
| **Vulnerable samples** | 474 (21.7%) |
| **Safe samples** | 1,707 (78.3%) |
| **Safe class breakdown** | 1,643 benign_sibling + 64 benign_control |
| **Unique CWE types** | 93 |
| **Unique CVEs** | 470 |
| **Chunk size** | 64 lines with 8-line overlap |
| **Hunk-centering** | Enabled |
| **Near-duplicate threshold** | 0.75 token-similarity |
| **License** | MIT (CEVuD wrapper); check upstream CVEfixes for source license |

---

## Dataset Creation

### Source Data

CVEfixes is a parquet-backed HuggingFace dataset that streams row-by-row. Each row contains:

- `vulnerable_code`: the vulnerable code snippet (pre-fix)
- `fixed_code`: the patched code snippet (post-fix)
- `repo_url`: GitHub repository URL
- `hash`: commit SHA where the fix was applied
- `cve_id`: CVE identifier
- `cwe_id`: CWE identifier
- `cvss2_base_score` / `cvss3_base_score`: CVSS scores
- `diff_with_context`: unified diff with surrounding context
- `language`: programming language

### Curation Pipeline

The raw CVEfixes data is converted and filtered through a multi-stage pipeline implemented in `src/scripts/convert_cvefixes.py` and `src/training/dataset_builder.py`.

**Stage 1 — Language and validity filtering**
1. Keep only Python rows.
2. Require non-empty vulnerable code, valid repo URL, and resolvable `.py` file path.
3. Skip documentation, test, packaging, and version-only files.

**Stage 2 — Signal and trivial-change filtering**
4. Require at least 2 lines of real code signal.
5. Drop (vulnerable, safe) pairs that differ only in non-semantic ways.

**Stage 3 — Deduplication**
6. Exact deduplication of normalized vulnerable snippets.
7. Contradiction removal: drop identical text appearing with both labels.

**Stage 4 — Safe-class construction**

CVEfixes provides only vulnerable samples. A genuine safe class is constructed from two sources:

- **Benign siblings** (1,643 samples): Functions from the same file, in commits the fix did not touch.
- **Benign controls** (64 samples): Functions from files the fix commit never touched, mined from verified-benign repositories.

The **post-fix function is explicitly not used as `label=0`** — it is a near-duplicate of its vulnerable twin (median token-similarity ≈ 0.94) and would collapse training to `P = 0.5`.

**Stage 5 — Enrichment**

Each sample is enriched with the full enclosing function (AST-expanded) and module-level imports, matching the inference-time context exactly.

**Stage 6 — Chunking**

Functions are cut into uniform 64-line windows with 8-line overlap. For vulnerable samples, only chunks overlapping the diff hunk are kept (hunk-centering).

**Stage 7 — Quality guards**

Any safe chunk >0.75 token-similar to a vulnerable chunk in the same project is dropped. Hard contradictions are also removed.

**Stage 8 — Splitting**

Project-level 60/20/20 split with `seed=42`. No project appears in more than one split.

---

## Dataset Structure

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
| `sample_subtype` | str | `vulnerable`, `benign_sibling`, or `benign_control` |
| `chunk_index` | int | Chunk index within the function |
| `chunk_start` | int | Chunk start line |
| `chunk_end` | int | Chunk end line |
| `hunk_text_start` | int | Diff hunk start offset within `text` |
| `hunk_text_end` | int | Diff hunk end offset within `text` |

---

## Data Splits

| Split | Samples | Vulnerable | Safe | Projects |
|-------|---------|------------|------|----------|
| Train | 1,464 | 316 | 1,148 | 330 |
| Validation | 358 | 76 | 282 | — |
| Test | 359 | 82 | 277 | — |

---

## Key Statistics

- **Class imbalance**: ~1 : 3.6 vulnerable/safe
- **Unique CWEs**: 93
- **Top CWEs**: CWE-79 (30), CWE-22 (27), CWE-20 (22), CWE-601 (17), CWE-918 (16)
- **Chunking**: Uniform 64-line windows with 8-line overlap
- **Hunk-centering**: Enabled
- **Near-duplicate guard**: 0.75 token-similarity threshold

---

## Intended Use

This dataset is intended for:

- **Training the CEVuD Stage-2 classifier**: The primary use case. The classifier is fine-tuned on this dataset to produce `P(vulnerable)` scores for code chunks.
- **Validation and testing**: Held-out project splits provide unbiased estimates of classifier performance.
- **Research**: Studying class imbalance, safe-class construction, and chunking strategies for vulnerability detection.

It is **not** intended for:

- Training the gate weights (use the VUDENC-based pipeline dataset instead)
- Standalone vulnerability detection without the gated pipeline
- Cross-language transfer without retraining

---

## Limitations

- **Python-only**: Contains only Python functions.
- **CWE imbalance**: 93 unique CWEs but distribution is skewed; rare types have few samples.
- **Temporal bias**: Spans many years of CVEs; older patterns may not reflect modern code.
- **Safe-class construction**: The safe class does not include post-fix code. Benign siblings and controls are used instead to avoid near-duplicate contradictions.
- **Chunk-level labels**: Vulnerabilities spanning multiple chunks may be missed if no single chunk contains the complete pattern.

---

## Citation

```bibtex
@misc{cevud2026,
  title={CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning},
  author={CEVuD Authors},
  year={2026},
  note={Dataset: Denash/cevud-training-dataset; Model: Denash/codebert-vuln-classifier}
}
```

---

## Related Resources

| Resource | Link |
|----------|------|
| **Model** | [`Denash/codebert-vuln-classifier`](https://huggingface.co/Denash/codebert-vuln-classifier) |
| **Pipeline dataset (VUDENC)** | [`Denash/cevud-pipeline-dataset`](https://huggingface.co/datasets/Denash/cevud-pipeline-dataset) |
| **Source dataset (CVEfixes)** | [`hitoshura25/cvefixes`](https://huggingface.co/datasets/hitoshura25/cvefixes) |
| **CEVuD GitHub** | https://github.com/Denash/CEVuD |

**Point of contact**: Open an issue on the CEVuD GitHub repository.
