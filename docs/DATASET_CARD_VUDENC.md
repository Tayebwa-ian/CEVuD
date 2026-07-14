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
  - 10K<n<100K
---

# CEVuD Pipeline Dataset (VUDENC-based)

> HuggingFace-ready dataset card for the VUDENC-based pipeline corpus.
> Published at [`Denash/cevud-pipeline-dataset`](https://huggingface.co/datasets/Denash/cevud-pipeline-dataset).

This dataset is the **held-out corpus for the CEVuD gate study**. It tunes and evaluates the full three-stage pipeline (Semgrep + small model + gating + LLM) on code the classifier never trained on. It is derived from [VUDENC](https://huggingface.co/datasets/DetectVul/Vudenc) (Wartschinski et al., *Information & Software Technology*, 2022).

The CEVuD Stage-2 classifier trained on this dataset is [`Denash/codebert-vuln-classifier`](https://huggingface.co/Denash/codebert-vuln-classifier).

---

## Dataset Summary

| Property | Value |
|----------|-------|
| **HF Dataset ID** | [`Denash/cevud-pipeline-dataset`](https://huggingface.co/datasets/Denash/cevud-pipeline-dataset) |
| **Source dataset** | [`DetectVul/Vudenc`](https://huggingface.co/datasets/DetectVul/Vudenc) |
| **Original paper** | Wartschinski et al., "Vulnerability Detection with Deep Learning on a Natural Codebase", *Information & Software Technology*, 2022 |
| **Language** | Python |
| **Task** | Binary sequence classification (vulnerable vs. safe) at function level |
| **Label scheme** | `0` = safe, `1` = vulnerable (derived from per-line annotations) |
| **Total samples** | 7,680 |
| **Projects (vulnerability types)** | 12 |
| **Vulnerable samples** | 1,587 (20.7%) |
| **Safe samples** | 6,093 (79.3%) |
| **License** | Same as VUDENC (check upstream) |

---

## Dataset Creation

### Source Data

VUDENC is a HuggingFace dataset containing Python functions with per-line vulnerability annotations. Each row contains:

- `raw_lines`: original source code lines
- `lines`: tokenized/normalized code lines
- `label`: per-line binary vulnerability label (1/0)
- `type`: per-line vulnerability category string

The seven vulnerability categories in VUDENC are:

1. SQL injection (`sql`)
2. Cross-Site Scripting (`xss`)
3. Command injection (`command`)
4. Cross-Site Request Forgery (`xsrf`)
5. Remote Code Execution (`remote_code_execution`)
6. Path Disclosure (`path_disclosure`)
7. Open Redirect (`open_redirect`)

### Curation Pipeline

The raw VUDENC data is converted through a pipeline implemented in `src/scripts/convert_vudenc.py`.

**Stage 1 — Function-level labeling**

1. A function is labeled `vulnerable` (`1`) if *any* of its lines is labeled vulnerable. Otherwise `safe` (`0`).
2. The dominant vulnerability type is extracted from the most-common per-line `type` string.

**Stage 2 — Source reconstruction**

3. `raw_lines` are joined back into a full function string.

**Stage 3 — Minimum signal filter**

4. Functions with fewer than 2 lines of real code signal are dropped.

**Stage 4 — Deduplication**

5. Exact deduplication of function strings.
6. Contradiction removal: drop identical text appearing with both labels.

**Stage 5 — Project grouping**

7. Samples are grouped by their dominant vulnerability type (e.g. `vudenc_sql`, `vudenc_xss`) to form logical "projects" for leakage-safe splitting.

**Stage 6 — Optional benign controls**

8. Verified-benign control samples can be merged in as `local_source` projects for the gate study, enabling precision/recall computation.

**Stage 7 — Splitting**

9. Project-level splitting with `seed=42`. No vulnerability type appears in more than one split.

---

## Dataset Structure

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

**Note**: VUDENC ships no repository or commit metadata, so provenance fields (`repo_url`, `commit_id`, `target_commit`, `cve_id`, `cvss_score`, `diff_with_context`, `fixed_code`) are present for schema consistency but are empty.

---

## Data Splits

| Split | Samples | Vulnerable | Safe | Projects |
|-------|---------|------------|------|----------|
| Train | 1,863 | 456 | 1,407 | 8 |
| Validation | 4,996 | 1,026 | 3,970 | 2 |
| Test | 821 | 105 | 716 | 3 |

---

## Per-Project Breakdown

| Project | Total | Vulnerable | Safe |
|---------|-------|------------|------|
| vudenc_AnnAssign' | 2 | 0 | 2 |
| vudenc_Assert' | 22 | 5 | 17 |
| vudenc_Assign' | 2,952 | 850 | 2,102 |
| vudenc_AsyncFunctionDef' | 18 | 1 | 17 |
| vudenc_AugAssign' | 19 | 3 | 16 |
| vudenc_Condition | 789 | 101 | 688 |
| vudenc_Expr' | 1,255 | 295 | 960 |
| vudenc_For | 32 | 4 | 28 |
| vudenc_FunctionDef' | 2,044 | 176 | 1,868 |
| vudenc_Import' | 85 | 25 | 60 |
| vudenc_ImportFrom' | 197 | 98 | 99 |
| vudenc_Return' | 265 | 29 | 236 |

---

## Key Statistics

- **Class distribution**: ~1 : 4 vulnerable/safe overall
- **Vulnerability type diversity**: 7 categories
- **Most common type**: `vudenc_Assign` (2,952 samples, 850 vulnerable)
- **Least common type**: `vudenc_AnnAssign'` (2 samples, 0 vulnerable)
- **Function size**: Average ~20–40 lines per function

---

## Intended Use

This dataset is intended for:

- **Gate tuning**: Selecting the linear gate weights `(W₁, W₂, T_escalation)` that maximize F2 on the validation split.
- **Pipeline evaluation**: Measuring the full CEVuD pipeline's recall, precision, TRR, and cost reduction on unseen data.
- **Benchmarking**: Comparing CEVuD against baseline strategies (Semgrep only, small model only, logistic regression, etc.).

It is **not** intended for:

- Training the small model (use the CVEfixes-based training dataset instead)
- Standalone vulnerability detection
- Training on the test split (reserved for final evaluation)

---

## Limitations

- **Function-level granularity**: A function is labeled vulnerable if any line is vulnerable. This may over-label functions containing both vulnerable and safe code.
- **Python-only**: All samples are Python functions.
- **Limited vulnerability diversity**: Only 7 vulnerability types, with `vudenc_Assign` dominating (2,952 of 7,680 samples).
- **No repository context**: VUDENC provides isolated functions, not full repositories. Cross-file dependencies are not captured.
- **Class imbalance**: The safe class dominates (79.3%), affecting precision/recall trade-offs.
- **Temporal bias**: Samples were collected at a specific point in time and may not reflect current vulnerability patterns.
- **Label noise**: Per-line annotations may contain errors.

---

## Citation

```bibtex
@misc{cevud2026,
  title={CEVuD: Cost-Effective Vulnerability Detection via Gated Static-Neural Reasoning},
  author={CEVuD Authors},
  year={2026},
  note={Dataset: Denash/cevud-pipeline-dataset; Model: Denash/codebert-vuln-classifier}
}

@article{wartschinski2022vudenc,
  title={Vulnerability Detection with Deep Learning on a Natural Codebase},
  author={Wartschinski, Laura and others},
  journal={Information and Software Technology},
  year={2022}
}
```

---

## Related Resources

| Resource | Link |
|----------|------|
| **Model** | [`Denash/codebert-vuln-classifier`](https://huggingface.co/Denash/codebert-vuln-classifier) |
| **Training dataset (CVEfixes)** | [`Denash/cevud-training-dataset`](https://huggingface.co/datasets/Denash/cevud-training-dataset) |
| **Source dataset (VUDENC)** | [`DetectVul/Vudenc`](https://huggingface.co/datasets/DetectVul/Vudenc) |
| **CEVuD GitHub** | https://github.com/Denash/CEVuD |

**Point of contact**: Open an issue on the CEVuD GitHub repository.
