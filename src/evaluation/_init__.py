"""
CEVuD Comparative Evaluation Suite
===================================
This package is fully decoupled from the production pipeline (`triage_orchestrator.py`,
`agent.py`). It exists to answer the scientific questions raised in review:

    1. How does the staged CEVuD gate compare against simpler baselines
       (Semgrep-only, CodeSheriff-only, always-LLM, OR-gate, ablations)?
    2. Were the gate weights/threshold tuned on a held-out validation split,
       and how sensitive are results to those choices?
    3. Is a linear gate justified, or would a learned non-linear boundary
       (logistic regression) do meaningfully better?
    4. What is the empirical, quantified effect of the static/SLM override rule?

Design principle
-----------------
Running Semgrep and the SLM classifier is the only *expensive* part of evaluation.
Everything else — every baseline, every ablation, every point in a weight/threshold
grid search — is a deterministic function of two cached numbers per sample:
``(severity_weight, slm_score)``. So this package extracts those two numbers ONCE
per labeled sample (`raw_score_extractor.py`) and persists them
(`RawScoreRecord`, see `schema.py`). All comparative analysis
(`gate_strategies.py`, `grid_search.py`, `sensitivity_analysis.py`,
`linearity_check.py`) then operates purely on that cached table, with zero
additional model or subprocess calls.

Module map
----------
- schema.py                  : Shared dataclasses (BenchmarkSample, RawScoreRecord, ...).
- benchmark_manifest.py       : Loads/validates the multi-project labeled benchmark manifest.
- repo_provider.py            : Clones a repo URL to a temp dir, yields it, always cleans up.
- raw_score_extractor.py      : Runs Semgrep + SLM once per sample, caches raw scores.
- gate_strategies.py          : Registry of pure escalation-decision functions (baselines).
- metrics.py                  : Precision/recall/F-beta/confusion-matrix computations.
- dataset_splitter.py         : Leakage-safe train/validation/test split.
- grid_search.py              : Weight/threshold grid search on the validation split only.
- sensitivity_analysis.py     : Heatmap + line-plot visualizations of the grid.
- linearity_check.py          : Logistic-regression comparison to justify the linear gate.
- run_comparative_evaluation.py : CLI orchestrator tying all of the above together.
"""