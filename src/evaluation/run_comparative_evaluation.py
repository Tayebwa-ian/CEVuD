"""
run_comparative_evaluation.py
===============================
The single entry point for the comparative evaluation study. Ties together
every module in this package into one run:

    1. Extract raw (severity_weight, slm_score, label) scores once per
       sample across every project in the benchmark manifest (cloning each
       project's repo into a temp dir and deleting it immediately after —
       see repo_provider.py).
    2. Split into train/validation/test (validation and test are held out
       from each other and, ideally, cover disjoint projects).
    3. Grid-search the linear gate's (weight_static, escalation_threshold)
       on VALIDATION only; generate sensitivity plots.
    4. Fit a logistic-regression comparison gate on VALIDATION; evaluate
       linear-vs-logistic on TEST (the "is linearity justified" question).
    5. Evaluate every baseline/ablation strategy on TEST (aggregate + per-project).
    6. Write a JSON report (machine-readable) and a Markdown report (for
       direct inclusion in the paper), including an explicit "gate
       provenance" section.

Usage:
    python src/evaluation/run_comparative_evaluation.py \\
        --manifest benchmark_manifest.json \\
        --config config.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List

from schema import RawScoreRecord
from benchmark_manifest import load_manifest, manifest_summary
from raw_score_extractor import RawScoreExtractor
from dataset_splitter import split_by_project, split_stratified, apply_split
from grid_search import run_grid_search
from sensitivity_analysis import generate_all_sensitivity_plots
from linearity_check import compare_linear_vs_logistic
from gate_strategies import GATE_STRATEGIES
from metrics import compute_metrics, per_group_metrics


def _resolve_eval_dir(config: Dict[str, Any], config_path: str) -> str:
    """Mirrors evaluate_pipeline.py's timestamped run-directory convention,
    under the same config-driven `evaluations_subdir`, so both harnesses'
    outputs live side by side.
    """
    ws_root = config["paths"]["workspace_root"]
    evals_sub = config["paths"]["evaluations_subdir"]
    if not os.path.isabs(ws_root):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        ws_root = os.path.join(config_dir, ws_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(ws_root, evals_sub, f"comparative_eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def _build_split_assignment(records: List[RawScoreRecord], eval_config: Dict[str, Any]) -> Dict[str, str]:
    """Chooses split_by_project when there are enough distinct projects,
    falling back to split_stratified (with a warning) otherwise. See
    dataset_splitter.py for the generalization-vs-sample-size tradeoff
    between the two strategies.
    """
    num_projects = len({r.project for r in records})
    strategy = eval_config.get("split_strategy", "by_project")

    if strategy == "by_project" and num_projects < 3:
        print(
            f"[!] Only {num_projects} distinct project(s) in the benchmark; "
            f"split_by_project requires >= 3. Falling back to split_stratified. "
            f"NOTE: this is a weaker generalization test — add more projects "
            f"before drawing strong cross-project conclusions."
        )
        strategy = "stratified"

    kwargs = dict(
        val_frac=eval_config.get("validation_fraction", 0.2),
        test_frac=eval_config.get("test_fraction", 0.2),
        seed=eval_config.get("random_seed", 42),
    )
    if strategy == "by_project":
        return split_by_project(records, **kwargs), strategy
    return split_stratified(records, **kwargs), strategy


def _evaluate_all_strategies(
    test_records: List[RawScoreRecord],
    production_gate_params: Dict[str, Any],
    tuned_gate_params: Dict[str, Any],
    logistic_coefficients: Dict[str, float],
    beta: float,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates every registered baseline/ablation, plus three CEVuD
    variants, on the held-out test split. Returns aggregate + per-project
    metrics for each.
    """
    labels = [r.label for r in test_records]
    severities = [r.severity_weight for r in test_records]
    slm_scores = [r.slm_score for r in test_records]
    groups = [r.project for r in test_records]

    variants: Dict[str, Dict[str, Any]] = {
        "semgrep_only": {"fn": GATE_STRATEGIES["semgrep_only"].fn, "params": {}},
        "small_model": {"fn": GATE_STRATEGIES["small_model_only"].fn, "params": {}},
        "always_llm": {"fn": GATE_STRATEGIES["always_llm"].fn, "params": {}},
        "semgrep_or_small_model": {"fn": GATE_STRATEGIES["semgrep_or_small_model"].fn, "params": {}},
        "cevud_production_defaults": {
            "fn": GATE_STRATEGIES["cevud_full"].fn,
            "params": production_gate_params,
        },
        "cevud_tuned_with_override": {
            "fn": GATE_STRATEGIES["cevud_full"].fn,
            "params": {**tuned_gate_params, "override_enabled": True},
        },
        "cevud_tuned_no_override": {
            "fn": GATE_STRATEGIES["cevud_no_override"].fn,
            "params": {**tuned_gate_params, "override_enabled": False},
        },
        "logistic_regression": {
            "fn": GATE_STRATEGIES["logistic_regression"].fn,
            "params": {
                "coefficients": (
                    logistic_coefficients["bias"],
                    logistic_coefficients["weight_severity"],
                    logistic_coefficients["weight_slm"],
                ),
                "decision_threshold": 0.5,
            },
        },
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, spec in variants.items():
        predictions = [spec["fn"](sev, slm, spec["params"]) for sev, slm in zip(severities, slm_scores)]
        results[name] = {
            "params": spec["params"],
            "aggregate": compute_metrics(predictions, labels, beta=beta),
            "per_project": per_group_metrics(predictions, labels, groups, beta=beta),
        }
    return results


def _render_markdown_report(
    eval_dir: str,
    dataset_summary: Dict[str, Any],
    split_sizes: Dict[str, int],
    split_strategy: str,
    best_grid_entry: Dict[str, Any],
    selection_metric: str,
    strategy_results: Dict[str, Dict[str, Any]],
    linearity_report: Dict[str, Any],
    override_ablation_delta: Dict[str, float],
    production_gate_params: Dict[str, Any],
) -> str:
    """Builds the human-readable Markdown report, including the explicit
    gate-provenance subsection the paper's methodology section needs.
    """
    lines: List[str] = []
    lines.append("# CEVuD Comparative Evaluation Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    lines.append("## 1. Dataset\n")
    lines.append(f"- Projects: {dataset_summary['num_projects']}")
    lines.append(f"- Total labeled samples: {dataset_summary['total_samples']}")
    lines.append(f"- Split strategy: `{split_strategy}` "
                  f"(see dataset_splitter.py for the by_project vs. stratified tradeoff)")
    lines.append(f"- Split sizes: train={split_sizes['train']}, "
                  f"validation={split_sizes['validation']}, test={split_sizes['test']}\n")
    lines.append("| Project | Total | Vulnerable | Safe | Source |")
    lines.append("|---|---|---|---|---|---|---|")
    for proj, s in dataset_summary["per_project"].items():
        lines.append(f"| {proj} | {s['total']} | {s['vulnerable']} | {s['safe']} | {s['source']} |")
    lines.append("")

    lines.append("## 2. Baseline / Ablation Comparison (held-out TEST split)\n")
    lines.append(f"Selection metric: `{selection_metric}` "
                  f"(F-beta with beta weighting recall over precision — see metrics.py).\n")
    lines.append("| Strategy | Precision | Recall | F1 | " + selection_metric.upper() +
                  " | Escalation rate | TRR | Cost reduction |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in strategy_results.items():
        m = res["aggregate"]
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m[selection_metric]:.3f} | {m['escalation_rate']:.3f} | {m['token_reduction_rate']:.3f} | {m['cost_savings_ratio']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Escalation rate is the fraction of samples sent to the Stage 3 LLM — the pipeline's "
        "direct cost proxy. **TRR** (Token Reduction Rate, `token_reduction_rate`) = 1 - "
        "escalation_rate: the share of samples — and, under CEVuD's uniform per-snippet token "
        "assumption, the share of tokens — that never reach the LLM. **Cost reduction** "
        "(`cost_savings_ratio`) is the *monetary* saving vs the Always-LLM baseline and is "
        "deliberately distinct from TRR: the gated pipeline still runs a cheap local scan "
        "(Semgrep + edge SLM, ~2% of an LLM call) on every snippet, so "
        "Cost reduction = TRR × (1 − cost_ratio) and sits slightly *below* TRR. As the local "
        "scan cost → 0 (the paper's 'zero marginal-cost edge' idealisation) the two converge. "
        "See docs/METRICS.md for the full derivation and justifications. "
        "`always_llm` is the recall/cost upper bound (TRR=0, Cost reduction=0); `semgrep_only` "
        "and `small model` are equivalent to the 'skip one stage' ablations requested in "
        "review (see gate_strategies.py docstring for why those pairs collapse to the same rule).\n"
    )

    lines.append("## 3. Per-Project Breakdown (CEVuD, tuned, with override)\n")
    lines.append("| Project | Precision | Recall | F1 | " + selection_metric.upper() + " | N |")
    lines.append("|---|---|---|---|---|---|")
    per_proj = strategy_results["cevud_tuned_with_override"]["per_project"]
    for proj, m in per_proj.items():
        lines.append(
            f"| {proj} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m[selection_metric]:.3f} | {m['total_samples']} |"
        )
    lines.append(
        "\nIf performance varies sharply across projects here, that indicates the gate (or its "
        "tuned weights) does not generalize uniformly — report this spread explicitly rather than "
        "only the aggregate row above.\n"
    )

    lines.append("## 4. Gate Tuning and Sensitivity\n")
    lines.append(
        f"The linear gate's `weight_static` and `escalation_threshold` were selected by grid "
        f"search on the VALIDATION split only (never the test split), maximizing `{selection_metric}`. "
        f"Selected configuration: `weight_static={best_grid_entry['weight_static']}`, "
        f"`weight_slm={best_grid_entry['weight_slm']}`, "
        f"`escalation_threshold={best_grid_entry['escalation_threshold']}`, achieving "
        f"`{selection_metric}={best_grid_entry['metrics'][selection_metric]:.4f}` on validation.\n"
    )
    lines.append("Sensitivity plots (saved alongside this report):")
    lines.append("- `gate_sensitivity_heatmap.png` — full grid over (weight_static, threshold)")
    lines.append("- `gate_threshold_sensitivity.png` — metric vs. threshold at the selected weight")
    lines.append("- `gate_weight_sensitivity.png` — metric vs. weight at the selected threshold\n")

    lines.append("## 5. Linearity Justification\n")
    coeffs = linearity_report["logistic_coefficients"]
    lines.append(
        f"A logistic regression gate (bias={coeffs['bias']:.4f}, "
        f"weight_severity={coeffs['weight_severity']:.4f}, weight_slm={coeffs['weight_slm']:.4f}) "
        f"was fit on the validation split and compared against the tuned linear gate on the test split.\n"
    )
    lines.append(linearity_report["verdict"] + "\n")

    lines.append("## 6. Override Rule: Provenance and Ablation\n")
    lines.append(
        "**Provenance.** The static/SLM override "
        f"(`static_override_value={production_gate_params.get('static_override_value', 1.0)}`, "
        f"`slm_override_threshold={production_gate_params.get('slm_override_threshold', 0.90)}`) "
        "originated as an engineering safety rule, not a value tuned on data: the concern was that "
        "a catastrophic static finding (Semgrep severity ERROR) or an extremely confident SLM "
        "prediction (>90%) could, in principle, still fall below the linear threshold if the other "
        "signal were low, silently suppressing escalation of a likely-real vulnerability. The "
        "override forces escalation in exactly those two cases, independent of the linear formula.\n"
    )
    lines.append(
        "**Ablation (empirical).** Grid search above tuned the linear gate WITHOUT the override "
        "(`override_enabled=False` for every grid point — see grid_search.py), so weight/threshold "
        "selection cannot be confounded with the override's effect. The override was then re-enabled "
        "on top of the already-tuned gate and measured on the test split:\n"
    )
    lines.append("| Metric | Without override | With override | Delta |")
    lines.append("|---|---|---|---|")
    for k, v in override_ablation_delta.items():
        lines.append(f"| {k} | {v['without']:.4f} | {v['with']:.4f} | {v['delta']:+.4f} |")

    report = "\n".join(lines)
    report_path = os.path.join(eval_dir, "comparative_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[+] Wrote Markdown report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="CEVuD comparative baseline / gate evaluation suite")
    parser.add_argument("--manifest", required=True, help="Path to the benchmark manifest JSON (see benchmark_manifest.py)")
    parser.add_argument("--config", default="config.json", help="Path to the master config.json")
    parser.add_argument("--cache", default=None, help="Path to cache/reuse raw_scores.json (skips re-extraction if present)")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore any existing raw score cache")
    parser.add_argument("--inline", action="store_true",
                        help="Score git_source projects from their embedded source_code/fixed_code "
                             "instead of cloning the real repo. Skips every git clone — the "
                             "fastest mode and the only one that runs offline / air-gapped.")
    parser.add_argument("--weight-step", type=float, default=0.05,
                        help="Grid-search step for weight_static in [0,1] (default 0.05 => 21x21 grid). "
                             "Use 0.1 for a coarse 11x11 grid during quick iteration.")
    parser.add_argument("--threshold-step", type=float, default=0.05,
                        help="Grid-search step for escalation_threshold in [0,1] (default 0.05). "
                             "Use 0.1 to coarsen the threshold axis.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    eval_config = config.get("evaluation", {})
    beta = eval_config.get("fbeta", 2.0)
    selection_metric = f"f{beta:g}"

    eval_dir = _resolve_eval_dir(config, args.config)
    cache_path = args.cache or os.path.join(eval_dir, "raw_scores_cache.json")

    # --- Step 1: raw score extraction (the only expensive step) ---
    projects = load_manifest(args.manifest)
    dataset_summary = manifest_summary(projects)
    print(f"[*] Benchmark manifest: {dataset_summary['num_projects']} projects, "
          f"{dataset_summary['total_samples']} samples")

    extractor = RawScoreExtractor(config_path=args.config)
    if args.cache and os.path.exists(cache_path) and not args.force_recompute:
        print(
            "[!] USING CACHE: Semgrep AND the SLM are SKIPPED — the "
            "reported severities/slm_scores are reused from "
            f"{cache_path}. Do NOT use --cache for the *reported* gate-study "
            "numbers; it is only a fast re-grid/plot path. Run without "
            "--cache (clone + real Semgrep) for the paper's metrics."
        )
    if args.inline:
        print(
            "[!] INLINE MODE: repos are NOT cloned; Semgrep still runs, but "
            "only over the *isolated* materialised snippets (no surrounding "
            "repo context). Use the default (git clone) path for the reported "
            "gate study so Semgrep scans the real project root."
        )
    records = extractor.extract(
        args.manifest, cache_path=cache_path,
        force_recompute=args.force_recompute, force_inline=args.inline,
    )

    # --- Step 2: leakage-safe split ---
    split_assignment, split_strategy = _build_split_assignment(records, eval_config)
    splits = apply_split(records, split_assignment)
    split_sizes = {k: len(v) for k, v in splits.items()}
    print(f"[*] Split sizes: {split_sizes}")

    # --- Step 3: grid search + sensitivity plots (validation only) ---
    # Coarsen the grid via the --weight-step / --threshold-step flags when
    # iterating quickly; the default 0.05 gives a 21x21 grid. Both axes
    # are cheap to sweep because every cell is a pure function over the
    # *cached* (severity_weight, slm_score) arrays — no model/Semgrep
    # calls happen inside the grid search itself.
    def _grid_axis(step: float) -> List[float]:
        step = max(1e-6, min(1.0, step))
        return [round(i * step, 6) for i in range(int(round(1.0 / step)) + 1)]

    weight_grid = _grid_axis(args.weight_step)
    threshold_grid = _grid_axis(args.threshold_step)
    grid_results, best = run_grid_search(
        splits["validation"], beta=beta, selection_metric=selection_metric,
        weight_grid=weight_grid, threshold_grid=threshold_grid,
    )
    generate_all_sensitivity_plots(grid_results, best, selection_metric, eval_dir)

    tuned_gate_params = {
        "weight_static": best["weight_static"],
        "weight_slm": best["weight_slm"],
        "escalation_threshold": best["escalation_threshold"],
        "static_override_value": 1.0,  # ERROR severity, matches production semantics
        "slm_override_threshold": config["gate_parameters"].get("slm_override_threshold", 0.90),
    }

    # --- Step 4: linearity check (fit on validation, compare on test) ---
    linearity_report = compare_linear_vs_logistic(
        splits["validation"], splits["test"], tuned_gate_params, beta=beta
    )

    # --- Step 5: evaluate every baseline/ablation on test ---
    production_gate_params = {
        "weight_static": config["gate_parameters"]["weight_static"],
        "weight_slm": config["gate_parameters"]["weight_slm"],
        "escalation_threshold": config["gate_parameters"]["escalation_threshold"],
        "override_enabled": True,
        "static_override_value": 1.0,
        "slm_override_threshold": config["gate_parameters"].get("slm_override_threshold", 0.90),
    }
    strategy_results = _evaluate_all_strategies(
        splits["test"],
        production_gate_params,
        tuned_gate_params,
        linearity_report["logistic_coefficients"],
        beta,
    )

    with_override = strategy_results["cevud_tuned_with_override"]["aggregate"]
    without_override = strategy_results["cevud_tuned_no_override"]["aggregate"]
    override_ablation_delta = {
        metric: {
            "without": without_override[metric],
            "with": with_override[metric],
            "delta": with_override[metric] - without_override[metric],
        }
        for metric in ("precision", "recall", selection_metric)
    }

    # --- Step 6: write JSON + Markdown reports ---
    json_report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_summary": dataset_summary,
        "split_strategy": split_strategy,
        "split_sizes": split_sizes,
        "grid_search": {"best": best, "full_grid_size": len(grid_results)},
        "linearity_check": linearity_report,
        "override_ablation": override_ablation_delta,
        "strategy_results": strategy_results,
    }
    json_path = os.path.join(eval_dir, "comparative_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"[+] Wrote JSON report: {json_path}")

    _render_markdown_report(
        eval_dir, dataset_summary, split_sizes, split_strategy, best, selection_metric,
        strategy_results, linearity_report, override_ablation_delta, production_gate_params,
    )

    print(f"\n[+] Comparative evaluation complete. All artifacts saved to: {eval_dir}")


if __name__ == "__main__":
    main()
