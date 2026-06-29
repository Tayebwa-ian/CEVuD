"""
Pipeline Evaluation Harness
===========================
Computes comprehensive detection effectiveness and cost-efficiency metrics
by running the full Stage 1 + Stage 2 pipeline against a curated ground-truth
dataset (gold_standard.json).

Persisted Artifacts (per evaluation run):
    - summary.json            : Quantitative metrics (Recall, Precision, Accuracy, F1, TRR, CSR)
    - detailed_findings.json  : Per-case risk scores, severities, and escalation decisions
    - semgrep_eval_results.json : Raw Stage 1 Semgrep output
    - confusion_matrix.png    : Visual heatmap of TP/FP/TN/FN
    - risk_distribution.png   : Histogram of risk scores with threshold overlay
    - category_performance.png: Per-vulnerability-category detection bar chart
    - input_ledger_snapshot.json : Frozen copy of the test data used for the run
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/headless environments
import matplotlib.pyplot as plt

from triage_orchestrator import TriageOrchestrator
from vector_store import LocalVectorStore


# ---------------------------------------------------------------------------
# Default path to the gold-standard ground-truth dataset.
# This file lives in tests/data/ so that it is co-located with the test suite
# and can be referenced by both pytest and this evaluator.
# ---------------------------------------------------------------------------
_DEFAULT_GOLD_STANDARD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "data", "gold_standard.json",
)


class PipelineEvaluator:
    """
    End-to-end benchmark harness for the CEVuD security pipeline.

    Workflow:
        1. Materialise gold-standard code snippets to temporary files on disk.
        2. Execute a live Semgrep scan (Stage 1) against those files.
        3. For every test case, run the CodeBERT SLM inference (Stage 2).
        4. Apply the mathematical gating formula to compute a risk score.
        5. Compare the pipeline's escalation decision against the ground-truth
           label to populate a confusion matrix.
        6. Derive Recall, Precision, Accuracy, F1, Token Reduction Rate (TRR),
           and Cost Savings Ratio (CSR).
        7. Persist all numerical results and visual charts to a versioned
           directory under ``workspace_storage/evaluation_runs/``.
    """

    def __init__(self, config_path: str):
        """
        Initialise the evaluator with pipeline weights and a unique run directory.

        Args:
            config_path: Absolute or relative path to the master ``config.json``.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Unpack gating parameters for quick access during scoring
        self.w1 = self.config["gate_parameters"]["weight_static"]
        self.w2 = self.config["gate_parameters"]["weight_slm"]
        self.threshold = self.config["gate_parameters"]["escalation_threshold"]

        # Instantiate the real orchestrator so we can call slm_inference()
        self.orchestrator = TriageOrchestrator(config_path)
        self.vector_store = LocalVectorStore(config_path)

        # Create a timestamped evaluation directory for this run.
        # workspace_root and evaluations_subdir may both be absolute (test config)
        # or relative (production config) - handle both cases.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_id = f"eval_{timestamp}"
        ws_root = self.config["paths"]["workspace_root"]
        evals_sub = self.config["paths"]["evaluations_subdir"]

        if os.path.isabs(evals_sub):
            # evaluations_subdir is already absolute (test config)
            self.eval_dir = os.path.join(evals_sub, self.eval_id)
        elif os.path.isabs(ws_root):
            # workspace_root is absolute, evaluations_subdir is relative to it
            self.eval_dir = os.path.join(ws_root, evals_sub, self.eval_id)
        else:
            # Both relative — join with cwd (production default)
            self.eval_dir = os.path.join(ws_root, evals_sub, self.eval_id)

        os.makedirs(self.eval_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _prepare_benchmark_files(self, test_cases: list) -> str:
        """
        Write each gold-standard snippet to its own ``.py`` file so that
        Semgrep can perform a real filesystem scan.

        IMPORTANT: Files are written to a ``tempfile.mkdtemp()`` directory
        **outside the project tree** (e.g. ``/tmp/cevud_bench_XXXX``).
        This guarantees that Semgrep's ``.gitignore`` / ``.semgrepignore``
        integration — which would otherwise silently skip files nested under
        ``workspace_storage/`` — cannot block the scan.

        Args:
            test_cases: List of dicts from ``gold_standard.json``.

        Returns:
            Absolute path to the temporary directory containing the files.
        """
        import tempfile
        # Write outside the project tree so .gitignore/.semgrepignore never applies
        temp_src_dir = tempfile.mkdtemp(prefix="cevud_bench_")

        for i, case in enumerate(test_cases):
            safe_fn = case["function_name"].replace(" ", "_")
            file_name = f"case_{i}_{safe_fn}.py"
            with open(os.path.join(temp_src_dir, file_name), "w") as f:
                f.write(case["source_code"])

        print(f"[*] Benchmark files materialised to: {temp_src_dir}")
        return temp_src_dir

    def _run_live_semgrep(self, target_dir: str) -> dict:
        """
        Execute the Semgrep CLI against *target_dir* using the same rule-set
        as the production CI pipeline.

        Args:
            target_dir: Directory containing the materialised benchmark files.

        Returns:
            Parsed JSON output from Semgrep (or an empty results dict on failure).
        """
        output_path = os.path.join(self.eval_dir, "semgrep_eval_results.json")

        # Resolve the custom taint-rules YAML relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_rules_path = os.path.join(
            base_dir, "semgrep_rules", "custom_appsec_rules.yaml"
        )

        # Find all python files to scan. Passing files directly to Semgrep
        # bypasses its default ignore patterns (like excluding directories named "tests")
        target_files = [
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.endswith(".py")
        ]

        cmd = [
            "semgrep",
            "--config", "p/python",
            "--config", custom_rules_path,
            "--no-git-ignore",
            "--json",
            "--output", output_path,
        ] + target_files

        print(f"[*] Executing Stage 1 live scan on {len(target_files)} benchmark files...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[-] Semgrep execution failed (Return Code: {result.returncode})")
            if result.stderr:
                print(f"[-] Semgrep Error Output:\n{result.stderr}")

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                return json.load(f)
        return {"results": []}

    # ------------------------------------------------------------------
    # Core evaluation loop
    # ------------------------------------------------------------------

    def run_evaluation(self, ledger_path: str):
        """
        Run every gold-standard test case through the real pipeline and
        compare the outcome against the ground-truth label.

        This method:
            * Materialises snippets → runs Semgrep → runs CodeBERT inference.
            * Computes a full confusion matrix and derived metrics.
            * Writes ``summary.json``, ``detailed_findings.json``, and charts
              into the versioned evaluation directory.

        Args:
            ledger_path: Path to the JSON file containing the gold-standard
                         test cases (``tests/data/gold_standard.json``).
        """
        # Freeze a copy of the input data alongside the results for auditability
        shutil.copy2(
            ledger_path,
            os.path.join(self.eval_dir, "input_ledger_snapshot.json"),
        )

        with open(ledger_path, "r") as f:
            test_cases: List[Dict[str, Any]] = json.load(f)

        bench_dir = self._prepare_benchmark_files(test_cases)

        try:
            semgrep_results = self._run_live_semgrep(bench_dir)

            # Build a lookup: benchmark filename → highest Semgrep severity string.
            # A file may match multiple rules; we keep the worst (highest weight) result.
            _sev_rank = {"ERROR": 3, "WARNING": 2, "INFO": 1, "NONE": 0}
            findings_map: Dict[str, str] = {}
            for res in semgrep_results.get("results", []):
                fname = os.path.basename(res["path"])
                sev = res["extra"]["severity"]
                # Only promote, never demote
                if _sev_rank.get(sev, 0) > _sev_rank.get(findings_map.get(fname, "NONE"), 0):
                    findings_map[fname] = sev
            print(f"[*] Semgrep matched {len(findings_map)} unique benchmark files.")

            # --- Confusion matrix accumulators ---
            tp = fp = fn = tn = 0
            escalations = 0
            all_risk_scores: List[float] = []
            detailed_findings: List[Dict[str, Any]] = []

            print(
                f"\n=== Running Pipeline Evaluation Loop "
                f"({len(test_cases)} cases) ==="
            )

            for i, case in enumerate(test_cases):
                # 1. Resolve which Semgrep severity was found for this case
                safe_fn = case["function_name"].replace(" ", "_")
                target_fname = f"case_{i}_{safe_fn}.py"
                sev_str = findings_map.get(target_fname, "NONE")
                s_sev = self.config["semgrep_severity_map"].get(sev_str, 0.0)

                # --------------------------------------------------------------
                # Context-Aware Window Extension
                # Trace upstream callers and downstream sinks to build a multi-file window
                # --------------------------------------------------------------
                context_blocks = self.vector_store.get_explicit_flow_context(case["function_name"])
                
                # Pre-populate the input window with the target modification snippet
                unified_source_window = case["source_code"]
                for block in context_blocks:
                    unified_source_window += f"\n# Context Flow Lineage from File: {block['file_path']}\n"
                    unified_source_window += block['source_code']

                # 2. Run real CodeBERT inference on the expanded, unified source window
                p_slm = self.orchestrator.slm_inference(unified_source_window)

                # 3. Apply the gating formula: R = (W1 × S_sev) + (W2 × P_slm)
                risk_score = (self.w1 * s_sev) + (self.w2 * p_slm)
                all_risk_scores.append(risk_score)
                escalated = risk_score >= self.threshold

                if escalated:
                    escalations += 1

                # 4. Compare against ground-truth to populate confusion matrix
                is_vuln = case["is_vulnerable"] == 1
                if is_vuln and escalated:
                    tp += 1
                elif is_vuln and not escalated:
                    fn += 1
                elif not is_vuln and escalated:
                    fp += 1
                else:
                    tn += 1

                # 5. Record per-case detail for persistent storage
                detailed_findings.append({
                    "index": i,
                    "function_name": case["function_name"],
                    "file_path": case["file_path"],
                    "is_vulnerable": case["is_vulnerable"],
                    "semgrep_severity": sev_str,
                    "static_weight": round(s_sev, 4),
                    "slm_probability": round(p_slm, 4),
                    "risk_score": round(risk_score, 4),
                    "escalated": escalated,
                })

                print(
                    f"  [{i:02d}] {case['function_name']:25s} | "
                    f"Risk: {risk_score:.3f} | "
                    f"Escalated: {str(escalated):5s} | "
                    f"Ground Truth: {'VULN' if is_vuln else 'SAFE'}"
                )

            # ----------------------------------------------------------
            # Metric computation
            # ----------------------------------------------------------
            total = len(test_cases)

            # Detection effectiveness
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            accuracy = (tp + tn) / total if total > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Cost-efficiency
            token_reduction_rate = (1 - (escalations / total)) * 100 if total > 0 else 0.0
            # CSR estimates savings vs. sending every case to the frontier LLM
            cost_savings_ratio = (
                (total - escalations) / total * 100 if total > 0 else 0.0
            )

            # ----------------------------------------------------------
            # Persist quantitative summary
            # ----------------------------------------------------------
            summary = {
                "eval_id": self.eval_id,
                "timestamp": datetime.now().isoformat(),
                "total_cases": total,
                "escalations": escalations,
                "confusion_matrix": {
                    "tp": tp, "fp": fp,
                    "fn": fn, "tn": tn,
                },
                "metrics": {
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "accuracy": round(accuracy, 4),
                    "f1_score": round(f1_score, 4),
                    "specificity": round(specificity, 4),
                    "token_reduction_rate": round(token_reduction_rate, 2),
                    "cost_savings_ratio": round(cost_savings_ratio, 2),
                },
            }

            summary_path = os.path.join(self.eval_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Persist the per-case detailed findings
            details_path = os.path.join(self.eval_dir, "detailed_findings.json")
            with open(details_path, "w") as f:
                json.dump(detailed_findings, f, indent=2)

            # ----------------------------------------------------------
            # Generate visual chart artifacts
            # ----------------------------------------------------------
            self._generate_graphs(tp, fp, fn, tn, all_risk_scores, detailed_findings)

            # ----------------------------------------------------------
            # Print the dashboard
            # ----------------------------------------------------------
            print("\n" + "=" * 58)
            print("        PIPELINE EVALUATION DASHBOARD")
            print("=" * 58)
            print(f"  Run ID                  : {self.eval_id}")
            print(f"  Total Cases Tested      : {total}")
            print(f"  Total Escalations       : {escalations}")
            print(f"  ─── Detection Metrics ───")
            print(f"  Recall (Sensitivity)    : {recall * 100:.1f}%")
            print(f"  Precision               : {precision * 100:.1f}%")
            print(f"  Accuracy                : {accuracy * 100:.1f}%")
            print(f"  F1 Score                : {f1_score:.4f}")
            print(f"  Specificity             : {specificity * 100:.1f}%")
            print(f"  ─── Cost Metrics ────────")
            print(f"  Token Reduction Rate    : {token_reduction_rate:.1f}%")
            print(f"  Cost Savings Ratio      : {cost_savings_ratio:.1f}%")
            print("=" * 58)
            print(f"  [+] Artifacts saved to: {self.eval_dir}")

            # Validation guards for research targets
            if recall >= 0.95 and token_reduction_rate >= 50.0:
                print(
                    "  [+] Target Met: Pipeline is cost-efficient "
                    "and structurally secure."
                )
            else:
                print(
                    "  [⚠️] Optimisation Required: Adjust weights in "
                    "config.json to balance safety and cost."
                )

        finally:
            # Clean up the temporary benchmark source files
            if os.path.exists(bench_dir):
                print(f"[*] Cleaning up temporary benchmarks in {bench_dir}...")
                shutil.rmtree(bench_dir)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _generate_graphs(
        self,
        tp: int, fp: int, fn: int, tn: int,
        scores: List[float],
        detailed_findings: List[Dict[str, Any]],
    ):
        """
        Produce persistent visual artifacts for the evaluation run.

        Generated files:
            * ``risk_distribution.png``   – histogram of all risk scores
            * ``confusion_matrix.png``    – annotated heatmap
            * ``category_performance.png``– per-file-path detection accuracy

        Args:
            tp, fp, fn, tn: Confusion matrix counts.
            scores:         All computed risk scores from the evaluation loop.
            detailed_findings: Per-case records including file_path for grouping.
        """
        # ---- 1. Risk Score Distribution Histogram ----
        plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=15, color="skyblue", edgecolor="black", alpha=0.85)
        plt.axvline(
            self.threshold,
            color="red", linestyle="dashed", linewidth=2,
            label=f"Threshold ({self.threshold})",
        )
        plt.title("Risk Score Distribution across Test Suite")
        plt.xlabel("Calculated Risk (R)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "risk_distribution.png"), dpi=150)
        plt.close()

        # ---- 2. Confusion Matrix Heatmap ----
        cm = [[tn, fp], [fn, tp]]
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Annotate each cell with the count
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row][col]),
                        ha="center", va="center", fontsize=18, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Safe", "Predicted Vuln"])
        ax.set_yticklabels(["Actual Safe", "Actual Vuln"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # ---- 3. Per-Category Detection Bar Chart ----
        # Group findings by the source file path (e.g. app/database.py)
        # and compute per-category accuracy (correct decisions / total).
        category_stats: Dict[str, Dict[str, int]] = {}
        for finding in detailed_findings:
            # Use the file_path basename without extension as the category
            cat = os.path.splitext(os.path.basename(finding["file_path"]))[0]
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            category_stats[cat]["total"] += 1

            is_vuln = finding["is_vulnerable"] == 1
            correctly_classified = (
                (is_vuln and finding["escalated"])
                or (not is_vuln and not finding["escalated"])
            )
            if correctly_classified:
                category_stats[cat]["correct"] += 1

        categories = sorted(category_stats.keys())
        accuracies = [
            category_stats[c]["correct"] / category_stats[c]["total"] * 100
            for c in categories
        ]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(categories, accuracies, color="mediumseagreen", edgecolor="black")
        # Annotate each bar with its value
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0f}%", ha="center", va="bottom", fontsize=9,
            )
        plt.ylim(0, 115)
        plt.title("Per-Category Detection Accuracy")
        plt.xlabel("Vulnerability Category")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "category_performance.png"), dpi=150)
        plt.close()


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CEVuD Pipeline Evaluator — benchmark the detection pipeline "
                    "against the gold-standard ground-truth dataset."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to the master config.json (default: config.json)",
    )
    parser.add_argument(
        "--ledger", default=None,
        help="Path to the gold-standard JSON ledger. "
             "Defaults to tests/data/gold_standard.json.",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Run the dataset_ingest benchmark seeder before evaluation.",
    )

    args = parser.parse_args()

    # Resolve the ledger path
    ledger = args.ledger or _DEFAULT_GOLD_STANDARD
    if not os.path.exists(ledger):
        print(f"[-] Ledger file not found: {ledger}")
        raise SystemExit(1)

    # Optionally seed the vector store first
    if args.seed:
        print("[*] Seeding benchmark data before evaluation...")
        os.system(
            f"python src/dataset_ingest.py --mode benchmark --file {ledger}"
        )

    evaluator = PipelineEvaluator(args.config)
    evaluator.run_evaluation(ledger)
