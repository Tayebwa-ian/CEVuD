# src/evaluate_pipeline.py
"""
Pipeline Evaluation Harness
===========================
Computes comprehensive detection effectiveness and cost-efficiency metrics
by running the evaluation loop against a curated ground-truth dataset 
(gold_standard.json) via the production TriageOrchestrator process pipeline.

This module enforces a single source of truth by relying on the real Stage 1
and Stage 2 staging engines to produce authentic filesystem artifacts and reports.

Persisted Artifacts (per evaluation run):
    - summary.json              : Quantitative metrics (Recall, Precision, Accuracy, F1, TRR, CSR)
    - detailed_findings.json    : Per-case risk scores, severities, and escalation decisions
    - confusion_matrix.png      : Visual heatmap of TP/FP/TN/FN
    - risk_distribution.png     : Histogram of risk scores with threshold overlay
    - category_performance.png  : Per-vulnerability-category detection bar chart
    - input_ledger_snapshot.json: Frozen copy of the test data used for the run
"""

import json
import os
import sys
import shutil
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/headless environments
import matplotlib.pyplot as plt

from triage_orchestrator import TriageOrchestrator


# ---------------------------------------------------------------------------
# Default path to the gold-standard ground-truth dataset.
# ---------------------------------------------------------------------------
_DEFAULT_GOLD_STANDARD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "data", "gold_standard.json",
)


class PipelineEvaluator:
    """
    End-to-end benchmark harness for the CEVuD security pipeline.

    Workflow:
        1. Materialize gold-standard code snippets to temporary files on disk.
        2. Execute a live Semgrep subprocess over the generated codebase.
        3. Invoke the `TriageOrchestrator` to aggregate results and score risk metrics.
        4. Derive and graph performance and cost reduction metrics.
    """

    def __init__(self, config_path: str):
        """
        Initialize the evaluator and establish dynamic workspace routing.

        Args:
            config_path: Absolute or relative path to the master ``config.json``.
        """
        self._config_path = os.path.abspath(config_path)
        with open(self._config_path, "r") as f:
            self.config = json.load(f)

        self.threshold = self.config["gate_parameters"]["escalation_threshold"]

        # Establish timestamped evaluation run output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_id = f"eval_{timestamp}"
        
        ws_root = self.config["paths"]["workspace_root"]
        evals_sub = self.config["paths"]["evaluations_subdir"]

        # Resolve evaluations destination folder cleanly
        if os.path.isabs(evals_sub):
            self.eval_dir = os.path.join(evals_sub, self.eval_id)
        elif os.path.isabs(ws_root):
            self.eval_dir = os.path.join(ws_root, evals_sub, self.eval_id)
        else:
            self.eval_dir = os.path.join(ws_root, evals_sub, self.eval_id)

        # Enforce evaluation folder existence for summaries and charts
        os.makedirs(self.eval_dir, exist_ok=True)

    def _prepare_benchmark_files(self, test_cases: List[Dict[str, Any]]) -> str:
        """
        Write each gold-standard snippet to its own .py file inside an isolated
        temporary folder outside the project tree to bypass ignore-mask rules.
        """
        # BORROWED KNOWLEDGE: Write outside the project tree so local ignore rules (.gitignore/.semgrepignore) cannot mask evaluation vectors
        bench_src_dir = tempfile.mkdtemp(prefix="cevud_bench_")

        for i, case in enumerate(test_cases):
            safe_fn = case["function_name"].replace(" ", "_")
            file_name = f"case_{i}_{safe_fn}.py"
            with open(os.path.join(bench_src_dir, file_name), "w", encoding="utf-8") as f:
                f.write(case["source_code"])

        print(f"[*] Benchmark test cases materialized to isolated path: {bench_src_dir}")
        return bench_src_dir

    def _run_live_semgrep(self, target_dir: str, output_path: str) -> dict:
        """
        Execute the Semgrep CLI against *target_dir* using the same rule-set
        as the production CI pipeline.
        """
        # Resolve custom appsec rules relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_rules_path = os.path.join(
            base_dir, "semgrep_rules", "custom_appsec_rules.yaml"
        )

        # Explicitly collect files to bypass directory filtering rules
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
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"results": []}

    def _execute_orchestration_run(self, bench_dir: str):
        """
        Invoke the primary production orchestrator pipeline over the evaluation workspace.
        """
        run_id = f"run_{os.getenv('GITHUB_RUN_ID') or os.getenv('GITHUB_SHA') or 'local-dev-run'}"
        target_artifact_dir = os.path.join(self.eval_dir, self.config["paths"]["artifacts_subdir"], run_id)
        os.makedirs(target_artifact_dir, exist_ok=True)

        # Enforce that Semgrep outputs to the directory where TriageOrchestrator looks
        semgrep_filename = self.config["paths"]["semgrep_output"]
        target_semgrep_json = os.path.join(bench_dir, semgrep_filename)
        
        self._run_live_semgrep(bench_dir, target_semgrep_json)

        print("[*] Executing production TriageOrchestrator over benchmark folder...")
        orchestrator = TriageOrchestrator(
            config_path=self._config_path,
            workspace_path=bench_dir
        )
        
        orchestrator.artifact_dir = target_artifact_dir
        orchestrator.process_pipeline()

    def _parse_triage_report(self) -> Dict[str, Any]:
        """
        Locate and load the generated triage report from the orchestrator's run directory.

        Returns:
            The parsed JSON content from the generated stage 2 triage report.
        """
        artifacts_subdir = self.config["paths"]["artifacts_subdir"]
        artifacts_path = os.path.join(self.eval_dir, artifacts_subdir)

        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Orchestrator artifact root missing from evaluation runtime: {artifacts_path}")

        run_dirs = [d for d in os.listdir(artifacts_path) if d.startswith("run_")]
        if not run_dirs:
            raise FileNotFoundError(f"No triage run execution directories discovered under {artifacts_path}")

        # Target the most recently generated runtime path
        latest_run_dir = max(
            [os.path.join(artifacts_path, d) for d in run_dirs],
            key=os.path.getmtime
        )
        triage_report_path = os.path.join(latest_run_dir, self.config["paths"]["triage_report"])

        if not os.path.exists(triage_report_path):
            raise FileNotFoundError(f"Stage 2 triage report data file not found: {triage_report_path}")

        with open(triage_report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_metrics(self, test_cases: List[Dict[str, Any]], triage_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[float], List[Dict[str, Any]]]:
        """
        Align the static evaluation ground truth cases against the dynamic orchestrator findings.

        Args:
            test_cases: The ground truth test records ledger data.
            triage_data: The deserialized triage report produced by the orchestrator loop.
        """
        findings_map = {}
        for finding in triage_data.get("findings", []):
            findings_map[finding["evaluated_file"]] = finding

        tp = fp = fn = tn = 0
        escalations = 0
        all_risk_scores: List[float] = []
        detailed_findings: List[Dict[str, Any]] = []

        print(f"\n=== Mapping Pipeline Triage Decisions to Ground Truth ({len(test_cases)} cases) ===")

        for i, case in enumerate(test_cases):
            func_name = case["function_name"]
            is_vuln = case["is_vulnerable"] == 1
            
            safe_fn = func_name.replace(" ", "_")
            expected_filename = f"case_{i}_{safe_fn}.py"

            matching_finding = None
            for file_key, finding_payload in findings_map.items():
                if file_key.endswith(expected_filename):
                    matching_finding = finding_payload
                    break

            if matching_finding is not None:
                escalated = matching_finding["escalate"]
                risk_score = matching_finding["metrics"]["calculated_combined_risk"]
                semgrep_sev = matching_finding["metrics"].get("semgrep_severity_score", 0.0)
                slm_prob = matching_finding["metrics"].get("slm_threat_probability", 0.0)
            else:
                escalated = False
                risk_score = 0.0
                semgrep_sev = 0.0
                slm_prob = 0.0

            all_risk_scores.append(risk_score)
            if escalated:
                escalations += 1

            if is_vuln and escalated:
                tp += 1
            elif is_vuln and not escalated:
                fn += 1
            elif not is_vuln and escalated:
                fp += 1
            else:
                tn += 1

            detailed_findings.append({
                "index": i,
                "function_name": func_name,
                "file_path": case["file_path"],
                "is_vulnerable": case["is_vulnerable"],
                "semgrep_severity": semgrep_sev,
                "slm_probability": slm_prob,
                "risk_score": round(risk_score, 4),
                "escalated": escalated
            })

            print(
                f"  [{i:02d}] {func_name:25s} | "
                f"Risk: {risk_score:.3f} | "
                f"Escalated: {str(escalated):5s} | "
                f"Ground Truth: {'VULN' if is_vuln else 'SAFE'}"
            )

        total = len(test_cases)
        metrics = {
            "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
            "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4),
            "accuracy": round((tp + tn) / total if total > 0 else 0.0, 4),
            "f1_score": round(2 * (tp / (tp + fp) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0, 4),
            "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
            "token_reduction_rate": round((1 - (escalations / total)) * 100 if total > 0 else 0.0, 2),
            "cost_savings_ratio": round(((total - escalations) / total) * 100 if total > 0 else 0.0, 2)
        }

        confusion_matrix = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        return {"confusion_matrix": confusion_matrix, "metrics": metrics, "total_cases": total, "escalations": escalations}, all_risk_scores, detailed_findings

    def _save_and_display_results(self, summary_payload: Dict[str, Any], detailed_findings: List[Dict[str, Any]], all_risk_scores: List[float]):
        """Save results down to files and dump the final metric visualization dashboard."""
        with open(os.path.join(self.eval_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        with open(os.path.join(self.eval_dir, "detailed_findings.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_findings, f, indent=2)

        cm = summary_payload["confusion_matrix"]
        self._generate_graphs(cm["tp"], cm["fp"], cm["fn"], cm["tn"], all_risk_scores, detailed_findings)

        metrics = summary_payload["metrics"]
        print("\n" + "=" * 58)
        print("        PIPELINE EVALUATION DASHBOARD")
        print("=" * 58)
        print(f"  Run ID                  : {self.eval_id}")
        print(f"  Total Cases Tested      : {summary_payload['total_cases']}")
        print(f"  Total Escalations       : {summary_payload['escalations']}")
        print(f"  ─── Detection Metrics ───")
        print(f"  Recall (Sensitivity)    : {metrics['recall'] * 100:.1f}%")
        print(f"  Precision               : {metrics['precision'] * 100:.1f}%")
        print(f"  Accuracy                : {metrics['accuracy'] * 100:.1f}%")
        print(f"  F1 Score                : {metrics['f1_score']:.4f}")
        print(f"  Specificity             : {metrics['specificity'] * 100:.1f}%")
        print(f"  ─── Cost Metrics ────────")
        print(f"  Token Reduction Rate    : {metrics['token_reduction_rate']:.1f}%")
        print(f"  Cost Savings Ratio      : {metrics['cost_savings_ratio']:.1f}%")
        print("=" * 58)
        print(f"  [+] Artifacts saved to: {self.eval_dir}")

        if metrics['recall'] >= 0.95 and metrics['token_reduction_rate'] >= 50.0:
            print("  [+] Target Met: Pipeline is cost-efficient and structurally secure.")
        else:
            print("  [⚠️] Optimization Required: Adjust parameters inside config.json.")

    def run_evaluation(self, ledger_path: str):
        """
        Execute the modular pipeline evaluation harness against ground truth inputs
        and mirror real-time findings to the fallback staging folder for E2E tests.
        """
        # Snapshot input data ledger
        shutil.copy2(ledger_path, os.path.join(self.eval_dir, "input_ledger_snapshot.json"))

        with open(ledger_path, "r", encoding="utf-8") as f:
            test_cases: List[Dict[str, Any]] = json.load(f)

        # 1. Materialize source text blocks outside repo tree boundaries
        bench_dir = self._prepare_benchmark_files(test_cases)

        try:
            # 2. Run the dynamic execution loop using the isolated benchmark path
            self._execute_orchestration_run(bench_dir)

            # 3. Process calculations against the generated workspace reports
            triage_data = self._parse_triage_report()
            summary_payload, risk_scores, detailed_findings = self._compute_metrics(test_cases, triage_data)
            
            summary_payload["eval_id"] = self.eval_id
            summary_payload["timestamp"] = datetime.now().isoformat()

            # 4. Save metrics visualizations down into evaluation_runs storage
            self._save_and_display_results(summary_payload, detailed_findings, risk_scores)

            # 5. Mirror runtime artifacts back into production workspace tree
            # This bridges the gap between evaluation runs and the downstream Phase 3 E2E test.
            try:
                run_id = triage_data.get("run_id", "run_local-dev-run")
                
                # Resolve the configuration paths directly from config file definitions
                ws_root = self.config["paths"].get("workspace_root", ".")
                if not os.path.isabs(ws_root):
                    # Fall back relative to the active master configuration directory layout
                    config_dir = os.path.dirname(self._config_path)
                    ws_root = os.path.abspath(os.path.join(config_dir, ws_root))

                # Build production destination: workspace_storage_e2e/artifacts/run_local-dev-run/
                prod_artifact_dir = os.path.join(ws_root, self.config["paths"]["artifacts_subdir"], run_id)
                eval_artifact_dir = os.path.join(self.eval_dir, self.config["paths"]["artifacts_subdir"], run_id)

                if os.path.exists(eval_artifact_dir):
                    os.makedirs(prod_artifact_dir, exist_ok=True)
                    for file_name in os.listdir(eval_artifact_dir):
                        src_f = os.path.join(eval_artifact_dir, file_name)
                        dst_f = os.path.join(prod_artifact_dir, file_name)
                        if os.path.isfile(src_f):
                            shutil.copy2(src_f, dst_f)
                    print(f"[+] Successfully synced validation findings to fallback test root: {prod_artifact_dir}")
            except Exception as sync_err:
                print(f"[!] Target mirroring warning encountered: {sync_err}")

        finally:
            # Clear down runtime files out of volatile directories safely
            if os.path.exists(bench_dir):
                print(f"[*] Cleaning up temporary benchmark files from {bench_dir}...")
                shutil.rmtree(bench_dir)

    def _generate_graphs(self, tp: int, fp: int, fn: int, tn: int, scores: List[float], detailed_findings: List[Dict[str, Any]]):
        """Produce visualization charts for risk trends and classification accuracy results."""
        # Risk Distribution Histogram Chart
        plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=15, color="skyblue", edgecolor="black", alpha=0.85)
        plt.axvline(self.threshold, color="red", linestyle="dashed", linewidth=2, label=f"Threshold ({self.threshold})")
        plt.title("Evaluation Run Risk Score Distribution")
        plt.xlabel("Calculated Risk Metric Score")
        plt.ylabel("Sample Density / Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "risk_distribution.png"), dpi=150)
        plt.close()

        # Heatmap Confusion Matrix Presentation
        cm = [[tn, fp], [fn, tp]]
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(cax)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row][col]), ha="center", va="center", fontsize=18, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Safe", "Predicted Vuln"])
        ax.set_yticklabels(["Actual Safe", "Actual Vuln"])
        ax.set_xlabel("Pipeline Judgment Label")
        ax.set_ylabel("Ground Truth Validation Label")
        ax.set_title("Confusion Matrix Dashboard Metric View")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # Category Performance Breakdown Charts
        category_stats: Dict[str, Dict[str, int]] = {}
        for finding in detailed_findings:
            cat = os.path.splitext(os.path.basename(finding["file_path"]))[0]
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            category_stats[cat]["total"] += 1
            is_vuln = finding["is_vulnerable"] == 1
            if (is_vuln and finding["escalated"]) or (not is_vuln and not finding["escalated"]):
                category_stats[cat]["correct"] += 1

        categories = sorted(category_stats.keys())
        accuracies = [(category_stats[c]["correct"] / category_stats[c]["total"] * 100) for c in categories]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(categories, accuracies, color="mediumseagreen", edgecolor="black")
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.0f}%", ha="center", va="bottom", fontsize=9)
        plt.ylim(0, 115)
        plt.title("Per-Category Vulnerability Detection Accuracy Breakdown")
        plt.xlabel("Target Vulnerability Module Class Type")
        plt.ylabel("Accuracy Level Percentage (%)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "category_performance.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEVuD Pipeline Evaluator — production verification harness.")
    parser.add_argument("--config", default="config.json", help="Path to the master config.json file.")
    parser.add_argument("--ledger", default=None, help="Path to the gold-standard validation file registry ledger.")
    parser.add_argument("--seed", action="store_true", help="Run the dataset_ingest benchmark database seeder prior to processing evaluation.")

    args = parser.parse_args()
    ledger_target = args.ledger or _DEFAULT_GOLD_STANDARD

    if not os.path.exists(ledger_target):
        print(f"[-] Evaluation Ledger Manifest registry missing: {ledger_target}")
        raise SystemExit(1)

    if args.seed:
        print("[*] Pre-seeding database vector blocks prior to launch initialization...")
        try:
            subprocess.run([sys.executable, "src/dataset_ingest.py", "--mode", "benchmark", "--file", ledger_target], check=True, capture_output=True, text=True)
            print("[+] Target metrics vectors populated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[-] Database pre-seeding pipeline failure encountered: {e.stderr}")
            raise SystemExit(1)

    evaluator = PipelineEvaluator(args.config)
    evaluator.run_evaluation(ledger_target)
