"""Pipeline test harness to compute accuracy, token savings, and performance metrics."""

import json
import os
import shutil
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
from triage_orchestrator import TriageOrchestrator

class PipelineEvaluator:
    """
    Harness for evaluating the end-to-end performance of the security pipeline.
    It compares pipeline decisions (escalate vs. ignore) against ground-truth labels
    to generate metrics like Recall, Precision, and Token Reduction.
    """

    def __init__(self, config_path: str):
        """
        Initializes the evaluator with necessary weights and a versioned run directory.

        Args:
            config_path (str): Path to the master JSON configuration file.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.w1 = self.config["gate_parameters"]["weight_static"]
        self.w2 = self.config["gate_parameters"]["weight_slm"]
        self.threshold = self.config["gate_parameters"]["escalation_threshold"]
        self.orchestrator = TriageOrchestrator(config_path)

        # Setup versioned evaluation directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_id = f"eval_{timestamp}"
        self.eval_dir = os.path.join(
            self.config["paths"]["workspace_root"], 
            self.config["paths"]["evaluations_subdir"], 
            self.eval_id
        )
        os.makedirs(self.eval_dir, exist_ok=True)

    def _prepare_benchmark_files(self, test_cases: list) -> str:
        """
        Writes benchmark code snippets to physical files so static analysis 
        tools can scan them.
        """
        temp_src_dir = os.path.join(self.eval_dir, "transient_benchmarks")
        os.makedirs(temp_src_dir, exist_ok=True)
        
        for i, case in enumerate(test_cases):
            # Create a unique filename that allows us to map results back
            safe_fn = case["function_name"].replace(" ", "_")
            file_name = f"case_{i}_{safe_fn}.py"
            with open(os.path.join(temp_src_dir, file_name), "w") as f:
                f.write(case["source_code"])
        
        return temp_src_dir

    def _run_live_semgrep(self, target_dir: str) -> dict:
        """
        Executes the actual Semgrep CLI against the benchmark files.
        """
        output_path = os.path.join(self.eval_dir, "semgrep_eval_results.json")
        
        # Resolve the absolute path to the custom rules relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_rules_path = os.path.join(base_dir, "semgrep_rules", "custom_appsec_rules.yaml")

        # Using the same config as USAGE.md/CI
        cmd = [
            "semgrep",
            "--config", "p/python",
            "--config", custom_rules_path,
            "--json",
            "--output", output_path,
            target_dir
        ]
        
        print(f"[*] Executing Stage 1 live scan on {target_dir}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for execution errors (e.g., semgrep not installed or config error)
        if result.returncode != 0:
            print(f"[-] Semgrep execution failed (Return Code: {result.returncode})")
            if result.stderr:
                print(f"[-] Semgrep Error Output:\n{result.stderr}")
        
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                return json.load(f)
        return {"results": []}

    def run_evaluation(self, ledger_path: str):
        """Runs testing data through the real pipeline logic and scores performance."""
        # Archive a snapshot of the ledger for this run
        shutil.copy2(ledger_path, os.path.join(self.eval_dir, "input_ledger_snapshot.json"))

        with open(ledger_path, "r") as f:
            test_cases = json.load(f)

        # NEW: Materialize snippets to disk and run Stage 1
        bench_dir = self._prepare_benchmark_files(test_cases)
        try:
            semgrep_results = self._run_live_semgrep(bench_dir)
            
            # Map findings to cases by filename
            findings_map = {}
            for res in semgrep_results.get("results", []):
                fname = os.path.basename(res["path"])
                findings_map[fname] = res["extra"]["severity"]

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0
            escalations = 0
            all_risk_scores = []

            print(f"\n=== Running Pipeline Evaluation Loop ({len(test_cases)} cases) ===")
            
            for i, case in enumerate(test_cases):
                # Map the actual Semgrep severity found on disk
                safe_fn = case["function_name"].replace(" ", "_")
                target_fname = f"case_{i}_{safe_fn}.py"
                sev_str = findings_map.get(target_fname, "NONE")
                s_sev = self.config["semgrep_severity_map"].get(sev_str, 0.0)
                
                # EXECUTE ACTUAL INFERENCE: No more mocking.
                # We pass the real source code to CodeBERT to see how it performs.
                p_slm = self.orchestrator.slm_inference(case["source_code"])
                
                # Apply our core gating formula: R = (W1 * S_sev) + (W2 * P_slm)
                # The static weight (w1) comes from the hardcoded severity in our test ledger
                risk_score = (self.w1 * s_sev) + (self.w2 * p_slm) 
                all_risk_scores.append(risk_score)
                escalated = risk_score >= self.threshold
                
                if escalated:
                    escalations += 1
                
                # Map pipeline decisions against ground truths
                if case["is_vulnerable"] == 1:
                    if escalated:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if escalated:
                        false_positives += 1
                    else:
                        true_negatives += 1

                print(f"Func: {case['function_name']:20} | Risk: {risk_score:.2f} | Escalated: {str(escalated):5} | Ground Truth Vuln: {case['is_vulnerable']}")

            # Compute performance metrics, safely handling edge cases like division by zero
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            token_reduction_rate = (1 - (escalations / len(test_cases))) * 100

            # Save quantitative summary
            summary = {
                "metrics": {
                    "recall": recall,
                    "precision": precision,
                    "token_reduction_rate": token_reduction_rate
                },
                "counts": {
                    "tp": true_positives, "fp": false_positives,
                    "fn": false_negatives, "tn": true_negatives
                }
            }
            
            # Persist summary results for CI/CD tracking or historical comparison
            with open(os.path.join(self.eval_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # Generate Visual Artifacts
            self._generate_graphs(true_positives, false_positives, false_negatives, true_negatives, all_risk_scores)

            # Output the performance dashboard
            print("\n==================================================")
            print("          PIPELINE EVALUATION DASHBOARD           ")
            print("==================================================")
            print(f" Run ID: {self.eval_id}")
            print(f" Total Code Elements Tested : {len(test_cases)}")
            print(f" Total Frontier Escalations : {escalations}")
            print(f" Token Reduction Rate (TRR) : {token_reduction_rate:.1f}%")
            print(f" Pipeline Detection Recall  : {recall * 100:.1f}%")
            print(f" Pipeline Alert Precision  : {precision * 100:.1f}%")
            print("==================================================")
            print(f"[+] All evaluation artifacts saved to: {self.eval_dir}")
            
            # Enforce validation guards
            if recall >= 0.95 and token_reduction_rate >= 50.0:
                print("[+] Target Met: Pipeline is cost-efficient and structurally secure.")
            else:
                print("[⚠️] Optimization Required: Adjust weights in config.json to balance safety and cost.")
        finally:
            # Cleanup temporary source files created for the live scan
            if os.path.exists(bench_dir):
                print(f"[*] Cleaning up temporary benchmark files in {bench_dir}...")
                shutil.rmtree(bench_dir)

    def _generate_graphs(self, tp, fp, fn, tn, scores):
        """
        Produces visual artifacts (histograms and heatmaps) for the evaluation run.

        Args:
            tp, fp, fn, tn (int): Confusion matrix counts.
            scores (List[float]): All risk scores calculated during the run.
        """
        # 1. Risk Score Distribution
        plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=15, color='skyblue', edgecolor='black')
        plt.axvline(self.threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({self.threshold})')
        plt.title("Risk Score Distribution across Test Suite")
        plt.xlabel("Calculated Risk (R)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(self.eval_dir, "risk_distribution.png"))
        plt.close()

        # 2. Confusion Matrix Plot
        cm = [[tn, fp], [fn, tp]]
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks([0, 1], ["Safe", "Vulnerable"])
        plt.yticks([0, 1], ["Safe", "Vulnerable"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.eval_dir, "confusion_matrix.png"))
        plt.close()

if __name__ == "__main__":
    # Seed the mock database and evaluation matrix files
    os.system("python src/dataset_ingest.py --mode benchmark --file src/data/gold_standard.json")
    
    # Run the automated performance evaluation evaluation loop
    evaluator = PipelineEvaluator("config.json")
    evaluator.run_evaluation("workspace_storage/evaluation_ledger.json")
