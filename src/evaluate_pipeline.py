"""Pipeline test harness to compute accuracy, token savings, and performance metrics."""

import json
import os
from src.triage_orchestrator import TriageOrchestrator

class PipelineEvaluator:
    """Computes precision, recall, and cost reduction data across test inputs."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.w1 = self.config["gate_parameters"]["weight_static"]
        self.w2 = self.config["gate_parameters"]["weight_slm"]
        self.threshold = self.config["gate_parameters"]["escalation_threshold"]
        self.orchestrator = TriageOrchestrator(config_path)

    def run_evaluation(self, ledger_path: str):
        """Runs testing data through the pipeline gating matrix and scores performance."""
        with open(ledger_path, "r") as f:
            test_cases = json.load(f)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        escalations = 0

        print(f"\n=== Running Pipeline Evaluation Loop ({len(test_cases)} cases) ===")
        
        for case in test_cases:
            # Simulate Stage 1: Map severity values to weight mappings
            sev_str = case["severity"]
            s_sev = self.config["semgrep_severity_map"].get(sev_str, 0.0)
            
            # EXECUTE ACTUAL INFERENCE: No more mocking.
            # We pass the real source code to CodeBERT to see how it performs.
            p_slm = self.orchestrator.slm_inference(case["source_code"])
            
            # Apply our core gating formula: R = (W1 * S_sev) + (W2 * P_slm)
            # The static weight (w1) comes from the hardcoded severity in our test ledger
            risk_score = (self.w1 * s_sev) + (self.w2 * p_slm) 
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

        # Output the performance dashboard
        print("\n==================================================")
        print("          PIPELINE EVALUATION DASHBOARD           ")
        print("==================================================")
        print(f" Total Code Elements Tested : {len(test_cases)}")
        print(f" Total Frontier Escalations : {escalations}")
        print(f" Token Reduction Rate (TRR) : {token_reduction_rate:.1f}%")
        print(f" Pipeline Detection Recall  : {recall * 100:.1f}%")
        print(f" Pipeline Alert Precision  : {precision * 100:.1f}%")
        print("==================================================")
        
        # Enforce validation guards
        if recall >= 0.95 and token_reduction_rate >= 50.0:
            print("[+] Target Met: Pipeline is cost-efficient and structurally secure.")
        else:
            print("[⚠️] Optimization Required: Adjust weights in config.json to balance safety and cost.")

if __name__ == "__main__":
    # Seed the mock database and evaluation matrix files
    os.system("python src/dataset_ingest.py")
    
    # Run the automated performance evaluation evaluation loop
    evaluator = PipelineEvaluator("config.json")
    evaluator.run_evaluation("workspace_storage/evaluation_ledger.json")
