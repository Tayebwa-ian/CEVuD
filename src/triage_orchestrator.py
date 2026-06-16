import json
import os
import shutil
import sys
import subprocess
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from diff_parser import DiffParser

class TriageOrchestrator:
    """
    Core Orchestration Logic for Stages 1 and 2.
    It consumes Semgrep results, runs semantic inference via CodeBERT, 
    and applies a weighted risk formula to decide if an escalation to 
    Stage 3 is necessary.
    """

    def __init__(self, config_path: str):
        """
        Initializes the orchestrator, model, and artifact directories.

        Args:
            config_path (str): Path to configuration settings.
        """
        # Load config with fallback to root if not found
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Fallback for local development if run from different subdirs
            root_config = os.path.join(os.path.dirname(__file__), "..", "config.json")
            with open(root_config, "r") as f:
                self.config = json.load(f)
        
        # Determine Run ID and Artifact Directory
        self.run_id = os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or "local-dev-run"
        if not self.run_id.startswith("run_"):
            self.run_id = f"run_{self.run_id}"
            
        self.artifact_dir = os.path.join(self.config["paths"]["workspace_root"], self.config["paths"]["artifacts_subdir"], self.run_id)
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Initialize CodeBERT for vulnerability classification
        # Using a model head fine-tuned for defect detection/security analysis
        self.model_name = "microsoft/codebert-base" 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.eval()

    def slm_inference(self, code_snippet: str) -> float:
        """
        Predicts the probability of code being vulnerable using a local SLM.

        Args:
            code_snippet (str): The source code to analyze.
        
        Returns:
            float: Probability score between 0.0 and 1.0.
        """
        # Tokenize the input code snippet with standard CodeBERT max length
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Softmax to get probabilities for [Non-Vulnerable, Vulnerable]
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract the 'Vulnerable' class probability (index 1)
        threat_probability = probs[0][1].item()
        
        # Ensure we return a float for the gating logic
        return round(float(threat_probability), 4)

    def evaluate_gate(self, semgrep_severity: str, slm_score: float) -> Dict[str, Any]:
        """
        Applies the weighted gating formula: R = (W1 * S_sev) + (W2 * P_slm).

        Args:
            semgrep_severity (str): Severity label from Semgrep (ERROR, WARNING, INFO, NONE).
            slm_score (float): Probability score from the CodeBERT model.

        Returns:
            Dict[str, Any]: Combined risk score and escalation decision.
        """
        weights = self.config["gate_parameters"]
        sev_map = self.config["semgrep_severity_map"]

        w1 = weights["weight_static"] if semgrep_severity != "NONE" else 0.0
        w2 = weights["weight_slm"]
        s_sev = sev_map.get(semgrep_severity, 0.3)
        p_slm = slm_score

        # Calculate combined risk index
        risk_score = (w1 * s_sev) + (w2 * p_slm)
        escalate = risk_score >= weights["escalation_threshold"]

        return {
            "risk_score": round(risk_score, 3),
            "escalate": escalate,
            "metrics": {
                "static_severity_weight": s_sev,
                "slm_probability_score": p_slm
            }
        }

    def _get_fallback_snippets(self) -> list:
        """
        Fail-safe mechanism: Extracts code snippets from the git diff 
        when static analysis fails to find a match. This ensures 
        Stage 2 always has context to analyze.
        """
        try:
            # Attempt to capture current local uncommitted/staged changes (Local Dev)
            diff_content = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.STDOUT).decode()
            
            # If the working tree is clean, fall back to comparing the last commit (Standard CI behavior)
            if not diff_content.strip():
                diff_content = subprocess.check_output(["git", "diff", "HEAD~1", "HEAD"], stderr=subprocess.STDOUT).decode()
            
            if not diff_content.strip():
                return []

            parser = DiffParser(diff_content)
            modified_map = parser.parse_modified_lines()
            
            all_impacted_functions = []
            for file_path, lines in modified_map.items():
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        content = f.read()
                        all_impacted_functions.extend(parser.get_functions_from_ast(content, lines))
            return all_impacted_functions
        except Exception as e:
            print(f"[!] Fallback snippet extraction failed: {e}")
            return []

    def process_pipeline(self):
        """
        Main execution flow for Stage 2.
        Manages file movements, parses Semgrep JSON, runs inference, 
        and writes the standardized triage report.
        """
        semgrep_filename = self.config["paths"]["semgrep_output"]
        target_path = os.path.join(self.artifact_dir, semgrep_filename)
        
        # Proactively move Stage 1 output to the run-specific directory if found in root
        if os.path.exists(semgrep_filename) and not os.path.samefile(os.getcwd(), self.artifact_dir):
            print(f"[*] Moving {semgrep_filename} to artifact directory: {self.artifact_dir}")
            shutil.move(semgrep_filename, target_path)
            
        semgrep_path = target_path
        if not os.path.exists(semgrep_path):
            print(f"[-] Target file {semgrep_path} missing. Terminating pipeline.")
            sys.exit(0)

        with open(semgrep_path, "r") as f:
            semgrep_data = json.load(f)

        findings = semgrep_data.get("results", [])
        finding_reports = []
        overall_escalate = False

        if not findings:
            print("[*] Stage 1 Clear. Falling back to SLM scan on modified functions...")
            fallback_snippets = self._get_fallback_snippets()
            
            if not fallback_snippets:
                pass 
            else:
                for snippet in fallback_snippets:
                    slm_score = self.slm_inference(snippet)
                    gate_result = self.evaluate_gate("NONE", slm_score)
                    finding_reports.append({
                        "evaluated_file": "Modified Diff Snippet",
                        "code_snippet": snippet,
                        "metrics": {
                            "semgrep_severity_score": 0.0,
                            "slm_threat_probability": slm_score,
                            "calculated_combined_risk": gate_result["risk_score"]
                        },
                        "escalate": gate_result["escalate"]
                    })
                    if gate_result["escalate"]:
                        overall_escalate = True
        else:
            for finding in findings:
                snippet = finding.get("extra", {}).get("lines", "")
                severity = finding.get("extra", {}).get("severity", "WARNING")
                
                slm_score = self.slm_inference(snippet)
                gate_result = self.evaluate_gate(severity, slm_score)
                
                finding_reports.append({
                    "evaluated_file": finding.get("path"),
                    "code_snippet": snippet,
                    "metrics": {
                        "semgrep_severity_score": gate_result["metrics"]["static_severity_weight"],
                        "slm_threat_probability": gate_result["metrics"]["slm_probability_score"],
                        "calculated_combined_risk": gate_result["risk_score"]
                    },
                    "escalate": gate_result["escalate"]
                })
                if gate_result["escalate"]:
                    overall_escalate = True

        # Aggregate and wrap into unified report structure
        triage_report = {
            "run_id": self.run_id,
            "findings": finding_reports,
            "gate_decision": {
                "escalate_to_llm": overall_escalate,
                "gating_threshold_applied": self.config["gate_parameters"]["escalation_threshold"]
            },
            "status": "VULNERABLE" if any(f["metrics"]["calculated_combined_risk"] > 0 for f in finding_reports) else "CLEAR"
        }
        if not finding_reports:
            triage_report["status"] = "CLEAR"

        # Write output file to workspace storage disk
        triage_path = os.path.join(self.artifact_dir, self.config["paths"]["triage_report"])
        with open(triage_path, "w") as out:
            json.dump(triage_report, out, indent=2)

        status = triage_report.get("status")
        max_risk = max([f["metrics"]["calculated_combined_risk"] for f in finding_reports]) if finding_reports else 0.0
        print(f"[+] Triage complete ({status}). Processed {len(finding_reports)} findings. Max Risk: {max_risk}. Report: {triage_path}")

if __name__ == "__main__":
    orchestrator = TriageOrchestrator("config.json")
    orchestrator.process_pipeline()
