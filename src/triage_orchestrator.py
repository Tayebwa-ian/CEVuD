import json
import os
import sys
from typing import Dict, Any

class TriageOrchestrator:
    """Manages Stage 1 & 2 pipeline data processing, executes the mathematical

    gating logic, and outputs standardized audit logs.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def mock_slm_inference(self, code_snippet: str) -> float:
        """A free local semantic check.

        In production, this loads your local CodeBERT model via onnxruntime.
        """
        clean_snippet = code_snippet.lower()
        if "mock_" in clean_snippet or "test_" in clean_snippet:
            return 0.15  # Low threat probability for test environments
        if "execute" in clean_snippet or "raw" in clean_snippet:
            return 0.85  # High threat probability for unparameterized sinks
        return 0.40

    def evaluate_gate(self, semgrep_severity: str, slm_score: float) -> Dict[str, Any]:
        """Applies the mathematical formula: R = (W1 * S_sev) + (W2 * P_slm)"""
        weights = self.config["gate_parameters"]
        sev_map = self.config["semgrep_severity_map"]

        w1 = weights["weight_static"]
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

    def process_pipeline(self):
        """Runs the orchestration process over the generated Semgrep inputs."""
        semgrep_path = self.config["paths"]["semgrep_output"]
        
        if not os.path.exists(semgrep_path):
            print(f"[-] Target file {semgrep_path} missing. Terminating pipeline.")
            sys.exit(0)

        with open(semgrep_path, "r") as f:
            semgrep_data = json.load(f)

        findings = semgrep_data.get("results", [])
        if not findings:
            print("[+] Stage 1 Clear: Zero structural rules matched. Pipeline ending.")
            sys.exit(0)

        # Evaluate the highest priority finding
        primary_finding = findings[0]
        snippet = primary_finding.get("extra", {}).get("lines", "")
        severity = primary_finding.get("extra", {}).get("severity", "WARNING")

        # Execute Stage 2 (Local SLM Model Verification)
        slm_score = self.mock_slm_inference(snippet)

        # Run Gating Mathematical Engine
        gate_result = self.evaluate_gate(severity, slm_score)

        # Reconstruct standardized triage data structural payload
        triage_report = {
            "run_id": os.getenv("GITHUB_RUN_ID", "local-dev-run"),
            "evaluated_file": primary_finding.get("path"),
            "metrics": {
                "semgrep_severity_score": gate_result["metrics"]["static_severity_weight"],
                "slm_threat_probability": gate_result["metrics"]["slm_probability_score"],
                "calculated_combined_risk": gate_result["risk_score"]
            },
            "gate_decision": {
                "escalate_to_llm": gate_result["escalate"],
                "gating_threshold_applied": self.config["gate_parameters"]["escalation_threshold"]
            }
        }

        # Write output file to workspace storage disk
        with open(self.config["paths"]["triage_report"], "w") as out:
            json.dump(triage_report, out, indent=2)

        print(f"[+] Triage step complete. Escalation Status: {gate_result['escalate']} (Risk: {gate_result['risk_score']})")

if __name__ == "__main__":
    orchestrator = TriageOrchestrator("config.json")
    orchestrator.process_pipeline()
