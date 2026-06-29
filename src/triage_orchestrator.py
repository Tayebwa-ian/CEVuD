import os
import json
import ast
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TriageOrchestrator:
    """
    Orchestrates the Stage 2 triage workflow by parsing static analysis findings,
    extracting pristine source code blocks directly from the repository filesystem context,
    scoring snippets independently via a fine-tuned Small Language Model (SLM), and combining 
    the distinct security telemetry points into a unified risk dossier.
    """

    def __init__(self, config_path: str, workspace_path: str = None):
        """
        Initializes configuration environments, path routing blocks, and security models.

        Args:
            config_path (str): File system path to the main test manifest configuration.
            workspace_path (str, optional): Target directory path of the codebase under analysis.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Establish base root context coordinates
        self.workspace_path = workspace_path or self.config["paths"].get("workspace_root", ".")
        self.artifact_dir = os.path.join(self.workspace_path, self.config["paths"]["artifacts_subdir"], "run_local-dev-run")

        # Initialize the target sequence classifier model checkpoint
        self.model_name = "jayansh21/codesheriff-bug-classifier"
        print(f"[*] Initializing Security SLM Classifier: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def extract_source_snippet(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Parses a physical source file into an Abstract Syntax Tree (AST) to identify
        and extract the complete, unbroken function block containing the matched lines.
        
        Ensures input context parity with the evaluation pipeline.

        Args:
            file_path (str): Relative or absolute target file path to parse.
            start_line (int): The starting line pointer from Semgrep (1-indexed).
            end_line (int): The ending line pointer from Semgrep (1-indexed).

        Returns:
            str: Clean, complete function string block starting at 'def ' and ending at scope close.
        """
        resolved_path = file_path if os.path.isabs(file_path) else os.path.join(self.workspace_path, file_path)
        
        if not os.path.exists(resolved_path):
            print(f"[!] Warning: Source file missing during extraction: {resolved_path}")
            return ""

        try:
            with open(resolved_path, "r", encoding="utf-8") as source_file:
                source_code = source_file.read()
                file_lines = source_code.splitlines()

            # Build the Abstract Syntax Tree from the target file source
            tree = ast.parse(source_code, filename=resolved_path)
            
            target_function_node = None
            
            # Walk through all nodes inside the syntax tree structure
            for node in ast.walk(tree):
                # Target both standard functions and async class/module methods
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if Semgrep's match line sits securely inside this function's block limits
                    # Note: node.end_lineno is supported natively in Python 3.8+
                    if node.lineno <= start_line <= node.end_lineno:
                        target_function_node = node
                        break # Exact matching functional scope located

            # If a valid function envelope is found, slice the exact node layout lines
            if target_function_node:
                # AST line offsets are 1-indexed; convert to 0-indexed slice bounds
                slice_start = target_function_node.lineno - 1
                slice_end = target_function_node.end_lineno
                
                # Reconstruct the pristine functional unit block string
                pristine_function = "\n".join(file_lines[slice_start:slice_end])
                return pristine_function.strip()
                
            # Fallback: If the finding is at a module level (outside a function), slice standard boundaries
            print(f"[*] Match line {start_line} outside function scope. Falling back to line slice.")
            slice_start = max(0, start_line - 1)
            slice_end = min(len(file_lines), end_line)
            return "\n".join(file_lines[slice_start:slice_end]).strip()

        except Exception as err:
            print(f"[!] AST parsing or slice exception on target asset file: {err}")
            return ""

    def slm_inference(self, code_snippet: str) -> float:
        """
        Executes independent local forward-pass tokenization and inference 
        over raw text data to map real threat probabilities.

        Args:
            code_snippet (str): Pristine code snippet source context string.

        Returns:
            float: Distributed risk scale soft probability bound ranging between [0.0, 1.0].
        """
        if not code_snippet.strip():
            return 0.0

        # Encode input code string matching base tokenization model thresholds
        inputs = self.tokenizer(
            code_snippet, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Map raw logits to a soft probability distribution curve via spatial softmax activation
            probabilities = torch.softmax(outputs.logits, dim=1).flatten()
            
        # Target Index 1 returns the explicit positive class continuous scalar distribution
        return round(probabilities[1].item(), 4)

    def evaluate_gate(self, semgrep_severity: str, slm_score: float) -> dict:
        """
        Combines isolated telemetry signals from Stage 1 and Stage 2 
        into a composite risk score matrix.

        Args:
            semgrep_severity (str): The raw diagnostic classification level from the static engine.
            slm_score (float): Independent soft probability generated by the neural network block.

        Returns:
            dict: Structured results matching the exact schema requirements of process_pipeline.
        """
        # Map configured weightings from the environment manifest config
        severity_weight = self.config["semgrep_severity_map"].get(semgrep_severity, 0.0)
        w1 = self.config["gate_parameters"]["weight_static"]
        w2 = self.config["gate_parameters"]["weight_slm"]
        
        # Calculate final continuous risk boundary parameters
        risk_score = (w1 * severity_weight) + (w2 * slm_score)
        escalate = risk_score >= self.config["gate_parameters"]["escalation_threshold"]
        
        return {
            "risk_score": round(risk_score, 4),
            "escalate": escalate,
            "metrics": {
                "static_severity_weight": round(severity_weight, 4),
                "slm_probability_score": round(slm_score, 4)
            }
        }

    def process_pipeline(self):
        """
        Main orchestration execution loop. Decouples structural data streams 
        by using Semgrep solely for file coordinates while reading code 
        blocks directly from the physical codebase. Writes out the final consolidated 
        stage1_2_triage.json dossier file.
        """
        semgrep_filename = self.config["paths"]["semgrep_output"]
        target_path = os.path.join(self.artifact_dir, semgrep_filename)
        semgrep_workspace_path = os.path.join(self.workspace_path, semgrep_filename)

        # Normalize artifact locations across directories
        if not os.path.exists(semgrep_workspace_path) and os.path.exists(semgrep_filename):
            semgrep_workspace_path = semgrep_filename

        if os.path.exists(semgrep_workspace_path):
            if not os.path.exists(target_path) or not os.path.samefile(semgrep_workspace_path, target_path):
                print(f"[*] Moving {semgrep_workspace_path} to artifact directory: {self.artifact_dir}")
                os.makedirs(self.artifact_dir, exist_ok=True)
                shutil.move(semgrep_workspace_path, target_path)

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target static file asset missing at path location: {target_path}")

        with open(target_path, "r") as f:
            semgrep_data = json.load(f)

        findings = semgrep_data.get("results", [])
        finding_reports = []
        overall_escalate = False

        # Loop through static matches to locate and extract targets independently
        for finding in findings:
            file_target_path = finding.get("path", "")
            
            # Extract line boundary metrics logged by the scanner pass
            start_line = finding.get("start", {}).get("line", 1)
            end_line = finding.get("end", {}).get("line", 1)
            severity = finding.get("extra", {}).get("severity", "WARNING")

            # Extract pure repository source strings instead of using semgrep snippet elements
            snippet = self.extract_source_snippet(file_target_path, start_line, end_line)

            # Pass the extracted source snippet to the neural model block independently
            slm_score = self.slm_inference(snippet)
            
            # Combine individual scores via triage gate mechanics
            gate_result = self.evaluate_gate(severity, slm_score)

            if gate_result["escalate"]:
                overall_escalate = True

            finding_reports.append({
                "evaluated_file": file_target_path,
                "code_snippet": snippet,
                "metrics": {
                    "semgrep_severity_score": gate_result["metrics"]["static_severity_weight"],
                    "slm_threat_probability": gate_result["metrics"]["slm_probability_score"],
                    "calculated_combined_risk": gate_result["risk_score"]
                },
                "escalate": gate_result["escalate"]
            })

        # Pack and serialize the unified pipeline triage data payload safely to disk
        triage_report = {
            "run_id": "run_local-dev-run",
            "gate_decision": {
                "escalate_to_llm": overall_escalate,
                "gating_threshold_applied": self.config["gate_parameters"]["escalation_threshold"]
            },
            "findings": finding_reports,
            "status": "VULNERABLE" if overall_escalate else "SAFE"
        }

        triage_output_path = os.path.join(self.artifact_dir, self.config["paths"]["triage_report"])
        with open(triage_output_path, "w") as out:
            json.dump(triage_report, out, indent=2)
            
        print(f"[+] Successfully wrote decoupled Stage 2 triage report data to: {triage_output_path}")
