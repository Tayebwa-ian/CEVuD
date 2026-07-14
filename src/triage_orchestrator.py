import os
import json
import ast
import shutil
import torch
import sys
from typing import List, Dict, Any
from model_manager import ModelManager
from vector_store import LocalVectorStore
from run_context import resolve_run_id, get_artifact_dir, get_vector_db_dir, get_model_cache_dir, get_semgrep_output_path, get_triage_report_path

# ---------------------------------------------------------------------------
# Single source of truth for the gate formula: src/evaluation/gate_strategies.py
# defines `linear_weighted_gate`, which is imported here rather than
# reimplemented, so the production gate and the gate evaluated in the
# comparative evaluation suite (src/evaluation/run_comparative_evaluation.py)
# can never silently drift apart.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
try:
    from evaluation.gate_strategies import linear_weighted_gate
except ImportError:
    from gate_strategies import linear_weighted_gate

# Reuse the SAME snippet assembly used to build the training data so the SLM
# sees function + module imports at inference time (train/inference parity).
try:
    from evaluation.code_context import collect_module_imports, build_context_snippet
except ImportError:
    from code_context import collect_module_imports, build_context_snippet

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
        # Set environment variable for ModelManager to locate config
        os.environ["CEVUD_CONFIG_PATH"] = os.path.abspath(config_path)
        self._config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Establish base root context coordinates
        self.workspace_path = workspace_path or self.config["paths"].get("workspace_root", ".")
        # Single source of truth for the run id: Stage 1/2/3 must agree on the
        # artifact directory or the Stage-2 ledger is never found by Stage 3.
        ws_root = self.config["paths"].get("workspace_root", "workspace_storage")
        run_id = resolve_run_id(self.workspace_path, ws_root)
        self.run_id = run_id

        # All workspace_storage subdirectories are computed by run_context.py
        # helpers from config.json → paths.  This is the single source of truth;
        # the workflows, agent.py, and model_manager.py all use the same helpers.
        self.artifact_dir = get_artifact_dir(self.workspace_path, self.config, self.run_id)
        self.vector_db_dir = get_vector_db_dir(self.workspace_path, self.config)
        self.model_cache_dir = get_model_cache_dir(self.workspace_path, self.config)
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Initialize ModelManager — no model loaded yet, it's lazy-loaded on first use
        self.model_manager = ModelManager()

        print(f"[*] TriageOrchestrator initialized. Model will load on first inference.")

    def extract_source_snippet(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Parses a physical source file into an Abstract Syntax Tree (AST) to identify
        and extract the complete, unbroken function block containing the matched lines.
        Includes a path-resilience fallback mechanism for staging environments.
        """
        # Try resolving via standard rules first
        resolved_path = file_path if os.path.isabs(file_path) else os.path.join(self.workspace_path, file_path)
        
        # SENIOR ARCHITECTURE FIX: Fallback lookup for shifting temporary execution directories
        if not os.path.exists(resolved_path):
            fallback_path = os.path.join(self.workspace_path, os.path.basename(file_path))
            if os.path.exists(fallback_path):
                resolved_path = fallback_path
            else:
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
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.lineno <= start_line <= node.end_lineno:
                        target_function_node = node
                        break

            if target_function_node:
                slice_start = target_function_node.lineno - 1
                slice_end = target_function_node.end_lineno
                pristine_function = "\n".join(file_lines[slice_start:slice_end])
                return pristine_function.strip()
                
            print(f"[*] Match line {start_line} outside function scope. Falling back to line slice.")
            slice_start = max(0, start_line - 1)
            slice_end = min(len(file_lines), end_line)
            return "\n".join(file_lines[slice_start:slice_end]).strip()

        except Exception as err:
            print(f"[!] AST parsing or slice exception on target asset file: {err}")
            return ""

    def _resolve_source(self, file_path: str):
        """Resolve ``file_path`` to an absolute path and return (path, source)."""
        resolved_path = file_path if os.path.isabs(file_path) else os.path.join(self.workspace_path, file_path)
        if not os.path.exists(resolved_path):
            fallback_path = os.path.join(self.workspace_path, os.path.basename(file_path))
            if os.path.exists(fallback_path):
                resolved_path = fallback_path
            else:
                return resolved_path, None
        try:
            with open(resolved_path, "r", encoding="utf-8") as fh:
                return resolved_path, fh.read()
        except Exception:
            return resolved_path, None

    def _function_block(self, resolved_path: str, source: str, start_line: int, end_line: int):
        """Return ``(function_code, func_start, func_end)`` for the function
        enclosing ``start_line`` (mirrors ``extract_source_snippet`` but also
        returns the line span so the SLM input can be re-assembled with imports)."""
        if source is None:
            return None, None, None
        try:
            tree = ast.parse(source, filename=resolved_path)
        except Exception:
            return None, None, None
        target = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= start_line <= node.end_lineno:
                    target = node
                    break
        if target is None:
            return None, None, None
        lines = source.splitlines()
        lo = target.lineno - 1
        hi = target.end_lineno
        return "\n".join(lines[lo:hi]).strip(), target.lineno, target.end_lineno

    def _slm_input(self, source: str, func_start: int, func_end: int) -> str:
        """Assemble the exact input the SLM is trained on: module imports +
        the enclosing function. Cross-file context is deliberately excluded
        here — it is added later, only when a finding is escalated to the LLM."""
        imports = collect_module_imports(source)
        return build_context_snippet(source, (func_start, func_end), imports, {})

    def _collect_context(self, func_name: str) -> List[str]:
        """Return cross-file context blocks (callers / callees) for ``func_name``
        via the vector store. Used only for LLM escalation, not for SLM scoring."""
        blocks: List[str] = []
        if not func_name:
            return blocks
        try:
            flow = self.vector_store.get_explicit_flow_context(func_name)
            for block in flow:
                blocks.append(
                    f"\n# Context: {block['relationship']} | {block['file_path']} | "
                    f"{block['function_name']}\n{block['source_code']}"
                )
        except Exception:
            pass
        return blocks

    def slm_inference_batch(self, code_snippets: List[str]) -> List[float]:
        """
        Performs batched inference on multiple code snippets using the fine-tuned SLM.
        This is the optimized version that replaces the slow per-snippet inference.

        Args:
            code_snippets (List[str]): List of clean, complete function source code strings.

        Returns:
            List[float]: List of risk probabilities (0.0 to 1.0) for each snippet.
        """
        probabilities = self.model_manager.get_classifier_inference(code_snippets)
        return [round(p, 4) for p in probabilities]

    def evaluate_gate(self, semgrep_severity: str, slm_score: float) -> dict:
        """
        Combines isolated static and neural telemetry signals into a composite risk score.
        
        The risk score is computed as a weighted linear combination of the static
        severity and the SLM probability. The escalation decision uses the tuned
        linear gate without override (the override was evaluated as an ablation and
        removed from production because it had zero measurable effect on metrics).

        Args:
            semgrep_severity (str): The raw diagnostic classification level from the static engine.
            slm_score (float): Independent soft probability generated by the neural network block.

        Returns:
            dict: Structured results matching pipeline schema requirements.
        """
        # Map configured weightings from the environment manifest config
        severity_map = self.config.get("semgrep_severity_map", {})
        severity_weight = severity_map.get(semgrep_severity, 0.0)
        w1 = self.config["gate_parameters"]["weight_static"]
        w2 = self.config["gate_parameters"]["weight_slm"]

        risk_score = (w1 * severity_weight) + (w2 * slm_score)

        gate_params = {
            "weight_static": w1,
            "weight_slm": w2,
            "escalation_threshold": self.config["gate_parameters"]["escalation_threshold"],
            "override_enabled": False,
        }
        forced_escalation = linear_weighted_gate(severity_weight, slm_score, gate_params)

        return {
            "risk_score": round(risk_score, 4),
            "escalate": forced_escalation,
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

        Optimization Strategy:
        1. Extract ALL code snippets with full context first (no model calls yet).
        2. Run ONE batched SLM inference on all enriched snippets (massive speed gain).
        3. Apply gating logic.
        4. Write unified triage report.

        Now: Each snippet includes:
            - Original function code
            - Upstream callers (functions that call this one)
            - Downstream callees (functions this one calls)
        """
        semgrep_filename = self.config["paths"]["semgrep_output"]
        target_path = os.path.join(self.artifact_dir, semgrep_filename)
        semgrep_workspace_path = os.path.join(self.workspace_path, semgrep_filename)

        if not os.path.exists(semgrep_workspace_path) and os.path.exists(semgrep_filename):
            semgrep_workspace_path = semgrep_filename

        # Fallback: the comparative evaluator writes Semgrep results under
        # the evaluations subtree (e.g. evaluation_runs/.../). Reuse the
        # most recent one so a Stage-2 run can consume evaluator output
        # without re-running Semgrep, and so the triage the Stage-3
        # agent reads is populated with the produced findings. Copy (do
        # not move) to avoid disturbing the evaluator's own artifacts.
        if not os.path.exists(semgrep_workspace_path):
            found = self._find_semgrep_results(semgrep_filename)
            if found:
                os.makedirs(self.artifact_dir, exist_ok=True)
                shutil.copy2(found, target_path)
                semgrep_workspace_path = target_path

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

        # Read chunking / aggregation config for the SLM (falling back to
        # sensible defaults when the block is absent in config.json).
        slm_cfg = self.config.get("slm_inference", {})
        chunk_max_lines = slm_cfg.get("chunk_max_lines", 64)
        chunk_overlap = slm_cfg.get("chunk_overlap", 8)
        chunk_min_code_lines = slm_cfg.get("min_code_lines", 2)
        aggregation = slm_cfg.get("aggregation", "max")
        top_chunks = slm_cfg.get("top_chunks_for_llm", 5)

        print(f"[*] Extracting {len(findings)} functions (SLM scored on uniform chunks)...")
        self.vector_store = LocalVectorStore(self._config_path, self.workspace_path)

        snippets = []
        finding_metadata = []
        for finding in findings:
            file_target_path = finding.get("path", "")
            start_line = finding.get("start", {}).get("line", 1)
            end_line = finding.get("end", {}).get("line", 1)
            severity = finding.get("extra", {}).get("severity", "WARNING")

            resolved_path, source = self._resolve_source(file_target_path)
            function_code, func_start, func_end = self._function_block(
                resolved_path, source, start_line, end_line
            )
            if not function_code:
                continue

            # Function name (simple def/async def heuristic) for cross-file lookup.
            func_name = ""
            for line in function_code.splitlines():
                s = line.strip()
                if s.startswith("def ") or s.startswith("async def "):
                    func_name = s.split("(", 1)[0].replace("def ", "").replace("async def ", "").strip()
                    break

            # Cross-file context is for the LLM ONLY — kept out of the SLM input.
            context_blocks = self._collect_context(func_name)

            # SLM sees function + module imports (matches the training format);
            # cross-file context is not baked in.
            slm_input = self._slm_input(source, func_start, func_end)

            snippets.append(slm_input)
            finding_metadata.append({
                "path": file_target_path,
                "start_line": start_line,
                "end_line": end_line,
                "severity": severity,
                "function_name": func_name,
                "function_code": function_code,
                "context_blocks": context_blocks,
            })

        print(f"[*] Running chunked SLM inference on {len(snippets)} functions...")
        chunk_results = self.model_manager.get_classifier_chunk_scores(
            snippets,
            chunk_max_lines=chunk_max_lines,
            chunk_overlap=chunk_overlap,
            min_code_lines=chunk_min_code_lines,
            aggregation=aggregation,
        )

        print(f"[*] Applying risk gate logic to {len(findings)} findings...")
        for metadata, result in zip(finding_metadata, chunk_results):
            severity = metadata["severity"]
            slm_score = result["score"]
            gate_result = self.evaluate_gate(severity, slm_score)
            if gate_result["escalate"]:
                overall_escalate = True

            # On escalation, hand the LLM the suspicious chunks (highest SLM
            # probability) plus the cross-file context — the "cross-context
            # argumentation" the Stage-3 agent reasons over.
            suspicious = sorted(
                result["chunks"], key=lambda c: c["prob"], reverse=True
            )[:top_chunks]

            finding_reports.append({
                "evaluated_file": metadata["path"],
                "function_name": metadata["function_name"],
                "code_snippet": metadata["function_code"],
                "metrics": {
                    "semgrep_severity_score": gate_result["metrics"]["static_severity_weight"],
                    "slm_threat_probability": gate_result["metrics"]["slm_probability_score"],
                    "slm_chunk_scores": [c["prob"] for c in result["chunks"]],
                    "calculated_combined_risk": gate_result["risk_score"],
                },
                "escalate": gate_result["escalate"],
                "suspicious_chunks": suspicious if gate_result["escalate"] else [],
                "cross_file_context": (
                    "".join(metadata["context_blocks"]) if gate_result["escalate"] else ""
                ),
            })

        triage_report = {
            "run_id": self.artifact_dir.split("/")[-1],  # Use dynamic run_id
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
        print(f"[+] Total findings: {len(findings)}, Escalated: {sum(1 for f in finding_reports if f['escalate'])}")


    def _find_semgrep_results(self, semgrep_filename: str):
        """Locate a Semgrep results file produced by the comparative
        evaluator under the evaluations subtree, returning the most recent
        match (or None). Used as a fallback so a Stage-2 run can
        consume evaluator output without re-running Semgrep.
        """
        eval_sub = self.config["paths"].get("evaluations_subdir", "evaluation_runs")
        eval_root = os.path.join(self.workspace_path, eval_sub)
        if not os.path.isdir(eval_root):
            return None
        candidate = None
        candidate_mtime = -1.0
        for root, _dirs, files in os.walk(eval_root):
            if semgrep_filename in files:
                p = os.path.join(root, semgrep_filename)
                m = os.path.getmtime(p)
                if m > candidate_mtime:
                    candidate, candidate_mtime = p, m
        return candidate

def _cli_main() -> None:
    """Command-line entry point so the orchestrator can be run as
    `python src/triage_orchestrator.py --workspace <dir> --config <cfg>`.

    NOTE: directory exclusion is applied at the Stage 1 Semgrep layer
    (see the CI workflow's `semgrep --exclude ...` invocation), not
    here — this flag is accepted only so the documented/CI command
    line keeps working.
    """
    import argparse
    parser = argparse.ArgumentParser(description="CEVuD Stage 2: Local Triage & Gating")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--workspace", default=".", help="Path to the target workspace/codebase")
    parser.add_argument(
        "--exclude-dirs", default="",
        help="Comma-separated dirs excluded at the Semgrep layer (ignored here).",
    )
    args = parser.parse_args()
    orchestrator = TriageOrchestrator(config_path=args.config, workspace_path=args.workspace)
    orchestrator.process_pipeline()


if __name__ == "__main__":
    _cli_main()
