"""Stage 3 Context Reasoning Engine running via DeepAgent task decomposition loops."""

import json
import os
import torch
from typing import List, Dict, Any, Optional
from deepagents import create_deep_agent
from llm_factory import LLMFactory
from vector_store import LocalVectorStore
from model_manager import ModelManager  # ✅ Centralized model access


class DeepAppSecAgent:
    """
    Stage 3 Synthesis Agent.
    This class handles the most expensive part of the pipeline: using a frontier LLM
    to perform deep task decomposition, cross-file data flow tracing, and remediation 
    generation for high-risk flaws escalated by the Stage 2 gating loop.

    Optimizations:
    - Uses ModelManager singleton to avoid redundant model loads.
    - Caches CodeBERT embeddings per function_name to avoid recomputation.
    - Caches hybrid context lookups (graph + semantic) per function_name.
    - Robust error handling for LLM/tool failures.
    - Clean, deterministic prompt engineering for consistent output.
    """

    def __init__(self, config_path: str, workspace_path: str = None):
        """
        Sets up the agent with vector store access and run-specific artifact paths.

        Args:
            config_path (str): Configuration mapping for models and storage.
            workspace_path (str, optional): Target workspace path under analysis. Defaults to None (current dir).
        """
        # Resolve target codebase environment variables cleanly
        self.workspace_path = os.path.abspath(workspace_path) if workspace_path else os.getcwd()
        
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Fallback for portable local development and testing lookup structures
            root_config = os.path.join(os.path.dirname(__file__), "..", "config.json")
            with open(root_config, "r") as f:
                self.config = json.load(f)

        # Connect the unified data graph database engine
        self.vector_store = LocalVectorStore(config_path, self.workspace_path)

        # Resolve unique execution pass identifier tags to prevent multi-tenant overwrite collisions
        self.run_id = os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or "local-dev-run"
        if not self.run_id.startswith("run_"):
            self.run_id = f"run_{self.run_id}"

        # Resolve workspace_root path configurations dynamically
        ws_root_cfg = self.config["paths"]["workspace_root"]
        if os.path.isabs(ws_root_cfg):
            effective_ws_root = ws_root_cfg
        else:
            effective_ws_root = os.path.join(self.workspace_path, ws_root_cfg)

        # Establish deterministic artifact delivery coordinates
        self.artifact_dir = os.path.join(effective_ws_root, self.config["paths"]["artifacts_subdir"], self.run_id)
        os.makedirs(self.artifact_dir, exist_ok=True)

        # ✅ Use singleton ModelManager for embeddings — no duplicate loads
        self.model_manager = ModelManager()
        self.embedding_tokenizer, self.embedding_model = self.model_manager.get_embedding_model()

        # ✅ In-memory cache for embeddings and context lookups (per run)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._context_cache: Dict[str, str] = {}

    def _generate_mean_pooled_embedding(self, text: str) -> List[float]:
        """
        Generates standard mean-pooled hidden-state vectors from CodeBERT.
        Uses cached results to avoid recomputing embeddings for the same text.

        Args:
            text (str): Source text or symbol identifier to vectorize.

        Returns:
            List[float]: A 768-dimensional list of floating-point numbers.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        inputs = self.embedding_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Apply mean-pooling across temporal dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        self._embedding_cache[text] = embedding
        return embedding

    def _get_context_tool(self, function_name: str) -> str:
        """
        Hybrid routing agent tool. Synthesizes explicit structural dependency lines 
        with standard text vector searches to trace data lineage across file paths.
        Uses caching to avoid repeated queries for the same function_name.

        Args:
            function_name (str): The name of the function/symbol to trace across files.

        Returns:
            str: A formatted string containing call-graph and semantic context blocks.
        """
        if function_name in self._context_cache:
            return self._context_cache[function_name]

        context_str = "--- Hybrid Codebase Lineage Trace ---\n"
        
        # Method A: Core Static Lineage Lookups using graph links
        try:
            structural_flow = self.vector_store.get_explicit_flow_context(function_name)
            if structural_flow:
                context_str += "[Graph Match Results - Explicit Upstream/Downstream Calls]\n"
                for node in structural_flow:
                    context_str += f"[{node['relationship']}] File: {node['file_path']} | Function: {node['function_name']}\n"
                    context_str += f"Code:\n{node['source_code']}\n--------------------\n"
        except Exception as e:
            context_str += f"[⚠️ Graph Lookup Error] {str(e)}\n"

        # Method B: Semantic Neighborhood Evaluation (CodeBERT Vector Matching)
        try:
            query_vector = self._generate_mean_pooled_embedding(function_name)
            semantic_matches = self.vector_store.query_cross_file_context(query_vector, limit=2)
            
            if semantic_matches:
                context_str += "\n[Semantic Proximity Matches - Relevant Shared Variables/Types]\n"
                for item in semantic_matches:
                    context_str += f"File: {item['file_path']} | Function: {item['function_name']} (Similarity: {item['similarity']:.3f})\n"
                    context_str += f"Code:\n{item['source_code']}\n--------------------\n"
        except Exception as e:
            context_str += f"[⚠️ Semantic Lookup Error] {str(e)}\n"

        if not any([
            any("Graph Match Results" in line for line in context_str.splitlines()),
            any("Semantic Proximity Matches" in line for line in context_str.splitlines())
        ]):
            context_str += f"No cross-file lineage connections found for target symbol identifier: '{function_name}'"

        # Cache result for future use
        self._context_cache[function_name] = context_str
        return context_str

    def execute_deep_analysis(self) -> None:
        """
        Orchestrates the Stage 3 workflow.
        Reads the triage gate results and, if escalated, invokes the DeepAgent
        to produce a detailed remediation dossier.
        """
        triage_file = os.path.join(self.artifact_dir, self.config["paths"]["triage_report"])

        if not os.path.exists(triage_file):
            raise FileNotFoundError(f"Upstream Stage 2 triage report ledger missing at target destination: {triage_file}")

        with open(triage_file, "r") as f:
            triage_data = json.load(f)

        # Gate Check: Stop expensive LLM processing immediately if Stage 2 gave a SAFE verdict
        if not triage_data.get("gate_decision", {}).get("escalate_to_llm", False):
            print("[+] Gating threshold conditions not breached. Halting agent execution loop.")
            return
            
        # Isolate entries flagged for human or autonomous code generation review
        escalated_findings = [f for f in triage_data.get("findings", []) if f.get("escalate")]
        if not escalated_findings:
            print("[+] No findings marked for escalation. Skipping LLM analysis.")
            return

        print(f"[*] Escalating {len(escalated_findings)} high-risk findings to DeepAgent engine blocks...")

        # Initialize the underlying frontier reasoning model
        llm_cfg = self.config.get("stage3_llm", {})
        if not llm_cfg:
            raise ValueError("Missing 'stage3_llm' configuration in config.json")

        try:
            base_model = LLMFactory.get_model(
                provider=llm_cfg.get("provider", "openai"),
                model_name=llm_cfg.get("model_name", "gpt-4o"),
                temperature=llm_cfg.get("temperature", 0.1)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM model: {str(e)}")

        # Define an explicit wrapper function to ensure clean tool signatures for the agent framework
        def context_tracing_tool(function_name: str) -> str:
            """Queries the local codebase call-graph data structures and vector context spaces."""
            if not isinstance(function_name, str) or not function_name.strip():
                return "[ERROR] Invalid function name provided to context_tracing_tool."
            return self._get_context_tool(function_name.strip())

        # Initialize the advanced task-breaking agent harness
        print("[*] Instantiating DeepAgent workspace environment...")
        try:
            security_agent = create_deep_agent(
                model=base_model,
                tools=[context_tracing_tool],  # Pass clean explicit function reference wrapper
                system_prompt=(
                    "You are an elite Application Security Vulnerability Engineer. "
                    "Your objective is to systematically review a list of high-risk code findings, "
                    "plan your analysis to cover all of them, break down the structural interaction paths "
                    "into explicit tasks, and consolidate your findings into a single, comprehensive "
                    "Remediation Dossier. For each finding, include sections for 'Vulnerability Analysis', "
                    "'Source/Sink Lineage', 'Exploit Proof-of-Concept Steps', and 'Remediation Patch'. "
                    "Use your tools to query code context for function/symbol names to determine cross-file issues. "
                    "DO NOT generate placeholder text. If context is missing, state it clearly. "
                    "Output MUST be valid Markdown with exactly four sections per finding."
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DeepAgent: {str(e)}")

        # Structure the payload string block cleanly
        findings_input = ""
        for idx, finding in enumerate(escalated_findings):
            # Extract target name safely with clean string fallbacks
            func_name = finding.get("function_name", "unknown_symbol").strip()
            file_path = finding.get("evaluated_file", "unknown_file").strip()
            snippet = finding.get("code_snippet", "").strip()
            risk_score = finding.get("metrics", {}).get("calculated_combined_risk", 0.0)

            if not func_name or not file_path:
                print(f"[!] Skipping malformed finding at index {idx}")
                continue

            findings_input += f"### Finding {idx+1}:\n"
            findings_input += f"Target Function: {func_name}\n"
            findings_input += f"File location: {file_path}\n"
            findings_input += f"Isolated Code Snippet:\n```python\n{snippet}\n```\n"
            findings_input += f"Calculated Combined Risk Index: {risk_score:.3f}\n\n"

        if not findings_input.strip():
            print("[!] No valid findings to analyze. Skipping LLM generation.")
            return

        execution_query = {
            "messages": (
                f"Task: Analyze the following list of high-risk security findings. "
                f"For each finding, provide a detailed analysis including Source/Sink Lineage, "
                f"Exploit Proof-of-Concept Steps, and a Remediation Patch. "
                f"Consolidate all analyses into a single, well-structured Markdown report.\n\n"
                f"High-Risk Findings Context Block:\n{findings_input}"
            )
        }

        # Dispatch execution task loop to the reasoning agent
        try:
            print("[*] Invoking DeepAgent for remediation synthesis...")
            response = security_agent.invoke(execution_query)
            final_dossier = response["messages"][-1].content
        except Exception as e:
            final_dossier = (
                "# 🚨 Remediation Dossier Generation Failed\n\n"
                f"## Error\n\n"
                f"DeepAgent failed to generate output: `{str(e)}`\n\n"
                f"## Action Required\n\n"
                f"Check LLM API key, network connectivity, or model availability.\n"
            )
            print(f"[!] DeepAgent invocation failed: {str(e)}")

        # Save the structured remediation portfolio down to persistent file disk mounts
        report_path = os.path.join(self.artifact_dir, "remediation_dossier.md")
        try:
            with open(report_path, "w", encoding="utf-8") as out:
                out.write(final_dossier)
            print(f"[+] Consolidated remediation dossier archived securely inside: {report_path}")
        except Exception as e:
            print(f"[!] Failed to write remediation dossier: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CEVuD Stage 3 Reasoning Agent Extraction Utility")
    parser.add_argument("--config", default="config.json", help="Path to config.json environment config")
    parser.add_argument("--workspace", default=".", help="Path to the workspace codebase folder")
    
    args = parser.parse_args()
    
    orchestrator_agent = DeepAppSecAgent(
        config_path=args.config,
        workspace_path=args.workspace
    )
    orchestrator_agent.execute_deep_analysis()
