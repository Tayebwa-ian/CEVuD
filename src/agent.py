"""Stage 3 Context Reasoning Engine running via DeepAgent task decomposition loops."""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from deepagents import create_deep_agent
from llm_factory import LLMFactory
from vector_store import LocalVectorStore

class DeepAppSecAgent:
    """
    Stage 3 Synthesis Agent.
    This class handles the most expensive part of the pipeline: using a frontier LLM
    to perform deep task decomposition and remediation generation for high-risk flaws.
    """

    def __init__(self, config_path: str, workspace_path: str = None):
        """
        Sets up the agent with vector store access and run-specific artifact paths.

        Args:
            config_path (str): Configuration mapping for models and storage.
            workspace_path (str, optional): Target workspace path under analysis. Defaults to None (current dir).
        """
        self.workspace_path = os.path.abspath(workspace_path) if workspace_path else os.getcwd()
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Portable configuration lookup
            root_config = os.path.join(os.path.dirname(__file__), "..", "config.json")
            with open(root_config, "r") as f:
                self.config = json.load(f)

        self.vector_store = LocalVectorStore(config_path, self.workspace_path)

        # Resolve the specific run directory for inputs and outputs
        self.run_id = os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or "local-dev-run"
        if not self.run_id.startswith("run_"):
            self.run_id = f"run_{self.run_id}"
            
        self.artifact_dir = os.path.join(self.workspace_path, self.config["paths"]["workspace_root"], self.config["paths"]["artifacts_subdir"], self.run_id)
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        # Initialize CodeBERT for feature extraction (embeddings)
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def _get_context_tool(self, query_text: str):
        """
        Semantic Search tool utilized by the DeepAgent's reasoning loop.
        It generates an embedding for the query and fetches cross-file context.

        Args:
            query_text (str): The natural language or code query from the agent.
        
        Returns:
            str: Formatted context string for the LLM prompt.
        """
        # Generate semantic embedding for the query text
        inputs = self.tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the pooled output or CLS token as the representative 768-dim vector
            # For CodeBERT, the first token ([CLS]) serves as the aggregate representation
            query_embedding = outputs.last_hidden_state[0][0].numpy().tolist()
        
        # Query the SQLite vector store with the generated embedding
        related_code = self.vector_store.query_cross_file_context(query_embedding, limit=3)
        
        if not related_code:
            return "No additional relevant context found in the local vector store."
            
        context_str = "--- Additional Repository Context ---\n"
        for item in related_code:
            context_str += f"File: {item['file_path']} | Function: {item['function_name']}\n"
            context_str += f"Code:\n{item['source_code']}\n"
            context_str += "------------------------------------\n"
        
        return context_str

    def execute_deep_analysis(self) -> None:
        """
        Orchestrates the Stage 3 workflow.
        Reads the triage gate results and, if escalated, invokes the DeepAgent
        to produce a detailed remediation dossier.
        """
        triage_file = os.path.join(self.artifact_dir, self.config["paths"]["triage_report"])

        with open(triage_file, "r") as f:
            triage_data = json.load(f)

        if not triage_data.get("gate_decision", {}).get("escalate_to_llm", False):
            print("[+] Gating threshold conditions not breached. Halting agent execution.")
            return
            
        escalated_findings = [f for f in triage_data.get("findings", []) if f.get("escalate")]
        print(f"[*] Escalating {len(escalated_findings)} high-risk findings to DeepAgent...")

        # Initialize the agnostic model instantiation wrapper (once)
        llm_cfg = self.config["stage3_llm"]
        base_model = LLMFactory.get_model(
            provider=llm_cfg["provider"],
            model_name=llm_cfg["model_name"],
            temperature=llm_cfg["temperature"]
        )

        # Initialize the advanced task-breaking agent harness (once)
        print("[*] Instantiating DeepAgent workspace environment to handle multiple findings...")
        security_agent = create_deep_agent(
            model=base_model,
            tools=[self._get_context_tool],
            system_prompt=(
                "You are an elite Application Security Vulnerability Engineer. "
                "Your objective is to systematically review a list of high-risk code findings, "
                "plan your analysis to cover all of them, break down the structural interaction paths "
                "into explicit tasks, and consolidate your findings into a single, comprehensive "
                "Remediation Dossier. For each finding, include sections for 'Vulnerability Analysis', "
                "'Source/Sink Lineage', 'Exploit PoC Steps', and 'Remediation Patch'. "
                "Use your tools to find related code context when necessary."
            )
        )

        # Prepare the consolidated input for the agent
        findings_input = ""
        for idx, finding in enumerate(escalated_findings):
            findings_input += f"### Finding {idx+1}:\n"
            findings_input += f"File: {finding['evaluated_file']}\n"
            findings_input += f"Code Snippet:\n```python\n{finding['code_snippet']}\n```\n"
            findings_input += f"Calculated Risk: {finding['metrics']['calculated_combined_risk']:.3f}\n\n"

        execution_query = {
            "messages": (
                f"Task: Analyze the following list of high-risk security findings. "
                f"For each finding, provide a detailed analysis including Source/Sink Lineage, "
                f"Exploit Proof-of-Concept Steps, and a Remediation Patch. "
                f"Consolidate all analyses into a single, well-structured Markdown report.\n\n"
                f"High-Risk Findings:\n{findings_input}"
            )
        }

        # Run model task interaction loops (once for all findings)
        response = security_agent.invoke(execution_query)
        final_dossier = response["messages"][-1].content

        # Save consolidated output dossier to artifact vault paths
        report_path = os.path.join(self.artifact_dir, "remediation_dossier.md")
        with open(report_path, "w") as out:
            out.write(final_dossier)

        print(f"[+] Consolidated remediation dossier archived securely inside: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CEVuD Stage 3 Reasoning Agent")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--workspace", default=".", help="Path to the workspace codebase under analysis")
    
    args = parser.parse_args()
    
    orchestrator_agent = DeepAppSecAgent(
        config_path=args.config,
        workspace_path=args.workspace
    )
    orchestrator_agent.execute_deep_analysis()
