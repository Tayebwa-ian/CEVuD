"""Stage 3 Context Reasoning Engine running via DeepAgent task decomposition loops."""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from deepagents import create_deep_agent
from .llm_factory import LLMFactory
from .vector_store import LocalVectorStore

class DeepAppSecAgent:
    """Decomposes massive file contents into structured tasks for secure resolution."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.vector_store = LocalVectorStore(config_path)
        
        # Initialize CodeBERT for feature extraction (embeddings)
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def _get_context_tool(self, query_text: str):
        """
        Tool for the DeepAgent to retrieve semantically relevant code snippets.
        Generates real vectors using CodeBERT to query the LocalVectorStore.
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
        """Assembles complex structural code inputs and passes them to a deep task-handling agent."""
        triage_file = self.config["paths"]["triage_report"]

        with open(triage_file, "r") as f:
            triage_data = json.load(f)

        if not triage_data.get("gate_decision", {}).get("escalate_to_llm", False):
            print("[+] Gating threshold conditions not breached. Halting agent execution.")
            return

        # Initialize the agnostic model instantiation wrapper
        llm_cfg = self.config["stage3_llm"]
        base_model = LLMFactory.get_model(
            provider=llm_cfg["provider"],
            model_name=llm_cfg["model_name"],
            temperature=llm_cfg["temperature"]
        )

        # Initialize the advanced task-breaking agent harness
        print("[*] Instantiating DeepAgent workspace environment to handle long context slices...")
        security_agent = create_deep_agent(
            model=base_model,
            tools=[], # Can be augmented with custom file-reading/repo tools
            # Integrated the LocalVectorStore as a tool for the agent
            tools=[self._get_context_tool], 
            system_prompt=(
                "You are an elite Application Security Vulnerability Engineer. "
                "Your objective is to systematically review large codebase diff slices, "
                "break down the structural interaction paths into explicit tasks using your internal todos, "
                "and assemble a clean, definitive Remediation Report."
                "and assemble a clean, definitive Remediation Report. Use your tools to find related code context."
            )
        )

        # Extract target vulnerability metadata parameters
        semgrep_raw_path = self.config["paths"]["semgrep_output"]
        with open(semgrep_raw_path, "r") as f:
            raw_lines = json.load(f).get("results", [])[0]["extra"]["lines"]

        # Build task execution payload query
        execution_query = {
            "messages": (
                f"Task: Analyze this structural code change for a security flaw:\n\n"
                f"Code Snippet:\n{raw_lines}\n\n"
                f"Instructions: Use your internal planning tools to systematically trace the data lineage, "
                f"identify potential edge cases across files, and output a markdown report with sections for "
                f"Source/Sink Lineage, Exploit PoC Steps, and a Remediation Patch."
            )
        }

        # Run model task interaction loops
        response = security_agent.invoke(execution_query)

        # Create systematic runtime artifact save directory paths
        run_id = triage_data.get("run_id", "fallback_run")
        artifact_dir = f"./workspace_storage/artifacts/{run_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        # Save output documents to artifact vault paths
        report_path = os.path.join(artifact_dir, "remediation_dossier.md")
        with open(report_path, "w") as out:
            out.write(response["messages"][-1].content)

        print(f"[+] Systematic artifact report archived securely inside: {report_path}")

if __name__ == "__main__":
    orchestrator_agent = DeepAppSecAgent("config.json")
    orchestrator_agent.execute_deep_analysis()
