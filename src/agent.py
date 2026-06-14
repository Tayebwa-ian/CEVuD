"""Stage 3 Context Reasoning Engine running via DeepAgent task decomposition loops."""

import json
import os
from deepagents import create_deep_agent
from .llm_factory import LLMFactory
from .vector_store import LocalVectorStore

class DeepAppSecAgent:
    """Decomposes massive file contents into structured tasks for secure resolution."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.vector_store = LocalVectorStore(config_path)

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
            system_prompt=(
                "You are an elite Application Security Vulnerability Engineer. "
                "Your objective is to systematically review large codebase diff slices, "
                "break down the structural interaction paths into explicit tasks using your internal todos, "
                "and assemble a clean, definitive Remediation Report."
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
