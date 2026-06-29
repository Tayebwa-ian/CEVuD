"""
End-to-End Live Pipeline Integration Regression Suite
======================================================
"""

import os
import json
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_ingest import IngestManager
from evaluate_pipeline import PipelineEvaluator
from triage_orchestrator import TriageOrchestrator
from agent import DeepAppSecAgent

# ---------------------------------------------------------------------------
# Setup Phase: Physical Code Generation from Gold Standard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def seed_physical_codebase_from_gold_data(real_test_config, gold_standard_path):
    """
    Parses the gold standard data array to build real python source files
    inside tests/workspace_storage/ before any tests execute.

    Returns an **absolute** path to the workspace root so that downstream
    fixtures (Semgrep invocations, TriageOrchestrator) all resolve paths
    consistently regardless of which directory pytest was launched from.
    """
    with open(gold_standard_path, "r") as f:
        dataset = json.load(f)

    # Resolve workspace_root as an absolute path anchored to this file's location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.join(repo_root, "tests", "workspace_storage")
    os.makedirs(workspace_root, exist_ok=True)

    # Track written unique code signatures per file paths
    file_contents = {}
    for entry in dataset:
        path = os.path.join(workspace_root, entry["file_path"])
        if path not in file_contents:
            file_contents[path] = []
        file_contents[path].append(entry["source_code"])

    # Physically save the scripts to disk
    for path, code_blocks in file_contents.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as source_file:
            source_file.write("\n\n".join(code_blocks))

    return workspace_root

# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

def test_live_dataset_ingestion_and_vector_db(real_test_config, gold_standard_path):
    """Phase 1: Ingests benchmark rules metadata into the persistent SQLite context database."""
    manager = IngestManager(real_test_config)
    manager.ingest_benchmark_json(gold_standard_path)
    
    assert os.path.exists(manager.db.db_path)
    # The DB should live inside the test workspace directory
    assert "tests" in manager.db.db_path and "workspace_storage" in manager.db.db_path


def test_live_stage1_and_stage2_real_workflow(real_test_config, gold_standard_path, seed_physical_codebase_from_gold_data):
    """
    Phase 2: Runs the actual, un-mocked evaluation loop.
    Triggers Semgrep, runs the fine-tuned Small Language Model (SLM),
    plots graphs, and compiles the real multi-entry triage json report.
    """
    workspace_root = seed_physical_codebase_from_gold_data

    # 1. Trigger live pipeline analyzer execution
    evaluator = PipelineEvaluator(real_test_config)
    evaluator.run_evaluation(gold_standard_path)
    
    # Assert evaluation folder structure and all generated persistent artifacts exist
    assert os.path.isdir(evaluator.eval_dir)
    assert os.path.exists(os.path.join(evaluator.eval_dir, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(evaluator.eval_dir, "risk_distribution.png"))
    assert os.path.exists(os.path.join(evaluator.eval_dir, "category_performance.png"))
    assert os.path.exists(os.path.join(evaluator.eval_dir, "summary.json"))
    assert os.path.exists(os.path.join(evaluator.eval_dir, "detailed_findings.json"))

    # Run Semgrep on the newly generated codebase to produce the required semgrep_results.json
    import subprocess
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    custom_rules_path = os.path.join(base_dir, "semgrep_rules", "custom_appsec_rules.yaml")
    semgrep_output_path = os.path.join(workspace_root, "semgrep_results.json")
    
    # Find all Python files recursively under the generated workspace_root
    py_files = []
    for root, _, files in os.walk(workspace_root):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    cmd = [
        "semgrep",
        "--config", "p/python",
        "--config", custom_rules_path,
        "--no-git-ignore",
        "--json",
        "--output", semgrep_output_path,
    ] + py_files
    subprocess.run(cmd, capture_output=True, text=True)

    # 2. Run the orchestrator to aggregate and gate the results
    orchestrator = TriageOrchestrator(config_path=real_test_config, workspace_path=workspace_root)
    orchestrator.process_pipeline()

    # 3. Verify that the generated triage report captures the full multi-function codebase
    triage_path = os.path.join(orchestrator.artifact_dir, orchestrator.config["paths"]["triage_report"])
    assert os.path.exists(triage_path), f"Expected triage report file missing at {triage_path}"

    with open(triage_path, "r") as f:
        triage_report = json.load(f)
    
    # 🎯 Print the absolute, exact location of the file Python just read:
    print(f"\n[FOUND IT!] Stage 2 report is physically located at: {os.path.abspath(triage_path)}")
        
    # Confirm it records multiple true generated findings from the codebase scan
    assert isinstance(triage_report["findings"], list)
    assert len(triage_report["findings"]) > 1, "Triage report must record multiple findings from your generated code files."
    
    # Validate real structural metrics are saved
    first_finding = triage_report["findings"][0]
    assert "evaluated_file" in first_finding
    assert "calculated_combined_risk" in first_finding["metrics"]


@patch("agent.create_deep_agent")
@patch("llm_factory.LLMFactory.get_model")
def test_stage3_remediation_handling_with_live_triage_report(mock_get_model, mock_agent_factory, real_test_config, seed_physical_codebase_from_gold_data):
    """
    Phase 3: Automatically feeds the un-mocked triage data compiled in Phase 2 into Stage 3.
    Mocks only the remote model connection to keep the test local and fast.
    """
    workspace_root = seed_physical_codebase_from_gold_data

    # Configure a safe model stub so internal validation comparisons don't crash
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Stubbed LLM Patch Response Blueprint Context")
    mock_get_model.return_value = mock_llm
    
    agent = DeepAppSecAgent(real_test_config, workspace_path=workspace_root)
    triage_path = os.path.join(agent.artifact_dir, agent.config["paths"]["triage_report"])
    
    # This assertion ensures that Stage 3 reads the genuine report created by Stage 1 & 2
    assert os.path.exists(triage_path), "Pipeline triage data from the actual Stage 1 and 2 run must exist."

    with open(triage_path, "r") as f:
        live_data = json.load(f)
    assert len(live_data["findings"]) > 1

    # Stub out the agent runner factory to block raw outbound API calls during test verification
    mock_agent_instance = MagicMock()
    mock_agent_instance.invoke.return_value = {
        "messages": [MagicMock(content="# Automated Vulnerability Remediation Dossier Output\nSuccessfully generated.")]
    }
    mock_agent_factory.return_value = mock_agent_instance
    
    # Execute the deep analysis over the real findings ledger
    agent.execute_deep_analysis()
    
    # Assert the final remediation markdown dossier artifact is saved to disk
    expected_dossier_path = os.path.join(agent.artifact_dir, "remediation_dossier.md")
    assert os.path.exists(expected_dossier_path)
