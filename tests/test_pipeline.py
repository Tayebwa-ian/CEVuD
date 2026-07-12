"""
End-to-End Live Pipeline Integration Regression Suite
======================================================
Only runs if --run-e2e flag is passed.
Uses real models and real files — stores artifacts persistently for audit.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_ingest import IngestManager
from evaluate_pipeline import PipelineEvaluator
from triage_orchestrator import TriageOrchestrator
from agent import DeepAppSecAgent


def pytest_addoption(parser):
    # NOTE: --run-e2e is actually registered in tests/conftest.py so the
    # skip-gating hook applies. This stub is kept only for backward-compat
    # if the module is imported standalone.
    pass

@pytest.fixture(scope="module")
def e2e_workspace():
    """Create persistent workspace for E2E tests — only if enabled."""
    
    # Use a fixed, persistent path under tests/
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace_storage_e2e"))
    os.makedirs(workspace_root, exist_ok=True)
    
    # Clear old artifacts to avoid contamination
    artifacts = os.path.join(workspace_root, "artifacts")
    if os.path.exists(artifacts):
        shutil.rmtree(artifacts)
    
    return workspace_root

@pytest.fixture(scope="module")
def e2e_config(e2e_workspace):
    """Generate config pointing to E2E workspace."""
    config = {
        "paths": {
            "workspace_root": e2e_workspace,
            "vector_db_dir": os.path.join(e2e_workspace, "codebase_vectors"),
            "evaluations_subdir": os.path.join(e2e_workspace, "evaluation_runs"),
            "artifacts_subdir": "artifacts",
            "semgrep_output": "semgrep_results.json",
            "triage_report": "stage1_2_triage.json"
        },
        "gate_parameters": {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "slm_override_threshold": 0.90
        },
        "semgrep_severity_map": {
            "INFO": 0.3,
            "WARNING": 0.7,
            "ERROR": 1.0,
            "NONE": 0.0
        },
        "stage3_llm": {
            "provider": "unipassau",
            "model_name": "qwen3-next-80b-a3b-instruct",
            "temperature": 0.1
        }
    }
    config_path = os.path.join(e2e_workspace, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


@pytest.fixture(scope="module")
def gold_standard_path():
    """Return path to gold standard test data."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "gold_standard.json"))

def test_e2e_dataset_ingestion(e2e_workspace, e2e_config, gold_standard_path):
    """Phase 1: Ingest gold standard into vector DB."""
    
    manager = IngestManager(e2e_config)
    manager.ingest_benchmark_json(gold_standard_path)
    
    db_path = os.path.join(e2e_workspace, "codebase_vectors", "codebase_context.db")
    assert os.path.exists(db_path), "Vector DB should be created"
    
    # Should have seeded 24 entries (from gold_standard.json)
    from vector_store import LocalVectorStore
    db = LocalVectorStore(e2e_config)
    with db.conn:
        count = db.conn.execute("SELECT COUNT(*) FROM codebase_embeddings").fetchone()[0]
    assert count == 24, "Should have 24 benchmark entries"

def test_e2e_stage1_and_stage2(e2e_workspace, e2e_config, gold_standard_path):
    """Phase 2: Run live Semgrep + TriageOrchestrator."""
    
    # 1. Run evaluator to generate gold-standard files
    evaluator = PipelineEvaluator(e2e_config)
    evaluator.run_evaluation(gold_standard_path)
    
    # 2. Verify evaluation artifacts exist
    eval_dir = evaluator.eval_dir
    assert os.path.exists(os.path.join(eval_dir, "summary.json"))
    assert os.path.exists(os.path.join(eval_dir, "detailed_findings.json"))
    assert os.path.exists(os.path.join(eval_dir, "confusion_matrix.png"))
    
    # 3. The evaluator already ran TriageOrchestrator live (Step 2 of
    #    run_evaluation) on the benchmark dir that actually holds the
    #    case_*.py source files, and mirrored its artifacts into the
    #    workspace artifact dir (artifacts/run_local-dev-run/). Do NOT
    #    re-run process_pipeline() here: run_evaluation's `finally`
    #    block already rmtree'd that benchmark dir, so a second pass
    #    would find no source files, skip every finding, and
    #    overwrite the good triage with an empty one -- which is
    #    exactly what Stage-3 then reads and halts on.
    run_id = os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or "local-dev-run"
    if not run_id.startswith("run_"):
        run_id = f"run_{run_id}"
    triage_path = os.path.join(
        e2e_workspace, "artifacts", run_id,
        e2e_config and json.load(open(e2e_config))["paths"]["triage_report"]
    )
    assert os.path.exists(triage_path), "Stage2 triage report must be generated"

    with open(triage_path, "r") as f:
        triage = json.load(f)

    assert triage["gate_decision"]["escalate_to_llm"] is True, "At least one finding should escalate"
    assert len(triage["findings"]) > 0

def test_e2e_stage3_remediation(e2e_workspace, e2e_config, gold_standard_path):
    """Phase 3: Run DeepAppSecAgent with mocked LLM."""
    # Mock the LLM to avoid external calls
    with patch("agent.LLMFactory.get_model") as mock_get_model, \
         patch("agent.create_deep_agent") as mock_agent_factory:
        
        # Mock LLM response
        mock_llm = MagicMock()
        mock_get_model.return_value = mock_llm
        
        # Mock agent response
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="# Remediation Dossier\n\n## Vulnerability Analysis\n\nTest response.\n")]
        }
        mock_agent_factory.return_value = mock_agent
        
        # Run agent
        agent = DeepAppSecAgent(config_path=e2e_config, workspace_path=e2e_workspace)
        agent.execute_deep_analysis()
        
        # Verify output
        dossier_path = os.path.join(agent.artifact_dir, "remediation_dossier.md")
        assert os.path.exists(dossier_path), "Remediation dossier must be saved"
        
        with open(dossier_path, "r") as f:
            content = f.read()
        
        assert "Vulnerability Analysis" in content
