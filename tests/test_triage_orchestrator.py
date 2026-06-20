import os
import sys
import json
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from src.triage_orchestrator import TriageOrchestrator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory and clean it up after testing."""
    workspace_dir = tempfile.mkdtemp()
    yield workspace_dir
    shutil.rmtree(workspace_dir)

@pytest.fixture
def mock_config(temp_workspace):
    """Generate a mock configuration JSON file inside the temporary workspace."""
    config = {
        "gate_parameters": {
            "weight_static": 0.3,
            "weight_slm": 0.7,
            "escalation_threshold": 0.55
        },
        "paths": {
            "workspace_root": "workspace_storage",
            "artifacts_subdir": "artifacts",
            "evaluations_subdir": "evaluation_runs",
            "semgrep_output": "semgrep_results.json",
            "triage_report": "stage1_2_triage.json",
            "vector_db_dir": "workspace_storage/codebase_vectors"
        },
        "semgrep_severity_map": {
            "ERROR": 1.0,
            "WARNING": 0.7,
            "INFO": 0.3
        },
        "stage3_llm": {
            "provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.0
        }
    }
    config_path = os.path.join(temp_workspace, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path, config

@patch("src.triage_orchestrator.AutoTokenizer")
@patch("src.triage_orchestrator.AutoModelForSequenceClassification")
def test_evaluate_gate_logic(mock_model_class, mock_tokenizer_class, mock_config, temp_workspace):
    """Verify combined risk calculation and escalation decisions on static weights."""
    config_path, config = mock_config
    
    # Initialize mock tokenizer & model objects
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    
    orchestrator = TriageOrchestrator(config_path, workspace_path=temp_workspace)
    
    # ERROR (weight 1.0) and high SLM score -> Score: (0.3 * 1.0) + (0.7 * 0.8) = 0.86 (Escalates)
    decision1 = orchestrator.evaluate_gate("ERROR", 0.8)
    assert decision1["risk_score"] == 0.86
    assert decision1["escalate"] is True

    # ERROR (weight 1.0) and low SLM score -> Score: (0.3 * 1.0) + (0.7 * 0.2) = 0.44 (Does not escalate)
    decision2 = orchestrator.evaluate_gate("ERROR", 0.2)
    assert decision2["risk_score"] == 0.44
    assert decision2["escalate"] is False

    # NONE (fail-safe mode, static weight 0) and high SLM score -> Score: (0 * 0) + (0.7 * 0.9) = 0.63 (Escalates)
    decision3 = orchestrator.evaluate_gate("NONE", 0.9)
    assert decision3["risk_score"] == 0.63
    assert decision3["escalate"] is True

@patch("src.triage_orchestrator.AutoTokenizer")
@patch("src.triage_orchestrator.AutoModelForSequenceClassification")
def test_evaluation_on_gold_standard_dataset(mock_model_class, mock_tokenizer_class, mock_config, temp_workspace):
    """Evaluate gating decisions on the gold_standard dataset using mocked SLM probabilities."""
    config_path, config = mock_config
    
    # Setup mock tokenizer and model objects
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    
    orchestrator = TriageOrchestrator(config_path, workspace_path=temp_workspace)
    
    # Load gold standard data
    gold_std_path = os.path.join(os.path.dirname(__file__), "data", "gold_standard.json")
    with open(gold_std_path, "r") as f:
        test_cases = json.load(f)
        
    evaluation_records = []
    
    for case in test_cases:
        severity = case.get("severity", "NONE")
        is_vulnerable = case["is_vulnerable"]
        
        # Simulate local SLM predictions:
        # If is_vulnerable = 1, yield high risk prob (e.g. 0.85), else yield low risk (e.g. 0.12)
        mock_slm_score = 0.85 if is_vulnerable == 1 else 0.12
        
        gate_decision = orchestrator.evaluate_gate(severity, mock_slm_score)
        
        record = {
            "function_name": case["function_name"],
            "is_vulnerable": is_vulnerable,
            "semgrep_severity": severity,
            "mocked_slm_score": mock_slm_score,
            "calculated_risk": gate_decision["risk_score"],
            "escalated": gate_decision["escalate"]
        }
        evaluation_records.append(record)
        
        # In a real environment, vulnerable cases must escalate, secure cases must not
        if is_vulnerable == 1:
            # Gating calculation score should exceed escalation threshold (0.55)
            assert gate_decision["escalate"] is True
        else:
            assert gate_decision["escalate"] is False

    # Store evaluation outcomes to tests audit dir for record keeping
    audit_dir = os.path.join(temp_workspace, "workspace_storage", "test_runs")
    os.makedirs(audit_dir, exist_ok=True)
    audit_report_path = os.path.join(audit_dir, "gold_standard_triage_audit.json")
    
    with open(audit_report_path, "w") as f:
        json.dump(evaluation_records, f, indent=2)
        
    assert os.path.exists(audit_report_path)
