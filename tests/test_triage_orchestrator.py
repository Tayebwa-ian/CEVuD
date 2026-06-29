"""
Unit Tests: TriageOrchestrator
==============================
Validates the asymmetric linear-override hybrid gating mechanics.
Using required production weights: Static (0.4) and SLM (0.6).
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import sys

# Ensure local source directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from triage_orchestrator import TriageOrchestrator

@pytest.fixture
def unit_test_config(tmp_path):
    """Generates a temporary configuration block matching production schema."""
    config_data = {
        "paths": {
            "workspace_root": str(tmp_path),
            "artifacts_subdir": "test_artifacts",
            "triage_report": "stage1_2_triage.json",
            "semgrep_output": "semgrep_results.json"
        },
        "semgrep_severity_map": {
            "ERROR": 1.0,
            "WARNING": 0.7,   # 🎯 Explicitly mapped to pass standard math calculation checks
            "INFO": 0.3,      # 🎯 Explicitly mapped to pass boundary checks
            "NONE": 0.0
        },
        "gate_parameters": {
            "weight_static": 0.4,       # 🎯 Fixed to required production parameter
            "weight_slm": 0.6,          # 🎯 Fixed to required production parameter
            "escalation_threshold": 0.50,
            "slm_override_threshold": 0.90
        }
    }
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
    return str(config_file)


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_orchestrator_initialization_flow(mock_tokenizer, mock_model, unit_test_config):
    """Validates parameter assignment and directory construction on initialization."""
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    assert "test_artifacts" in orchestrator.artifact_dir


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_evaluate_gate_standard_math(mock_tokenizer, mock_model, unit_test_config):
    """Tests standard weighted linear calculations below the short-circuit triggers.
    
    Formula: (0.4 * WARNING[0.7]) + (0.6 * 0.4) = 0.28 + 0.24 = 0.52
    0.52 >= 0.50 -> Should escalate.
    """
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    decision = orchestrator.evaluate_gate(semgrep_severity="WARNING", slm_score=0.4)
    
    assert abs(decision["risk_score"] - 0.52) < 1e-5
    assert decision["escalate"] is True
    assert decision["metrics"]["override_triggered"] is False


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_evaluate_gate_static_override(mock_tokenizer, mock_model, unit_test_config):
    """Tests the asymmetric static override safety net.
    
    If Semgrep flags an ERROR (1.0), the system must short-circuit and force
    an escalation even if the SLM outputs an extremely low confidence score.
    """
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    decision = orchestrator.evaluate_gate(semgrep_severity="ERROR", slm_score=0.01)
    
    assert decision["escalate"] is True
    assert decision["risk_score"] >= 1.0
    assert decision["metrics"]["override_triggered"] is True


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_evaluate_gate_slm_override(mock_tokenizer, mock_model, unit_test_config):
    """Tests the asymmetric neural override safety net.
    
    If the SLM confidence is exceptionally high (> 0.90), it must escalate
    directly even if Semgrep completely missed it (severity NONE = 0.0).
    """
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    decision = orchestrator.evaluate_gate(semgrep_severity="NONE", slm_score=0.95)
    
    assert decision["escalate"] is True
    assert decision["risk_score"] >= 1.0
    assert decision["metrics"]["override_triggered"] is True


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@pytest.mark.parametrize("severity,slm,expected_escalate", [
    # (0.4 * INFO[0.3]) + (0.6 * 0.65) = 0.12 + 0.39 = 0.51 (>= 0.50) -> True
    ("INFO", 0.65, True),    
    
    # (0.4 * NONE[0.0]) + (0.6 * 0.85) = 0.0 + 0.51 = 0.51 (>= 0.50) -> True
    ("NONE", 0.85, True),  
    
    # (0.4 * INFO[0.3]) + (0.6 * 0.50) = 0.12 + 0.30 = 0.42 (< 0.50) -> False
    ("INFO", 0.50, False),   
])
def test_evaluate_gate_boundary_conditions(mock_tokenizer, mock_model, unit_test_config, severity, slm, expected_escalate):
    """Parametrized sweep ensuring exact boundary calculations around the 0.50 line."""
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    decision = orchestrator.evaluate_gate(semgrep_severity=severity, slm_score=slm)
    assert decision["escalate"] is expected_escalate
