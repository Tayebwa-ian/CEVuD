"""
Unit Tests: TriageOrchestrator
==============================
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from triage_orchestrator import TriageOrchestrator

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_orchestrator_initialization_flow(mock_tokenizer, mock_model, unit_test_config):
    """Validates parameter assignment and directory construction on initialization."""
    orchestrator = TriageOrchestrator(config_path=unit_test_config, workspace_path="/tmp/fake_work")
    assert orchestrator.workspace_path == "/tmp/fake_work"
    assert "test_artifacts" in orchestrator.artifact_dir

@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_evaluate_gate_risk_math(mock_tokenizer, mock_model, unit_test_config):
    """Tests the math inside the risk score equation."""
    orchestrator = TriageOrchestrator(config_path=unit_test_config)
    decision = orchestrator.evaluate_gate(semgrep_severity="ERROR", slm_score=0.8)
    assert decision["risk_score"] == 0.88
    assert decision["escalate"] is True
