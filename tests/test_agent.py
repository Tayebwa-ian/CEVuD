import os
import sys
import json
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from src.agent import DeepAppSecAgent

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

@patch("src.agent.AutoTokenizer")
@patch("src.agent.AutoModel")
def test_agent_initialization(mock_model_class, mock_tokenizer_class, mock_config, temp_workspace):
    """Verify standard initialization of the DeepAppSecAgent client and path binding."""
    config_path, config = mock_config
    
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    
    agent = DeepAppSecAgent(config_path, workspace_path=temp_workspace)
    
    assert agent.workspace_path == temp_workspace
    assert agent.artifact_dir.startswith(temp_workspace)
    assert os.path.exists(agent.artifact_dir)
