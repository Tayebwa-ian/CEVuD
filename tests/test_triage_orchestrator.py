import os
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock
from triage_orchestrator import TriageOrchestrator
from model_manager import ModelManager

# Test configuration for triage orchestrator
TEST_CONFIG = {
    "paths": {
        "workspace_root": ".",
        "vector_db_dir": os.path.join(".", "codebase_vectors"),
        "artifacts_subdir": "artifacts",
        "model_cache_dir": "workspace_storage/model_cache",
        "semgrep_output": "semgrep_results.json",
        "triage_report": "stage1_2_triage.json"
    },
    "semgrep_severity_map": {
        "ERROR": 1.0,
        "WARNING": 0.7,
        "INFO": 0.3
    },
    "gate_parameters": {
        "weight_static": 0.4,
        "weight_slm": 0.6,
        "escalation_threshold": 0.52
    }
}

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create required directory structure
        artifacts_dir = os.path.join(temp_dir, "artifacts", "run_local-dev-run")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Create a sample source file
        source_file = os.path.join(temp_dir, "test_file.py")
        with open(source_file, "w") as f:
            f.write('''
def vulnerable_function():
    import os
    os.system("rm -rf /")  # Security issue
    return "hello"

def safe_function():
    return "world"
''')
        
        # Create semgrep output with findings
        semgrep_output = os.path.join(temp_dir, "semgrep_results.json")
        semgrep_data = {
            "results": [
                {
                    "path": "test_file.py",
                    "start": {"line": 2},
                    "end": {"line": 5},
                    "extra": {
                        "severity": "ERROR",
                        "message": "Command injection vulnerability"
                    }
                },
                {
                    "path": "test_file.py",
                    "start": {"line": 7},
                    "end": {"line": 9},
                    "extra": {
                        "severity": "WARNING",
                        "message": "Function does nothing"
                    }
                }
            ]
        }
        
        with open(semgrep_output, "w") as f:
            json.dump(semgrep_data, f)
        
        # Create config file
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(TEST_CONFIG, f)
        
        yield {
            "workspace": temp_dir,
            "config_path": config_path,
            "semgrep_path": semgrep_output,
            "source_file": source_file,
            "artifacts_dir": artifacts_dir
        }

@pytest.fixture
def triage_orchestrator(temp_workspace):
    """Create a TriageOrchestrator instance for testing."""
    return TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )

def test_triage_orchestrator_initialization(temp_workspace):
    """Test that TriageOrchestrator initializes correctly."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    assert orchestrator.config == TEST_CONFIG
    assert orchestrator.workspace_path == temp_workspace["workspace"]
    assert orchestrator.artifact_dir == os.path.join(temp_workspace["workspace"], "artifacts", "run_local-dev-run")
    assert isinstance(orchestrator.model_manager, ModelManager)

def test_extract_source_snippet_function_scope(temp_workspace):
    """Test extracting a complete function from source code."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    # Test extracting the vulnerable function (lines 2-5)
    snippet = orchestrator.extract_source_snippet(
        "test_file.py", 
        start_line=2, 
        end_line=5
    )
    
    assert "def vulnerable_function():" in snippet
    assert "os.system(\"rm -rf /\")" in snippet
    assert "return \"hello\"" in snippet
    assert len(snippet.splitlines()) >= 4

def test_extract_source_snippet_safe_function(temp_workspace):
    """Test extracting a safe function from source code."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    # Test extracting the safe function (lines 7-9)
    snippet = orchestrator.extract_source_snippet(
        "test_file.py", 
        start_line=7, 
        end_line=9
    )
    
    assert "def safe_function():" in snippet
    assert "return \"world\"" in snippet
    assert "os.system" not in snippet

def test_slm_inference_batch_empty_list():
    """Test batch inference with empty list returns empty list."""
    orchestrator = TriageOrchestrator("config.json")
    
    with patch.object(orchestrator.model_manager, 'get_classifier_inference') as mock_inference:
        mock_inference.return_value = []
        result = orchestrator.slm_inference_batch([])
        assert result == []
        mock_inference.assert_called_once_with([])

def test_slm_inference_batch_single_snippet():
    """Test batch inference with a single snippet."""
    orchestrator = TriageOrchestrator("config.json")
    
    with patch.object(orchestrator.model_manager, 'get_classifier_inference') as mock_inference:
        mock_inference.return_value = [0.85]
        result = orchestrator.slm_inference_batch(["def test(): pass"])
        assert result == [0.85]
        mock_inference.assert_called_once_with(["def test(): pass"])

def test_slm_inference_batch_multiple_snippets():
    """Test batch inference with multiple snippets."""
    orchestrator = TriageOrchestrator("config.json")
    
    with patch.object(orchestrator.model_manager, 'get_classifier_inference') as mock_inference:
        mock_inference.return_value = [0.2, 0.7, 0.95]
        snippets = ["def a(): pass", "def b(): pass", "def c(): pass"]
        result = orchestrator.slm_inference_batch(snippets)
        assert result == [0.2, 0.7, 0.95]
        mock_inference.assert_called_once_with(snippets)

def test_evaluate_gate_standard_case():
    """Test standard risk calculation without overrides."""
    orchestrator = TriageOrchestrator("config.json")
    
    # Test case: WARNING severity (0.7) + SLM 0.6 score
    # risk_score = (0.4 * 0.7) + (0.6 * 0.6) = 0.64
    # threshold = 0.52, so should escalate
    result = orchestrator.evaluate_gate("WARNING", 0.6)
    
    assert result["risk_score"] == 0.64
    assert result["escalate"] is True
    assert result["metrics"]["static_severity_weight"] == 0.7
    assert result["metrics"]["slm_probability_score"] == 0.6
    assert result["metrics"]["override_triggered"] is False

def test_evaluate_gate_static_override():
    """Test static severity override (ERROR = 1.0)."""
    orchestrator = TriageOrchestrator("config.json")
    
    # Test case: ERROR severity (1.0) + SLM 0.1 score (low)
    # This should trigger override even though combined score is low
    result = orchestrator.evaluate_gate("ERROR", 0.1)
    
    assert result["risk_score"] == 1.0  # Forced to 1.0 due to override
    assert result["escalate"] is True   # Override triggers escalation
    assert result["metrics"]["static_severity_weight"] == 1.0
    assert result["metrics"]["slm_probability_score"] == 0.1
    assert result["metrics"]["override_triggered"] is True

def test_evaluate_gate_slm_override():
    """Test SLM probability override (> 0.9)."""
    orchestrator = TriageOrchestrator("config.json")
    
    # Test case: WARNING severity (0.7) + SLM 0.95 score (> 0.9)
    # This should trigger override even if combined score is below threshold
    result = orchestrator.evaluate_gate("WARNING", 0.95)
    
    assert result["risk_score"] == 1.0  # Forced to 1.0 due to override
    assert result["escalate"] is True   # Override triggers escalation
    assert result["metrics"]["static_severity_weight"] == 0.7
    assert result["metrics"]["slm_probability_score"] == 0.95
    assert result["metrics"]["override_triggered"] is True

def test_evaluate_gate_combined_override():
    """Test both static and SLM overrides simultaneously."""
    orchestrator = TriageOrchestrator("config.json")
    
    # Test case: ERROR severity (1.0) + SLM 0.95 score (> 0.9)
    # Both overrides should trigger
    result = orchestrator.evaluate_gate("ERROR", 0.95)
    
    assert result["risk_score"] == 1.0
    assert result["escalate"] is True
    assert result["metrics"]["static_severity_weight"] == 1.0
    assert result["metrics"]["slm_probability_score"] == 0.95
    assert result["metrics"]["override_triggered"] is True

def test_process_pipeline_basic_functionality(temp_workspace):
    """Test the full pipeline execution with minimal mocking."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    # Mock the model manager's inference to return predictable values
    with patch.object(orchestrator.model_manager, 'get_classifier_inference') as mock_inference:
        # Simulate SLM scores: high for vulnerable function, low for safe one
        mock_inference.return_value = [0.95, 0.1]
        
        # Run the pipeline
        orchestrator.process_pipeline()
        
        # Verify output file was created
        triage_output = os.path.join(temp_workspace["artifacts_dir"], "stage1_2_triage.json")
        assert os.path.exists(triage_output)
        
        # Load and validate the output
        with open(triage_output, "r") as f:
            output = json.load(f)
        
        # Validate structure
        assert "run_id" in output
        assert "gate_decision" in output
        assert "findings" in output
        assert "status" in output
        
        # Validate escalation triggered (due to ERROR + SLM 0.95 override)
        assert output["gate_decision"]["escalate_to_llm"] is True
        assert output["status"] == "VULNERABLE"
        
        # Validate findings
        assert len(output["findings"]) == 2
        assert output["findings"][0]["escalate"] is True  # Vulnerable function
        assert output["findings"][1]["escalate"] is False  # Safe function

def test_process_pipeline_no_findings(temp_workspace):
    """Test pipeline with empty semgrep results."""
    # Create empty semgrep output
    semgrep_output = os.path.join(temp_workspace["workspace"], "semgrep_results.json")
    with open(semgrep_output, "w") as f:
        json.dump({"results": []}, f)
    
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    # Mock model inference (should not be called)
    with patch.object(orchestrator.model_manager, 'get_classifier_inference') as mock_inference:
        mock_inference.return_value = []
        
        # Run the pipeline
        orchestrator.process_pipeline()
        
        # Verify output file was created
        triage_output = os.path.join(temp_workspace["artifacts_dir"], "stage1_2_triage.json")
        assert os.path.exists(triage_output)
        
        # Load and validate the output
        with open(triage_output, "r") as f:
            output = json.load(f)
        
        # Validate structure
        assert output["gate_decision"]["escalate_to_llm"] is False
        assert output["status"] == "SAFE"
        assert len(output["findings"]) == 0

def test_process_pipeline_missing_semgrep_file(temp_workspace):
    """Test pipeline with missing semgrep output file."""
    # Remove semgrep file
    os.remove(os.path.join(temp_workspace["workspace"], "semgrep_results.json"))
    
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    with pytest.raises(FileNotFoundError):
        orchestrator.process_pipeline()

def test_extract_source_snippet_file_not_found(temp_workspace):
    """Test extraction when file doesn't exist."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    snippet = orchestrator.extract_source_snippet("nonexistent.py", 1, 5)
    assert snippet == ""

def test_extract_source_snippet_syntax_error(temp_workspace):
    """Test extraction when file has syntax error."""
    # Create a file with syntax error
    bad_file = os.path.join(temp_workspace["workspace"], "bad_file.py")
    with open(bad_file, "w") as f:
        f.write("def bad_function(  # Missing closing parenthesis")
    
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    snippet = orchestrator.extract_source_snippet("bad_file.py", 1, 1)
    assert snippet == ""  # Should return empty string on error

def test_extract_source_snippet_outside_function_scope(temp_workspace):
    """Test extraction when line is outside any function scope."""
    orchestrator = TriageOrchestrator(
        config_path=temp_workspace["config_path"],
        workspace_path=temp_workspace["workspace"]
    )
    
    # Test with line outside any function (line 1, which is just a comment)
    snippet = orchestrator.extract_source_snippet("test_file.py", 1, 1)
    assert snippet == ""  # Should return empty since no code at line 1
    # Note: In our test file, line 1 is empty, so this should return empty
    # But let's test with a line that's just a comment
    snippet = orchestrator.extract_source_snippet("test_file.py", 1, 1)
    assert snippet == ""
    
    # Test with line 6 (between functions)
    snippet = orchestrator.extract_source_snippet("test_file.py", 6, 6)
    assert snippet == ""  # Empty line between functions
