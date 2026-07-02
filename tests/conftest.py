"""
Pytest Local Workspace Setup Configuration
===========================================
Establishes a localized testing storage directory framework inside the repo layout.
"""

import os
import json
import pytest

@pytest.fixture(scope="session")
def repository_root() -> str:
    """Resolves the absolute location path matching the code repository root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def real_test_config(repository_root) -> str:
    """
    Builds a config file pointing to an explicit, persistent workspace
    environment located inside the repository's tests folder.

    All paths are written as **absolute** so that no component (TriageOrchestrator,
    LocalVectorStore, PipelineEvaluator) double-appends directories when
    workspace_path is also passed as an absolute path.
    """
    test_storage_root = os.path.join(repository_root, "tests", "workspace_storage")
    test_db_dir = os.path.join(test_storage_root, "codebase_vectors")
    test_evals_dir = os.path.join(test_storage_root, "evaluation_runs")
    test_artifacts_dir = os.path.join(test_storage_root, "artifacts")

    # Ensure physical folder structures exist ahead of execution sequences
    for d in (test_storage_root, test_db_dir, test_evals_dir, test_artifacts_dir):
        os.makedirs(d, exist_ok=True)

    config_data = {
        "paths": {
            # Absolute paths so components resolve correctly regardless of cwd
            "workspace_root": test_storage_root,
            "vector_db_dir": test_db_dir,
            "evaluations_subdir": test_evals_dir,
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
            "provider": "openai",
            "model_name": "mock-gpt-4",
            "temperature": 0.0
        }
    }

    config_file = os.path.join(repository_root, "tests", "test_config_manifest.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return str(config_file)

@pytest.fixture(scope="session")
def unit_test_config(repository_root) -> str:
    """Provides configuration paths for standard unit testing scenarios."""
    config_file = os.path.join(repository_root, "tests", "unit_test_config_manifest.json")
    config_data = {
        "paths": {
            "workspace_root": "tests/workspace_storage",
            "artifacts_subdir": "test_artifacts",
            "semgrep_output": "semgrep.json",
            "triage_report": "stage1_2_triage.json"
        },
        "gate_parameters": {
            "weight_static": 0.4,
            "weight_slm": 0.6,
            "escalation_threshold": 0.52,
            "slm_override_threshold": 0.90
        },
        "semgrep_severity_map": {
            "ERROR": 1.0, "WARNING": 0.7, "INFO": 0.3, "NONE": 0.0
        }
    }
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    return str(config_file)

@pytest.fixture(scope="session")
def gold_standard_path(repository_root) -> str:
    """Resolves your absolute local gold standard validation benchmark ledger asset."""
    target_path = os.path.join(repository_root, "tests", "data", "gold_standard.json")
    if not os.path.exists(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Seed a valid fallback JSON scheme structure if empty or missing
        with open(target_path, "w") as seed_file:
            json.dump([
                {
                    "path": "vulnerable_sample.py",
                    "code": "def process(user_input):\n    eval(user_input)",
                    "expected_vulnerability": True,
                    "severity": "ERROR"
                }
            ], seed_file, indent=2)
    return target_path
