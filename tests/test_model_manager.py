"""
Unit Tests: ModelManager
========================
Validates the singleton pattern, model caching, and batched inference behavior
of the centralized model loader used across Stage 2 and evaluation pipelines.
"""

import os
import json
import tempfile
import shutil
import pytest
import torch
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys_path_backup = sys.path.copy()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
os.environ["CEVUD_CONFIG_PATH"] = "/tmp/config.json"
from model_manager import ModelManager

# Restore sys.path after import
sys.path = sys_path_backup

@pytest.fixture(autouse=True)
def reset_singleton_permanently():
    from model_manager import ModelManager
    ModelManager._instance = None
    yield
    ModelManager._instance = None

@pytest.fixture(scope="module")
def temp_cache_dir():
    """Create a temporary cache directory for model downloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def mock_config(temp_cache_dir):
    """Generate a minimal config that points to the temp cache directory."""
    config = {
        "paths": {
            "workspace_root": temp_cache_dir,
            "model_cache_dir": "model_cache"
        }
    }
    config_path = os.path.join(temp_cache_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    os.environ["CEVUD_CONFIG_PATH"] = config_path
    yield config_path
    # Use pop with a default fallback to prevent KeyErrors if another test popped it
    os.environ.pop("CEVUD_CONFIG_PATH", None)


def test_model_manager_singleton_pattern(mock_config):
    """Verify that ModelManager enforces singleton behavior."""
    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2, "ModelManager must be a singleton"


def test_model_manager_initializes_cache_directory(mock_config, temp_cache_dir):
    """Ensure cache directory is created if it doesn't exist."""
    manager = ModelManager()
    expected_cache = os.path.join(temp_cache_dir, "model_cache")
    assert os.path.exists(expected_cache), "Cache directory should be auto-created"


def test_model_manager_loads_classifier_once(mock_config):
    """Verify classifier model is loaded only once across multiple calls."""
    manager = ModelManager()
    
    # First call: should trigger download/load
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model:
        tokenizer, classifier = manager.get_classifier()
        mock_tok.assert_called_once()
        mock_model.assert_called_once()
    
    # Second call: should NOT reload
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok2, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model2:
        tokenizer2, classifier2 = manager.get_classifier()
        mock_tok2.assert_not_called()
        mock_model2.assert_not_called()
    
    assert tokenizer is tokenizer2, "Tokenizer should be cached"
    assert classifier is classifier2, "Classifier should be cached"


def test_model_manager_loads_embedding_model_once(mock_config):
    """Verify embedding model is loaded only once."""
    manager = ModelManager()
    
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:
        tokenizer, embedder = manager.get_embedding_model()
        mock_tok.assert_called_once()
        mock_model.assert_called_once()
    
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok2, \
         patch("transformers.AutoModel.from_pretrained") as mock_model2:
        tokenizer2, embedder2 = manager.get_embedding_model()
        mock_tok2.assert_not_called()
        mock_model2.assert_not_called()
    
    assert tokenizer is tokenizer2
    assert embedder is embedder2


def test_model_manager_cpu_device(mock_config):
    """Ensure models are explicitly moved to CPU for CI/CD compatibility."""
    manager = ModelManager()
    
    with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model:
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        _, classifier = manager.get_classifier()
        mock_instance.to.assert_called_with("cpu")


def test_model_manager_empty_batch_returns_empty_list(mock_config):
    """Ensure empty input list returns empty output."""
    manager = ModelManager()
    probs = manager.get_classifier_inference([])
    assert probs == []
