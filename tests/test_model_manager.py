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


def test_model_manager_multilabel_scoring(mock_config):
    """A multi-label vulnerability classifier scores P_slm as
    ``max(1 - P(safe), max_i P(CWE_i))`` -- the per-CWE signal,
    not the (often-saturated) ``safe`` class, drives the gate.

    NOTE: ``_configure_scoring`` runs inside ``get_classifier_inference``
    (not ``get_classifier``), so the model's forward output must be wired
    before the inference call, and the scoring-mode assertions come after it.
    """
    manager = ModelManager()
    manager._scoring_mode = None  # force auto-detection

    fake_model = MagicMock()
    fake_model.config.id2label = {
        0: "safe", 1: "CWE-79", 2: "CWE-89",  # 31-class multi-label style
    }
    # Row 0 = vulnerable: P(safe)=0.27 -> 1-P=0.73, CWE=0.95 -> P_slm = max(0.73, 0.95) = 0.95
    # Row 1 = safe:      P(safe)=0.99 -> 1-P=0.007, CWE=0.007 -> P_slm = 0.007
    # logit for prob p is ln(p/(1-p)); -1 -> 0.269, 3 -> 0.9526, 5 -> 0.9933, -5 -> 0.0067
    logits = torch.tensor([[-1.0, 3.0, -5.0], [5.0, -5.0, -5.0]])

    class _FakeTok:
        def __call__(self, texts, **kw):
            return {"input_ids": torch.zeros((len(texts), 2), dtype=torch.long)}

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTok()), \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
        tokenizer, model = manager.get_classifier()
        # Force the model forward path to return our crafted logits.
        model.return_value = MagicMock(logits=logits)
        probs = manager.get_classifier_inference(["x = 1", "y = 2"])

    assert manager._scoring_mode == "multilabel"
    assert manager._safe_class_idx == 0
    # vulnerable row escalates, safe row stays near zero
    assert probs[0] == pytest.approx(0.95, abs=1e-2)
    assert probs[1] == pytest.approx(0.007, abs=1e-2)


def test_model_manager_chunk_scores_aggregates(mock_config):
    """``get_classifier_chunk_scores`` cuts a long snippet into uniform chunks,
    scores each chunk, and aggregates (max by default). Mirrors how the Stage-2
    gate feeds the small model at inference time.

    The real model load is bypassed by mocking ``get_classifier_inference``;
    we just need the chunking + aggregation orchestration to be correct.
    """
    manager = ModelManager()

    # ~180 lines of real code -> several 64-line windows with 8-line overlap.
    long_code = "".join(f"def f_{i}():\n    return {i}\n" for i in range(60))

    def fake_inference(texts, **kwargs):
        n = len(texts)
        return [round((i + 1) / n, 4) for i in range(n)]

    with patch.object(manager, "get_classifier_inference", side_effect=fake_inference):
        results = manager.get_classifier_chunk_scores(
            [long_code], chunk_max_lines=64, chunk_overlap=8, min_code_lines=2
        )

    assert len(results) == 1
    r = results[0]
    assert r["score"] == pytest.approx(1.0, abs=1e-3)
    assert len(r["chunks"]) >= 2
    for c in r["chunks"]:
        assert set(c.keys()) >= {"start_line", "end_line", "prob", "text"}


def test_model_manager_chunk_scores_empty(mock_config):
    """An empty snippet list returns an empty result without touching the model."""
    manager = ModelManager()
    assert manager.get_classifier_chunk_scores([]) == []
