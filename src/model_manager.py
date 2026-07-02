# src/model_manager.py
"""
ModelManager: Centralized Singleton for Efficient Model Loading and Inference

This module provides a thread-safe, lazy-loaded singleton for managing Hugging Face
models used throughout the CEVuD pipeline. It eliminates redundant downloads and
memory duplication by ensuring each model is loaded exactly once per process.

Design Goals:
- Zero redundant downloads of model weights.
- Zero redundant model loading into RAM.
- Support for both classification and embedding models.
- Automatic caching to local disk via Hugging Face's cache system.
- ONNX-ready interface (future optimization).

Usage:
    manager = ModelManager()
    tokenizer, classifier = manager.get_classifier()
    tokenizer, embedder = manager.get_embedding_model()
"""

import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import Tuple, Optional

class ModelManager:
    """
    Singleton class for managing Hugging Face models used in CEVuD.
    Ensures models are loaded only once per Python process.
    """
    
    _instance = None
    _initialized = False

    def __new__(cls):
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model paths and cache directories once per effective config."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.getenv("CEVUD_CONFIG_PATH", os.path.join(repo_root, "config.json"))
        if not os.path.exists(config_path):
            default_config = {
                "paths": {
                    "workspace_root": "workspace_storage",
                    "model_cache_dir": "model_cache"
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
                }
            }
            os.makedirs(repo_root, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
        self._config_path = config_path
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            ws_root = config["paths"]["workspace_root"]
            cache_sub = config["paths"]["model_cache_dir"]
            if not os.path.isabs(ws_root):
                ws_root = os.path.join(repo_root, ws_root)
            self.cache_dir = os.path.abspath(os.path.join(ws_root, cache_sub))
        except Exception:
            self.cache_dir = os.path.join(repo_root, "workspace_storage", "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.classifier_model_name = "jayansh21/codesheriff-bug-classifier"
        self.embedding_model_name = "microsoft/codebert-base"
        self._classifier_tokenizer = None
        self._classifier_model = None
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._initialized = True
        self._vuln_class_idx = None  # Initialize for auto-detection

    def get_classifier(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Returns the pre-trained CodeBERT classifier for vulnerability scoring.
        Loads and caches the model if not already loaded.

        Returns:
            Tuple[AutoTokenizer, AutoModelForSequenceClassification]: Tokenizer and classifier model.
        """
        if self._classifier_tokenizer is None or self._classifier_model is None:
            print(f"[*] Loading CodeBERT Classifier: {self.classifier_model_name} (first use)")
            self._classifier_tokenizer = AutoTokenizer.from_pretrained(
                self.classifier_model_name,
                cache_dir=self.cache_dir
            )
            self._classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.classifier_model_name,
                cache_dir=self.cache_dir
            )
            self._classifier_model.eval()
            self._classifier_model.to("cpu")
        return self._classifier_tokenizer, self._classifier_model

    def get_embedding_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Returns the pre-trained CodeBERT embedding model for vector generation.
        Loads and caches the model if not already loaded.

        Returns:
            Tuple[AutoTokenizer, AutoModel]: Tokenizer and embedding model.
        """
        if self._embedding_tokenizer is None or self._embedding_model is None:
            print(f"[*] Loading CodeBERT Embedding Model: {self.embedding_model_name} (first use)")
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                cache_dir=self.cache_dir
            )
            self._embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                cache_dir=self.cache_dir
            )
            self._embedding_model.eval()
            self._embedding_model.to("cpu")
        return self._embedding_tokenizer, self._embedding_model

    def get_classifier_inference(self, code_snippets: list) -> list:
        """
        Performs batched inference on a list of code snippets using the classifier.
        This is the optimized, high-speed version for Stage 2 triage.

        Args:
            code_snippets (list of str): List of clean function source code strings.

        Returns:
            list of float: List of risk probabilities (0.0 to 1.0) for each snippet.
        """
        if not code_snippets:
            return []

        tokenizer, model = self.get_classifier()

        # Ensure we have vuln_class_idx from model config
        if not hasattr(self, "_vuln_class_idx") or self._vuln_class_idx is None:
            self._detect_vulnerability_class(model)

        inputs = tokenizer(
            code_snippets,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = None

            if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
                logits = outputs.logits
            else:
                try:
                    outputs = model.forward(**inputs)
                    if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
                        logits = outputs.logits
                    elif torch.is_tensor(outputs):
                        logits = outputs
                except Exception:
                    pass

            if logits is None and torch.is_tensor(outputs):
                logits = outputs

            if logits is None:
                raise TypeError("Could not obtain logits from model output")

            probabilities = torch.softmax(logits, dim=1)
            vuln_probs = probabilities[:, self._vuln_class_idx].tolist()
            return vuln_probs
    
    def _detect_vulnerability_class(self, model):
        """Auto-detect which output class corresponds to 'Security Vulnerability'."""
        if not hasattr(model.config, "id2label"):
            print("[!] Model has no id2label mapping. Falling back to index 1.")
            self._vuln_class_idx = 1
            return

        id2label = model.config.id2label
        print(f"[*] Model labels: {id2label}")

        # Look for any label containing security-relevant keywords
        vuln_keywords = ["security", "vuln", "bug", "exploit", "attack", "risk"]
        for idx, label in id2label.items():
            if any(kw in label.lower() for kw in vuln_keywords):
                self._vuln_class_idx = idx
                print(f"[+] Auto-detected vulnerability class: {idx} → '{label}'")
                return

        # Fallback: if none found, use index 1 (historical default)
        print("[!] No vulnerability label detected. Falling back to index 1.")
        self._vuln_class_idx = 1
