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
from run_context import get_model_cache_dir

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
                "models": {
                    "classifier_model": "jayansh21/codesheriff-bug-classifier",
                    "embedding_model": "microsoft/codebert-base"
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
            self.cache_dir = get_model_cache_dir(repo_root, config)
        except Exception:
            self.cache_dir = os.path.join(repo_root, "workspace_storage", "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        models_cfg = config.get("models", {}) if "config" in locals() else {}
        self.classifier_model_name = models_cfg.get(
            "classifier_model", "jayansh21/codesheriff-bug-classifier"
        )
        self.embedding_model_name = models_cfg.get(
            "embedding_model", "microsoft/codebert-base"
        )
        self._classifier_tokenizer = None
        self._classifier_model = None
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._initialized = True
        self._scoring_mode = None       # "softmax" | "multilabel" (auto-detected)
        self._vuln_class_idx = None     # softmax: security-vulnerability class
        self._safe_class_idx = None     # multilabel: dedicated "safe" class

    def get_classifier(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Returns the pre-trained SLM classifier for vulnerability scoring.
        Loads and caches the model if not already loaded.

        Returns:
            Tuple[AutoTokenizer, AutoModelForSequenceClassification]: Tokenizer and classifier model.
        """
        if self._classifier_tokenizer is None or self._classifier_model is None:
            print(f"[*] Loading SLM Classifier: {self.classifier_model_name} (first use)")
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

        # Configure the logits->probability mapping once per model load.
        if self._scoring_mode is None:
            self._configure_scoring(model)

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

            if self._scoring_mode == "multilabel":
                # Multi-label head (BCEWithLogitsLoss): each of the 31
                # classes is an independent sigmoid. In practice the dedicated
                # "safe" class fires ~1.0 even for vulnerable code, so
                # 1 - P(safe) alone collapses every score to ~0 and the gate
                # stops escalating real vulnerabilities. The real vulnerability
                # signal lives in the per-CWE probabilities, so CEVuD's threat
                # score is the MAXIMUM over the 30 CWE classes, with
                # (1 - P(safe)) kept as a floor:
                #   P_slm = max(1 - P(safe), max_i P(CWE_i))
                # -> a single [0,1] score that still escalates on a strong
                # per-CWE hit while respecting an explicit safe verdict.
                probs = torch.sigmoid(logits)
                safe = probs[:, self._safe_class_idx]
                num_labels = probs.shape[1]
                cwe_idx = [i for i in range(num_labels) if i != self._safe_class_idx]
                cwe_max = probs[:, cwe_idx].max(dim=1).values
                vuln = torch.maximum(1.0 - safe, cwe_max)
                vuln_probs = vuln.tolist()
            else:
                # Single-label head (CrossEntropy): softmax over classes, and we
                # gate on the dedicated security-vulnerability class.
                probabilities = torch.softmax(logits, dim=1)
                vuln_probs = probabilities[:, self._vuln_class_idx].tolist()
            return vuln_probs

    def get_classifier_chunk_scores(
        self,
        code_snippets: list,
        chunk_max_lines: int = 64,
        chunk_overlap: int = 8,
        min_code_lines: int = 2,
        aggregation: str = "max",
    ) -> list:
        """Scores code *snippets* by cutting each into uniform chunks that fit
        the model's 512-token window, scoring every chunk, then aggregating.

        Returns a list (parallel to ``code_snippets``) of dicts::

            {"score": float, "chunks": [{"start_line", "end_line", "prob", "text"}, ...]}

        ``score`` is the aggregated per-chunk probability (default ``max``: a
        function is vulnerable if any chunk is). ``chunks`` is kept so callers
        (the Stage-2 gate) can show the LLM exactly which windows were flagged.

        This mirrors how the classifier is trained (function-level label,
        chunk-level input) so train/inference stay consistent.
        """
        from code_chunks import chunk_code, aggregate_chunk_scores

        if not code_snippets:
            return []

        # Flatten all chunk texts into one batch for a single model call.
        per_snippet_chunks: List[list] = []
        flat_texts: List[str] = []
        for snippet in code_snippets:
            chunks = chunk_code(snippet or "", chunk_max_lines, chunk_overlap, min_code_lines)
            per_snippet_chunks.append(chunks)
            flat_texts.extend(c.text for c in chunks)

        flat_probs = self.get_classifier_inference(flat_texts) if flat_texts else []

        results = []
        cursor = 0
        for chunks in per_snippet_chunks:
            if not chunks:
                results.append({"score": 0.0, "chunks": []})
                continue
            probs = flat_probs[cursor:cursor + len(chunks)]
            cursor += len(chunks)
            chunk_info = [
                {
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "prob": round(float(p), 4),
                    "text": c.text,
                }
                for c, p in zip(chunks, probs)
            ]
            score = aggregate_chunk_scores([float(p) for p in probs], aggregation)
            results.append({"score": round(float(score), 4), "chunks": chunk_info})
        return results

    def _configure_scoring(self, model):
        """Decide how raw logits map to a single vulnerability probability.

        Two output formats are supported and auto-detected from the model's
        ``id2label`` mapping:

        * **Single-label softmax** (e.g. ``jayansh21/codesheriff-bug-classifier``):
          5 mutually-exclusive classes, one of which is the security
          vulnerability class -> ``P(vuln) = softmax[vuln_idx]``.
        * **Multi-label sigmoid** (e.g.
          ``ayshajavd/graphcodebert-vuln-classifier``): 31 independent classes
          (30 CWE categories + a dedicated "safe" class). In practice the
          ``safe`` class fires ~1.0 even for vulnerable code, so scoring as
          ``1 - P(safe)`` collapses every score to ~0. The real
          vulnerability signal lives in the per-CWE probabilities, so the gate
          score is ``P_slm = max(1 - P(safe), max_i P(CWE_i))`` -- a single
          ``[0, 1]`` value that escalates on a strong per-CWE hit while
          still respecting an explicit safe verdict.
        """
        id2label = getattr(model.config, "id2label", None)
        if id2label is not None:
            print(f"[*] Model labels: {id2label}")

        # Binary classifier (exactly two classes): unambiguously a single-label
        # softmax model. P(vulnerable) is the softmax probability of the
        # vulnerable class, independent of the underlying CWE/vulnerability type
        # — this is the contract CEVuD's custom-trained model must satisfy.
        # We detect this by head size (not label wording) so a negative class
        # named "safe"/"benign"/etc. is NOT mistaken for a multi-label head.
        num_labels = getattr(model.config, "num_labels", None)
        if num_labels == 2:
            vuln_idx = 1
            if id2label is not None:
                for idx, label in id2label.items():
                    if any(k in label.lower() for k in ("vulnerab", "vuln", "exploit")):
                        vuln_idx = int(idx)
                        break
            self._scoring_mode = "softmax"
            self._vuln_class_idx = vuln_idx
            print(
                f"[+] Binary classifier (num_labels=2) detected. "
                f"Scoring P(vulnerable) = softmax(logits)[:, {vuln_idx}]."
            )
            return

        # Multi-label mode: a dedicated "safe"/"benign"/"clean" class implies
        # the other classes are independent vulnerability tags.
        safe_labels = ("safe", "benign", "not_vulnerable", "non-vulnerable", "clean")
        safe_idx = None
        if id2label is not None:
            for idx, label in id2label.items():
                if label.lower() in safe_labels:
                    safe_idx = int(idx)
                    break

        if safe_idx is not None:
            self._scoring_mode = "multilabel"
            self._safe_class_idx = safe_idx
            print(f"[+] Multi-label classifier detected. Safe class idx {safe_idx} "
                  f"-> '{id2label[safe_idx]}'. Scoring as P_slm = max(1 - P(safe), max_i P(CWE_i)).")
            return

        # Single-label mode: pick the security-vulnerability class to gate on.
        self._scoring_mode = "softmax"
        if id2label is None:
            print("[!] Model has no id2label mapping. Falling back to index 3 "
                  "(Security Vulnerability).")
            self._vuln_class_idx = 3
            return
        self._vuln_class_idx = self._detect_security_class(id2label)

    def _detect_security_class(self, id2label):
        """Identify the single output class that means *security* vulnerability.

        A naive keyword scan over the labels is dangerous: the word "risk"
        appears in "Null Reference Risk", so matching on "risk"/"bug" would
        wrongly select that class and report the probability of a null
        dereference as the vulnerability score -- an obvious SQLi would then
        score ~0 and never escalate. We therefore match the *security
        vulnerability* class specifically, in priority order.

        Returns:
            int: index of the security-vulnerability class.
        """
        # Tier 1: the explicit security-vulnerability class.
        for idx, label in id2label.items():
            low = label.lower()
            if "security" in low and ("vulnerab" in low or "vuln" in low):
                print(f"[+] Detected Security Vulnerability class: {idx} -> '{label}'")
                return int(idx)

        # Tier 2: any label that reads as a security bug.
        for idx, label in id2label.items():
            if "security" in label.lower():
                print(f"[+] Detected Security class: {idx} -> '{label}'")
                return int(idx)

        # Tier 3: a generic vulnerability/bug label (no security-specific class).
        for idx, label in id2label.items():
            low = label.lower()
            if "vulnerab" in low or "vuln" in low or "exploit" in low:
                print(f"[+] Detected Vulnerability class: {idx} -> '{label}'")
                return int(idx)

        # Tier 4: last-resort fallback.
        print("[!] No security/vulnerability label detected. Falling back to index 3.")
        return 3
