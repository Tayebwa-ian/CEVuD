"""Automated utility to fetch, construct, and seed our local evaluation test matrix."""

import os
import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModel
from .diff_parser import DiffParser
from src.vector_store import LocalVectorStore

class IngestManager:
    """Handles different data ingestion modes: Gold Standard Benchmarks or Repository Crawling."""
    
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        os.makedirs(self.config["paths"]["vector_db_dir"], exist_ok=True)
        self.db = LocalVectorStore(config_path)
        
        # Initialize CodeBERT for real embeddings during ingestion
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def _generate_embedding(self, text: str):
        """Generates a real 768-dim vector using CodeBERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[0][0].numpy().tolist()

    def ingest_benchmark_json(self, json_path: str):
        """Loads a Gold Standard JSON and seeds the evaluation ledger."""
        print(f"[*] Ingesting benchmark data from: {json_path}")
        with open(json_path, "r") as f:
            test_suite = json.load(f)
        
        for item in test_suite:
            vector = self._generate_embedding(item["source_code"])
            self.db.insert_code_block(
                file_path=item["file_path"],
                func_name=item["function_name"],
                source=item["source_code"],
                embedding=vector
            )
        
        # Save for evaluation script
        os.makedirs("workspace_storage", exist_ok=True)
        with open("workspace_storage/evaluation_ledger.json", "w") as f:
            json.dump(test_suite, f, indent=2)
        print(f"[+] Successfully seeded {len(test_suite)} benchmark cases.")

    def ingest_repository(self, repo_path: str):
        """Walks a local directory, extracts all functions, and indexes them."""
        print(f"[*] Crawling repository at: {repo_path}")
        count = 0
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", errors="ignore") as f:
                        content = f.read()
                    
                    # Use DiffParser's AST logic to extract all functions
                    # We pass a set of all lines to capture every function
                    all_lines = set(range(1, content.count('\n') + 2))
                    functions = DiffParser.get_functions_from_ast(content, all_lines)
                    
                    for func_code in functions:
                        # Quick extraction of function name for metadata
                        name_match = re.search(r"def\s+(\w+)\s*\(", func_code)
                        func_name = name_match.group(1) if name_match else "unknown"
                        
                        vector = self._generate_embedding(func_code)
                        self.db.insert_code_block(
                            file_path=os.path.relpath(full_path, repo_path),
                            func_name=func_name,
                            source=func_code,
                            embedding=vector
                        )
                        count += 1
        print(f"[+] Repository ingestion complete. Indexed {count} functions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEVuD Data Ingestion Utility")
    parser.add_argument("--mode", choices=["benchmark", "repo"], required=True, help="Ingestion mode")
    parser.add_argument("--file", help="Path to Gold Standard JSON (for benchmark mode)")
    parser.add_argument("--path", help="Path to local repository root (for repo mode)")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    
    args = parser.parse_args()
    manager = IngestManager(args.config)
    
    if args.mode == "benchmark":
        if not args.file:
            print("[-] Error: --file is required for benchmark mode.")
        else:
            manager.ingest_benchmark_json(args.file)
    elif args.mode == "repo":
        if not args.path:
            print("[-] Error: --path is required for repo mode.")
        else:
            manager.ingest_repository(args.path)
