"""Automated utility to fetch, construct, and seed our local evaluation test matrix."""

import os
import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModel
from diff_parser import DiffParser
from vector_store import LocalVectorStore

class IngestManager:
    """
    Context Management utility for seeding the LocalVectorStore.
    Supports benchmarking with ground-truth data or indexing a live repository.
    """
    
    def __init__(self, config_path: str):
        """
        Initializes models and database connections for data ingestion.

        Args:
            config_path (str): Configuration file path.
        """
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
        """
        Computes a semantic vector using CodeBERT.

        Args:
            text (str): Source code or query text.

        Returns:
            List[float]: 768-dimensional embedding vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[0][0].numpy().tolist()

    def ingest_benchmark_json(self, json_path: str):
        """
        Seeds the database from a JSON manifest of known vulnerabilities.

        Args:
            json_path (str): Path to the gold_standard.json file.
        """
        print(f"[*] Ingesting benchmark data from: {json_path}")
        with open(json_path, "r") as f:
            test_suite = json.load(f)
        
        # Materialize snippets to a local folder so they can be scanned manually by Stage 1
        samples_dir = "vulnerability_samples"
        os.makedirs(samples_dir, exist_ok=True)
        print(f"[*] Materializing benchmark snippets to: {samples_dir}/")

        for item in test_suite:
            # Create a safe filename for the snippet
            safe_fn = item["function_name"].replace(" ", "_")
            file_path = os.path.join(samples_dir, f"{safe_fn}.py")
            
            with open(file_path, "w") as f:
                f.write(item["source_code"])

            vector = self._generate_embedding(item["source_code"])
            self.db.insert_code_block(
                file_path=item["file_path"],
                func_name=item["function_name"],
                source=item["source_code"],
                embedding=vector
            )
        
        print(f"[+] Successfully seeded {len(test_suite)} benchmark cases.")

    def ingest_repository(self, repo_path: str):
        """
        Performs a full crawl of a repository to build a semantic RAG index.

        Args:
            repo_path (str): Local root directory of the repository to index.
        """
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
