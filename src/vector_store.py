import json
import os
import sqlite3
import struct
from typing import List, Tuple, Dict, Any

class LocalVectorStore:
    """Manages an embedded, zero-cost codebase semantic context index inside SQLite."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        db_dir = config["paths"]["vector_db_dir"]
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = os.path.join(db_dir, "codebase_context.db")
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Constructs the standard schema foundations for storing code structures."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS codebase_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_lookup ON codebase_embeddings(file_path);")

    @staticmethod
    def _serialize_vector(vector: List[float]) -> bytes:
        """Converts an array of floating-point numbers into compact raw bytes."""
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def _deserialize_vector(blob: bytes) -> List[float]:
        """Restores a compact binary payload back into a python float array."""
        num_floats = len(blob) // 4
        return list(struct.unpack(f"{num_floats}f", blob))

    @staticmethod
    def _compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates the exact angular similarity profile between two dense vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def insert_code_block(self, file_path: str, func_name: str, source: str, embedding: List[float]) -> None:
        """Saves a code block along with its vector signature to the local index."""
        blob = self._serialize_vector(embedding)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO codebase_embeddings (file_path, function_name, source_code, embedding_blob) VALUES (?, ?, ?, ?)",
                (file_path, func_name, source, blob)
            )

    def query_cross_file_context(self, query_embedding: List[float], limit: int = 2) -> List[Dict[str, Any]]:
        """Scans the database to find the most semantically relevant code structures.

        Args:
            query_embedding (List[float]): The vector representation of the target snippet.
            limit (int): Total context elements to extract.

        Returns:
            List[Dict[str, Any]]: Chronologically ranked structural database entries.
        """
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file_path, function_name, source_code, embedding_blob FROM codebase_embeddings")
            for file_path, func_name, source, blob in cursor.fetchall():
                db_vector = self._deserialize_vector(blob)
                similarity = self._compute_cosine_similarity(query_embedding, db_vector)
                results.append({
                    "file_path": file_path,
                    "function_name": func_name,
                    "source_code": source,
                    "similarity": similarity
                })
        
        # Sort by closest match descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
