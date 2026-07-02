import json
import os
import sqlite3
import struct
from typing import List, Tuple, Dict, Any


class LocalVectorStore:
    """
    Persistent context store using SQLite. 
    Stores source code alongside binary embeddings to facilitate 
    semantic search and RAG for the Stage 3 agent.

    Key Features:
    - Stores function-level code blocks with 768-dim embeddings.
    - Tracks upstream/downstream call relationships (call graph).
    - Supports `file_hash` for incremental embedding (avoid re-embedding unchanged files).
    - Enables semantic similarity search for cross-file context retrieval.
    """

    def __init__(self, config_path: str, workspace_path: str = None):
        """
        Initializes the LocalVectorStore client.

        Args:
            config_path (str): Path to the config file.
            workspace_path (str, optional): Target workspace path. Defaults to None (current working dir).
        """
        self.workspace_path = os.path.abspath(workspace_path) if workspace_path else os.getcwd()
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            root_config = os.path.join(os.path.dirname(__file__), "..", "config.json")
            with open(root_config, "r") as f:
                config = json.load(f)
        
        db_dir = config["paths"]["vector_db_dir"]
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(self.workspace_path, db_dir)
            
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = os.path.join(db_dir, "codebase_context.db")
        self._conn = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Constructs relational schema foundations with relational call-graph dimensions."""
        with sqlite3.connect(self.db_path) as conn:
            # Create table with new `file_hash` column (backward compatible)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS codebase_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    calls_out TEXT,           -- JSON string array of invoked functions
                    calls_in TEXT,            -- JSON string array of dependent upstream callers
                    file_hash TEXT            -- ✅ SHA-256 hash of entire file content for change detection
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_lookup ON codebase_embeddings(file_path);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_func_lookup ON codebase_embeddings(function_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON codebase_embeddings(file_hash);")  # ✅ For fast lookups
            conn.commit()

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
        """
        Calculates cosine similarity between two vectors.

        Args:
            vec_a, vec_b: List of floating point numbers.
        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    @property
    def conn(self):
        return self._get_connection()

    def _get_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row  # Enable dict-like access
        return self._conn

    def insert_code_block(self, file_path: str, func_name: str, source: str, 
                          embedding: List[float], calls_out: List[str] = None, 
                          calls_in: List[str] = None, file_hash: str = "") -> None:
        """Saves a code block along with its vector signature and static call arrays.

        Args:
            file_path (str): Relative path to the source file.
            func_name (str): Name of the function.
            source (str): Source code of the function.
            embedding (List[float]): 768-dimensional embedding vector.
            calls_out (List[str], optional): List of function names called by this function.
            calls_in (List[str], optional): List of function names that call this function.
            file_hash (str, optional): SHA-256 hash of the entire file content. Used for change detection.
        """
        blob = self._serialize_vector(embedding)
        c_out = json.dumps(calls_out or [])
        c_in = json.dumps(calls_in or [])
        with sqlite3.connect(self.db_path) as conn:
            if file_hash:
                existing = conn.execute(
                    "SELECT 1 FROM codebase_embeddings WHERE file_hash = ? LIMIT 1",
                    (file_hash,)
                ).fetchone()
                if existing:
                    return
            conn.execute(
                """INSERT INTO codebase_embeddings 
                   (file_path, function_name, source_code, embedding_blob, calls_out, calls_in, file_hash) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_path, func_name, source, blob, c_out, c_in, file_hash)
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
            cursor = conn.execute("SELECT file_path, function_name, source_code, embedding_blob, calls_out, calls_in FROM codebase_embeddings")
            for file_path, func_name, source, blob, c_out, c_in in cursor.fetchall():
                db_vector = self._deserialize_vector(blob)
                similarity = self._compute_cosine_similarity(query_embedding, db_vector)
                results.append({
                    "file_path": file_path,
                    "function_name": func_name,
                    "source_code": source,
                    "similarity": similarity,
                    "calls_out": json.loads(c_out),
                    "calls_in": json.loads(c_in)
                })
        
        # Sort by closest match descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def get_explicit_flow_context(self, function_name: str) -> List[Dict[str, Any]]:
        """Retrieves direct callers and callees crossing file boundaries via graph links."""
        context_nodes = []
        with sqlite3.connect(self.db_path) as conn:
            # Step A: Locate target function data
            cursor = conn.execute(
                "SELECT calls_out, calls_in FROM codebase_embeddings WHERE function_name = ?", (function_name,)
            )
            row = cursor.fetchone()
            if not row:
                return []
            
            calls_out = json.loads(row[0])
            calls_in = json.loads(row[1])
            
            # Step B: Hydrate nodes
            all_targets = list(set(calls_out + calls_in))
            if not all_targets:
                return []
                
            placeholders = ",".join(["?"] * len(all_targets))
            cursor = conn.execute(
                f"SELECT file_path, function_name, source_code FROM codebase_embeddings WHERE function_name IN ({placeholders})",
                all_targets
            )
            for file_path, fn_name, source in cursor.fetchall():
                context_nodes.append({
                    "file_path": file_path,
                    "function_name": fn_name,
                    "source_code": source,
                    "relationship": "upstream-caller" if fn_name in calls_in else "downstream-sink"
                })
        return context_nodes

    def get_code_block_by_file_and_func(self, file_path: str, func_name: str) -> Dict[str, Any] | None:
        """
        Retrieves a specific code block from a file and function.
        Used by dataset_ingest.py to check if a file has already been embedded and if its content has changed.

        Args:
            file_path (str): Relative path to the source file.
            func_name (str): Specific function name.

        Returns:
            Dict[str, Any] | None: Record with file_path, function_name, source_code, file_hash, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path, function_name, source_code, embedding_blob, file_hash FROM codebase_embeddings WHERE file_path = ? AND function_name = ?",
                (file_path, func_name)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "file_path": row[0],
                    "function_name": row[1],
                    "source_code": row[2],
                    "source": row[2],
                    "embedding": self._deserialize_vector(row[3]),
                    "file_hash": row[4]
                }
            return None
