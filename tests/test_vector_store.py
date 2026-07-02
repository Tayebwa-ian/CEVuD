import os
import sys
import json
import tempfile
import shutil
import pytest
import hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vector_store import LocalVectorStore

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory and clean it up after testing."""
    workspace_dir = tempfile.mkdtemp()
    yield workspace_dir
    shutil.rmtree(workspace_dir)

@pytest.fixture
def mock_config(temp_workspace):
    """Generate a mock configuration JSON file inside the temporary workspace."""
    config = {
        "paths": {
            "vector_db_dir": "workspace_storage/codebase_vectors"
        }
    }
    config_path = os.path.join(temp_workspace, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path

def test_vector_store_initialization(mock_config, temp_workspace):
    """Test that database initializes schema and sets paths relative to workspace."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    assert os.path.exists(store.db_path)
    assert store.db_path.startswith(temp_workspace)

def test_serialization_deserialization(mock_config, temp_workspace):
    """Verify that serialization preserves floating-point vector dimensions."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    original_vector = [0.123, -0.456, 0.789]
    serialized = store._serialize_vector(original_vector)
    deserialized = store._deserialize_vector(serialized)
    
    assert len(deserialized) == len(original_vector)
    assert pytest.approx(deserialized[0]) == 0.123
    assert pytest.approx(deserialized[1]) == -0.456

def test_cosine_similarity_math():
    """Verify trigonometric similarity computation behaves correctly."""
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert pytest.approx(LocalVectorStore._compute_cosine_similarity(v1, v2)) == 1.0

    v3 = [0.0, 1.0]
    assert pytest.approx(LocalVectorStore._compute_cosine_similarity(v1, v3)) == 0.0

    v_zero = [0.0, 0.0]
    assert LocalVectorStore._compute_cosine_similarity(v1, v_zero) == 0.0

def test_insert_and_query_vectors(mock_config, temp_workspace):
    """Test inserting and semantic ranking inside the database."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    
    vec_a = [0.9, 0.1]
    vec_b = [0.1, 0.9]
    
    store.insert_code_block(
        file_path="app/auth.py",
        func_name="login",
        source="def login(): pass",
        embedding=vec_a
    )
    store.insert_code_block(
        file_path="app/db.py",
        func_name="query",
        source="def query(): pass",
        embedding=vec_b
    )
    
    # Query with target close to vec_a
    query_vec = [0.85, 0.15]
    results = store.query_cross_file_context(query_vec, limit=2)
    
    assert len(results) == 2
    assert results[0]["file_path"] == "app/auth.py"
    assert results[0]["function_name"] == "login"
    assert results[0]["similarity"] > results[1]["similarity"]

def test_get_code_block_by_file_and_func(mock_config, temp_workspace):
    """Test retrieving a specific code block by file path and function name."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    
    # Insert a code block
    embedding = [0.5, 0.5]
    store.insert_code_block(
        file_path="app/auth.py",
        func_name="login",
        source="def login(username, password): pass",
        embedding=embedding
    )
    
    # Retrieve by file and function name
    result = store.get_code_block_by_file_and_func("app/auth.py", "login")
    
    assert result is not None
    assert result["file_path"] == "app/auth.py"
    assert result["function_name"] == "login"
    assert result["source"] == "def login(username, password): pass"
    assert result["embedding"] == embedding
    
    # Test non-existent entry
    result = store.get_code_block_by_file_and_func("app/auth.py", "logout")
    assert result is None
    
    # Test with different file
    result = store.get_code_block_by_file_and_func("app/db.py", "login")
    assert result is None

def test_file_hash_persistence(mock_config, temp_workspace):
    """Test that file_hash is stored and retrieved correctly."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    
    # Create test file content and calculate its hash
    test_content = "def test_function():\n    return 'hello'\n"
    file_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()
    
    # Insert code block with file_hash
    embedding = [0.3, 0.7]
    store.insert_code_block(
        file_path="app/test.py",
        func_name="test_function",
        source=test_content,
        embedding=embedding,
        file_hash=file_hash
    )
    
    # Verify the file_hash was stored
    result = store.get_code_block_by_file_and_func("app/test.py", "test_function")
    assert result is not None
    assert result["file_hash"] == file_hash

def test_incremental_embedding_skip_logic(mock_config, temp_workspace):
    """Test that duplicate file_hash entries are skipped during insertion."""
    store = LocalVectorStore(mock_config, workspace_path=temp_workspace)
    
    # Create test content and its hash
    test_content = "def test_function():\n    return 'hello'\n"
    file_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()
    
    # First insertion
    embedding1 = [0.3, 0.7]
    store.insert_code_block(
        file_path="app/test.py",
        func_name="test_function",
        source=test_content,
        embedding=embedding1,
        file_hash=file_hash
    )
    
    # Count total entries
    conn = store._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM codebase_embeddings")
    initial_count = cursor.fetchone()[0]
    
    # Second insertion with same file_hash (should be skipped)
    embedding2 = [0.4, 0.6]  # Different embedding
    store.insert_code_block(
        file_path="app/test.py",
        func_name="test_function",
        source=test_content,
        embedding=embedding2,
        file_hash=file_hash
    )
    
    # Count entries after second insertion - should be same as initial
    cursor.execute("SELECT COUNT(*) FROM codebase_embeddings")
    final_count = cursor.fetchone()[0]
    
    assert final_count == initial_count, "Duplicate file_hash should not create new entry"
    
    # Verify the embedding was NOT updated (still has original value)
    result = store.get_code_block_by_file_and_func("app/test.py", "test_function")
    assert result["embedding"] == pytest.approx(embedding1, rel=1e-5), "Existing entry should retain original embedding"
    