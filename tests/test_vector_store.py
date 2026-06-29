import os
import sys
import json
import tempfile
import shutil
import pytest
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
