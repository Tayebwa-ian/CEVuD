"""
Unit Tests: DiffParser
======================
Validates AST parsing, hunk boundary calculation, path normalization, and caching.
"""

import os
import sys
import pytest
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from diff_parser import DiffParser

def test_diff_parser_extracts_modified_lines():
    """Verifies that line numbers are accurately captured from Git diff hunks."""
    raw_diff = (
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,4 +10,6 @@\n"
        " def execute_payload(data):\n"
        "+    # Added security control check\n"
        "+    sanitize(data)\n"
        "     return os.system(data)\n"
    )
    
    parser = DiffParser(raw_diff)
    result = parser.parse_modified_lines()
    
    assert "src/app.py" in result
    assert 11 in result["src/app.py"]
    assert 12 in result["src/app.py"]

def test_diff_parser_respects_exclusions():
    """Ensures that paths matching exclusion rules are skipped during parsing."""
    raw_diff = (
        "--- a/tests/test_mock.py\n"
        "+++ b/tests/test_mock.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+# Test change\n"
    )
    
    parser = DiffParser(raw_diff, exclude_prefixes=["tests/"])
    result = parser.parse_modified_lines()
    assert "tests/test_mock.py" not in result

def test_diff_parser_path_normalization():
    """Verifies that file paths are normalized to handle Windows/Linux path differences."""
    # Test with forward slashes (Unix style)
    raw_diff_forward = (
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+# Test change\n"
    )
    
    # Test with backslashes (Windows style)
    raw_diff_backward = (
        "--- a\\src\\app.py\n"
        "+++ b\\src\\app.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+# Test change\n"
    )
    
    parser_forward = DiffParser(raw_diff_forward)
    parser_backward = DiffParser(raw_diff_backward)
    
    result_forward = parser_forward.parse_modified_lines()
    result_backward = parser_backward.parse_modified_lines()
    
    # Both should normalize to the same path format
    assert "src/app.py" in result_forward
    assert "src/app.py" in result_backward
    assert result_forward["src/app.py"] == result_backward["src/app.py"]

def test_get_functions_from_ast_boundary():
    """Validates function extraction boundaries within accurate AST scopes."""
    source_code = (
        "def entry_point():\n"
        "    print('Hello')\n"
        "\n"
        "def target_func(x):\n"
        "    payload = x * 2\n"
        "    return payload\n"
    )
    
    # Line 5 points specifically to the body of target_func
    extracted = DiffParser.get_functions_from_ast("test_file.py", source_code, {5})
    assert len(extracted) == 1
    assert "def target_func" in extracted[0]
    assert "entry_point" not in extracted[0]

def test_get_functions_from_ast_cache():
    """Verifies that AST parsing is cached to avoid redundant parsing."""
    source_code = (
        "def test_func():\n"
        "    return 42\n"
    )
    
    # First call should parse and cache
    result1 = DiffParser.get_functions_from_ast("test_file.py", source_code, {2})
    
    # Second call with same file should use cache
    # We'll use a mock to track if parse_ast is called
    original_parse_ast = DiffParser._parse_ast
    parse_call_count = 0
    
    def mock_parse_ast(file_path, file_content):
        nonlocal parse_call_count
        parse_call_count += 1
        return original_parse_ast(file_path, file_content)
    
    # Patch the method to count calls
    DiffParser._parse_ast = mock_parse_ast
    
    try:
        result2 = DiffParser.get_functions_from_ast("test_file.py", source_code, {2})
        assert len(result1) == len(result2) == 1
        assert parse_call_count == 1  # Should only be called once due to caching
    finally:
        # Restore original method
        DiffParser._parse_ast = original_parse_ast

def test_get_functions_from_ast_syntax_error_handling():
    """Verifies that syntax errors in files are handled gracefully without crashing."""
    invalid_source_code = "def test_func()  # Missing colon\n    return 42\n"
    
    # Should not raise exception, just return empty list
    result = DiffParser.get_functions_from_ast("bad_file.py", invalid_source_code, {2})
    assert len(result) == 0

def test_get_functions_from_ast_empty_modified_lines():
    """Verifies that empty modified_lines returns empty list."""
    source_code = "def test_func():\n    return 42\n"
    
    result = DiffParser.get_functions_from_ast("test_file.py", source_code, set())
    assert len(result) == 0

def test_get_functions_from_ast_multiple_functions():
    """Verifies that multiple functions impacted by changes are all returned."""
    source_code = (
        "def func1():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def func2():\n"
        "    y = 2\n"
        "    return y\n"
        "\n"
        "def func3():\n"
        "    z = 3\n"
        "    return z\n"
    )
    
    # Modified lines in func1 and func2
    modified_lines = {2, 7}
    
    DiffParser._ast_cache.clear()
    
    result = DiffParser.get_functions_from_ast("test_file.py", source_code, modified_lines)
    assert len(result) == 2
    assert "def func1" in result[0] or "def func1" in result[1]
    assert "def func2" in result[0] or "def func2" in result[1]
    assert "def func3" not in result[0] and "def func3" not in result[1]

    @pytest.fixture(autouse=True)
    def clean_ast_cache():
        """Flushes the static AST parsing cache between unit test executions."""
        from diff_parser import DiffParser
        DiffParser._ast_cache.clear()
        yield
        DiffParser._ast_cache.clear()
