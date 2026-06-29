"""
Unit Tests: DiffParser
======================
Validates AST parsing, hunk boundary calculation, and path exclusions.
"""

import os
import sys
import pytest
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
    extracted = DiffParser.get_functions_from_ast(source_code, {5})
    assert len(extracted) == 1
    assert "def target_func" in extracted[0]
    assert "entry_point" not in extracted[0]
