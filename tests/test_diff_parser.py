import os
import sys
import pytest
from src.diff_parser import DiffParser

# Setup path context for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_diff_parser_init():
    """Test DiffParser constructor defaults and custom parameter assignments."""
    parser = DiffParser("some diff content")
    assert parser.diff_content == "some diff content"
    assert parser.exclude_prefixes == []

    parser_custom = DiffParser("some diff", exclude_prefixes=["src/", "tests/"])
    assert parser_custom.exclude_prefixes == ["src/", "tests/"]

def test_diff_parser_line_extraction():
    """Verify DiffParser isolates modified lines correctly from a complex diff payload."""
    diff_payload = (
        "diff --git a/app/main.py b/app/main.py\n"
        "index 1234567..abcdefg 100644\n"
        "--- a/app/main.py\n"
        "+++ b/app/main.py\n"
        "@@ -5,4 +5,6 @@\n"
        " def check_user():\n"
        "-    print('old')\n"
        "+    print('new line 1')\n"
        "+    print('new line 2')\n"
        "     return True\n"
        "diff --git a/tests/test_main.py b/tests/test_main.py\n"
        "--- a/tests/test_main.py\n"
        "+++ b/tests/test_main.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+def test_func():\n"
    )

    # Exclude tests/ directory
    parser = DiffParser(diff_payload, exclude_prefixes=["tests/"])
    modified = parser.parse_modified_lines()

    assert "tests/test_main.py" not in modified
    assert "app/main.py" in modified
    assert 5 in modified["app/main.py"]
    assert 6 in modified["app/main.py"]
    assert 7 not in modified["app/main.py"]

def test_diff_parser_function_ast_extraction():
    """Verify AST parsing identifies function boundaries and extracts complete function code."""
    source_code = (
        "import sys\n"
        "\n"
        "def helper_method(val):\n"
        "    res = val * 2\n"
        "    return res\n"
        "\n"
        "class MyClass:\n"
        "    def method_one(self):\n"
        "        # line 9\n"
        "        pass\n"
    )

    # Line 4 is inside helper_method
    funcs = DiffParser.get_functions_from_ast(source_code, {4})
    assert len(funcs) == 1
    assert "def helper_method" in funcs[0]
    assert "res = val * 2" in funcs[0]
    assert "method_one" not in funcs[0]

    # Line 9 is inside method_one
    funcs_class = DiffParser.get_functions_from_ast(source_code, {9})
    assert len(funcs_class) == 1
    assert "def method_one" in funcs_class[0]
    assert "helper_method" not in funcs_class[0]

def test_diff_parser_syntax_error_handling():
    """Verify AST parser returns empty list gracefully on invalid Python syntax."""
    invalid_code = "def incomplete_func(\n"
    funcs = DiffParser.get_functions_from_ast(invalid_code, {1})
    assert funcs == []
