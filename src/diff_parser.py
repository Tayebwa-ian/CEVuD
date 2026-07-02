import ast
import os
import re
from typing import Dict, List, Set

class DiffParser:
    """
    Static utility class for decomposing Git diffs and Python source code.
    It identifies which specific lines changed and uses AST to map those lines 
    to discrete function/method blocks.

    Optimization Highlights:
    - Caches parsed ASTs per file to avoid redundant parsing.
    - Normalizes file paths to avoid mismatches in git diff vs filesystem.
    - Only parses files that are modified and not excluded.
    - Handles encoding and syntax errors gracefully.
    """

    # ✅ Class-level cache for AST trees: {norm_path: (tree, file_content)}
    _ast_cache: Dict[str, tuple] = {}

    def __init__(self, diff_content: str, exclude_prefixes: List[str] = None):
        """
        Initializes the DiffParser.

        Args:
            diff_content (str): The raw git diff output to parse.
            exclude_prefixes (List[str], optional): Prefixes of file paths to exclude from analysis.
                                                    Defaults to None.
        """
        self.diff_content = diff_content
        self.exclude_prefixes = exclude_prefixes if exclude_prefixes is not None else []

    def parse_modified_lines(self) -> Dict[str, Set[int]]:
        """
        Processes a raw 'git diff' string to isolate modified line numbers.

        Returns:
            Dict[str, Set[int]]: Mapping of normalized file paths to a set of added/modified line numbers.
        """
        modified_files: Dict[str, Set[int]] = {}
        current_file = None
        current_line = 0

        file_re = file_re = re.compile(r"^[\+]{3} [ab][/\\](.*)")
        hunk_re = re.compile(r"^@@ -\d+,\d+ \+(\d+),\d+ @@")

        for line in self.diff_content.splitlines():
            file_match = file_re.match(line)
            if file_match:
                # Normalize path: replace \ with /, then normpath, then ensure forward slashes
                raw_path = file_match.group(1).replace("\\", "/")
                current_file = os.path.normpath(raw_path).replace("\\", "/")  # ✅ Ensure forward slashes
                normalized_excludes = [prefix.replace("\\", "/") for prefix in self.exclude_prefixes]
                if current_file.endswith(".py"):
                    should_exclude = any(current_file.startswith(prefix) for prefix in normalized_excludes)
                    if not should_exclude:
                        modified_files[current_file] = set()
                else:
                    current_file = None
                continue

            if current_file and current_file in modified_files:
                hunk_match = hunk_re.match(line)
                if hunk_match:
                    current_line = int(hunk_match.group(1))
                    continue

                if line.startswith("+") and not line.startswith("+++"):
                    modified_files[current_file].add(current_line)
                    current_line += 1
                elif not line.startswith("-"):
                    current_line += 1

        return {k: v for k, v in modified_files.items() if v}

    @staticmethod
    def _parse_ast(file_path: str, file_content: str) -> ast.AST:
        """
        Parses a Python file into AST with caching. Avoids re-parsing same file.

        Args:
            file_path (str): Normalized file path.
            file_content (str): Source code content.

        Returns:
            ast.AST: Parsed AST tree.
        """
        norm_path = os.path.normpath(file_path)
        if norm_path in DiffParser._ast_cache:
            return DiffParser._ast_cache[norm_path][0]

        try:
            tree = ast.parse(file_content)
            DiffParser._ast_cache[norm_path] = (tree, file_content)
            return tree
        except SyntaxError as e:
            # Log and return None — file is malformed, skip
            print(f"[!] SyntaxError parsing {norm_path}: {e}")
            return None
        except Exception as e:
            print(f"[!] Unexpected error parsing {norm_path}: {e}")
            return None

    @staticmethod
    def get_functions_from_ast(file_path: str, file_content: str, modified_lines: Set[int]) -> List[str]:
        """
        Matches modified line numbers to the boundaries of function definitions.
        Ensures that only the logical blocks impacted by changes are analyzed.

        Args:
            file_path (str): Normalized file path of the source file.
            file_content (str): Complete source code of the modified file.
            modified_lines (Set[int]): Line numbers that were added or modified.

        Returns:
            List[str]: Raw source code strings of the impacted functions.
        """
        if not modified_lines:
            return []

        # Parse AST once per file, using cache
        tree = DiffParser._parse_ast(file_path, file_content)
        if tree is None:
            return []

        impacted_functions = []

        # Standardized AST traversal to find function definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                # Use end_lineno (Python 3.8+) for precise slicing. Fallback if missing.
                end_line = getattr(node, "end_lineno", start_line + 1)
                # Check if any modified line falls within this function's scope
                if any(start_line <= line <= end_line for line in modified_lines):
                    # Extract the raw source code lines for this specific logical block
                    lines = file_content.splitlines()[start_line - 1:end_line]
                    impacted_functions.append("\n".join(lines))

        return impacted_functions
