import ast
import re
from typing import Dict, List, Set

class DiffParser:
    """Handles the processing of git diffs and extracts structural code blocks

    using Python's native Abstract Syntax Tree (AST) module.
    """

    def __init__(self, diff_content: str):
        self.diff_content = diff_content

    def parse_modified_lines(self) -> Dict[str, Set[int]]:
        """Parses a raw git diff string to extract modified file paths and line numbers.

        Returns:
            Dict[str, Set[int]]: Mapping of file paths to a set of added/modified line numbers.
        """
        modified_files: Dict[str, Set[int]] = {}
        current_file = None
        current_line = 0

        # Regular expressions to parse git diff structural markers
        file_re = re.compile(r"^[\+]{3} b/(.*)")
        hunk_re = re.compile(r"^@@ -\d+,\d+ \+(\d+),\d+ @@")

        for line in self.diff_content.splitlines():
            file_match = file_re.match(line)
            if file_match:
                current_file = file_match.group(1)
                if current_file.endswith(".py"):
                    modified_files[current_file] = set()
                continue

            if current_file and current_file.endswith(".py"):
                hunk_match = hunk_re.match(line)
                if hunk_match:
                    current_line = int(hunk_match.group(1))
                    continue

                if line.startswith("+") and not line.startswith("+++"):
                    modified_files[current_file].add(current_line)
                    current_line += 1
                elif not line.startswith("-"):
                    current_line += 1

        # Clean out files that had no actual line additions (e.g., pure deletions)
        return {k: v for k, v in modified_files.items() if v}

    @staticmethod
    def get_functions_from_ast(file_content: str, modified_lines: Set[int]) -> List[str]:
        """Uses Python AST to match modified line numbers to specific function blocks.

        Args:
            file_content (str): Complete source code of the modified file.
            modified_lines (Set[int]): Line numbers that were added or modified.

        Returns:
            List[str]: Raw source code strings of the impacted functions.
        """
        try:
            tree = ast.parse(file_content)
        except SyntaxError:
            return []  # Return empty if code doesn't parse due to incomplete PR states

        impacted_functions = []

        # Standardized AST traversal to find function definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine the start and end boundaries of the function block.
                # decorators are included in the lineno of the FunctionDef node.
                start_line = node.lineno
                
                # Use end_lineno (Python 3.8+) for precise slicing. 
                # Fallback to a default span if the attribute is missing.
                end_line = getattr(node, "end_lineno", start_line + 20) 
                
                # Check if any modified lines fall within this function's scope
                if any(start_line <= line <= end_line for line in modified_lines):
                    # Extract the raw source code lines for this specific logical block
                    lines = file_content.splitlines()[start_line - 1:end_line]
                    impacted_functions.append("\n".join(lines))

        return impacted_functions
