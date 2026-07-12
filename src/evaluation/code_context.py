"""
code_context.py
================
Helpers that turn a CVEfixes row (a unified diff + a pre-extracted
vulnerable snippet) into the *real, complete* code context that the static
analyzer (Semgrep), the local SLM classifier, and the Stage-3 LLM
actually need:

    * the FULL enclosing function (not just the 1-3 changed lines),
    * the module-level imports it depends on, and
    * best-effort CROSS-FILE context (the source of locally imported modules),
      so cross-file taint / data-flow can be reasoned about.

Why this module exists
----------------------
The CVEfixes `vulnerable_code` field is an unreliable, sometimes
concatenated fragment. The authoritative source is the real repository at the
fix commit. So the evaluation pipeline (raw_score_extractor.py) CLONES the
repo, checks out the vulnerable commit, and uses the helpers here to expand
a diff-hunk anchor into a complete, context-rich snippet.

All functions in this module are side-effect free and take an explicit
``read_file(rel_path) -> Optional[str]`` callback for the cross-file
collector, so the logic is fully unit-testable without any network or
git dependency.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

# A callback that returns the textual content of ``rel_path`` inside the
# cloned repository (e.g. backed by ``git show <commit>:<rel_path>``).
ReadFileFn = Callable[[str], Optional[str]]


# ---------------------------------------------------------------------------
# 1. Diff -> function anchor
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)(?:\n|$)")


def parse_diff_anchors(diff: str) -> Dict[str, Dict[str, object]]:
    """Parses a unified diff and returns, per changed file, an anchor dict::

        {rel_file_path: {
            "pre_start": int,   # 1-indexed start line in the PRE-image
            "pre_end":   int,   # 1-indexed inclusive end in the PRE-image
            "post_start": int,  # 1-indexed start line in the POST-image (fixed)
            "post_end":   int,  # 1-indexed inclusive end in the POST-image
            "has_def":  bool,   # did a hunk header mention a `def`?
            "def_line": int|None,
        }}

    The "PRE-image" numbers refer to the vulnerable (pre-fix) file, which
    is the version we want to analyze for detection. The "POST-image" numbers
    refer to the fixed (post-fix) file and are used to build the *safe*
    (label=0) sample for a balanced dataset. We pick, per file, the
    hunk whose ``@@ ... @@ <ctx>`` header contains ``def `` / ``async def ``
    (that hunk locates the vulnerable function); if none does, we fall back to
    the first hunk.
    """
    anchors: Dict[str, Dict[str, object]] = {}
    current_file: Optional[str] = None
    # Per-file, track candidate hunks: list of (pre_start, pre_len, ctx)
    pending: Dict[str, List[Tuple[int, int, str]]] = {}

    for line in (diff or "").splitlines():
        m_file = _DIFF_FILE_RE.match(line)
        if m_file:
            current_file = m_file.group(2).strip()
            pending.setdefault(current_file, [])
            continue
        m_hunk = _HUNK_RE.match(line)
        if m_hunk and current_file is not None:
            pre_start = int(m_hunk.group(1))
            pre_len = int(m_hunk.group(2)) if m_hunk.group(2) else 1
            post_start = int(m_hunk.group(3))
            post_len = int(m_hunk.group(4)) if m_hunk.group(4) else 1
            # A pure-addition hunk has pre_len==0 (no pre-image lines);
            # clamp so pre_end >= pre_start (a 1-line anchor).
            pre_end = pre_start + max(pre_len, 1) - 1
            # A pure-deletion hunk has post_len==0 (no post-image lines);
            # clamp so post_end >= post_start (a 1-line anchor).
            post_end = post_start + max(post_len, 1) - 1
            ctx = m_hunk.group(5).strip()
            pending[current_file].append((pre_start, pre_len, post_start, post_len, ctx))

    for fpath, hunks in pending.items():
        if not hunks:
            # Diff header with no hunks (e.g. a pure rename).
            continue
        chosen = None
        for h in hunks:
            if "def " in h[4] or "async def " in h[4]:
                chosen = h
                break
        if chosen is None:
            chosen = hunks[0]
        pre_start, pre_len, post_start, post_len, ctx = chosen
        has_def = ("def " in ctx) or ("async def " in ctx)
        def_line = pre_start if has_def else None
        anchors[fpath] = {
            "pre_start": pre_start,
            "pre_end": pre_start + max(pre_len, 1) - 1,
            "post_start": post_start,
            "post_end": post_start + max(post_len, 1) - 1,
            "has_def": has_def,
            "def_line": def_line,
        }
    return anchors


# ---------------------------------------------------------------------------
# 2. Expand an anchor line to the enclosing function (Python AST)
# ---------------------------------------------------------------------------

def expand_to_function(source: str, anchor_line: int) -> Tuple[int, int]:
    """Given file ``source`` and a 1-indexed ``anchor_line`` (the line the
    vulnerable hunk starts on), returns the 1-indexed inclusive
    ``(start_line, end_line)`` of the smallest enclosing Python function.

    If the anchor is not inside any function (top-level code such as a
    module-level ``__version__`` assignment), we fall back to a small
    context window around the anchor so the SLM/LLM still get surrounding
    lines rather than a single isolated line.
    """
    if anchor_line < 1:
        anchor_line = 1
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return (anchor_line, anchor_line)

    best: Optional[Tuple[int, int]] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            if start <= anchor_line <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end)

    if best is not None:
        return best
    # Fallback window for non-function (module-level) code.
    lines = source.splitlines()
    lo = max(1, anchor_line - 3)
    hi = min(len(lines), anchor_line + 8)
    return (lo, hi)


# ---------------------------------------------------------------------------
# 3. Module-level imports + cross-file context
# ---------------------------------------------------------------------------

def collect_module_imports(source: str) -> str:
    """Returns the concatenated module-level ``import`` / ``from ... import``
    statements found in ``source`` (so the SLM/LLM know the dependencies
    of the function being scored).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(ast.get_source_segment(source, node) or "")
    return "\n".join(s for s in out if s).strip()


def _resolve_local_module_path(
    import_node: ast.ImportFrom, file_path: str, repo_root: str
) -> Optional[str]:
    """Resolves a (possibly relative) ``from ... import`` statement to a
    concrete ``.py`` path inside ``repo_root``, or None if it is a
    third-party / stdlib import we cannot resolve locally.
    """
    level = import_node.level  # 0 = absolute, 1 = '.', 2 = '..', ...
    module = (import_node.module or "").split(".")

    base_dir = os.path.dirname(file_path)
    if level > 0:
        # Go up ``level`` directories from the importing file.
        for _ in range(level - 1):
            base_dir = os.path.dirname(base_dir)
        parts = module
    else:
        if not module:
            return None
        parts = module

    if not parts:
        return None

    rel_candidates = [
        os.path.normpath(os.path.join(base_dir, *parts) + ".py").replace("\\", "/"),
        os.path.normpath(os.path.join(base_dir, *parts, "__init__.py")).replace("\\", "/"),
    ]
    # Whether this file actually exists in the repo is decided by
    # collect_cross_file_context's read_file gate (it returns None for
    # anything absent at the commit, including stdlib/third-party imports
    # like `os` or `requests`). So just return the well-formed
    # repo-relative path; the caller drops it if read_file says no.
    return rel_candidates[0]


def collect_cross_file_context(
    source: str,
    file_path: str,
    repo_root: str,
    read_file: ReadFileFn,
    max_modules: int = 3,
    max_lines_per_module: int = 200,
) -> Dict[str, str]:
    """Collects best-effort cross-file context: for every LOCAL (intra-repo)
    module imported by ``source``, fetches that module's source via
    ``read_file`` and returns ``{rel_path: trimmed_source}``.

    This is what lets the SLM/LLM (and Semgrep, if the materialized
    snippets are scanned together) reason about data flow across files.
    Third-party / stdlib imports are ignored. Results are bounded by
    ``max_modules`` and ``max_lines_per_module`` to keep context sizes sane.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    collected: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        rel = _resolve_local_module_path(node, file_path, repo_root)
        if rel is None or rel in collected:
            continue
        if rel == file_path.replace("\\", "/"):
            continue  # skip self-imports
        content = read_file(rel)
        if content is None:
            continue
        lines = content.splitlines()
        if len(lines) > max_lines_per_module:
            lines = lines[:max_lines_per_module]
        collected[rel] = "\n".join(lines)
        if len(collected) >= max_modules:
            break
    return collected


def build_context_snippet(
    source: str,
    function_range: Tuple[int, int],
    imports: str,
    cross_file: Dict[str, str],
) -> str:
    """Assembles the final snippet handed to the SLM/LLM::

        <module imports>

        <enclosing function source>

        # ---- cross-file context ----
        # <mod_a.py>
        <source of mod_a>
        ...
    """
    lo, hi = function_range
    func_lines = source.splitlines()[lo - 1 : hi]
    func_src = "\n".join(func_lines)

    parts: List[str] = []
    if imports:
        parts.append(imports)
    parts.append(func_src)
    if cross_file:
        parts.append("# ---- cross-file context ----")
        for mod_path, mod_src in cross_file.items():
            parts.append(f"# <{mod_path}>")
            parts.append(mod_src)
    return "\n\n".join(parts).strip() + "\n"
