"""
data_quality.py
===============
Shared, dependency-free helpers that keep label *noise* out of the CEVuD
training / evaluation corpora.

WHY THIS MODULE EXISTS
----------------------
The original CVEfixes pipeline paired a pre-fix (vulnerable, ``label=1``) and a
post-fix (safe, ``label=0``) **function** for every fix commit. For a large
fraction of CVEs the actual patch is a *trivial* change inside a large function
(a version bump in ``package_data.py``, a one-line docstring edit, a
``__version__`` assignment, …). After the dataset builder expands each side to
the full enclosing function, the two samples become **near-identical text with
opposite labels** — and in many cases *byte-for-byte identical*. A classifier
trained on such pairs cannot learn anything: the loss plateaus at ``ln(2) ≈
0.693`` and ROC-AUC stays at ~0.5 (exactly the symptom the user observed).

The functions here let every stage of the pipeline reject those noisy samples:

* ``_is_noise_file(path)``         — drop docs/tests/packaging/version files.
* ``_has_code_signal(line)``       — does a line carry *learnable* code (not just
                                     a comment, docstring, or version assignment)?
* ``code_signal_line_count(text)`` — how many such lines a snippet has.
* ``texts_are_equivalent(a, b)``   — identical once blank/whitespace is ignored.
* ``is_trivial_change(vuln, safe)``— True when the *only* difference between the
                                     vulnerable and fixed snippet is non-semantic
                                     (comments / docstrings / version bumps).
* ``find_contradictions(records)`` — yield normalized texts that appear with
                                     BOTH labels (hard contradictions).

These are deliberately heuristic and conservative: they never drop a sample
that carries a real, learnable code change. See ``docs/DATA_QUALITY.md`` for the
full rationale and tuning guidance.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# 1. File-level noise: paths that almost never carry a learnable vuln signal
# ---------------------------------------------------------------------------
# Substrings (case-insensitive) that, if present in a file path, mark the file
# as non-source / packaging / documentation. Including these as (vuln, safe)
# pairs only trains the model on version bumps and README edits.
NOISE_PATH_PATTERNS: Tuple[str, ...] = (
    "/docs/", "/doc/", "/documentation/",
    "/tests/", "/test/", "/spec/", "/specs/",
    "/examples/", "/example/", "/migrations/",
    "conftest.py",
    "setup.py", "setup.cfg", "pyproject.toml", "poetry.lock",
    "package_data", "_version", "versioneer", "pkg_resources",
    "conf.py",
    "changelog", "changes", "history", "news",
    "readme", "authors", "license", "requirements",
    "tox.ini", "noxfile",
)

# File extensions that are never Python source code.
NOISE_EXTENSIONS: Tuple[str, ...] = (
    ".md", ".rst", ".txt", ".cfg", ".ini", ".toml",
    ".json", ".yaml", ".yml", ".lock", ".csv", ".html",
    ".svg", ".png", ".pdf",
)


def _is_noise_file(path: str) -> bool:
    """Return True if ``path`` is a file we should not treat as learnable
    source code (docs, tests, packaging, version files, non-code extensions)."""
    p = (path or "").lower()
    if p.endswith(NOISE_EXTENSIONS):
        return True
    return any(pat in p for pat in NOISE_PATH_PATTERNS)


# ---------------------------------------------------------------------------
# 2. Line-level signal: is a line real code, or just noise?
# ---------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"^\s*#")
_DOCSTR_RE = re.compile(r'^\s*("""|\'\'\'|"|\')')
_VERSION_ASSIGN_RE = re.compile(r"^\s*(__version__|version|release|build|tag)\s*=")
_CODE_KEYWORDS = (
    "def ", "class ", "if ", "elif ", "else", "for ", "while ", "try",
    "except", "finally", "with ", "return", "yield", "await", "raise",
    "assert", "import ", "from ", "del ", "global ", "nonlocal ",
    "lambda", "pass", "break", "continue", "async ",
)
_CALL_RE = re.compile(r"[A-Za-z_]\w*\s*\(")
_AUG_RE = re.compile(r"(\+=|-=|\*=|/=|//=|%=|&=|\|=|\^=|<<=|>>=|:=)")
_BINOP_RE = re.compile(r"(==|!=|<=|>=|<<|>>|\band\b|\bor\b|\bin\b|\bnot\b|\bis\b)")


def _has_code_signal(line: str) -> bool:
    """Return True if ``line`` is executable code (a learnable signal) rather
    than a comment, docstring, or bare version assignment."""
    s = line.strip()
    if not s:
        return False
    if _COMMENT_RE.match(s):
        return False
    if _DOCSTR_RE.match(s):
        return False
    if _VERSION_ASSIGN_RE.match(s):
        return False
    if any(s.startswith(k) for k in _CODE_KEYWORDS):
        return True
    if _CALL_RE.search(s):
        return True
    if _AUG_RE.search(s):
        return True
    if _BINOP_RE.search(s):
        return True
    # Plain assignment to a non-version variable, e.g. ``x = foo()``.
    if "=" in s and not s.startswith("="):
        return True
    return False


def code_signal_line_count(text: str) -> int:
    """Count lines in ``text`` that carry learnable code signal."""
    return sum(1 for ln in (text or "").splitlines() if _has_code_signal(ln))


# ---------------------------------------------------------------------------
# 3. Equivalence + trivial-change detection
# ---------------------------------------------------------------------------

def normalize_code(text: str) -> str:
    """Strip blank lines and surrounding whitespace so two snippets compare
    equal when they differ only by formatting/blank lines."""
    return "\n".join(ln.strip() for ln in (text or "").splitlines() if ln.strip())


def texts_are_equivalent(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` are identical once whitespace/blank lines are
    ignored (e.g. ``__version__ = '3.6'`` vs ``__version__ = '3.7'`` are NOT
    equivalent, but two byte-identical functions are)."""
    if not a or not b:
        return False
    return normalize_code(a) == normalize_code(b)


def is_trivial_change(vuln: str, safe: str) -> bool:
    """Return True when the only differences between the vulnerable and fixed
    snippet are non-semantic — i.e. the change carries **no learnable
    vulnerability signal** and the (vuln, safe) pair is pure label noise.

    Specifically it is trivial when:
      * the two snippets are equivalent (identical text), OR
      * every line that actually changed is a comment, docstring, or version
        assignment (no executable code signal on either side of the diff).
    """
    if texts_are_equivalent(vuln, safe):
        return True

    v_lines = [ln.rstrip() for ln in (vuln or "").splitlines()]
    s_lines = [ln.rstrip() for ln in (safe or "").splitlines()]
    sm = difflib.SequenceMatcher(None, v_lines, s_lines)
    changed: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed.extend(v_lines[i1:i2])
        changed.extend(s_lines[j1:j2])

    if not changed:
        return True
    # Non-trivial if ANY changed line carries real code signal.
    return not any(_has_code_signal(ln) for ln in changed)


# ---------------------------------------------------------------------------
# 4. Contradiction scan across a corpus
# ---------------------------------------------------------------------------

def find_contradictions(records: Iterable[Tuple[str, int]]) -> List[str]:
    """Given ``(text, label)`` pairs, return the normalized texts that appear
    with **both** labels. These are hard contradictions (identical input,
    opposite ground truth) and must be dropped before training."""
    by_text: Dict[str, int] = {}
    contradictions: List[str] = []
    for text, label in records:
        norm = normalize_code(text)
        if not norm:
            continue
        if norm in by_text:
            if by_text[norm] != label:
                contradictions.append(norm)
        else:
            by_text[norm] = label
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in contradictions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
