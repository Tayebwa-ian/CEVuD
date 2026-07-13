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
 * ``changed_line_ratio(a, b)`` — fraction of lines that differ between two
                                      snippets (0.0–1.0). Used by the
                                      safe-counterpart diagnostic.
 * ``is_substantial_change(a, b, t)`` — True when the post-fix function differs
                                      from its vulnerable twin across more than
                                      ``t`` of their lines (a bundled refactor /
                                      rework, not just the security patch).
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


def changed_line_ratio(a: str, b: str) -> float:
    """Fraction of lines that differ between ``a`` and ``b`` (0.0–1.0).

    Computed with difflib.SequenceMatcher over rstripped lines. Used by the
    safe-counterpart diagnostic (``src/scripts/diagnose_safe_counterparts.py``)
    to flag post-fix functions that were *heavily* reworked beyond the actual
    security patch — a sign the ``label=0`` sample may carry spurious,
    non-security differences from its vulnerable twin (which would teach the
    classifier the *refactor* rather than the *vulnerability*).
    """
    if not a or not b:
        return 1.0
    a_lines = [ln.rstrip() for ln in a.splitlines()]
    b_lines = [ln.rstrip() for ln in b.splitlines()]
    if not a_lines and not b_lines:
        return 0.0
    sm = difflib.SequenceMatcher(None, a_lines, b_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed += (i2 - i1) + (j2 - j1)
    total = len(a_lines) + len(b_lines)
    return (changed / total) if total else 0.0


def is_substantial_change(a: str, b: str, threshold: float = 0.5) -> bool:
    """True when ``a`` and ``b`` differ across more than ``threshold`` of their
    (combined) lines — i.e. the fix commit did more than patch the
    vulnerability (it refactored, reformatted, or reworked the function).

    Such pairs are a contamination risk for the (vulnerable, safe) training
    signal: the classifier may learn the *refactor* rather than the
    *vulnerability*. See ``docs/SAFE_COUNTERPARTS.md``.
    """
    return changed_line_ratio(a, b) > threshold


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


# ---------------------------------------------------------------------------
# 5. Token-level near-duplicate detection (the safe-counterpart guard)
# ---------------------------------------------------------------------------
# The contradiction scan above only catches *byte-identical* (normalized) text
# with opposite labels. The dominant failure mode of the CVEfixes "safe = the
# post-fix twin" design is subtler: the safe function is 1–2 lines different
# from its vulnerable twin — near-identical text with opposite labels. A
# classifier collapses to P=0.5 on such pairs exactly as it does on exact
# contradictions. The helpers below quantify that *near*-duplication with a
# cheap token-based similarity so the miner / builder / trainer can drop a
# candidate safe sample that is too similar to any vulnerable sample. See
# docs/SAFE_COUNTERPARTS.md and docs/DATA_QUALITY.md.

# Identifiers/keywords OR single non-space punctuation tokens. Operating on
# tokens (not raw characters) makes the ratio robust to whitespace/formatting
# and reflects real code-structure overlap.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s\w]")


def tokenize_code(text: str) -> List[str]:
    """Tokenize ``text`` into identifiers, numbers, and punctuation tokens.

    Comments and blank lines are ignored so the similarity reflects executable
    structure rather than documentation. Used by :func:`token_similarity`.
    """
    out: List[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or _COMMENT_RE.match(s):
            continue
        out.extend(_TOKEN_RE.findall(s))
    return out


def _sim_from_tokens(ta: List[str], tb: List[str], floor: float = 0.0) -> float:
    """SequenceMatcher token ratio with cheap upper-bound short-circuits.

    ``floor`` lets callers skip the expensive ``ratio()`` when even the cheap
    length/quick bounds already fall at or below the best score seen so far.
    """
    la, lb = len(ta), len(tb)
    if la == 0 or lb == 0:
        return 0.0
    # ratio() <= 2*min/(la+lb); bail early when it cannot beat the floor.
    if (2.0 * min(la, lb)) / (la + lb) <= floor:
        return 0.0
    sm = difflib.SequenceMatcher(None, ta, tb)
    if sm.real_quick_ratio() <= floor:
        return 0.0
    if sm.quick_ratio() <= floor:
        return 0.0
    return sm.ratio()


def token_similarity(a: str, b: str) -> float:
    """Return the token-level similarity ratio of two code snippets in [0, 1].

    1.0 means the two snippets tokenize identically; ~0.94 is the median
    vuln↔safe similarity measured on the broken CVEfixes twins. Values above
    ~0.75 indicate the two snippets differ by only a handful of tokens — the
    near-duplicate contradiction we must keep out of the (vuln, safe) split.
    """
    return _sim_from_tokens(tokenize_code(a), tokenize_code(b))


def max_token_similarity(text: str, ref_token_lists: Iterable[List[str]]) -> float:
    """Highest token similarity between ``text`` and any pre-tokenized snippet
    in ``ref_token_lists`` (see :func:`tokenize_code`).

    Pre-tokenizing the reference set once (the vulnerable snippets) makes the
    guard practical to run for every candidate safe sample.
    """
    tt = tokenize_code(text)
    if not tt:
        return 0.0
    best = 0.0
    for rt in ref_token_lists:
        r = _sim_from_tokens(tt, rt, floor=best)
        if r > best:
            best = r
            if best >= 1.0:
                break
    return best


def count_cross_label_near_duplicates(
    records: Iterable[Tuple[str, int]], threshold: float = 0.90
) -> int:
    """Count ``label=0`` texts that are >``threshold`` token-similar to some
    ``label=1`` text (near-duplicate contradictions across the class boundary).

    This is the *near*-duplicate complement of :func:`find_contradictions`
    (which only finds byte-identical opposite-label text). Used by the trainer's
    pre-flight guard to refuse a dataset whose safe class is just lightly-edited
    copies of the vulnerable class. To stay tractable on large splits the
    vulnerable side is pre-tokenized once and the length short-circuits in
    :func:`_sim_from_tokens` skip the O(n·m) worst case for most pairs.
    """
    vuln_tokens: List[List[str]] = []
    safe_texts: List[str] = []
    for text, label in records:
        if label == 1:
            toks = tokenize_code(text)
            if toks:
                vuln_tokens.append(toks)
        else:
            safe_texts.append(text)
    if not vuln_tokens:
        return 0
    count = 0
    for text in safe_texts:
        if max_token_similarity(text, vuln_tokens) > threshold:
            count += 1
    return count
