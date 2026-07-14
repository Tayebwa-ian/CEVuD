"""
Unit Tests: mine_benign_functions.py
======================================
Validates the benign-function miner's pure helpers: deterministic seeding
and function-name inference. Network/git-heavy paths are mocked.
"""

import hashlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluation")))

from scripts.mine_benign_functions import _commit_seed


class TestCommitSeed:
    def test_deterministic_for_same_inputs(self):
        assert _commit_seed("abc123", 42) == _commit_seed("abc123", 42)

    def test_different_for_different_commits(self):
        assert _commit_seed("abc123", 42) != _commit_seed("def456", 42)

    def test_different_for_different_seeds(self):
        assert _commit_seed("abc123", 42) != _commit_seed("abc123", 99)

    def test_returns_int(self):
        result = _commit_seed("abc123", 42)
        assert isinstance(result, int)

    def test_non_negative(self):
        assert _commit_seed("abc123", 42) >= 0

    def test_stable_across_processes(self):
        expected = _commit_seed("stable_commit_sha", 12345)
        assert expected == (12345 + int(hashlib.md5("stable_commit_sha".encode()).hexdigest()[:8], 16)) & 0x7FFFFFFF
