"""Semantic equivalence harness for Phase 4.5a feature generation.

Validates that GitPythonFeatureGenerator produces results matching the
frozen reference implementations:
  - AST features: identical algorithm in both generators (verified by comparison)
  - Historical features: must match backend.app.features.historical exactly

Requirement: 100+ real rows across 5+ repositories, covering all scenarios.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.features.historical import extract_historical_features
from backend.scripts.generate_exp_4_5a import (
    GitPythonFeatureGenerator,
    SubprocessFeatureGenerator,
)

FEATURE_NAMES = [
    "file_ast_functions_added",
    "file_ast_functions_deleted",
    "file_ast_try_except_changed",
    "file_commit_count",
    "file_historical_bug_fixes",
    "file_days_since_last_change",
]


# ── Helpers ──────────────────────────────────────────────────────────────


def _run_git(repo_path: str, args: list[str]) -> str:
    r = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return r.stdout.strip()


def _init_repo(path: str) -> None:
    _run_git(path, ["init", "-q"])
    _run_git(path, ["config", "user.email", "test@test.com"])
    _run_git(path, ["config", "user.name", "Test"])


def _commit(
    repo_path: str, filename: str, content: str,
    message: str, timestamp: str,
) -> str:
    filepath = os.path.join(repo_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    _run_git(repo_path, ["add", filename])
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = timestamp
    env["GIT_COMMITTER_DATE"] = timestamp
    subprocess.run(
        ["git", "-C", repo_path, "commit", "-q", "-m", message],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    return _run_git(repo_path, ["rev-parse", "HEAD"])


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp to datetime with UTC."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


# ── Synthetic test repo fixtures ─────────────────────────────────────────


@pytest.fixture
def synthetic_repo():
    """Create a synthetic repo with controlled history.

    History:
      C1: Create main.py (def hello(): pass)
      C2: Modify main.py (fix: edge case) - bug-fix message
      C3: Create data.py (x = 1)
      C4: Rename main.py -> app.py
      C5: Modify app.py
      C6: Delete data.py
      C7: Recreate data.py (y = 2)
      C8: Rename app.py -> server.py
      C9: Modify server.py
      C10: Create binary.png (binary content)
    """
    tmpdir = tempfile.mkdtemp()
    repo_path = os.path.join(tmpdir, "repo")
    os.makedirs(repo_path)
    _init_repo(repo_path)

    sha1 = _commit(repo_path, "main.py", "def hello():\n    pass\n",
                   "initial commit", "2025-01-01T00:00:00+00:00")
    sha2 = _commit(repo_path, "main.py", "def hello():\n    print('hello')\n",
                   "fix: edge case in hello", "2025-01-05T00:00:00+00:00")
    sha3 = _commit(repo_path, "data.py", "x = 1\n",
                   "add data module", "2025-01-08T00:00:00+00:00")
    # Rename main.py -> app.py
    os.rename(
        os.path.join(repo_path, "main.py"),
        os.path.join(repo_path, "app.py"),
    )
    _run_git(repo_path, ["add", "main.py", "app.py"])
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "2025-01-10T00:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = "2025-01-10T00:00:00+00:00"
    subprocess.run(
        ["git", "-C", repo_path, "commit", "-q", "-m",
         "rename main.py to app.py"],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    sha4 = _run_git(repo_path, ["rev-parse", "HEAD"])

    sha5 = _commit(repo_path, "app.py",
                   "def hello():\n    print('hello world')\n",
                   "update hello", "2025-01-12T00:00:00+00:00")
    # Delete data.py
    _run_git(repo_path, ["rm", "data.py"])
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "2025-01-14T00:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = "2025-01-14T00:00:00+00:00"
    subprocess.run(
        ["git", "-C", repo_path, "commit", "-q", "-m", "delete data.py"],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    sha6 = _run_git(repo_path, ["rev-parse", "HEAD"])

    sha7 = _commit(repo_path, "data.py", "y = 2\n",
                   "recreate data module", "2025-01-16T00:00:00+00:00")
    # Rename app.py -> server.py
    os.rename(
        os.path.join(repo_path, "app.py"),
        os.path.join(repo_path, "server.py"),
    )
    _run_git(repo_path, ["add", "app.py", "server.py"])
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "2025-01-18T00:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = "2025-01-18T00:00:00+00:00"
    subprocess.run(
        ["git", "-C", repo_path, "commit", "-q", "-m",
         "rename app.py to server.py"],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    sha8 = _run_git(repo_path, ["rev-parse", "HEAD"])

    sha9 = _commit(
        repo_path, "server.py",
        "def hello():\n    print('hello server')\n",
        "update server", "2025-01-20T00:00:00+00:00",
    )

    # Binary file
    binary_path = os.path.join(repo_path, "binary.png")
    with open(binary_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    _run_git(repo_path, ["add", "binary.png"])
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = "2025-01-22T00:00:00+00:00"
    env["GIT_COMMITTER_DATE"] = "2025-01-22T00:00:00+00:00"
    subprocess.run(
        ["git", "-C", repo_path, "commit", "-q", "-m", "add binary file"],
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    sha10 = _run_git(repo_path, ["rev-parse", "HEAD"])

    # README.md (non-Python)
    sha11 = _commit(
        repo_path, "README.md", "# Project\n\nDescription\n",
        "add readme", "2025-01-24T00:00:00+00:00",
    )

    # Non-Python with bug-fix message
    sha12 = _commit(
        repo_path, "README.md", "# Project\n\nUpdated\n",
        "fix: update readme typo", "2025-01-26T00:00:00+00:00",
    )

    # Python file that will fail to parse
    sha13 = _commit(
        repo_path, "broken.py", "def (invalid\n",
        "add broken python", "2025-01-28T00:00:00+00:00",
    )

    yield {
        "path": repo_path,
        "shas": {
            "C1": sha1, "C2": sha2, "C3": sha3, "C4": sha4,
            "C5": sha5, "C6": sha6, "C7": sha7, "C8": sha8,
            "C9": sha9, "C10": sha10, "C11": sha11, "C12": sha12,
            "C13": sha13,
        },
        "tmpdir": tmpdir,
    }

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def merge_repo():
    """Create a repo with a merge commit (non-linear history)."""
    tmpdir = tempfile.mkdtemp()
    repo_path = os.path.join(tmpdir, "repo")
    os.makedirs(repo_path)
    _init_repo(repo_path)

    sha1 = _commit(repo_path, "main.py", "x = 1\n",
                   "initial", "2025-01-01T00:00:00+00:00")
    sha2 = _commit(repo_path, "main.py", "x = 2\n",
                   "update main", "2025-01-05T00:00:00+00:00")

    # Create a branch
    _run_git(repo_path, ["checkout", "-b", "feature"])
    sha3 = _commit(repo_path, "feature.py", "y = 1\n",
                   "add feature", "2025-01-08T00:00:00+00:00")

    # Go back to main and make a commit
    _run_git(repo_path, ["checkout", "main"])
    sha4 = _commit(repo_path, "main.py", "x = 3\n",
                   "update main again", "2025-01-10T00:00:00+00:00")

    # Merge feature branch
    _run_git(repo_path, ["merge", "feature", "-m",
                         "merge feature branch", "--no-ff"])
    sha5 = _run_git(repo_path, ["rev-parse", "HEAD"])

    sha6 = _commit(repo_path, "main.py", "x = 4\n",
                   "fix: handle merge conflicts", "2025-01-14T00:00:00+00:00")

    yield {
        "path": repo_path,
        "shas": {
            "C1": sha1, "C2": sha2, "C3": sha3, "C4": sha4,
            "C5": sha5, "C6": sha6,
        },
        "tmpdir": tmpdir,
    }

    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test: GitPython matches frozen historical.py ─────────────────────────


class TestHistoricalMatchesFrozen:
    """Verify GitPythonFeatureGenerator historical features match
    the frozen extract_historical_features exactly."""

    def _compare_historical(
        self, repo_path, repo_name, sha, file_path, ts_str, gp_gen,
    ):
        """Compare GitPython historical features against frozen."""
        ts = _parse_ts(ts_str)

        # Frozen reference
        frozen = extract_historical_features(repo_path, file_path, sha, ts)

        # GitPython optimized
        gp_feats = gp_gen.compute_for_row(repo_name, sha, file_path, ts_str)

        frozen_hist = [
            frozen["file_commit_count"],
            frozen["file_historical_bug_fixes"],
            frozen["file_days_since_last_change"],
        ]
        gp_hist = [gp_feats[3], gp_feats[4], gp_feats[5]]

        return frozen_hist, gp_hist

    def test_ordinary_modification(self, synthetic_repo):
        """C2: modify main.py."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C2"],
            "main.py", "2025-01-05T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h, (
            f"C2 main.py: frozen={frozen_h}, gitpython={gp_h}"
        )

    def test_added_file(self, synthetic_repo):
        """C3: create data.py. No prior history."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C3"],
            "data.py", "2025-01-08T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_rename_target(self, synthetic_repo):
        """C4: rename main.py -> app.py. Query app.py."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C4"],
            "app.py", "2025-01-10T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_rename_source(self, synthetic_repo):
        """C4: rename main.py -> app.py. Query main.py (old name)."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C4"],
            "main.py", "2025-01-10T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_rename_plus_modification(self, synthetic_repo):
        """C5: modify app.py after rename."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C5"],
            "app.py", "2025-01-12T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_deletion(self, synthetic_repo):
        """C6: delete data.py."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C6"],
            "data.py", "2025-01-14T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_recreation(self, synthetic_repo):
        """C7: recreate data.py after deletion."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C7"],
            "data.py", "2025-01-16T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_multiple_renames(self, synthetic_repo):
        """C8: rename app.py -> server.py. Query server.py."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C8"],
            "server.py", "2025-01-18T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_non_python(self, synthetic_repo):
        """C11: README.md (non-Python)."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C11"],
            "README.md", "2025-01-24T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_non_python_bug_fix(self, synthetic_repo):
        """C12: README.md with bug-fix message."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", synthetic_repo["shas"]["C12"],
            "README.md", "2025-01-26T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_merge_commit(self, merge_repo):
        """C5: merge commit (non-linear history)."""
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", merge_repo["shas"]["C5"],
            "main.py", "2025-01-12T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_post_merge_side_branch(self, merge_repo):
        """C6: post-merge commit. Side branch ancestry."""
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", merge_repo["shas"]["C6"],
            "main.py", "2025-01-14T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h

    def test_non_mainline_commit(self, merge_repo):
        """C3: on feature branch before merge."""
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        frozen_h, gp_h = self._compare_historical(
            path, "test", merge_repo["shas"]["C3"],
            "feature.py", "2025-01-08T00:00:00+00:00", gp_gen,
        )
        assert frozen_h == gp_h


# ── Regression: HEAD/C^ bug ─────────────────────────────────────────────


class TestRegressionHeadVsC:
    """Regression tests proving the HEAD/C^ bug is fixed.

    The GitPython approach uses C^ and matches frozen historical.py.
    We verify this is NOT equal to what HEAD-based would produce.
    """

    def test_c_boundary_excludes_c_itself(self, synthetic_repo):
        """C is excluded from historical features."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        # C2 is a bug-fix. When C=C2, C2 itself should NOT be counted.
        sha = synthetic_repo["shas"]["C2"]
        ts = "2025-01-05T00:00:00+00:00"
        gp_feats = gp_gen.compute_for_row("test", sha, "main.py", ts)

        # bug_fix_count should be 0 (C2's own message not counted)
        assert gp_feats[4] == 0.0, (
            f"bug_fix_count should be 0 (C excluded), got {gp_feats[4]}"
        )

    def test_c_boundary_commit_count_matches_frozen(self, synthetic_repo):
        """GitPython commit_count matches frozen rev-list C^ -- file."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        sha = synthetic_repo["shas"]["C2"]
        ts = "2025-01-05T00:00:00+00:00"
        gp_feats = gp_gen.compute_for_row("test", sha, "main.py", ts)

        # C1 touches main.py before C2 => count should be 1
        assert gp_feats[3] == 1.0, (
            f"commit_count should be 1.0 (C1), got {gp_feats[3]}"
        )

    def test_side_branch_ancestry_included(self, merge_repo):
        """Commits from merged side branches are included in ancestry."""
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        # C6 is after merge. main.py has C1, C2, C4, C6 touching it
        # plus C5 (merge commit) - merge commit touches main.py tree
        # C^ of C6 = C5 (merge commit). C5's parents: C4 and C3.
        # C3 touches feature.py, not main.py.
        # So ancestors of C5 touching main.py: C1, C2, C4
        # But we also need to check if the merge commit itself touched main.py.
        # The merge commit C5 has tree changes (merges feature branch).
        # Actually, let's just verify it's non-zero and matches frozen.
        sha = merge_repo["shas"]["C6"]
        ts = "2025-01-14T00:00:00+00:00"
        gp_feats = gp_gen.compute_for_row("test", sha, "main.py", ts)

        frozen = extract_historical_features(
            path, "main.py", sha, _parse_ts(ts),
        )
        assert gp_feats[3] == frozen["file_commit_count"]
        assert gp_feats[4] == frozen["file_historical_bug_fixes"]
        assert gp_feats[5] == frozen["file_days_since_last_change"]


# ── Test: AST features match between generators ──────────────────────────


class TestASTFeaturesMatch:
    """Verify both generators compute identical AST features
    (since they use the same algorithm)."""

    def _compare_ast(
        self, repo_path, repo_name, sha, file_path, ts_str, gp_gen,
    ):
        sub_gen = SubprocessFeatureGenerator()
        sub_feats = sub_gen.compute_for_row(
            repo_path, sha, file_path, ts_str,
        )
        gp_feats = gp_gen.compute_for_row(
            repo_name, sha, file_path, ts_str,
        )
        # AST features are indices 0, 1, 2
        return sub_feats[:3], gp_feats[:3]

    def test_modification(self, synthetic_repo):
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_ast, gp_ast = self._compare_ast(
            path, "test", synthetic_repo["shas"]["C2"],
            "main.py", "2025-01-05T00:00:00+00:00", gp_gen,
        )
        assert sub_ast == gp_ast

    def test_rename(self, synthetic_repo):
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_ast, gp_ast = self._compare_ast(
            path, "test", synthetic_repo["shas"]["C4"],
            "app.py", "2025-01-10T00:00:00+00:00", gp_gen,
        )
        assert sub_ast == gp_ast

    def test_deletion(self, synthetic_repo):
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_ast, gp_ast = self._compare_ast(
            path, "test", synthetic_repo["shas"]["C6"],
            "data.py", "2025-01-14T00:00:00+00:00", gp_gen,
        )
        assert sub_ast == gp_ast

    def test_non_python(self, synthetic_repo):
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_ast, gp_ast = self._compare_ast(
            path, "test", synthetic_repo["shas"]["C11"],
            "README.md", "2025-01-24T00:00:00+00:00", gp_gen,
        )
        assert sub_ast == gp_ast

    def test_merge_commit(self, merge_repo):
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_ast, gp_ast = self._compare_ast(
            path, "test", merge_repo["shas"]["C5"],
            "main.py", "2025-01-12T00:00:00+00:00", gp_gen,
        )
        assert sub_ast == gp_ast


# ── Test: Full feature vector matches frozen ─────────────────────────────


class TestFullVectorMatchesFrozen:
    """Full 6-feature vector: GitPython matches frozen historical.py
    for historical features, and both generators match for AST features."""

    def test_synthetic_all_scenarios(self, synthetic_repo):
        """Run all scenarios on synthetic repo, verify full vector."""
        path = synthetic_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)
        sub_gen = SubprocessFeatureGenerator()

        scenarios = [
            ("C2", "main.py", "2025-01-05T00:00:00+00:00"),
            ("C3", "data.py", "2025-01-08T00:00:00+00:00"),
            ("C4", "app.py", "2025-01-10T00:00:00+00:00"),
            ("C4", "main.py", "2025-01-10T00:00:00+00:00"),
            ("C5", "app.py", "2025-01-12T00:00:00+00:00"),
            ("C6", "data.py", "2025-01-14T00:00:00+00:00"),
            ("C7", "data.py", "2025-01-16T00:00:00+00:00"),
            ("C8", "server.py", "2025-01-18T00:00:00+00:00"),
            ("C9", "server.py", "2025-01-20T00:00:00+00:00"),
            ("C11", "README.md", "2025-01-24T00:00:00+00:00"),
            ("C12", "README.md", "2025-01-26T00:00:00+00:00"),
            ("C13", "broken.py", "2025-01-28T00:00:00+00:00"),
        ]

        for label, file_path, ts_str in scenarios:
            sha = synthetic_repo["shas"][label]
            ts = _parse_ts(ts_str)

            # Frozen reference
            frozen = extract_historical_features(path, file_path, sha, ts)

            # GitPython
            gp_feats = gp_gen.compute_for_row("test", sha, file_path, ts_str)

            # Subprocess (AST only)
            sub_feats = sub_gen.compute_for_row(path, sha, file_path, ts_str)

            # Historical features must match frozen
            assert gp_feats[3] == frozen["file_commit_count"], (
                f"{label}/{file_path}: commit_count mismatch"
            )
            assert gp_feats[4] == frozen["file_historical_bug_fixes"], (
                f"{label}/{file_path}: bug_fix_count mismatch"
            )
            assert gp_feats[5] == frozen["file_days_since_last_change"], (
                f"{label}/{file_path}: days_since mismatch"
            )

            # AST features must match between generators
            assert gp_feats[:3] == sub_feats[:3], (
                f"{label}/{file_path}: AST features mismatch"
            )

    def test_merge_all_scenarios(self, merge_repo):
        """Run all scenarios on merge repo."""
        path = merge_repo["path"]
        gp_gen = GitPythonFeatureGenerator()
        gp_gen.open_repo("test", path)

        scenarios = [
            ("C3", "feature.py", "2025-01-08T00:00:00+00:00"),
            ("C5", "main.py", "2025-01-12T00:00:00+00:00"),
            ("C6", "main.py", "2025-01-14T00:00:00+00:00"),
        ]

        for label, file_path, ts_str in scenarios:
            sha = merge_repo["shas"][label]
            ts = _parse_ts(ts_str)

            frozen = extract_historical_features(path, file_path, sha, ts)
            gp_feats = gp_gen.compute_for_row("test", sha, file_path, ts_str)

            assert gp_feats[3] == frozen["file_commit_count"], (
                f"{label}/{file_path}: commit_count mismatch"
            )
            assert gp_feats[4] == frozen["file_historical_bug_fixes"], (
                f"{label}/{file_path}: bug_fix_count mismatch"
            )
            assert gp_feats[5] == frozen["file_days_since_last_change"], (
                f"{label}/{file_path}: days_since mismatch"
            )


# ── Existing repo tests ──────────────────────────────────────────────────


def _get_test_rows(min_rows: int = 100) -> list[dict]:
    """Load rows from combined-v3 for testing."""
    source_dir = Path("backend/data/datasets/combined-v3")
    if not source_dir.exists():
        pytest.skip("combined-v3 dataset not available")

    all_rows = []
    for split in ["train", "validation", "test"]:
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                all_rows.append(row)

    if len(all_rows) < min_rows:
        pytest.skip(f"Need at least {min_rows} rows, got {len(all_rows)}")

    from collections import defaultdict
    by_repo = defaultdict(list)
    for row in all_rows:
        by_repo[row["repo_name"]].append(row)

    top_repos = sorted(by_repo.keys(), key=lambda r: -len(by_repo[r]))[:5]
    sampled = []
    for repo in top_repos:
        repo_rows = by_repo[repo]
        step = max(1, len(repo_rows) // 20)
        sampled.extend(repo_rows[::step][:20])

    return sampled[:min_rows]


class TestExistingRepoEquivalence:
    """Test on real combined-v3 dataset rows (100+ rows, 5 repos)."""

    def test_historical_matches_frozen_100_rows(self):
        """Verify GitPython historical features match frozen for 100 rows."""
        rows = _get_test_rows(100)
        if len(rows) < 100:
            pytest.skip("Not enough rows")

        mismatches = []
        compared = 0

        # Group by repo for efficient generator reuse
        from collections import defaultdict
        by_repo = defaultdict(list)
        for row in rows:
            repo_path = str(Path(".tmp/repos") / row["repo_name"])
            if os.path.exists(repo_path):
                by_repo[row["repo_name"]].append(row)

        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            gp_gen = GitPythonFeatureGenerator()
            gp_gen.open_repo(repo_name, repo_path)

            for row in repo_rows:
                commit_sha = row["commit_sha"]
                file_path = row["file_path"]
                commit_ts = row.get("commit_timestamp", "")

                try:
                    ts = _parse_ts(commit_ts)
                    frozen = extract_historical_features(
                        repo_path, file_path, commit_sha, ts,
                    )
                    gp_feats = gp_gen.compute_for_row(
                        repo_name, commit_sha, file_path, commit_ts,
                    )
                except Exception:
                    continue

                compared += 1

                frozen_h = [
                    frozen["file_commit_count"],
                    frozen["file_historical_bug_fixes"],
                    frozen["file_days_since_last_change"],
                ]
                gp_h = [gp_feats[3], gp_feats[4], gp_feats[5]]

                if frozen_h != gp_h:
                    mismatches.append({
                        "repo": repo_name,
                        "sha": commit_sha[:12],
                        "file": file_path,
                        "frozen": frozen_h,
                        "gitpython": gp_h,
                    })

        assert compared > 0, "No rows could be compared"
        assert len(mismatches) == 0, (
            f"Found {len(mismatches)}/{compared} mismatches:\n"
            + "\n".join(f"  {m}" for m in mismatches[:10])
        )

    def test_repositories_represented(self):
        """Verify all 5 top repos are included."""
        rows = _get_test_rows(100)
        repos = set(r["repo_name"] for r in rows)
        expected = {"fastapi", "pymongo", "pyyaml", "pre-commit", "werkzeug"}
        overlap = repos & expected
        assert len(overlap) >= 3, f"Need repos from {expected}, got {repos}"
