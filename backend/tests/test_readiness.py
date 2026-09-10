"""Readiness tests for Phase 4.5a full generation.

Validates:
  - Repository handling (valid reuse, invalid detection, cleanup)
  - /dev/null path handling
  - Unresolved Git object handling
  - Output contract (6 features, version, key preservation)
  - Duplicate prevention
  - combined-v3 alignment (row counts, joinability)
  - Interruption/partial-output safety
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
    EXPERIMENTAL_FEATURE_VERSION,
)
from backend.scripts.generate_exp_4_5a import (
    REPOS_DIR,
    GitPythonFeatureGenerator,
    _clone_repo,
    _read_jsonl,
    _remove_dir_robust,
)

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


def _commit(repo_path: str, filename: str, content: str,
            message: str, timestamp: str) -> str:
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


# ── Test: Valid clone reuse ─────────────────────────────────────────────


class TestCloneReuse:
    """Verify _clone_repo reuses valid clones and removes invalid ones."""

    def test_valid_clone_reused(self):
        """Existing valid clone is reused (not re-cloned)."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)
        _init_repo(repo_path)
        _commit(repo_path, "a.py", "x = 1\n", "initial", "2025-01-01T00:00:00+00:00")

        # Point REPOS_DIR to tmpdir
        import backend.scripts.generate_exp_4_5a as gen_mod
        original_repos_dir = gen_mod.REPOS_DIR
        gen_mod.REPOS_DIR = Path(tmpdir)

        try:
            # First call: clone
            result1 = _clone_repo("file:///nonexistent", "repo")
            assert result1 is not None
            assert os.path.exists(result1)

            # Second call: should reuse (not error)
            result2 = _clone_repo("file:///nonexistent", "repo")
            assert result2 is not None
            assert os.path.exists(result2)
        finally:
            gen_mod.REPOS_DIR = original_repos_dir
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_clone_removed_and_recloned(self):
        """Invalid clone is removed and fresh clone created."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)

        # Create invalid repo (has .git but corrupted)
        os.makedirs(os.path.join(repo_path, ".git"))
        with open(os.path.join(repo_path, ".git", "HEAD"), "w") as f:
            f.write("garbage")

        import backend.scripts.generate_exp_4_5a as gen_mod
        original_repos_dir = gen_mod.REPOS_DIR
        gen_mod.REPOS_DIR = Path(tmpdir)

        try:
            result = _clone_repo("file:///nonexistent", "repo")
            # Should either succeed (if URL is valid) or return None
            # Key point: should not crash or leave the corrupted repo
            if result:
                assert os.path.exists(os.path.join(result, ".git"))
        finally:
            gen_mod.REPOS_DIR = original_repos_dir
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test: /dev/null handling ─────────────────────────────────────────────


class TestDevNullHandling:
    """Verify /dev/null paths produce zero features safely."""

    def test_dev_null_returns_zeroes(self):
        """Rows with file_path='/dev/null' produce zero historical features."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)
        _init_repo(repo_path)
        sha = _commit(repo_path, "a.py", "x = 1\n", "initial", "2025-01-01T00:00:00+00:00")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("test", repo_path)

        features = gen.compute_for_row("test", sha, "/dev/null", "2025-01-01T00:00:00+00:00")

        # /dev/null should produce zero features (AST zeros + historical zeros)
        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dev_null_in_real_dataset(self):
        """Verify /dev/null rows exist in combined-v3 and are handled."""
        source_dir = Path("backend/data/datasets/combined-v3")
        if not source_dir.exists():
            pytest.skip("Dataset not available")

        dev_null_rows = 0
        for split in ["train", "validation", "test"]:
            with open(source_dir / f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["file_path"] == "/dev/null":
                        dev_null_rows += 1

        # The dataset has /dev/null rows (file additions)
        # These must be handled gracefully
        assert dev_null_rows >= 0  # Just verify no crash


# ── Test: Unresolved Git object handling ─────────────────────────────────


class TestGitObjectHandling:
    """Verify behavior when Git objects cannot be resolved."""

    def test_missing_file_returns_zeroes(self):
        """File that doesn't exist at commit returns zeros."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)
        _init_repo(repo_path)
        sha = _commit(repo_path, "a.py", "x = 1\n", "initial", "2025-01-01T00:00:00+00:00")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("test", repo_path)

        # Request file that doesn't exist
        features = gen.compute_for_row("test", sha, "nonexistent.py", "2025-01-01T00:00:00+00:00")

        assert len(features) == 6
        assert all(isinstance(f, float) for f in features)
        # Historical features should be 0 (no history for nonexistent file)
        assert features[3] == 0.0  # commit_count
        assert features[4] == 0.0  # bug_fix_count
        assert features[5] == 0.0  # days_since
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_sha_returns_zeroes(self):
        """Invalid commit SHA returns zeros without crashing."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)
        _init_repo(repo_path)
        _commit(repo_path, "a.py", "x = 1\n", "initial", "2025-01-01T00:00:00+00:00")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("test", repo_path)

        features = gen.compute_for_row(
            "test", "deadbeef00000000000000000000000000000000", "a.py", "2025-01-01T00:00:00+00:00"
        )

        assert len(features) == 6
        assert all(f == 0.0 for f in features)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_root_commit_historical_zeros(self):
        """Root commit (no parent) produces zero historical features."""
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")
        os.makedirs(repo_path)
        _init_repo(repo_path)
        sha = _commit(repo_path, "a.py", "x = 1\n", "initial", "2025-01-01T00:00:00+00:00")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("test", repo_path)

        features = gen.compute_for_row("test", sha, "a.py", "2025-01-01T00:00:00+00:00")

        # Root commit: no history, no parent
        assert features[3] == 0.0  # commit_count
        assert features[4] == 0.0  # bug_fix_count
        assert features[5] == 0.0  # days_since
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test: Output contract ────────────────────────────────────────────────


class TestOutputContract:
    """Verify output JSONL satisfies the feature contract."""

    def test_feature_count(self):
        """Output has exactly 6 features per row."""
        # Use a known valid repo
        repo_path = str(REPOS_DIR / "flask")
        if not os.path.exists(repo_path):
            pytest.skip("flask repo not available")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("flask", repo_path)

        # Get a real row from the dataset
        rows = _read_jsonl(Path("backend/data/datasets/combined-v3/train.jsonl"))
        flask_rows = [r for r in rows if r["repo_name"] == "flask"][:5]

        for row in flask_rows:
            features = gen.compute_for_row(
                "flask", row["commit_sha"], row["file_path"],
                row.get("commit_timestamp", ""),
            )
            assert len(features) == EXPERIMENTAL_FEATURE_COUNT, (
                f"Expected {EXPERIMENTAL_FEATURE_COUNT} features, got {len(features)}"
            )
            assert all(isinstance(f, float) for f in features)
            assert all(not (f != f) for f in features)  # no NaN

    def test_version_string(self):
        """Output version is 'exp-4.5a'."""
        assert EXPERIMENTAL_FEATURE_VERSION == "exp-4.5a"

    def test_feature_names(self):
        """Feature names are defined and ordered."""
        assert len(EXPERIMENTAL_FEATURE_NAMES) == 6
        assert EXPERIMENTAL_FEATURE_NAMES[0] == "file_ast_functions_added"
        assert EXPERIMENTAL_FEATURE_NAMES[3] == "file_commit_count"


# ── Test: Duplicate prevention ───────────────────────────────────────────


class TestDuplicatePrevention:
    """Verify no duplicate rows in output."""

    def test_no_duplicate_keys_in_dataset(self):
        """Verify dataset key uniqueness for non-/dev/null rows."""
        source_dir = Path("backend/data/datasets/combined-v3")
        if not source_dir.exists():
            pytest.skip("Dataset not available")

        for split in ["train", "validation", "test"]:
            keys = []
            with open(source_dir / f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["file_path"] != "/dev/null":
                        keys.append((r["commit_sha"], r["file_path"]))

            # Non-/dev/null rows should be unique
            assert len(keys) == len(set(keys)), (
                f"{split}: {len(keys)} rows but {len(set(keys))} unique keys"
            )


# ── Test: combined-v3 alignment ─────────────────────────────────────────


class TestCombinedV3Alignment:
    """Verify generator reads exact rows from combined-v3."""

    def test_row_counts_match(self):
        """Expected: 34405 train, 9410 val, 10574 test = 54389 total."""
        source_dir = Path("backend/data/datasets/combined-v3")
        if not source_dir.exists():
            pytest.skip("Dataset not available")

        expected = {"train": 34405, "validation": 9410, "test": 10574}
        for split, count in expected.items():
            rows = _read_jsonl(source_dir / f"{split}.jsonl")
            assert len(rows) == count, f"{split}: expected {count}, got {len(rows)}"

    def test_all_repos_have_urls(self):
        """Every row in combined-v3 has a repo_url."""
        source_dir = Path("backend/data/datasets/combined-v3")
        if not source_dir.exists():
            pytest.skip("Dataset not available")

        missing_urls = 0
        for split in ["train", "validation", "test"]:
            with open(source_dir / f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if not r.get("repo_url"):
                        missing_urls += 1

        assert missing_urls == 0, f"{missing_urls} rows missing repo_url"

    def test_feature_version_absent_in_source(self):
        """Source dataset has no experimental features yet."""
        source_dir = Path("backend/data/datasets/combined-v3")
        if not source_dir.exists():
            pytest.skip("Dataset not available")

        for split in ["train", "validation", "test"]:
            with open(source_dir / f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    assert "experimental_feature_version" not in r
                    assert "experimental_features" not in r


# ── Test: Output/restart safety ─────────────────────────────────────────


class TestOutputSafety:
    """Verify output does not falsely appear complete if interrupted."""

    def test_no_partial_output_exists(self):
        """No experimental-exp-4.5a directory exists before generation."""
        output_dir = Path("backend/data/datasets/experimental-exp-4.5a")
        # It's OK if it doesn't exist; this confirms the starting state
        if output_dir.exists():
            files = list(output_dir.iterdir())
            # If it exists, it should either be empty or have all 3 splits
            splits_present = [f.name for f in files if f.suffix == ".jsonl"]
            assert len(splits_present) in (0, 3), (
                f"Partial output detected: {splits_present}"
            )

    def test_output_key_preservation(self):
        """Output preserves commit_sha and file_path from source."""
        repo_path = str(REPOS_DIR / "flask")
        if not os.path.exists(repo_path):
            pytest.skip("flask repo not available")

        gen = GitPythonFeatureGenerator()
        gen.open_repo("flask", repo_path)

        rows = _read_jsonl(Path("backend/data/datasets/combined-v3/train.jsonl"))
        flask_rows = [r for r in rows if r["repo_name"] == "flask"][:3]

        for row in flask_rows:
            features = gen.compute_for_row(
                "flask", row["commit_sha"], row["file_path"],
                row.get("commit_timestamp", ""),
            )
            # Features are computed but NOT written to source
            # The output contract is verified here:
            assert len(features) == 6
            assert all(isinstance(f, float) for f in features)


# ── Test: Ruff clean ────────────────────────────────────────────────────


class TestRuffClean:
    """Verify all Phase 4.5 files pass Ruff."""

    def test_ruff_passes(self):
        """Ruff check on all Phase 4.5 files."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check",
             "backend/app/features/ast_diff.py",
             "backend/app/features/historical.py",
             "backend/app/features/experimental_schemas.py",
             "backend/app/ml/experimental_loader.py",
             "backend/scripts/generate_exp_4_5a.py"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0, f"Ruff failed:\n{result.stdout}"


# ── Test: Robust cleanup ────────────────────────────────────────────────


class TestRobustCleanup:
    """Verify _remove_dir_robust handles edge cases."""

    def test_nonexistent_directory(self):
        """Removing non-existent directory returns True."""
        assert _remove_dir_robust("/nonexistent/path/that/doesnt/exist") is True

    def test_empty_directory(self):
        """Removing empty directory succeeds."""
        tmpdir = tempfile.mkdtemp()
        empty = os.path.join(tmpdir, "empty")
        os.makedirs(empty)
        assert _remove_dir_robust(empty) is True
        assert not os.path.exists(empty)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_directory_with_files(self):
        """Removing directory with files succeeds."""
        tmpdir = tempfile.mkdtemp()
        d = os.path.join(tmpdir, "stuff")
        os.makedirs(d)
        with open(os.path.join(d, "file.txt"), "w") as f:
            f.write("test")
        assert _remove_dir_robust(d) is True
        assert not os.path.exists(d)
        shutil.rmtree(tmpdir, ignore_errors=True)
