"""Tests for checkpoint/resume infrastructure.

Validates:
  - Atomic JSONL writes
  - Progress file save/load
  - Per-repo checkpoint streaming
  - Resume skips completed repos
  - Merge checkpoints into final output
  - Deterministic output (same input -> same features)
  - Clone retry with transient failure recovery
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.features.experimental_schemas import EXPERIMENTAL_FEATURE_COUNT
from backend.scripts.generate_exp_4_5a import (
    GitPythonFeatureGenerator,
    _append_jsonl,
    _atomic_write_jsonl,
    _clone_repo,
    _load_progress,
    _merge_checkpoints,
    _read_jsonl,
    _read_jsonl_lenient,
    _save_progress,
)

# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ── Atomic write tests ──────────────────────────────────────────────────


class TestAtomicWrite:
    def test_atomic_write_creates_file(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        rows = [{"a": 1}, {"b": 2}]
        _atomic_write_jsonl(path, rows)
        assert path.exists()
        loaded = _read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0] == {"a": 1}

    def test_atomic_write_no_temp_file_left(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        _atomic_write_jsonl(path, [{"x": 1}])
        tmp_files = list(tmp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_overwrites_existing(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        _atomic_write_jsonl(path, [{"old": True}])
        _atomic_write_jsonl(path, [{"new": True}])
        loaded = _read_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0] == {"new": True}


class TestAppendJsonl:
    def test_append_single_row(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        _append_jsonl(path, {"a": 1})
        _append_jsonl(path, {"b": 2})
        loaded = _read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0] == {"a": 1}
        assert loaded[1] == {"b": 2}

    def test_append_empty_file(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        _append_jsonl(path, {"x": 1})
        assert path.exists()
        loaded = _read_jsonl(path)
        assert len(loaded) == 1


# ── Lenient JSONL reader tests ──────────────────────────────────────────


class TestReadJsonlLenient:
    def test_valid_file_all_rows_recovered(self, tmp_dir):
        """Complete JSONL file — every row is returned."""
        path = tmp_dir / "ckpt.jsonl"
        rows = [
            {"commit_sha": "aaa", "file_path": "a.py"},
            {"commit_sha": "bbb", "file_path": "b.py"},
            {"commit_sha": "ccc", "file_path": "c.py"},
        ]
        _atomic_write_jsonl(path, rows)
        loaded = _read_jsonl_lenient(path)
        assert loaded == rows

    def test_truncated_final_line_recovered(self, tmp_dir):
        """Truncated trailing line from interrupted append — valid rows returned."""
        path = tmp_dir / "ckpt.jsonl"
        # Write two valid rows manually
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"commit_sha": "aaa", "file_path": "a.py"}\n')
            f.write('{"commit_sha": "bbb", "file_path": "b.py"}\n')
            # Simulate interrupted append: incomplete JSON
            f.write('{"commit_sha": "ccc", "file_pa')
        loaded = _read_jsonl_lenient(path)
        assert len(loaded) == 2
        assert loaded[0]["commit_sha"] == "aaa"
        assert loaded[1]["commit_sha"] == "bbb"

    def test_malformed_json_in_middle_raises(self, tmp_dir):
        """Malformed JSON in the middle of the file raises JSONDecodeError."""
        path = tmp_dir / "ckpt.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"commit_sha": "aaa", "file_path": "a.py"}\n')
            f.write('NOT VALID JSON\n')
            f.write('{"commit_sha": "ccc", "file_path": "c.py"}\n')
        with pytest.raises(json.JSONDecodeError):
            _read_jsonl_lenient(path)

    def test_blank_lines_skipped(self, tmp_dir):
        """Blank lines between valid rows are ignored."""
        path = tmp_dir / "ckpt.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"commit_sha": "aaa"}\n')
            f.write('\n')
            f.write('   \n')
            f.write('{"commit_sha": "bbb"}\n')
        loaded = _read_jsonl_lenient(path)
        assert len(loaded) == 2

    def test_empty_file_returns_empty(self, tmp_dir):
        """Empty file returns an empty list."""
        path = tmp_dir / "ckpt.jsonl"
        path.touch()
        loaded = _read_jsonl_lenient(path)
        assert loaded == []

    def test_resume_after_truncate_no_duplicates(self, tmp_dir):
        """Simulate resume after truncated checkpoint — no duplicate rows."""
        ckpt_dir = tmp_dir / "checkpoints"
        ckpt_dir.mkdir()
        repo_ckpt = ckpt_dir / "myrepo.jsonl"

        # Simulate: 3 commits completed, 4th interrupted mid-append
        rows = [
            {"commit_sha": "c1", "file_path": "a.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [0.0] * 6},
            {"commit_sha": "c2", "file_path": "b.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [1.0, 0.0, 0.0, 3.0, 1.0, 5.0]},
            {"commit_sha": "c3", "file_path": "c.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [0.0, 1.0, 0.0, 7.0, 0.0, 2.0]},
        ]
        with open(repo_ckpt, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            # Interrupted append for c4
            f.write('{"commit_sha": "c4", "file_path": "d.py", "exper')

        # Load checkpoint — should recover c1, c2, c3; drop c4
        loaded = _read_jsonl_lenient(repo_ckpt)
        assert len(loaded) == 3
        done_commits = {r["commit_sha"] for r in loaded}
        assert done_commits == {"c1", "c2", "c3"}
        assert "c4" not in done_commits

        # Simulate resume: c4 would be recomputed, c1-c3 skipped
        all_commits = ["c1", "c2", "c3", "c4"]
        recomputed = [c for c in all_commits if c not in done_commits]
        assert recomputed == ["c4"]


# ── Progress file tests ─────────────────────────────────────────────────


class TestProgressFile:
    def test_load_nonexistent(self, tmp_dir):
        progress = _load_progress(tmp_dir / "progress.json")
        assert progress == {}

    def test_save_and_load(self, tmp_dir):
        path = tmp_dir / "progress.json"
        progress = {"repo_a": {"done": True, "rows": 100}}
        _save_progress(path, progress)
        loaded = _load_progress(path)
        assert loaded == progress

    def test_atomic_save(self, tmp_dir):
        path = tmp_dir / "progress.json"
        _save_progress(path, {"r1": {"done": True, "rows": 50}})
        _save_progress(path, {"r1": {"done": True, "rows": 50}, "r2": {"done": True, "rows": 30}})
        loaded = _load_progress(path)
        assert "r2" in loaded
        assert loaded["r2"]["rows"] == 30

    def test_no_temp_file_left(self, tmp_dir):
        path = tmp_dir / "progress.json"
        _save_progress(path, {"r1": {"done": True}})
        tmp_files = list(tmp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


# ── Checkpoint merge tests ──────────────────────────────────────────────


class TestMergeCheckpoints:
    def test_merge_creates_split_files(self, tmp_dir):
        ckpt_dir = tmp_dir / "checkpoints"
        ckpt_dir.mkdir()
        # _merge_checkpoints writes to output_base / "combined-v3"
        output_base = tmp_dir / "output"
        output_base.mkdir()

        ckpt_rows = [
            {"commit_sha": "abc123", "file_path": "a.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [0.0] * EXPERIMENTAL_FEATURE_COUNT},
        ]
        _atomic_write_jsonl(ckpt_dir / "repo1.jsonl", ckpt_rows)

        split_data = {
            "train": [{"commit_sha": "abc123", "file_path": "a.py", "repo_name": "repo1"}],
            "validation": [],
            "test": [],
        }

        _merge_checkpoints(output_base, ckpt_dir, split_data)

        train_path = output_base / "combined-v3" / "train.jsonl"
        assert train_path.exists()
        train_rows = _read_jsonl(train_path)
        assert len(train_rows) == 1
        assert train_rows[0]["commit_sha"] == "abc123"

    def test_merge_preserves_all_rows(self, tmp_dir):
        ckpt_dir = tmp_dir / "checkpoints"
        ckpt_dir.mkdir()
        output_base = tmp_dir / "output"
        output_base.mkdir()

        rows = [
            {"commit_sha": "aaa", "file_path": "x.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [1.0, 0.0, 0.0, 5.0, 1.0, 3.0]},
            {"commit_sha": "bbb", "file_path": "y.py",
             "experimental_feature_version": "exp-4.5a",
             "experimental_features": [0.0, 1.0, 0.0, 3.0, 0.0, 7.0]},
        ]
        _atomic_write_jsonl(ckpt_dir / "repo1.jsonl", rows)

        split_data = {
            "train": [
                {"commit_sha": "aaa", "file_path": "x.py"},
                {"commit_sha": "bbb", "file_path": "y.py"},
            ],
            "validation": [],
            "test": [],
        }

        _merge_checkpoints(output_base, ckpt_dir, split_data)
        train_rows = _read_jsonl(output_base / "combined-v3" / "train.jsonl")
        assert len(train_rows) == 2


# ── Determinism tests ───────────────────────────────────────────────────


class TestDeterminism:
    def _get_valid_repo(self):
        """Find a valid repo for testing."""
        import git as gitpython

        from backend.scripts.generate_exp_4_5a import REPOS_DIR
        for name in ["twisted", "requests", "flask"]:
            repo_path = REPOS_DIR / name
            if repo_path.exists() and (repo_path / ".git").exists():
                try:
                    r = gitpython.Repo(str(repo_path))
                    _ = r.head.commit.hexsha
                    return name, str(repo_path)
                except Exception:
                    pass
        return None, None

    def test_feature_vector_deterministic(self, tmp_dir):
        """Same row processed twice produces identical features."""
        repo_name, repo_path = self._get_valid_repo()
        if not repo_path:
            pytest.skip("No valid repo clone available")

        gen = GitPythonFeatureGenerator()
        gen.open_repo(repo_name, repo_path)
        try:
            import git as gitpython
            repo = gitpython.Repo(repo_path)
            commit = next(repo.iter_commits(max_count=10))
            sha = commit.hexsha
            parent = commit.parents[0].hexsha if commit.parents else None

            if parent is None:
                pytest.skip("No parent commit found")

            diff = commit.diff(parent)
            if not diff:
                pytest.skip("No file changes found")

            file_path = diff[0].a_path

            features1 = gen.compute_for_row(repo_name, sha, file_path, "2025-01-01T00:00:00Z")
            features2 = gen.compute_for_row(repo_name, sha, file_path, "2025-01-01T00:00:00Z")
            assert features1 == features2
            assert len(features1) == EXPERIMENTAL_FEATURE_COUNT
        finally:
            gen.clear()

    def test_features_non_negative(self, tmp_dir):
        """All feature values are >= 0."""
        repo_name, repo_path = self._get_valid_repo()
        if not repo_path:
            pytest.skip("No valid repo clone available")

        gen = GitPythonFeatureGenerator()
        gen.open_repo(repo_name, repo_path)
        try:
            import git as gitpython
            repo = gitpython.Repo(repo_path)
            for commit in repo.iter_commits(max_count=5):
                parent = commit.parents[0].hexsha if commit.parents else None
                diff = commit.diff(parent)
                for d in diff:
                    fp = d.a_path
                    features = gen.compute_for_row(
                        repo_name, commit.hexsha, fp, "2025-01-01T00:00:00Z"
                    )
                    assert all(f >= 0.0 for f in features), (
                        f"Negative feature for {commit.hexsha[:8]}:{fp}: {features}"
                    )
        finally:
            gen.clear()


# ── Test: Clone retry with transient failure ──────────────────────────────


def _make_source_repo(tmpdir: str) -> str:
    """Create a bare git repo for cloning tests."""
    src = os.path.join(tmpdir, "source")
    os.makedirs(src)
    subprocess.run(
        ["git", "init", "-q", src], check=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    subprocess.run(
        ["git", "-C", src, "config", "user.email", "t@t.com"],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    subprocess.run(
        ["git", "-C", src, "config", "user.name", "T"],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    with open(os.path.join(src, "a.py"), "w") as f:
        f.write("x = 1\n")
    subprocess.run(
        ["git", "-C", src, "add", "a.py"], check=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    subprocess.run(
        ["git", "-C", src, "commit", "-q", "-m", "init"],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return src


class TestCloneRetry:
    """Verify _clone_repo retries on transient failure and cleans partial dest."""

    def test_clone_succeeds_after_transient_failure(self):
        """Clone succeeds on second attempt after transient first failure."""
        tmpdir = tempfile.mkdtemp()
        try:
            source_repo = _make_source_repo(tmpdir)
            source_url = "file:///" + source_repo.replace("\\", "/")

            import backend.scripts.generate_exp_4_5a as gen_mod
            original_repos_dir = gen_mod.REPOS_DIR
            gen_mod.REPOS_DIR = Path(tmpdir)

            call_count = 0
            original_run = subprocess.run

            def mock_run(cmd, **kwargs):
                nonlocal call_count
                if isinstance(cmd, list) and len(cmd) > 2 and cmd[1] == "clone":
                    call_count += 1
                    if call_count == 1:
                        r = subprocess.CompletedProcess(
                            cmd, returncode=1, stdout="", stderr="mock error",
                        )
                        return r
                return original_run(cmd, **kwargs)

            try:
                with patch("backend.scripts.generate_exp_4_5a.subprocess.run",
                           side_effect=mock_run):
                    result = _clone_repo(source_url, "myrepo")
                    assert result is not None
                    assert os.path.exists(os.path.join(result, ".git"))
                    assert call_count == 2
            finally:
                gen_mod.REPOS_DIR = original_repos_dir
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clone_cleans_partial_destination(self):
        """Partial destination (no .git) is cleaned before fresh clone."""
        tmpdir = tempfile.mkdtemp()
        try:
            source_repo = _make_source_repo(tmpdir)
            source_url = "file:///" + source_repo.replace("\\", "/")

            import backend.scripts.generate_exp_4_5a as gen_mod
            original_repos_dir = gen_mod.REPOS_DIR
            gen_mod.REPOS_DIR = Path(tmpdir)

            # Create partial destination (no .git)
            partial_dir = os.path.join(tmpdir, "myrepo")
            os.makedirs(partial_dir)
            with open(os.path.join(partial_dir, "junk.txt"), "w") as f:
                f.write("partial")

            try:
                result = _clone_repo(source_url, "myrepo")
                assert result is not None
                assert os.path.exists(os.path.join(result, ".git"))
                assert not os.path.exists(os.path.join(result, "junk.txt"))
            finally:
                gen_mod.REPOS_DIR = original_repos_dir
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clone_all_attempts_fail(self):
        """Returns None after 3 failed attempts."""
        tmpdir = tempfile.mkdtemp()
        try:
            import backend.scripts.generate_exp_4_5a as gen_mod
            original_repos_dir = gen_mod.REPOS_DIR
            gen_mod.REPOS_DIR = Path(tmpdir)

            call_count = 0
            original_run = subprocess.run

            def mock_run(cmd, **kwargs):
                nonlocal call_count
                if isinstance(cmd, list) and len(cmd) > 2 and cmd[1] == "clone":
                    call_count += 1
                return original_run(cmd, **kwargs)

            try:
                with patch("backend.scripts.generate_exp_4_5a.subprocess.run",
                           side_effect=mock_run):
                    result = _clone_repo("file:///nonexistent", "myrepo")
                    assert result is None
                    assert call_count == 3
            finally:
                gen_mod.REPOS_DIR = original_repos_dir
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_valid_clone_reuse_no_extra_clone(self):
        """Second call reuses valid clone without calling subprocess."""
        tmpdir = tempfile.mkdtemp()
        try:
            source_repo = _make_source_repo(tmpdir)
            source_url = "file:///" + source_repo.replace("\\", "/")

            import backend.scripts.generate_exp_4_5a as gen_mod
            original_repos_dir = gen_mod.REPOS_DIR
            gen_mod.REPOS_DIR = Path(tmpdir)

            clone_count = 0
            original_run = subprocess.run

            def mock_run(cmd, **kwargs):
                nonlocal clone_count
                if isinstance(cmd, list) and len(cmd) > 2 and cmd[1] == "clone":
                    clone_count += 1
                return original_run(cmd, **kwargs)

            try:
                with patch("backend.scripts.generate_exp_4_5a.subprocess.run",
                           side_effect=mock_run):
                    result1 = _clone_repo(source_url, "myrepo")
                    assert result1 is not None
                    assert clone_count == 1

                    result2 = _clone_repo(source_url, "myrepo")
                    assert result2 is not None
                    assert clone_count == 1
            finally:
                gen_mod.REPOS_DIR = original_repos_dir
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_partial_destination_without_git_cleaned(self):
        """Partial dir without .git is removed before clone attempt."""
        tmpdir = tempfile.mkdtemp()
        try:
            import backend.scripts.generate_exp_4_5a as gen_mod
            original_repos_dir = gen_mod.REPOS_DIR
            gen_mod.REPOS_DIR = Path(tmpdir)

            partial_dir = os.path.join(tmpdir, "myrepo")
            os.makedirs(partial_dir)
            with open(os.path.join(partial_dir, "data.txt"), "w") as f:
                f.write("incomplete")

            assert os.path.exists(partial_dir)
            assert not os.path.exists(os.path.join(partial_dir, ".git"))

            result = _clone_repo("file:///nonexistent", "myrepo")
            assert result is None
        finally:
            gen_mod.REPOS_DIR = original_repos_dir
            shutil.rmtree(tmpdir, ignore_errors=True)
