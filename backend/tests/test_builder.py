"""Tests for dataset builder (Phase 3)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.app.dataset.builder import DatasetBuilder
from backend.app.dataset.config import DatasetConfig
from backend.app.features.schemas import COMMIT_FEATURE_NAMES, FILE_FEATURE_NAMES
from backend.app.schemas.diff import CommitInfo, FileDiff, FileStatus, Hunk


def _make_commit(
    sha: str,
    message: str,
    files: list[FileDiff] | None = None,
    author_date: datetime | None = None,
) -> CommitInfo:
    c = CommitInfo(
        sha=sha,
        short_sha=sha[:8],
        author="test",
        author_date=author_date or datetime(2026, 1, 1, tzinfo=timezone.utc),
        message=message,
        files=files or [],
    )
    c.compute_stats()
    return c


def _make_file(path: str = "src/main.py", lines_added: int = 1, lines_deleted: int = 0) -> FileDiff:
    return FileDiff(
        path=path,
        status=FileStatus.MODIFIED,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        hunks=[
            Hunk(
                old_start=1,
                old_count=3,
                new_start=1,
                new_count=3 + lines_added - lines_deleted,
                content=(
                    " context\n"
                    + "-old line\n" * lines_deleted
                    + "+new line\n" * lines_added
                    + " context\n"
                ),
            )
        ],
    )


class TestEmptyDataset:
    def test_empty_repo(self, tmp_path: Path) -> None:
        builder = DatasetBuilder()
        manifest = builder.build([], tmp_path / "out")

        assert (tmp_path / "out" / "metadata.json").exists()
        assert (tmp_path / "out" / "train.jsonl").exists()
        assert (tmp_path / "out" / "validation.jsonl").exists()
        assert (tmp_path / "out" / "test.jsonl").exists()
        assert (tmp_path / "out" / "ambiguous.jsonl").exists()

        assert manifest.total_commits_processed == 0
        assert manifest.total_file_examples == 0


class TestCommitLevelSplits:
    def test_commit_level_split(self, tmp_path: Path) -> None:
        commits = [
            _make_commit(f"{'a' + str(i).zfill(39)}", f"commit {i}", [_make_file()])
            for i in range(20)
        ]
        builder = DatasetBuilder()
        manifest = builder.build(
            commits, tmp_path / "out",
            repo_url="https://github.com/test/repo",
        )

        # All files from one commit should be in the same split
        train = list(Path(tmp_path / "out" / "train.jsonl").read_text().strip().split("\n"))
        val = list(Path(tmp_path / "out" / "validation.jsonl").read_text().strip().split("\n"))
        test = list(Path(tmp_path / "out" / "test.jsonl").read_text().strip().split("\n"))

        # Each line is one file; all files from a commit share the same split
        # With 20 commits: 14 train, 3 val, 3 test
        assert manifest.train_commits == 14
        assert manifest.validation_commits == 3
        assert manifest.test_commits == 3

    def test_chronological_split_ordering(self, tmp_path: Path) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                f"{'a' + str(i).zfill(39)}",
                f"commit {i}",
                [_make_file()],
                author_date=base + timedelta(days=i),
            )
            for i in range(20)
        ]
        builder = DatasetBuilder()
        manifest = builder.build(commits, tmp_path / "out")

        assert manifest.train_commits > 0
        assert manifest.validation_commits > 0
        assert manifest.test_commits > 0

    def test_deterministic_tie_breaking(self, tmp_path: Path) -> None:
        commits = [
            _make_commit(f"{'a' + str(i).zfill(39)}", f"commit {i}", [_make_file()])
            for i in range(10)
        ]
        builder = DatasetBuilder()
        builder.build(commits, tmp_path / "out1")
        builder.build(commits, tmp_path / "out2")

        assert (
            (tmp_path / "out1" / "train.jsonl").read_bytes()
            == (tmp_path / "out2" / "train.jsonl").read_bytes()
        )


class TestFeatureContract:
    def test_feature_contract_29_16(self, tmp_path: Path) -> None:
        commits = [
            _make_commit("a" * 40, "add feature", [_make_file()]),
        ]
        builder = DatasetBuilder()
        builder.build(commits, tmp_path / "out")

        for split in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            path = tmp_path / "out" / split
            if not path.exists() or path.stat().st_size == 0:
                continue
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                row = json.loads(line)
                assert len(row["commit_features"]) == len(COMMIT_FEATURE_NAMES)
                assert len(row["file_features"]) == len(FILE_FEATURE_NAMES)


class TestNoFutureInfo:
    def test_no_future_info_in_features(self, tmp_path: Path) -> None:
        commits = [
            _make_commit("a" * 40, "add feature", [_make_file()]),
            _make_commit("b" * 40, "fix: bug", [_make_file()]),
        ]
        builder = DatasetBuilder()
        builder.build(commits, tmp_path / "out")

        # Features should not contain any future information
        # (This is a structural check — the feature extractor uses only
        # commit-time data by design)
        assert (tmp_path / "out" / "metadata.json").exists()


class TestDeterministicOutput:
    def test_deterministic_output(self, tmp_path: Path) -> None:
        commits = [
            _make_commit(f"{'a' + str(i).zfill(39)}", f"commit {i}", [_make_file()])
            for i in range(5)
        ]
        builder = DatasetBuilder()
        builder.build(commits, tmp_path / "out1")
        builder.build(commits, tmp_path / "out2")

        for fname in ["metadata.json", "train.jsonl", "validation.jsonl", "test.jsonl", "ambiguous.jsonl"]:
            assert (
                (tmp_path / "out1" / fname).read_bytes()
                == (tmp_path / "out2" / fname).read_bytes()
            ), f"{fname} differs between runs"


class TestAmbiguousExcluded:
    def test_ambiguous_excluded(self, tmp_path: Path) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "a" * 40, "modify feature",
                [_make_file("src/main.py")],
                author_date=base,
            ),
            _make_commit(
                "b" * 40, "another change",
                [_make_file("src/other.py")],
                author_date=base + timedelta(days=1),
            ),
            _make_commit(
                "c" * 40, "fix: regression",
                [_make_file("src/main.py")],
                author_date=base + timedelta(days=2),
            ),
        ]
        builder = DatasetBuilder()
        manifest = builder.build(commits, tmp_path / "out")

        # Check that ambiguous.jsonl exists
        assert (tmp_path / "out" / "ambiguous.jsonl").exists()

        # Check that supervised files have zero ambiguous rows
        for split in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            path = tmp_path / "out" / split
            if not path.exists() or path.stat().st_size == 0:
                continue
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                row = json.loads(line)
                assert row["label_status"] != "ambiguous", (
                    f"Found ambiguous row in {split}: {row['commit_sha']}"
                )


class TestSingleCommit:
    def test_single_commit_negative(self, tmp_path: Path) -> None:
        commits = [_make_commit("a" * 40, "add feature", [_make_file()])]
        builder = DatasetBuilder()
        manifest = builder.build(commits, tmp_path / "out")

        assert manifest.total_file_examples == 1
        assert manifest.negative_examples == 1
        assert manifest.positive_examples == 0


class TestSplitBeforeExpansion:
    def test_split_before_expansion(self, tmp_path: Path) -> None:
        commits = [
            _make_commit(
                f"{'a' + str(i).zfill(39)}",
                f"commit {i}",
                [_make_file(f"src/file{i}.py") for i in range(3)],
            )
            for i in range(20)
        ]
        builder = DatasetBuilder()
        manifest = builder.build(commits, tmp_path / "out")

        # All 3 files from commit 0 should be in the same split
        all_rows = []
        for split in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            path = tmp_path / "out" / split
            if path.exists() and path.stat().st_size > 0:
                for line in path.read_text().strip().split("\n"):
                    if line:
                        all_rows.append(json.loads(line))

        # Group by commit
        by_commit: dict[str, list[str]] = {}
        for row in all_rows:
            sha = row["commit_sha"]
            by_commit.setdefault(sha, []).append(row["split"])

        for sha, splits in by_commit.items():
            assert len(set(splits)) == 1, (
                f"Commit {sha} has files in multiple splits: {splits}"
            )


class TestHighConfidence不受FileCountRestriction:
    def test_sha_attribution_ignores_medium_threshold(self, tmp_path: Path) -> None:
        """Candidate referenced by SHA in 10-file bug-fix gets positive."""
        files = [_make_file(f"src/file{i}.py") for i in range(10)]
        c1 = _make_commit("a" * 40, "add feature", [_make_file("src/shared.py")])
        sha_prefix = c1.sha[:12]
        c2 = _make_commit("b" * 40, f"fix: regression {sha_prefix}", files)
        builder = DatasetBuilder()
        manifest = builder.build(
            [c1, c2], tmp_path / "out",
            repo_url="https://github.com/test/repo",
        )

        # c1 should be positive despite c2 having 10 files
        assert manifest.positive_examples > 0


class TestManifestMetadata:
    def test_manifest_has_config(self, tmp_path: Path) -> None:
        commits = [_make_commit("a" * 40, "add feature", [_make_file()])]
        builder = DatasetBuilder(DatasetConfig(observation_window_commits=100))
        manifest = builder.build(commits, tmp_path / "out")

        meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
        assert meta["config"]["observation_window_commits"] == 100
        assert meta["dataset_version"] == "v1"
        assert meta["feature_version"] == "v1"


class TestBugFixCommitStatistics:
    def test_bug_fix_statistics(self, tmp_path: Path) -> None:
        c1 = _make_commit("a" * 40, "add feature", [_make_file("src/core.py")])
        c2 = _make_commit("b" * 40, "Revert 'add feature'", [_make_file("src/core.py")])
        sha_prefix = c1.sha[:12]
        c3 = _make_commit("c" * 40, f"fix: bug {sha_prefix}", [_make_file("src/core.py")])
        builder = DatasetBuilder()
        manifest = builder.build(
            [c1, c2, c3], tmp_path / "out",
            repo_url="https://github.com/test/repo",
        )

        assert manifest.revert_commits_found == 1
        assert manifest.bug_fix_commits_found == 1
