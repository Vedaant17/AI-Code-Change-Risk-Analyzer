"""Tests for the Git service (schema validation and unit logic).

These tests validate schema construction and model behaviour without
requiring a real cloned repository.  Integration tests that actually
clone and diff are separate.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.schemas.diff import (
    CommitInfo,
    DiffStats,
    FileDiff,
    FileStatus,
    Hunk,
)


class TestFileDiffModel:
    def test_total_lines_changed(self) -> None:
        fd = FileDiff(path="a.py", status=FileStatus.MODIFIED, lines_added=5, lines_deleted=3)
        assert fd.total_lines_changed == 8

    def test_defaults(self) -> None:
        fd = FileDiff(path="x.py", status=FileStatus.ADDED)
        assert fd.lines_added == 0
        assert fd.lines_deleted == 0
        assert fd.is_binary is False
        assert fd.hunks == []


class TestDiffStats:
    def test_total_lines_changed(self) -> None:
        s = DiffStats(total_lines_added=10, total_lines_deleted=7)
        assert s.total_lines_changed == 17


class TestCommitInfo:
    def test_compute_stats(self) -> None:
        files = [
            FileDiff(path="a.py", status=FileStatus.ADDED, lines_added=10),
            FileDiff(path="b.py", status=FileStatus.MODIFIED, lines_added=5, lines_deleted=3),
            FileDiff(path="c.py", status=FileStatus.DELETED, lines_deleted=8),
            FileDiff(path="d.py", status=FileStatus.RENAMED),
            FileDiff(path="e.png", status=FileStatus.BINARY),
        ]
        info = CommitInfo(sha="abc123", files=files)
        stats = info.compute_stats()

        assert stats.total_files == 5
        assert stats.files_added == 1
        assert stats.files_modified == 1
        assert stats.files_deleted == 1
        assert stats.files_renamed == 1
        assert stats.files_binary == 1
        assert stats.total_lines_added == 15
        assert stats.total_lines_deleted == 11

    def test_empty_files(self) -> None:
        info = CommitInfo(sha="abc123")
        stats = info.compute_stats()
        assert stats.total_files == 0


class TestHunkModel:
    def test_hunk_creation(self) -> None:
        h = Hunk(old_start=1, old_count=5, new_start=1, new_count=6, content="@@ -1,5 +1,6 @@\n")
        assert h.old_start == 1
        assert h.new_count == 6
