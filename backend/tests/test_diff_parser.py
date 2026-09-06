"""Tests for the lightweight unified-diff parser."""

from __future__ import annotations

from backend.app.schemas.diff import FileStatus
from backend.app.services.diff_parser import parse_unified_diff


class TestModifiedFile:
    def test_parses_single_modified_file(self, modified_diff: str) -> None:
        files = parse_unified_diff(modified_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "src/main.py"
        assert f.status == FileStatus.MODIFIED
        assert f.lines_added >= 2
        assert f.lines_deleted >= 1
        assert len(f.hunks) == 2

    def test_hunk_headers_extracted(self, modified_diff: str) -> None:
        files = parse_unified_diff(modified_diff)
        hunk = files[0].hunks[0]
        assert hunk.old_start == 10
        assert hunk.old_count == 7
        assert hunk.new_start == 10
        assert hunk.new_count == 8


class TestAddedFile:
    def test_detects_added_file(self, added_diff: str) -> None:
        files = parse_unified_diff(added_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "new_module.py"
        assert f.status == FileStatus.ADDED
        assert f.lines_added == 3
        assert f.lines_deleted == 0


class TestDeletedFile:
    def test_detects_deleted_file(self, deleted_diff: str) -> None:
        files = parse_unified_diff(deleted_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "old_module.py"
        assert f.status == FileStatus.DELETED
        assert f.lines_added == 0
        assert f.lines_deleted == 3


class TestRenamedFile:
    def test_detects_rename(self, renamed_diff: str) -> None:
        files = parse_unified_diff(renamed_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "new_name.py"
        assert f.old_path == "old_name.py"
        assert f.status == FileStatus.RENAMED


class TestBinaryFile:
    def test_detects_binary_file(self, binary_diff: str) -> None:
        files = parse_unified_diff(binary_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "assets/image.png"
        assert f.is_binary is True
        assert f.status == FileStatus.BINARY


class TestMultipleFiles:
    def test_parses_multiple_files(self, multi_file_diff: str) -> None:
        files = parse_unified_diff(multi_file_diff)
        assert len(files) == 3
        paths = {f.path for f in files}
        assert "src/main.py" in paths
        assert "new_module.py" in paths
        assert "old_module.py" in paths

    def test_total_counts_aggregate(self, multi_file_diff: str) -> None:
        files = parse_unified_diff(multi_file_diff)
        total_added = sum(f.lines_added for f in files)
        total_deleted = sum(f.lines_deleted for f in files)
        assert total_added > 0
        assert total_deleted > 0


class TestPathsWithSpaces:
    def test_handles_paths_with_spaces(self, spaces_diff: str) -> None:
        files = parse_unified_diff(spaces_diff)
        assert len(files) == 1
        assert files[0].path == "path with spaces/file.py"


class TestNoNewline:
    def test_handles_no_newline_at_eof(self, no_newline_diff: str) -> None:
        files = parse_unified_diff(no_newline_diff)
        assert len(files) == 1
        f = files[0]
        assert f.path == "file.py"
        assert f.status == FileStatus.MODIFIED


class TestEdgeCases:
    def test_empty_diff_returns_empty_list(self) -> None:
        assert parse_unified_diff("") == []

    def test_whitespace_only_diff_returns_empty_list(self) -> None:
        assert parse_unified_diff("   \n  \n") == []

    def test_no_newline_marker_in_diff_path(self) -> None:
        raw = (
            "diff --git a/file.py b/file.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            "\\ No newline at end of file\n"
        )
        files = parse_unified_diff(raw)
        assert len(files) == 1
        assert files[0].path == "file.py"
