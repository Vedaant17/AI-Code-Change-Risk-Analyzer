"""Tests for file-level feature extraction."""

from __future__ import annotations

from backend.app.features.file_features import extract_file_features
from backend.app.features.schemas import FILE_FEATURE_NAMES, FileFeatures
from backend.app.schemas.diff import FileDiff, FileStatus, Hunk


class TestExtractFileFeatures:
    def test_modified_python_file(self) -> None:
        fd = FileDiff(
            path="src/main.py",
            status=FileStatus.MODIFIED,
            lines_added=5,
            lines_deleted=2,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=10,
                    new_start=1,
                    new_count=13,
                    content=(
                        " import os\n"
                        "+import sys\n"
                        "+import json\n"
                        " \n"
                        " def hello():\n"
                        "-    pass\n"
                        "+    print('hello')\n"
                        "+    return True\n"
                        " \n"
                        " class Foo:\n"
                        "     pass\n"
                    ),
                )
            ],
        )
        ff = extract_file_features(fd)

        assert ff.file_path == "src/main.py"
        assert ff.is_binary is False
        assert ff.is_test_file is False
        assert ff.lines_added == 5
        assert ff.lines_deleted == 2
        assert ff.total_lines_changed == 7
        assert ff.hunk_count == 1
        # Heuristic detection: def/class/import patterns
        assert ff.function_declarations_added >= 0
        assert ff.imports_added >= 0

    def test_added_file(self) -> None:
        fd = FileDiff(
            path="new_module.py",
            status=FileStatus.ADDED,
            lines_added=3,
            lines_deleted=0,
            hunks=[
                Hunk(
                    old_start=0,
                    old_count=0,
                    new_start=1,
                    new_count=3,
                    content=(
                        "+# New module\n"
                        "+def greet():\n"
                        "+    pass\n"
                    ),
                )
            ],
        )
        ff = extract_file_features(fd)
        assert ff.lines_added == 3
        assert ff.lines_deleted == 0
        assert ff.function_declarations_added >= 1

    def test_binary_file(self) -> None:
        fd = FileDiff(
            path="image.png",
            status=FileStatus.BINARY,
            is_binary=True,
        )
        ff = extract_file_features(fd)
        assert ff.is_binary is True
        assert ff.lines_added == 0
        assert ff.lines_deleted == 0
        assert ff.hunk_count == 0
        assert ff.function_declarations_added == 0
        assert ff.function_declarations_deleted == 0

    def test_test_file_detection(self) -> None:
        fd = FileDiff(
            path="tests/test_service.py",
            status=FileStatus.MODIFIED,
            lines_added=2,
            lines_deleted=1,
        )
        ff = extract_file_features(fd)
        assert ff.is_test_file is True

    def test_rename_file(self) -> None:
        fd = FileDiff(
            path="new_name.py",
            old_path="old_name.py",
            status=FileStatus.RENAMED,
            lines_added=1,
            lines_deleted=1,
            hunks=[
                Hunk(
                    old_start=5,
                    old_count=3,
                    new_start=5,
                    new_count=3,
                    content=(
                        " def func():\n"
                        "-    old line\n"
                        "+    new line\n"
                    ),
                )
            ],
        )
        ff = extract_file_features(fd)
        assert ff.file_path == "new_name.py"
        assert ff.total_lines_changed == 2

    def test_indentation_stats(self) -> None:
        fd = FileDiff(
            path="deep.py",
            status=FileStatus.MODIFIED,
            lines_added=2,
            lines_deleted=0,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=1,
                    new_start=1,
                    new_count=3,
                    content=(
                        "+        x = 1\n"  # 8 spaces indent
                        "+            y = 2\n"  # 12 spaces indent
                    ),
                )
            ],
        )
        ff = extract_file_features(fd)
        assert ff.max_changed_line_indent >= 12
        assert ff.avg_changed_line_indent > 0

    def test_no_hunks(self) -> None:
        fd = FileDiff(
            path="empty.py",
            status=FileStatus.MODIFIED,
            lines_added=0,
            lines_deleted=0,
        )
        ff = extract_file_features(fd)
        assert ff.hunk_count == 0
        assert ff.avg_hunk_size == 0.0

    def test_multiple_hunks(self) -> None:
        fd = FileDiff(
            path="multi.py",
            status=FileStatus.MODIFIED,
            lines_added=4,
            lines_deleted=2,
            hunks=[
                Hunk(old_start=1, old_count=3, new_start=1, new_count=4, content="+a\n-b\n"),
                Hunk(old_start=20, old_count=2, new_start=21, new_count=3, content="+c\n-d\n"),
            ],
        )
        ff = extract_file_features(fd)
        assert ff.hunk_count == 2
        assert ff.avg_hunk_size > 0


class TestFeatureVectorOrdering:
    def test_to_feature_vector_length(self) -> None:
        fd = FileDiff(
            path="test.py",
            status=FileStatus.MODIFIED,
            lines_added=1,
            lines_deleted=0,
            hunks=[Hunk(old_start=1, old_count=1, new_start=1, new_count=2, content="+x\n")],
        )
        ff = extract_file_features(fd)
        vec = ff.to_feature_vector()
        assert len(vec) == len(FILE_FEATURE_NAMES)

    def test_vector_matches_field_order(self) -> None:
        fd = FileDiff(
            path="test.py",
            status=FileStatus.MODIFIED,
            lines_added=5,
            lines_deleted=3,
            hunks=[Hunk(old_start=1, old_count=3, new_start=1, new_count=5, content="+a\n-b\n")],
        )
        ff = extract_file_features(fd)
        vec = ff.to_feature_vector()
        d = ff.model_dump()
        for i, name in enumerate(FILE_FEATURE_NAMES):
            assert vec[i] == float(d[name]), f"Mismatch at index {i} ({name})"

    def test_deterministic(self) -> None:
        fd = FileDiff(
            path="det.py",
            status=FileStatus.MODIFIED,
            lines_added=2,
            lines_deleted=1,
            hunks=[Hunk(old_start=1, old_count=2, new_start=1, new_count=3, content="+x\n-y\n")],
        )
        vec1 = extract_file_features(fd).to_feature_vector()
        vec2 = extract_file_features(fd).to_feature_vector()
        assert vec1 == vec2
