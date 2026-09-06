"""Shared fixtures for backend tests."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone

import pytest

from backend.app.schemas.diff import CommitInfo, FileDiff, FileStatus, Hunk
from backend.app.services.diff_parser import parse_unified_diff


# ---------------------------------------------------------------------------
# Sample unified diffs for testing
# ---------------------------------------------------------------------------

SAMPLE_MODIFIED_FILE = textwrap.dedent("""\
    diff --git a/src/main.py b/src/main.py
    index abc1234..def5678 100644
    --- a/src/main.py
    +++ b/src/main.py
    @@ -10,7 +10,8 @@ def hello():
         print("hello")
    -    x = 1
    +    x = 2
    +    y = 3
     
         return x
    @@ -20,4 +21,5 @@ def world():
         pass
     
    +    z = 0
     # end
""")

SAMPLE_ADDED_FILE = textwrap.dedent("""\
    diff --git a/new_module.py b/new_module.py
    new file mode 100644
    index 0000000..abc1234
    --- /dev/null
    +++ b/new_module.py
    @@ -0,0 +1,3 @@
    +# New module
    +def greet():
    +    pass
""")

SAMPLE_DELETED_FILE = textwrap.dedent("""\
    diff --git a/old_module.py b/old_module.py
    deleted file mode 100644
    index abc1234..0000000
    --- a/old_module.py
    +++ /dev/null
    @@ -1,3 +0,0 @@
    -# Old module
    -def old_func():
    -    pass
""")

SAMPLE_RENAMED_FILE = textwrap.dedent("""\
    diff --git a/old_name.py b/new_name.py
    similarity index 95%
    rename from old_name.py
    rename to new_name.py
    index abc1234..def5678 100644
    --- a/old_name.py
    +++ b/new_name.py
    @@ -5,3 +5,3 @@ def func():
    -old line
    +new line
""")

SAMPLE_BINARY_FILE = textwrap.dedent("""\
    diff --git a/assets/image.png b/assets/image.png
    index abc1234..def5678 100644
    GIT binary patch
    literal 1234
    some binary content here
""")

SAMPLE_MULTI_FILE = "\n".join([
    SAMPLE_MODIFIED_FILE,
    SAMPLE_ADDED_FILE,
    SAMPLE_DELETED_FILE,
])

SAMPLE_PATH_WITH_SPACES = textwrap.dedent("""\
    diff --git a/path with spaces/file.py b/path with spaces/file.py
    index abc1234..def5678 100644
    --- a/path with spaces/file.py
    +++ b/path with spaces/file.py
    @@ -1,3 +1,3 @@
    -old
    +new
     context
""")

SAMPLE_NO_NEWLINE = textwrap.dedent("""\
    diff --git a/file.py b/file.py
    index abc1234..def5678 100644
    --- a/file.py
    +++ b/file.py
    @@ -1,3 +1,3 @@
    -old line
    +new line
     context
    \\ No newline at end of file
""")


@pytest.fixture
def modified_diff() -> str:
    return SAMPLE_MODIFIED_FILE


@pytest.fixture
def added_diff() -> str:
    return SAMPLE_ADDED_FILE


@pytest.fixture
def deleted_diff() -> str:
    return SAMPLE_DELETED_FILE


@pytest.fixture
def renamed_diff() -> str:
    return SAMPLE_RENAMED_FILE


@pytest.fixture
def binary_diff() -> str:
    return SAMPLE_BINARY_FILE


@pytest.fixture
def multi_file_diff() -> str:
    return SAMPLE_MULTI_FILE


@pytest.fixture
def spaces_diff() -> str:
    return SAMPLE_PATH_WITH_SPACES


@pytest.fixture
def no_newline_diff() -> str:
    return SAMPLE_NO_NEWLINE


# ---------------------------------------------------------------------------
# Pre-built CommitInfo fixtures for feature extraction tests
# ---------------------------------------------------------------------------


@pytest.fixture
def single_python_commit_info() -> CommitInfo:
    """A CommitInfo with one modified Python file containing functions and imports."""
    files = [
        FileDiff(
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
                        " from os import path\n"
                        "-import sys\n"
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
    ]
    info = CommitInfo(
        sha="a" * 40,
        short_sha="a" * 8,
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message="test commit",
        files=files,
    )
    info.compute_stats()
    return info


@pytest.fixture
def multi_language_commit_info() -> CommitInfo:
    """A CommitInfo with Python and JavaScript files."""
    files = [
        FileDiff(
            path="app.py",
            status=FileStatus.MODIFIED,
            lines_added=3,
            lines_deleted=1,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=5,
                    new_start=1,
                    new_count=7,
                    content=(
                        " import os\n"
                        "+import sys\n"
                        " \n"
                        " def main():\n"
                        "-    pass\n"
                        "+    print('hi')\n"
                        "+    return 0\n"
                    ),
                )
            ],
        ),
        FileDiff(
            path="app.js",
            status=FileStatus.MODIFIED,
            lines_added=2,
            lines_deleted=1,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=4,
                    new_start=1,
                    new_count=5,
                    content=(
                        " function main() {\n"
                        "-    console.log('old');\n"
                        "+    console.log('new');\n"
                        "+    return 0;\n"
                        " }\n"
                    ),
                )
            ],
        ),
    ]
    info = CommitInfo(
        sha="b" * 40,
        short_sha="b" * 8,
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message="multi-language test",
        files=files,
    )
    info.compute_stats()
    return info


@pytest.fixture
def test_prod_coupling_commit_info() -> CommitInfo:
    """A CommitInfo with both test and production files changed."""
    files = [
        FileDiff(
            path="src/service.py",
            status=FileStatus.MODIFIED,
            lines_added=2,
            lines_deleted=1,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=3,
                    new_start=1,
                    new_count=4,
                    content=(
                        " def compute():\n"
                        "-    return 1\n"
                        "+    result = 42\n"
                        "+    return result\n"
                    ),
                )
            ],
        ),
        FileDiff(
            path="tests/test_service.py",
            status=FileStatus.MODIFIED,
            lines_added=3,
            lines_deleted=0,
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=3,
                    new_start=1,
                    new_count=6,
                    content=(
                        " from src.service import compute\n"
                        " \n"
                        "+def test_compute():\n"
                        "+    assert compute() == 42\n"
                        " \n"
                    ),
                )
            ],
        ),
    ]
    info = CommitInfo(
        sha="c" * 40,
        short_sha="c" * 8,
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message="test-prod coupling test",
        files=files,
    )
    info.compute_stats()
    return info


@pytest.fixture
def empty_commit_info() -> CommitInfo:
    """A CommitInfo with no files (e.g. merge commit with no diff)."""
    info = CommitInfo(
        sha="d" * 40,
        short_sha="d" * 8,
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message="empty commit",
        files=[],
    )
    info.compute_stats()
    return info


@pytest.fixture
def binary_only_commit_info() -> CommitInfo:
    """A CommitInfo with only a binary file change."""
    files = [
        FileDiff(
            path="assets/image.png",
            status=FileStatus.BINARY,
            is_binary=True,
        ),
    ]
    info = CommitInfo(
        sha="e" * 40,
        short_sha="e" * 8,
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message="binary only",
        files=files,
    )
    info.compute_stats()
    return info
