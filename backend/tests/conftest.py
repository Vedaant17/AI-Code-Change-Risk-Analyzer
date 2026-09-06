"""Shared fixtures for backend tests."""

from __future__ import annotations

import textwrap

import pytest

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
