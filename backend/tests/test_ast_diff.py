"""Tests for AST diff feature extraction (Phase 4.5a).

Tests that AST features measure function regions affected by changed
lines, not literal declaration additions/deletions.
"""

from __future__ import annotations

import textwrap

from backend.app.features.ast_diff import (
    _compute_changed_lines,
    _parse_ast,
    extract_ast_diff_features,
)

# -- Helpers -----------------------------------------------------------------


def _hunk(
    old_start: int,
    old_count: int,
    new_start: int,
    new_count: int,
    content: str,
) -> dict:
    """Build a hunk dict matching the structure from FileDiff.hunks."""
    return {
        "old_start": old_start,
        "old_count": old_count,
        "new_start": new_start,
        "new_count": new_count,
        "content": content,
    }


def _simple_hunk(
    old_start: int,
    old_count: int,
    new_start: int,
    new_count: int,
    added: list[str],
    removed: list[str],
    context: list[str] | None = None,
) -> dict:
    """Build a hunk from lists of added, removed, and context lines."""
    lines: list[str] = []
    # Interleave: for simplicity, put context first, then removed, then added
    # But a real diff interleaves them.  Let's build a simple representation.
    # We'll use a simpler approach: just list the lines in order.
    ctx = context or [" context"]
    for c in ctx:
        lines.append(f" {c}")
    for r in removed:
        lines.append(f"-{r}")
    for a in added:
        lines.append(f"+{a}")
    content = "\n".join(lines)
    return _hunk(old_start, old_count, new_start, new_count, content)


# -- Changed line computation tests -----------------------------------------


class TestComputeChangedLines:
    """Tests for _compute_changed_lines."""

    def test_added_lines(self):
        h = _hunk(1, 3, 1, 4, "+added1\n+added2\n context")
        added, removed = _compute_changed_lines([h])
        assert 1 in added
        assert 2 in added
        assert len(removed) == 0

    def test_deleted_lines(self):
        h = _hunk(1, 4, 1, 3, "-del1\n-del2\n context")
        added, removed = _compute_changed_lines([h])
        assert len(added) == 0
        assert 1 in removed
        assert 2 in removed

    def test_mixed_add_delete(self):
        h = _hunk(5, 4, 5, 5, " context1\n-old1\n+new1\n context2\n+new2")
        added, removed = _compute_changed_lines([h])
        assert 6 in added  # new1 at new line 6
        assert 8 in added  # new2 at new line 8
        assert 6 in removed  # old1 at old line 6

    def test_multiple_hunks(self):
        h1 = _hunk(1, 3, 1, 3, "+line1\n context\n context")
        h2 = _hunk(10, 3, 10, 4, " context\n context\n-line2\n+line3")
        added, removed = _compute_changed_lines([h1, h2])
        assert 1 in added
        assert 12 in removed
        assert 12 in added  # line3 at new line 12

    def test_no_changes(self):
        h = _hunk(1, 3, 1, 3, " context1\n context2\n context3")
        added, removed = _compute_changed_lines([h])
        assert len(added) == 0
        assert len(removed) == 0

    def test_skips_file_markers(self):
        h = _hunk(1, 1, 1, 1, "+++ b/file.py\n--- a/file.py\n+added")
        added, removed = _compute_changed_lines([h])
        # +++ and --- are skipped; +added is at new line 3 (markers don't advance counter)
        assert 3 in added
        assert len(added) == 1  # +++ and --- not counted


# -- AST parsing tests ------------------------------------------------------


class TestParseAst:
    """Tests for _parse_ast."""

    def test_valid_python(self):
        tree = _parse_ast("def foo():\n    pass\n")
        assert tree is not None

    def test_syntax_error(self):
        tree = _parse_ast("def foo(\n")
        assert tree is None

    def test_empty_string(self):
        tree = _parse_ast("")
        assert tree is not None  # empty module is valid


# -- Feature extraction tests -----------------------------------------------


class TestExtractAstDiffFeatures:
    """Tests for extract_ast_diff_features."""

    def test_function_added(self):
        """A new function in C with added lines overlapping it."""
        old_content = textwrap.dedent("""\
            def existing():
                pass
        """)
        new_content = textwrap.dedent("""\
            def existing():
                pass

            def new_func():
                x = 1
                return x
        """)
        # The diff adds lines 3-5 (new_func and its body)
        hunks = [_hunk(2, 1, 2, 5, " pass\n+\n+def new_func():\n+    x = 1\n+    return x")]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        assert result["file_ast_functions_added"] >= 1.0
        assert result["file_ast_functions_deleted"] == 0.0

    def test_function_deleted(self):
        """A function removed in C, detected in C^ AST."""
        old_content = textwrap.dedent("""\
            def existing():
                pass

            def removed_func():
                pass
        """)
        new_content = textwrap.dedent("""\
            def existing():
                pass
        """)
        # The diff deletes lines 3-4 (removed_func)
        hunks = [_hunk(2, 3, 2, 1, " pass\n-\n-def removed_func():\n-    pass\n")]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        assert result["file_ast_functions_deleted"] >= 1.0
        assert result["file_ast_functions_added"] == 0.0

    def test_function_body_changed(self):
        """Changing lines inside a function body (not the def line) still
        counts the function as affected if its range overlaps changed lines."""
        old_content = textwrap.dedent("""\
            def foo():
                x = 1
                return x
        """)
        new_content = textwrap.dedent("""\
            def foo():
                x = 2
                y = 3
                return x + y
        """)
        # Changes are inside foo's body (lines 2-3 in old, 2-4 in new)
        hunks = [_hunk(
            1, 3, 1, 4,
            " def foo():\n-    x = 1\n+    x = 2\n"
            "+    y = 3\n-    return x\n+    return x + y",
        )]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        # foo's range (1-3 in old, 1-4 in new) overlaps deleted/added lines
        assert result["file_ast_functions_added"] >= 1.0
        assert result["file_ast_functions_deleted"] >= 1.0

    def test_nested_function(self):
        """Nested def is correctly identified."""
        old_content = textwrap.dedent("""\
            def outer():
                pass
        """)
        new_content = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                inner()
        """)
        hunks = [_hunk(
            1, 2, 1, 5,
            " def outer():\n-    pass\n+    def inner():\n"
            "+        pass\n+    inner()",
        )]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        # Both outer (range overlaps) and inner (entirely new) should be counted
        assert result["file_ast_functions_added"] >= 1.0

    def test_try_except_changed(self):
        """Try/except block with changed lines."""
        old_content = textwrap.dedent("""\
            try:
                x = 1
            except ValueError:
                pass
        """)
        new_content = textwrap.dedent("""\
            try:
                x = 2
                y = 3
            except ValueError:
                pass
            except TypeError:
                pass
        """)
        hunks = [_hunk(
            1, 4, 1, 6,
            " try:\n-    x = 1\n+    x = 2\n+    y = 3\n"
            " except ValueError:\n     pass\n"
            "+except TypeError:\n+    pass",
        )]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        assert result["file_ast_try_except_changed"] >= 1.0

    def test_unchanged_nodes_not_counted(self):
        """A function that exists in both versions with no changes is NOT counted."""
        old_content = textwrap.dedent("""\
            def foo():
                pass

            def bar():
                pass
        """)
        new_content = textwrap.dedent("""\
            def foo():
                pass

            def bar():
                x = 1
                return x
        """)
        # Only bar is changed (lines 4-5 in old)
        hunks = [_hunk(4, 2, 4, 3, " def bar():\n-    pass\n+    x = 1\n+    return x")]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        # Only bar should be counted (foo is unchanged)
        assert result["file_ast_functions_added"] == 1.0
        assert result["file_ast_functions_deleted"] == 1.0

    def test_non_python_returns_zeros(self):
        """Non-Python files return all zeros."""
        result = extract_ast_diff_features(
            "function foo() { }",
            "function foo() { return 1; }",
            [_hunk(1, 1, 1, 2, "-old\n+new")],
            "test.js",
        )
        assert result["file_ast_functions_added"] == 0.0
        assert result["file_ast_functions_deleted"] == 0.0
        assert result["file_ast_try_except_changed"] == 0.0

    def test_binary_returns_zeros(self):
        """Binary files (None content) return zeros."""
        result = extract_ast_diff_features(None, None, [], "test.png")
        assert result["file_ast_functions_added"] == 0.0

    def test_newly_added_file(self):
        """Newly added file: all top-level functions counted as added."""
        new_content = textwrap.dedent("""\
            def func_a():
                pass

            def func_b():
                pass
        """)
        # All lines are added
        hunks = [_hunk(0, 0, 1, 6, "+def func_a():\n+    pass\n+\n+def func_b():\n+    pass\n+")]
        result = extract_ast_diff_features(None, new_content, hunks, "new.py")
        assert result["file_ast_functions_added"] == 2.0
        assert result["file_ast_functions_deleted"] == 0.0

    def test_deleted_file(self):
        """Deleted file: all top-level functions counted as deleted."""
        old_content = textwrap.dedent("""\
            def func_a():
                pass

            def func_b():
                pass
        """)
        hunks = [_hunk(1, 6, 0, 0, "-def func_a():\n-    pass\n-\n-def func_b():\n-    pass\n-")]
        result = extract_ast_diff_features(old_content, None, hunks, "old.py")
        assert result["file_ast_functions_deleted"] == 2.0
        assert result["file_ast_functions_added"] == 0.0

    def test_empty_file(self):
        """Empty file returns all zeros."""
        result = extract_ast_diff_features("", "", [], "empty.py")
        assert result["file_ast_functions_added"] == 0.0
        assert result["file_ast_functions_deleted"] == 0.0
        assert result["file_ast_try_except_changed"] == 0.0

    def test_syntax_error_returns_zeros(self):
        """File with syntax errors returns zeros (no regex fallback)."""
        old_content = "def foo(\n"
        new_content = "def bar():\n    pass\n"
        hunks = [_hunk(1, 1, 1, 2, "-def foo(\n+def bar():\n+    pass")]
        result = extract_ast_diff_features(old_content, new_content, hunks, "bad.py")
        assert result["file_ast_functions_added"] == 0.0
        assert result["file_ast_functions_deleted"] == 0.0

    def test_deterministic(self):
        """Same input produces identical output."""
        old_content = "def foo():\n    pass\n"
        new_content = "def foo():\n    x = 1\n"
        hunks = [_hunk(1, 2, 1, 2, " def foo():\n-    pass\n+    x = 1")]
        r1 = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        r2 = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        assert r1 == r2

    def test_async_function_counted(self):
        """AsyncFunctionDef is counted as a function."""
        old_content = ""
        new_content = textwrap.dedent("""\
            async def fetch():
                pass
        """)
        hunks = [_hunk(0, 0, 1, 2, "+async def fetch():\n+    pass")]
        result = extract_ast_diff_features(old_content, new_content, hunks, "test.py")
        assert result["file_ast_functions_added"] >= 1.0
