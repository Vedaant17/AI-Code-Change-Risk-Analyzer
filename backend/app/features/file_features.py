"""File-level feature extraction (Phase 2).

Extracts deterministic features from a single ``FileDiff`` by inspecting
hunk content lines.  All detection is regex-based and heuristic —
no AST parsing, no external dependencies.
"""

from __future__ import annotations

import re

from backend.app.features.language import detect_language_index, is_test_file
from backend.app.features.schemas import FileFeatures
from backend.app.schemas.diff import FileDiff, FileStatus

# ── Heuristic regex patterns for hunk content ──────────────────────────────
# These match declarations in **stripped** added/removed text (the + or -
# prefix has already been removed by _split_added_removed).
# They are intentionally simple and do not attempt full syntax parsing.

# Python-style declarations
_PY_DEF_RE = re.compile(r"^\s*def\s+\w+", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^\s*class\s+\w+", re.MULTILINE)
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+\S+\s+)?import\s+", re.MULTILINE,
)

# C-family / Java / JS / TS declarations
_C_FUNC_RE = re.compile(
    r"^\s*(?:function\s+\w+|(?:public|private|protected|static|async|export|const|let|var)\s+)*"
    r"\w+\s*\(.*\)\s*(?:\{|=>)",
    re.MULTILINE,
)
_C_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+",
    re.MULTILINE,
)
_C_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+\{|import\s+\w|from\s+\S+\s+import\s+)",
    re.MULTILINE,
)

# Go / Rust function-like
_GO_FUNC_RE = re.compile(
    r"^\s*func\s+\w+", re.MULTILINE
)
_GO_IMPORT_RE = re.compile(
    r"^\s*\"[\w/]+\"", re.MULTILINE
)

# Generic: any line that contains a function name, parens, and brace
_GENERIC_FUNC_RE = re.compile(
    r"^\s*\w+\s*\(.*\)\s*\{",
    re.MULTILINE,
)

# Generic import-like: import/from/include/require
_GENERIC_IMPORT_RE = re.compile(
    r"^\s*(?:import|from|include|require|#include)\s",
    re.MULTILINE,
)


def _count_pattern(pattern: re.Pattern[str], text: str) -> int:
    """Count non-overlapping matches of *pattern* in *text*."""
    return len(pattern.findall(text))


def _split_added_removed(content: str) -> tuple[str, str]:
    """Split hunk content into the added-lines text and removed-lines text."""
    added_lines: list[str] = []
    removed_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])  # strip the leading +
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(line[1:])
    return "\n".join(added_lines), "\n".join(removed_lines)


def _indent_of_line(line: str) -> int:
    """Return the number of leading whitespace characters in *line*."""
    return len(line) - len(line.lstrip())


def _collect_changed_lines(content: str) -> list[str]:
    """Return the code lines (without + / - prefix) that were changed."""
    result: list[str] = []
    for line in content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            result.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            result.append(line[1:])
    return result


def _detect_declarations(added_text: str, removed_text: str) -> dict[str, int]:
    """Heuristically count declaration-like lines in added/removed text."""
    # Try Python-style first (very common in our target repos)
    py_add = _count_pattern(_PY_DEF_RE, added_text)
    py_rem = _count_pattern(_PY_DEF_RE, removed_text)
    c_add = _count_pattern(_C_FUNC_RE, added_text)
    c_rem = _count_pattern(_C_FUNC_RE, removed_text)
    go_add = _count_pattern(_GO_FUNC_RE, added_text)
    go_rem = _count_pattern(_GO_FUNC_RE, removed_text)
    gen_add = _count_pattern(_GENERIC_FUNC_RE, added_text)
    gen_rem = _count_pattern(_GENERIC_FUNC_RE, removed_text)

    # Use the maximum across all detectors to avoid double-counting
    func_added = max(py_add, c_add, go_add, gen_add)
    func_deleted = max(py_rem, c_rem, go_rem, gen_rem)

    py_cls_add = _count_pattern(_PY_CLASS_RE, added_text)
    py_cls_rem = _count_pattern(_PY_CLASS_RE, removed_text)
    c_cls_add = _count_pattern(_C_CLASS_RE, added_text)
    c_cls_rem = _count_pattern(_C_CLASS_RE, removed_text)

    cls_added = max(py_cls_add, c_cls_add)
    cls_deleted = max(py_cls_rem, c_cls_rem)

    py_imp_add = _count_pattern(_PY_IMPORT_RE, added_text)
    py_imp_rem = _count_pattern(_PY_IMPORT_RE, removed_text)
    c_imp_add = _count_pattern(_C_IMPORT_RE, added_text)
    c_imp_rem = _count_pattern(_C_IMPORT_RE, removed_text)
    go_imp_add = _count_pattern(_GO_IMPORT_RE, added_text)
    go_imp_rem = _count_pattern(_GO_IMPORT_RE, removed_text)
    gen_imp_add = _count_pattern(_GENERIC_IMPORT_RE, added_text)
    gen_imp_rem = _count_pattern(_GENERIC_IMPORT_RE, removed_text)

    imports_added = max(py_imp_add, c_imp_add, go_imp_add, gen_imp_add)
    imports_deleted = max(py_imp_rem, c_imp_rem, go_imp_rem, gen_imp_rem)

    return {
        "function_declarations_added": func_added,
        "function_declarations_deleted": func_deleted,
        "class_declarations_added": cls_added,
        "class_declarations_deleted": cls_deleted,
        "imports_added": imports_added,
        "imports_deleted": imports_deleted,
    }


def _compute_indent_stats(all_changed_lines: list[str]) -> tuple[float, int]:
    """Return (avg_changed_line_indent, max_changed_line_indent).

    Indent is measured as the number of leading whitespace characters in
    each changed line.  This is a **structural proxy only** — it does not
    represent true nesting depth or cyclomatic complexity.
    """
    if not all_changed_lines:
        return 0.0, 0
    indents = [_indent_of_line(line) for line in all_changed_lines]
    return sum(indents) / len(indents), max(indents)


def extract_file_features(file_diff: FileDiff) -> FileFeatures:
    """Extract deterministic features from a single ``FileDiff``.

    Parameters
    ----------
    file_diff:
        A ``FileDiff`` object produced by the Phase 1 diff parser.

    Returns
    -------
    FileFeatures
        A fully-populated feature model.
    """
    # Language and test classification
    language_idx = detect_language_index(file_diff.path)
    test = is_test_file(file_diff.path)

    # Collect all changed lines across hunks
    all_changed: list[str] = []
    added_all: list[str] = []
    removed_all: list[str] = []
    for hunk in file_diff.hunks:
        added, removed = _split_added_removed(hunk.content)
        added_all.append(added)
        removed_all.append(removed)
        all_changed.extend(_collect_changed_lines(hunk.content))

    added_text = "\n".join(added_all)
    removed_text = "\n".join(removed_all)

    # Heuristic declaration detections
    decls = _detect_declarations(added_text, removed_text)

    # Indentation statistics
    avg_indent, max_indent = _compute_indent_stats(all_changed)

    # Hunk size average
    hunk_sizes = [h.old_count + h.new_count for h in file_diff.hunks]
    avg_hunk_size = (sum(hunk_sizes) / len(hunk_sizes)) if hunk_sizes else 0.0

    return FileFeatures(
        file_path=file_diff.path,
        language=language_idx,
        is_binary=file_diff.is_binary,
        is_test_file=test,
        lines_added=file_diff.lines_added,
        lines_deleted=file_diff.lines_deleted,
        total_lines_changed=file_diff.total_lines_changed,
        hunk_count=len(file_diff.hunks),
        function_declarations_added=decls["function_declarations_added"],
        function_declarations_deleted=decls["function_declarations_deleted"],
        class_declarations_added=decls["class_declarations_added"],
        class_declarations_deleted=decls["class_declarations_deleted"],
        imports_added=decls["imports_added"],
        imports_deleted=decls["imports_deleted"],
        avg_changed_line_indent=avg_indent,
        max_changed_line_indent=max_indent,
        avg_hunk_size=avg_hunk_size,
    )
