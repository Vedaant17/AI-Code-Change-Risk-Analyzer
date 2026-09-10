"""AST-based diff feature extraction for Phase 4.5a.

Analyzes structural properties of ACTUALLY CHANGED Python code by:

  1. Obtaining full file content at C (current) and C^ (parent)
  2. Parsing each with ``ast.parse()`` (complete, valid Python files)
  3. Mapping diff hunk line numbers to old-file / new-file coordinates
  4. Counting AST nodes whose source ranges overlap changed lines

This module does NOT parse isolated diff fragments.  If ``ast.parse()``
fails on a full file, the fallback is to return zeros -- not regex
counts mislabeled as AST features.

The three features produced are:

  file_ast_functions_added:
      Count of FunctionDef/AsyncFunctionDef in the new-file AST whose
      source range overlaps at least one *added* line.

  file_ast_functions_deleted:
      Count of FunctionDef/AsyncFunctionDef in the old-file AST whose
      source range overlaps at least one *deleted* line.

  file_ast_try_except_changed:
      Count of Try nodes in either AST whose source range overlaps *any*
      changed line (added or deleted).
"""

from __future__ import annotations

import ast
import logging

from backend.app.features.language import detect_language

logger = logging.getLogger(__name__)


# -- Line-number mapping from hunks -----------------------------------------


def _compute_changed_lines(
    hunks: list[dict],
) -> tuple[set[int], set[int]]:
    """Compute added line numbers (new-file) and deleted line numbers (old-file).

    Parameters
    ----------
    hunks:
        List of hunk dicts with keys: old_start, old_count, new_start,
        new_count, content.

    Returns
    -------
    (added_lines_new, deleted_lines_old)
        1-indexed line number sets in new-file and old-file coordinates.
    """
    added_lines_new: set[int] = set()
    deleted_lines_old: set[int] = set()

    for hunk in hunks:
        old_line = hunk["old_start"]
        new_line = hunk["new_start"]
        for raw_line in hunk["content"].splitlines():
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                added_lines_new.add(new_line)
                new_line += 1
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                deleted_lines_old.add(old_line)
                old_line += 1
            else:
                old_line += 1
                new_line += 1

    return added_lines_new, deleted_lines_old


# -- AST node collection ----------------------------------------------------


def _collect_nodes(
    tree: ast.Module,
    node_types: tuple[type[ast.AST], ...],
) -> list[ast.AST]:
    """Collect all AST nodes of the given types from the tree."""
    return [node for node in ast.walk(tree) if isinstance(node, node_types)]


def _overlaps(node: ast.AST, lines: set[int]) -> bool:
    """Check if an AST node's source range overlaps any line in *lines*.

    Uses ``lineno`` and ``end_lineno`` attributes (available on all
    nodes produced by ``ast.parse()`` in Python 3.8+).
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return False
    return any(line in lines for line in range(start, end + 1))


def _count_overlapping(
    tree: ast.Module,
    node_types: tuple[type[ast.AST], ...],
    lines: set[int],
) -> int:
    """Count nodes of *node_types* whose source range overlaps *lines*."""
    nodes = _collect_nodes(tree, node_types)
    return sum(1 for n in nodes if _overlaps(n, lines))


# -- Full-file AST parsing --------------------------------------------------


def _parse_ast(content: str) -> ast.Module | None:
    """Parse *content* as a Python module.  Returns None on failure."""
    try:
        return ast.parse(content)
    except SyntaxError:
        return None


# -- Public API --------------------------------------------------------------


_FUNC_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_TRY_NODE_TYPES = (ast.Try,)


def extract_ast_diff_features(
    old_content: str | None,
    new_content: str | None,
    hunks: list[dict],
    file_path: str,
) -> dict[str, float]:
    """Extract AST-based structural features from a file diff.

    Parameters
    ----------
    old_content:
        Full file content at C^ (parent).  None for newly added files.
    new_content:
        Full file content at C (current).  None for deleted files.
    hunks:
        List of hunk dicts (from ``FileDiff.hunks`` via ``model_dump()``).
    file_path:
        File path for language detection.

    Returns
    -------
    dict
        Keys: ``file_ast_functions_added``, ``file_ast_functions_deleted``,
        ``file_ast_try_except_changed``.  All values are floats >= 0.
    """
    zero = {
        "file_ast_functions_added": 0.0,
        "file_ast_functions_deleted": 0.0,
        "file_ast_try_except_changed": 0.0,
    }

    # Non-Python files: return zeros immediately
    lang = detect_language(file_path)
    if lang != "python":
        return zero

    # Compute changed line numbers from hunks
    added_lines_new, deleted_lines_old = _compute_changed_lines(hunks)

    # Parse full file contents
    old_ast = _parse_ast(old_content) if old_content is not None else None
    new_ast = _parse_ast(new_content) if new_content is not None else None

    # If either parse fails, return zeros (no regex fallback)
    if old_content is not None and old_ast is None:
        logger.debug("ast.parse failed for old content of %s, returning zeros", file_path)
        return zero
    if new_content is not None and new_ast is None:
        logger.debug("ast.parse failed for new content of %s, returning zeros", file_path)
        return zero

    # Count features
    funcs_added = 0.0
    funcs_deleted = 0.0
    try_changed = 0.0

    if new_ast is not None and added_lines_new:
        funcs_added = float(
            _count_overlapping(new_ast, _FUNC_NODE_TYPES, added_lines_new)
        )
        all_changed = added_lines_new | deleted_lines_old
        try_changed += _count_overlapping(
            new_ast, _TRY_NODE_TYPES, all_changed
        )

    if old_ast is not None and deleted_lines_old:
        funcs_deleted = float(
            _count_overlapping(old_ast, _FUNC_NODE_TYPES, deleted_lines_old)
        )
        all_changed = added_lines_new | deleted_lines_old
        try_changed += _count_overlapping(
            old_ast, _TRY_NODE_TYPES, all_changed
        )

    # Deduplicate try_except counts: a Try node may appear in both ASTs
    # at the same line range.  Only count it once if it overlaps the same
    # changed lines in both versions.
    if old_ast is not None and new_ast is not None:
        all_changed = added_lines_new | deleted_lines_old
        if all_changed:
            old_try_nodes = _collect_nodes(old_ast, _TRY_NODE_TYPES)
            new_try_nodes = _collect_nodes(new_ast, _TRY_NODE_TYPES)
            old_try_ranges = {
                (getattr(n, "lineno", 0), getattr(n, "end_lineno", 0))
                for n in old_try_nodes
                if _overlaps(n, all_changed)
            }
            new_try_ranges = {
                (getattr(n, "lineno", 0), getattr(n, "end_lineno", 0))
                for n in new_try_nodes
                if _overlaps(n, all_changed)
            }
            # Count unique line ranges (dedup across old/new)
            unique_try_ranges = old_try_ranges | new_try_ranges
            try_changed = float(len(unique_try_ranges))

    return {
        "file_ast_functions_added": funcs_added,
        "file_ast_functions_deleted": funcs_deleted,
        "file_ast_try_except_changed": try_changed,
    }
