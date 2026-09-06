"""Language detection and test-file identification (Phase 2).

Pure-utility module — no git or diff dependencies.  Maps file extensions to
language names and classifies files as test or production code based on
path conventions.

This module does **not** use ``linguist``, ``pygments``, or any external
library.  It is intentionally simple and deterministic.
"""

from __future__ import annotations

import os
import re

# ── Canonical language list ────────────────────────────────────────────────
# The position in this list encodes the numeric ``language`` feature.
# ``detect_language_index()`` returns a float in [0, 1] that represents
# the language's position in this list.

LANGUAGE_LIST: list[str] = [
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "cpp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "sql",
    "shell",
    "yaml",
    "json",
    "toml",
    "xml",
    "html",
    "css",
    "markdown",
    "text",
    "unknown",
]

# ── Extension → language mapping ───────────────────────────────────────────

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "markdown",
}

# ── Test-file path patterns ────────────────────────────────────────────────

# Path segments that indicate test directories
_TEST_DIR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)test_/"),
    re.compile(r"(^|/)_test/"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"(^|/)test/"),
]

# Filename patterns for test files
_TEST_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^test_[\w]+"),
    re.compile(r"^[\w]+_test\.[\w]+$"),
    re.compile(r"^[\w]+_spec\.[\w]+$"),
    re.compile(r"^[\w]+\.test\.[\w]+$"),
    re.compile(r"^[\w]+\.spec\.[\w]+$"),
]


def detect_language(path: str) -> str:
    """Return the language name for a file path based on its extension.

    Parameters
    ----------
    path:
        File path relative to the repository root.

    Returns
    -------
    str
        Language name (e.g. ``"python"``) or ``"unknown"``.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    return EXTENSION_MAP.get(ext, "unknown")


def detect_language_index(path: str) -> float:
    """Return a float in [0, 1] encoding the language's canonical position.

    The value is ``language_list_index / (len(LANGUAGE_LIST) - 1)``.
    This makes the feature deterministic and ordered.
    """
    lang = detect_language(path)
    try:
        idx = LANGUAGE_LIST.index(lang)
    except ValueError:
        idx = LANGUAGE_LIST.index("unknown")
    return idx / max(len(LANGUAGE_LIST) - 1, 1)


def is_test_file(path: str) -> bool:
    """Heuristic: is this path a test file?

    Checks both directory conventions (``tests/``, ``test_/``, ``spec/``)
    and filename conventions (``test_*.py``, ``*_test.py``, etc.).
    """
    # Check directory segments
    for pattern in _TEST_DIR_PATTERNS:
        if pattern.search(path):
            return True

    # Check filename (basename only)
    basename = os.path.basename(path)
    for pattern in _TEST_FILE_PATTERNS:
        if pattern.match(basename):
            return True

    return False


def is_prod_file(path: str) -> bool:
    """Heuristic: is this path a production (non-test) file?"""
    return not is_test_file(path)
