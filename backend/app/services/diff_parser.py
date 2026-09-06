"""Lightweight unified-diff parser (Phase 1).

Parses raw unified diff text into structured FileDiff objects using only
Python string operations and regex.  No third-party diff libraries required.

The parser is behind a clean ``parse_unified_diff`` function interface so it
can be swapped for ``unidiff`` or another library later without affecting
callers.
"""

from __future__ import annotations

import re

from backend.app.schemas.diff import FileDiff, FileStatus, Hunk

# Matches a diff-file header line:
#   diff --git a/old b/new
#   --- a/old
#   +++ b/new
_DIFF_HEADER_RE = re.compile(
    r"^diff\s+--git\s+a/(.+?)\s+b/(.+)$", re.MULTILINE
)

# Matches a hunk header:
#   @@ -old_start,old_count +new_start,new_count @@ [optional function context]
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$", re.MULTILINE
)

# Matches the "rename from / rename to" similarity lines
_RENAME_RE = re.compile(r"^rename from (.+)$\nrename to (.+)$", re.MULTILINE)

# Binary diff marker
_BINARY_MARKER = "Binary files"


def _count_lines(text: str) -> int:
    """Count non-empty lines in a hunk body (additions + deletions only)."""
    count = 0
    for line in text.splitlines():
        if line.startswith("+") or line.startswith("-"):
            if not line.startswith("+++") and not line.startswith("---"):
                count += 1
    return count


def _parse_single_file_diff(file_section: str) -> FileDiff | None:
    """Parse a single file's diff section into a FileDiff."""
    lines = file_section.splitlines(keepends=True)
    if not lines:
        return None

    # --- Extract default path from "diff --git a/old b/new" header ---
    diff_header_match = re.search(
        r"^diff\s+--git\s+a/(.+?)\s+b/(.+)$", file_section, re.MULTILINE
    )
    default_old_path: str | None = None
    default_new_path: str | None = None
    if diff_header_match:
        default_old_path = diff_header_match.group(1)
        default_new_path = diff_header_match.group(2)

    # --- Detect file path and status from headers ---
    path = ""
    old_path: str | None = None
    status = FileStatus.MODIFIED
    is_binary = False

    for i, line in enumerate(lines):
        line_stripped = line.rstrip("\n")

        # Binary file?
        if line_stripped.startswith(_BINARY_MARKER):
            is_binary = True
            status = FileStatus.BINARY
            # Try to extract path from "Binary files a/path and b/path differ"
            m = re.match(r"Binary files (.+) and (.+) differ", line_stripped)
            if m:
                path = m.group(1).removeprefix("a/")
            continue

        # Also detect "GIT binary patch" as binary indicator
        if line_stripped == "GIT binary patch":
            is_binary = True
            status = FileStatus.BINARY
            continue

        # --- / +++ header lines
        if line_stripped.startswith("--- "):
            old_path_candidate = line_stripped[4:]
            if old_path_candidate == "/dev/null":
                status = FileStatus.ADDED
            else:
                old_path_candidate = old_path_candidate.split("\t")[0]
                if old_path_candidate.startswith("a/"):
                    old_path_candidate = old_path_candidate[2:]
                old_path = old_path_candidate
            continue

        if line_stripped.startswith("+++ "):
            new_path_candidate = line_stripped[4:]
            if new_path_candidate == "/dev/null":
                status = FileStatus.DELETED
            else:
                new_path_candidate = new_path_candidate.split("\t")[0]
                if new_path_candidate.startswith("b/"):
                    new_path_candidate = new_path_candidate[2:]
                path = new_path_candidate
            continue

        # Rename detection
        rename_from_match = re.match(r"^rename from (.+)$", line_stripped)
        if rename_from_match:
            old_path = rename_from_match.group(1)
            status = FileStatus.RENAMED
            continue

        rename_to_match = re.match(r"^rename to (.+)$", line_stripped)
        if rename_to_match:
            path = rename_to_match.group(1)
            continue

    # Fall back to the diff --git header paths when ---/+++ didn't provide one
    if not path:
        if default_new_path:
            path = default_new_path
        elif default_old_path:
            path = default_old_path

    if not old_path and default_old_path:
        old_path = default_old_path

    if not path and not is_binary:
        return None

    # --- Parse hunks ---
    hunks: list[Hunk] = []
    lines_added = 0
    lines_deleted = 0

    hunk_positions = list(_HUNK_HEADER_RE.finditer(file_section))

    for idx, hunk_match in enumerate(hunk_positions):
        old_start = int(hunk_match.group(1))
        old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
        new_start = int(hunk_match.group(3))
        new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

        # Hunk body: from end of this header to start of next header (or EOF)
        body_start = hunk_match.end()
        body_end = hunk_positions[idx + 1].start() if idx + 1 < len(hunk_positions) else len(file_section)
        hunk_body = file_section[body_start:body_end]

        hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                content=hunk_body,
            )
        )

        # Count actual +/- lines in the body
        for hline in hunk_body.splitlines():
            if hline.startswith("+") and not hline.startswith("+++"):
                lines_added += 1
            elif hline.startswith("-") and not hline.startswith("---"):
                lines_deleted += 1

    return FileDiff(
        path=path,
        old_path=old_path,
        status=status,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        is_binary=is_binary,
        hunks=hunks,
    )


def parse_unified_diff(raw_diff: str) -> list[FileDiff]:
    """Parse a unified diff string into a list of FileDiff objects.

    This is the clean public interface.  It can be replaced with an
    ``unidiff``-based implementation later without changing callers.

    Parameters
    ----------
    raw_diff:
        The full output of ``git diff`` (unified format).

    Returns
    -------
    list[FileDiff]
        One entry per file changed in the diff.
    """
    if not raw_diff or not raw_diff.strip():
        return []

    file_diffs: list[FileDiff] = []

    # Split on the "diff --git" header lines to isolate per-file sections.
    parts = re.split(r"(?=^diff --git )", raw_diff, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part or not part.startswith("diff --git"):
            continue

        result = _parse_single_file_diff(part)
        if result is not None:
            file_diffs.append(result)

    return file_diffs
