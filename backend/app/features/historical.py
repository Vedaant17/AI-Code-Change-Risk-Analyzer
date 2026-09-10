"""Historical file feature extraction for Phase 4.5a.

Extracts per-file features from git history using information available
strictly before candidate commit C.  Every git command uses ``C^``
(parent of C) as the revision boundary, ensuring C itself is excluded.

The three features produced are:

  file_commit_count:
      Number of commits that touched this file before C.

  file_historical_bug_fixes:
      Number of commits touching this file before C whose messages match
      the frozen bug-fix patterns from ``labeling.py``.

  file_days_since_last_change:
      Days between the most recent commit touching this file before C
      and C's author timestamp.  0 for newly created files.

Information boundary
--------------------
For candidate commit C with parent C^:

  information(feature(C)) <= information available immediately before C

All git commands use ``C^`` as revision, which by definition excludes C.
No post-C commit, label, revert, defect evidence, or future history
enters any feature.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime

from backend.app.dataset.labeling import _BUG_FIX_PATTERNS

logger = logging.getLogger(__name__)


# -- Git primitives ----------------------------------------------------------


def _run_git(repo_path: str, args: list[str], timeout: int = 30) -> str:
    """Run a git command and return stripped stdout.

    Returns empty string on any failure (git not found, command error, etc.).
    """
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("git command failed: %s: %s", cmd, exc)
        return ""


def _get_parent_timestamp(repo_path: str, commit_sha: str) -> int:
    """Get the author timestamp of the parent of *commit_sha*.

    For root commits (no parent), returns 0.
    """
    ts_str = _run_git(
        repo_path,
        ["log", "-1", "--format=%at", f"{commit_sha}^"],
    )
    if ts_str:
        try:
            return int(ts_str)
        except ValueError:
            pass
    return 0


def _is_bug_fix_message(message: str) -> bool:
    """Check if a commit message matches the frozen bug-fix patterns.

    Uses the exact same ``_BUG_FIX_PATTERNS`` from ``labeling.py``.
    """
    return any(p.search(message) for p in _BUG_FIX_PATTERNS)


# -- Public API --------------------------------------------------------------


def extract_historical_features(
    repo_path: str,
    file_path: str,
    commit_sha: str,
    commit_timestamp: datetime | None,
) -> dict[str, float]:
    """Extract historical per-file features from git history.

    Parameters
    ----------
    repo_path:
        Path to the local git repository working directory.
    file_path:
        Path to the file relative to the repo root.
    commit_sha:
        Full SHA of the candidate commit C.
    commit_timestamp:
        Author date of C.  Used for computing days-since-last-change.

    Returns
    -------
    dict
        Keys: ``file_commit_count``, ``file_historical_bug_fixes``,
        ``file_days_since_last_change``.  All values are floats >= 0.
    """
    parent_ref = f"{commit_sha}^"

    # -- file_commit_count --
    # git rev-list --count C^ -- file
    # Counts commits touching file before C.  C^ excludes C.
    count_str = _run_git(
        repo_path,
        ["rev-list", "--count", parent_ref, "--", file_path],
    )
    try:
        commit_count = int(count_str) if count_str else 0
    except ValueError:
        commit_count = 0

    # -- file_historical_bug_fixes --
    # git log --format=%H C^ -- file  =>  get SHAs of all prior commits
    # Then check each message against _BUG_FIX_PATTERNS
    bug_fix_count = 0.0
    if commit_count > 0:
        log_output = _run_git(
            repo_path,
            ["log", "--format=%H", parent_ref, "--", file_path],
        )
        if log_output:
            shas = log_output.splitlines()
            for sha in shas:
                sha = sha.strip()
                if not sha:
                    continue
                msg = _run_git(
                    repo_path,
                    ["log", "-1", "--format=%s", sha],
                )
                if _is_bug_fix_message(msg):
                    bug_fix_count += 1.0

    # -- file_days_since_last_change --
    # git log -1 --format=%at C^ -- file  =>  timestamp of last prior commit
    last_ts_str = _run_git(
        repo_path,
        ["log", "-1", "--format=%at", parent_ref, "--", file_path],
    )
    days_since = 0.0
    if last_ts_str and commit_timestamp is not None:
        try:
            last_ts = int(last_ts_str)
            c_ts = int(commit_timestamp.replace(tzinfo=UTC).timestamp())
            delta_secs = max(0, c_ts - last_ts)
            days_since = float(delta_secs // 86400)
        except (ValueError, TypeError, OSError):
            days_since = 0.0

    return {
        "file_commit_count": float(commit_count),
        "file_historical_bug_fixes": bug_fix_count,
        "file_days_since_last_change": days_since,
    }
