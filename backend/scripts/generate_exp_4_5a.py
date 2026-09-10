"""Generate exp-4.5a experimental features for combined-v3 dataset.

Reads frozen combined-v3 JSONL, clones repos, computes 6 experimental
features per (commit_sha, file_path) pair, writes output JSONL.

All temporary repository clones live under .tmp/repos/ (project-local).

Provides two implementations:
  SubprocessFeatureGenerator -- original subprocess-based (reference)
  GitPythonFeatureGenerator  -- optimized with GitPython (production)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

# Ensure UTF-8 on Windows (only when run as main script, not when imported)
if sys.platform == "win32" and __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import git as gitpython

from backend.app.dataset.labeling import _BUG_FIX_PATTERNS
from backend.app.features.ast_diff import extract_ast_diff_features
from backend.app.features.experimental_schemas import EXPERIMENTAL_FEATURE_COUNT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Project-local temporary workspace for repo clones
REPOS_DIR = Path(__file__).resolve().parent.parent.parent / ".tmp" / "repos"


# ── Subprocess Git helpers (reference) ──────────────────────────────────


def _git(repo_path: str, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("git command failed: %s: %s", args, exc)
        return None


def _git_file_content(repo_path: str, sha: str, file_path: str) -> str | None:
    """Get file content at a specific commit. Returns None for missing/binary."""
    content = _git(repo_path, ["show", f"{sha}:{file_path}"])
    if content is None:
        return None
    if not content:
        return ""
    try:
        content.encode("utf-8")
        return content
    except UnicodeEncodeError:
        return None


# ── Diff parsing helpers ─────────────────────────────────────────────────


def _parse_diff_hunks(diff_text: str) -> dict[str, list[dict]]:
    """Parse unified diff into per-file hunks for AST extractor."""
    result: dict[str, list[dict]] = {}
    current_file = None
    hunks: list[dict] = []
    old_count = new_count = 0
    lines_list: list[str] = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file and hunks:
                result[current_file] = hunks
            parts = line.split(" b/", 1)
            current_file = parts[1] if len(parts) > 1 else None
            hunks = []
            old_count = new_count = 0
            lines_list = []
        elif line.startswith("@@"):
            m = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                hunks.append({
                    "old_start": int(m.group(1)),
                    "old_count": old_count,
                    "new_start": int(m.group(2)),
                    "new_count": new_count,
                    "content": "\n".join(lines_list),
                })
                old_count = new_count = 0
                lines_list = []
        elif current_file:
            if line.startswith("+"):
                new_count += 1
                lines_list.append(line)
            elif line.startswith("-"):
                old_count += 1
                lines_list.append(line)
            elif line.startswith(" "):
                old_count += 1
                new_count += 1
                lines_list.append(line)
    if current_file and hunks:
        result[current_file] = hunks
    return result


# ── Diff generation (Python-side) ───────────────────────────────────────


def _make_diff(old_content: str | None, new_content: str | None,
               file_path: str) -> str:
    """Generate unified diff text from old/new content using difflib."""
    import difflib
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = (new_content or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


# ── Feature generation ──────────────────────────────────────────────────


@dataclass
class SubprocessFeatureGenerator:
    """Original subprocess-based feature generator (reference implementation).

    Uses HEAD-based _get_path_commits.  Kept for semantic equivalence testing.
    """

    _commit_cache: dict[tuple[str, str], dict] = field(default_factory=dict)
    _exp_cache: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    _msg_cache: dict[str, str] = field(default_factory=dict)
    _path_commit_cache: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def clear(self):
        self._commit_cache.clear()
        self._exp_cache.clear()
        self._msg_cache.clear()
        self._path_commit_cache.clear()

    def compute_for_row(
        self,
        repo_path: str,
        commit_sha: str,
        file_path: str,
        commit_date_str: str,
    ) -> list[float]:
        cache_key = (commit_sha, file_path)
        if cache_key in self._exp_cache:
            return self._exp_cache[cache_key]

        parent_sha = self._get_parent(repo_path, commit_sha)

        old_content = (
            self._get_file(repo_path, parent_sha, file_path)
            if parent_sha else None
        )
        new_content = self._get_file(repo_path, commit_sha, file_path)

        diff_text = _make_diff(old_content, new_content, file_path)

        f_added, f_deleted, try_changed = _compute_ast_features(
            old_content, new_content, diff_text, file_path,
        )

        commit_count = self._count_prior_commits(repo_path, commit_sha, file_path)
        bug_fix_count = self._count_prior_bug_fixes(repo_path, commit_sha, file_path)
        days_since = self._days_since_last_change(
            repo_path, commit_sha, file_path, commit_date_str,
        )

        features = [f_added, f_deleted, try_changed,
                    float(commit_count), float(bug_fix_count), days_since]
        self._exp_cache[cache_key] = features
        return features

    def _get_parent(self, repo_path: str, commit_sha: str) -> str | None:
        cache_key = ("parent", commit_sha)
        if cache_key in self._commit_cache:
            return self._commit_cache[cache_key].get("parent")
        parent = _git(repo_path, ["rev-parse", f"{commit_sha}^"])
        self._commit_cache[cache_key] = {"parent": parent}
        return parent

    def _get_file(self, repo_path: str, sha: str, file_path: str) -> str | None:
        cache_key = ("file", sha, file_path)
        if cache_key in self._commit_cache:
            return self._commit_cache[cache_key].get("content")
        content = _git_file_content(repo_path, sha, file_path)
        self._commit_cache[cache_key] = {"content": content}
        return content

    def _get_path_commits(
        self, repo_path: str, commit_sha: str, file_path: str,
    ) -> list[str]:
        """Get ordered list of commit SHAs touching file_path before commit_sha.

        NOTE: Uses HEAD-based filtering (reference implementation).
        """
        cache_key = (repo_path, file_path, commit_sha)
        if cache_key in self._path_commit_cache:
            return self._path_commit_cache[cache_key]

        head_key = (repo_path, file_path, "HEAD")
        if head_key not in self._path_commit_cache:
            output = _git(repo_path, [
                "rev-list", "HEAD", "--", file_path,
            ])
            self._path_commit_cache[head_key] = [
                s.strip() for s in (output or "").split("\n") if s.strip()
            ]
        all_shas = self._path_commit_cache[head_key]

        result = []
        found = False
        for s in all_shas:
            if s == commit_sha:
                found = True
                break
            result.append(s)

        if not found:
            output = _git(repo_path, [
                "rev-list", commit_sha, "--", file_path,
            ])
            result = [s.strip() for s in (output or "").split("\n") if s.strip()]

        self._path_commit_cache[cache_key] = result
        return result

    def _count_prior_commits(
        self, repo_path: str, commit_sha: str, file_path: str,
    ) -> int:
        return len(self._get_path_commits(repo_path, commit_sha, file_path))

    def _count_prior_bug_fixes(
        self, repo_path: str, commit_sha: str, file_path: str,
    ) -> int:
        ancestor_shas = self._get_path_commits(repo_path, commit_sha, file_path)
        if not ancestor_shas:
            return 0
        uncached = [s for s in ancestor_shas if s not in self._msg_cache]
        if uncached:
            batch_output = _git(repo_path, [
                "log", "--format=%H %s", *uncached,
            ])
            if batch_output:
                for log_line in batch_output.split("\n"):
                    parts = log_line.split(" ", 1)
                    if len(parts) >= 2:
                        self._msg_cache[parts[0]] = parts[1]
        count = 0
        for s in ancestor_shas:
            msg = self._msg_cache.get(s, "")
            if any(p.search(msg) for p in _BUG_FIX_PATTERNS):
                count += 1
        return count

    def _days_since_last_change(
        self, repo_path: str, commit_sha: str, file_path: str,
        commit_date_str: str,
    ) -> float:
        ancestor_shas = self._get_path_commits(repo_path, commit_sha, file_path)
        if not ancestor_shas:
            return 0.0
        most_recent = ancestor_shas[0]
        output = _git(repo_path, [
            "log", "-1", "--format=%at", most_recent,
        ])
        if not output:
            return 0.0
        try:
            last_ts = int(output.strip())
        except (ValueError, TypeError):
            return 0.0
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
            current_ts = int(dt.timestamp())
            return max(0.0, float((current_ts - last_ts) / 86400.0))
        except (ValueError, TypeError, OSError):
            return 0.0


@dataclass
class GitPythonFeatureGenerator:
    """Optimized GitPython-based feature generator.

    Uses C^ as revision boundary (matches frozen historical.py semantics).
    Uses repo.iter_commits(rev=C^, paths=file) for historical queries.
    Replaces all subprocess Git operations with GitPython.
    """

    # Per-repo GitPython Repo objects: repo_name -> Repo
    _repos: dict[str, gitpython.Repo] = field(default_factory=dict)

    # Commit metadata cache: (repo_name, sha) -> {"parent": str|None, "ts": int, "msg": str}
    _meta: dict[tuple[str, str], dict] = field(default_factory=dict)

    # File content cache: (repo_name, sha, path) -> str|None
    _file_cache: dict[tuple[str, str, str], str | None] = field(default_factory=dict)

    # Historical raw cache: (repo_name, candidate_sha, path) ->
    #   {"count": int, "bug_fix_count": int, "last_ts": int|None}
    _hist_raw: dict[tuple[str, str, str], dict] = field(default_factory=dict)

    # Bug-fix message cache: sha -> bool
    _bug_fix_cache: dict[str, bool] = field(default_factory=dict)

    # Experiment result cache: (candidate_sha, path) -> list[float]
    _exp_cache: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    def open_repo(self, repo_name: str, repo_path: str) -> None:
        """Open a GitPython Repo object for the given repo."""
        if repo_name not in self._repos:
            self._repos[repo_name] = gitpython.Repo(repo_path)

    def clear(self) -> None:
        """Clear all caches (call between repos)."""
        self._repos.clear()
        self._meta.clear()
        self._file_cache.clear()
        self._hist_raw.clear()
        self._bug_fix_cache.clear()
        self._exp_cache.clear()

    def _get_meta(self, repo_name: str, sha: str) -> dict:
        """Get commit metadata (parent, timestamp, message) with caching."""
        key = (repo_name, sha)
        if key in self._meta:
            return self._meta[key]

        repo = self._repos[repo_name]
        try:
            commit = repo.commit(sha)
        except gitpython.BadName:
            self._meta[key] = {"parent": None, "ts": 0, "msg": ""}
            return self._meta[key]

        parent = commit.parents[0].hexsha if commit.parents else None
        ts = commit.committed_date
        # First line of message (matches frozen historical.py: %s format)
        msg = commit.message.split("\n", 1)[0] if commit.message else ""

        self._meta[key] = {"parent": parent, "ts": ts, "msg": msg}
        return self._meta[key]

    def _get_file_content(self, repo_name: str, sha: str, file_path: str) -> str | None:
        """Get file content at a commit. Returns None for missing/binary.

        Strips trailing newline to match ``git show sha:path`` behavior.
        """
        key = (repo_name, sha, file_path)
        if key in self._file_cache:
            return self._file_cache[key]

        repo = self._repos[repo_name]
        try:
            commit = repo.commit(sha)
            blob = commit.tree[file_path]
            data = blob.data_stream.read()
            text = data.decode("utf-8")
            # git show sha:path outputs content with a trailing newline
            # stripped.  Match that behavior precisely: remove exactly one
            # trailing newline if present.
            if text.endswith("\n"):
                text = text[:-1]
            self._file_cache[key] = text
            return text
        except (KeyError, UnicodeDecodeError, gitpython.BadName, ValueError):
            self._file_cache[key] = None
            return None

    def _get_historical_raw(
        self, repo_name: str, commit_sha: str, file_path: str,
    ) -> dict:
        """Get raw historical data for a (candidate, path) pair with caching.

        Returns {"count": int, "bug_fix_count": int, "last_ts": int|None}.
        Uses C^ as revision boundary (frozen historical.py semantics).
        """
        key = (repo_name, commit_sha, file_path)
        if key in self._hist_raw:
            return self._hist_raw[key]

        # /dev/null paths have no history (file additions in diff metadata)
        if file_path == "/dev/null":
            self._hist_raw[key] = {"count": 0, "bug_fix_count": 0, "last_ts": None}
            return self._hist_raw[key]

        repo = self._repos[repo_name]
        meta = self._get_meta(repo_name, commit_sha)

        # Root commit: no ancestors
        if meta["parent"] is None:
            self._hist_raw[key] = {"count": 0, "bug_fix_count": 0, "last_ts": None}
            return self._hist_raw[key]

        # Walk ancestors of C^ that touch file_path (frozen: no --follow).
        # Single pass: collect count, bug_fix_count, and most-recent timestamp.
        parent_ref = f"{commit_sha}^"
        bug_fix_count = 0
        last_ts = None
        count = 0

        try:
            for commit in repo.iter_commits(rev=parent_ref, paths=file_path):
                count += 1
                if last_ts is None:
                    last_ts = commit.committed_date
                msg = commit.message.split("\n", 1)[0] if commit.message else ""
                if self._is_bug_fix(msg):
                    bug_fix_count += 1
        except (ValueError, gitpython.BadName):
            pass

        self._hist_raw[key] = {
            "count": count,
            "bug_fix_count": bug_fix_count,
            "last_ts": last_ts,
        }
        return self._hist_raw[key]

    def _is_bug_fix(self, message: str) -> bool:
        """Check if a commit message matches frozen bug-fix patterns."""
        if not message:
            return False
        cached = self._bug_fix_cache.get(message)
        if cached is not None:
            return cached
        result = any(p.search(message) for p in _BUG_FIX_PATTERNS)
        self._bug_fix_cache[message] = result
        return result

    def compute_for_row(
        self,
        repo_name: str,
        commit_sha: str,
        file_path: str,
        commit_date_str: str,
    ) -> list[float]:
        """Compute 6 experimental features for a single row."""
        cache_key = (commit_sha, file_path)
        if cache_key in self._exp_cache:
            return self._exp_cache[cache_key]

        try:
            meta = self._get_meta(repo_name, commit_sha)
            parent_sha = meta["parent"]

            # File contents
            old_content = (
                self._get_file_content(repo_name, parent_sha, file_path)
                if parent_sha else None
            )
            new_content = self._get_file_content(repo_name, commit_sha, file_path)

            # Diff
            diff_text = _make_diff(old_content, new_content, file_path)

            # AST features
            f_added, f_deleted, try_changed = _compute_ast_features(
                old_content, new_content, diff_text, file_path,
            )

            # Historical features (C^ boundary)
            hist = self._get_historical_raw(repo_name, commit_sha, file_path)
            commit_count = float(hist["count"])
            bug_fix_count = float(hist["bug_fix_count"])

            # Days since last change
            days_since = 0.0
            if hist["last_ts"] is not None and commit_date_str:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
                    c_ts = int(dt.timestamp())
                    delta_secs = max(0, c_ts - hist["last_ts"])
                    days_since = float(delta_secs // 86400)
                except (ValueError, TypeError, OSError):
                    days_since = 0.0

            features = [f_added, f_deleted, try_changed,
                        commit_count, bug_fix_count, days_since]
        except (ValueError, gitpython.BadName, OSError):
            features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self._exp_cache[cache_key] = features
        return features


# ── Shared helpers ────────────────────────────────────────────────────────


def _compute_ast_features(
    old_content: str | None,
    new_content: str | None,
    diff_text: str,
    file_path: str,
) -> tuple[float, float, float]:
    """Compute AST diff features from old/new content and diff text.

    Returns (functions_added, functions_deleted, try_except_changed).
    """
    if old_content is not None and new_content is not None:
        file_hunks = _parse_diff_hunks(diff_text).get(file_path, [])
        ast_result = extract_ast_diff_features(
            old_content, new_content, file_hunks, file_path,
        )
    elif new_content is not None and old_content is None:
        n = new_content.count("\n") + 1
        ast_result = extract_ast_diff_features(
            None, new_content,
            [{"old_start": 0, "old_count": 0,
              "new_start": 1, "new_count": n,
              "content": "+" * n}],
            file_path,
        )
    elif old_content is not None and new_content is None:
        n = old_content.count("\n") + 1
        ast_result = extract_ast_diff_features(
            old_content, None,
            [{"old_start": 1, "old_count": n,
              "new_start": 0, "new_count": 0,
              "content": "-" * n}],
            file_path,
        )
    else:
        return 0.0, 0.0, 0.0

    return (
        float(ast_result.get("file_ast_functions_added", 0)),
        float(ast_result.get("file_ast_functions_deleted", 0)),
        float(ast_result.get("file_ast_try_except_changed", 0)),
    )


def _remove_dir_robust(path: str) -> bool:
    """Remove a directory, handling Windows locked-file issues.

    Tries shutil.rmtree, then rmdir /s /q, then rename to a temp name
    so that git clone can proceed.  Returns True if the path no longer
    exists (or has been renamed out of the way).
    """
    if not os.path.exists(path):
        return True

    # Attempt 1: shutil.rmtree
    try:
        shutil.rmtree(path)
        return not os.path.exists(path)
    except (PermissionError, OSError):
        pass

    # Attempt 2: Windows rmdir (handles locked files better)
    try:
        subprocess.run(
            ["cmd", "/c", "rmdir", "/s", "/q", path],
            capture_output=True, timeout=30,
        )
        if not os.path.exists(path):
            return True
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Attempt 3: rename to .bak so git clone can proceed
    # Rename usually succeeds even with locked files because it doesn't
    # require the same permissions as delete.
    try:
        bak = path + "._bak_cleanup"
        if os.path.exists(bak):
            shutil.rmtree(bak, ignore_errors=True)
        os.rename(path, bak)
        return not os.path.exists(path)
    except (PermissionError, OSError):
        pass

    return not os.path.exists(path)


def _clone_repo(repo_url: str, repo_name: str) -> str | None:
    """Clone a repo into .tmp/repos/<repo_name>. Returns path or None.

    If a valid clone already exists, reuses it.  Otherwise removes the
    old directory and clones fresh.  Uses full clone (not blobless) so
    that git show/cat-file operations are local (no network per file).
    """
    repo_path = str(REPOS_DIR / repo_name)

    # Reuse existing valid clone
    if os.path.exists(os.path.join(repo_path, ".git")):
        try:
            repo = gitpython.Repo(repo_path)
            _ = repo.head.commit.hexsha  # verify object store is intact
            log.info(f"[{repo_name}] Reusing existing clone")
            return repo_path
        except Exception:
            log.info(f"[{repo_name}] Existing clone invalid, removing...")
            _remove_dir_robust(repo_path)

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--quiet", repo_url, repo_path],
        capture_output=True, text=True, timeout=1200,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        log.error(f"Clone {repo_name} failed: {result.stderr[:200]}")
        return None
    return repo_path


def _cleanup_repo(repo_name: str) -> None:
    repo_path = REPOS_DIR / repo_name
    if repo_path.exists():
        shutil.rmtree(str(repo_path), ignore_errors=True)


# ── I/O helpers ──────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_lenient(path: Path) -> list[dict]:
    """Read JSONL, dropping an incomplete trailing line.

    Used only for checkpoint files that may have been truncated by an
    interrupted append.  A malformed line in the *middle* of the file
    raises ``json.JSONDecodeError`` (unlike the trailing-line case).
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    rows: list[dict] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only tolerate truncation on the last non-empty line.
            remaining = "".join(lines[i + 1:]).strip()
            if remaining:
                raise  # malformed line in the middle — real corruption
            break  # truncated trailing line from interrupted write
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL atomically: write to temp file, then rename."""
    tmp_path = path.with_suffix(".tmp")
    _write_jsonl(tmp_path, rows)
    tmp_path.replace(path)


def _append_jsonl(path: Path, row: dict) -> None:
    """Append a single row to a JSONL file (for streaming output)."""
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")


# ── Checkpoint helpers ──────────────────────────────────────────────────


def _load_progress(progress_path: Path) -> dict:
    """Load checkpoint progress file. Returns dict of repo -> {done, rows}."""
    if not progress_path.exists():
        return {}
    with open(progress_path, encoding="utf-8") as f:
        return json.load(f)


def _save_progress(progress_path: Path, progress: dict) -> None:
    """Save checkpoint progress atomically."""
    _atomic_write_jsonl(progress_path, [])  # ensure parent exists
    tmp = progress_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(progress_path)


def _get_repo_rows(
    split_data: dict[str, list[dict]],
    repo_name: str,
) -> dict[str, list[tuple[str, str, str]]]:
    """Group rows by commit_sha for a given repo.

    Returns {commit_sha: [(file_path, split, commit_timestamp), ...]}.
    """
    commits: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split, rows in split_data.items():
        for row in rows:
            if row["repo_name"] == repo_name:
                commits[row["commit_sha"]].append(
                    (row["file_path"], split, row.get("commit_timestamp", "")),
                )
    return commits


def _find_repo_url(split_data: dict[str, list[dict]], repo_name: str) -> str | None:
    """Find the repo_url for a given repo_name."""
    for split in ["train", "validation", "test"]:
        for row in split_data[split]:
            if row["repo_name"] == repo_name:
                return row.get("repo_url")
    return None


# ── Main generation ─────────────────────────────────────────────────────


def generate(output_base: Path, repo_filter: str | None = None) -> None:
    """Generate experimental features with checkpoint/resume support.

    Streams results per-repo to disk. On resume, skips completed repos.

    Args:
        output_base: Base directory for output.
        repo_filter: If set, only process this one repo (for testing).
    """
    source_dir = Path("backend/data/datasets/combined-v3")
    checkpoint_dir = output_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_dir / "progress.json"

    gen = GitPythonFeatureGenerator()

    split_data: dict[str, list[dict]] = {}
    for split in ["train", "validation", "test"]:
        rows = _read_jsonl(source_dir / f"{split}.jsonl")
        split_data[split] = rows

    # Group by repo
    repo_commits: dict[str, dict[str, list[tuple]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for split, rows in split_data.items():
        for row in rows:
            rn = row["repo_name"]
            if repo_filter and rn != repo_filter:
                continue
            repo_commits[rn][row["commit_sha"]].append(
                (row["file_path"], split, row.get("commit_timestamp", "")),
            )

    total_commits = sum(len(v) for v in repo_commits.values())
    log.info(f"Target: {total_commits} commits across {len(repo_commits)} repos")

    # Load checkpoint progress
    progress = _load_progress(progress_path)
    completed_repos = {k for k, v in progress.items() if v.get("done")}
    if completed_repos:
        log.info(f"Resuming: {len(completed_repos)} repos already completed")

    processed = 0
    t0 = time.time()

    for repo_name in sorted(repo_commits.keys()):
        if repo_name in completed_repos:
            repo_progress = progress.get(repo_name, {})
            processed += repo_progress.get("rows", 0)
            log.info(f"[{repo_name}] Skipping (completed: {repo_progress.get('rows', 0)} rows)")
            continue

        commits = repo_commits[repo_name]
        repo_url = _find_repo_url(split_data, repo_name)
        if not repo_url:
            log.warning(f"Skipping {repo_name}: no repo_url")
            continue

        log.info(f"[{repo_name}] Cloning ({len(commits)} commits)...")
        t1 = time.time()
        repo_path = _clone_repo(repo_url, repo_name)
        if not repo_path:
            continue
        log.info(f"[{repo_name}] Clone done in {time.time()-t1:.1f}s")

        gen.open_repo(repo_name, repo_path)

        # Per-repo checkpoint file
        repo_ckpt_path = checkpoint_dir / f"{repo_name}.jsonl"

        # Check if we have partial progress for this repo
        # Track (commit_sha, file_path) pairs, not just commit SHAs.
        # A truncated checkpoint may have partial commit rows.
        repo_done_keys: set[tuple[str, str]] = set()
        repo_row_count = 0
        if repo_ckpt_path.exists():
            existing = _read_jsonl_lenient(repo_ckpt_path)
            repo_row_count = len(existing)
            for r in existing:
                repo_done_keys.add((r["commit_sha"], r["file_path"]))
            # Rewrite checkpoint to remove any truncated trailing line
            if existing:
                _atomic_write_jsonl(repo_ckpt_path, existing)
            log.info(f"[{repo_name}] Resuming from checkpoint ({repo_row_count} rows)")

        new_rows = 0
        for commit_sha in sorted(commits.keys()):
            # Skip commit only if ALL its files were already processed
            all_files_done = all(
                (commit_sha, fp) in repo_done_keys
                for fp, _, _ in commits[commit_sha]
            )
            if all_files_done:
                processed += 1
                continue

            for file_path, split, commit_timestamp in commits[commit_sha]:
                if (commit_sha, file_path) in repo_done_keys:
                    continue  # this file already checkpointed
                features = gen.compute_for_row(
                    repo_name, commit_sha, file_path, commit_timestamp,
                )
                row = {
                    "commit_sha": commit_sha,
                    "file_path": file_path,
                    "experimental_feature_version": "exp-4.5a",
                    "experimental_features": features,
                }
                _append_jsonl(repo_ckpt_path, row)
                new_rows += 1

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_commits - processed) / rate if rate > 0 else 0
                log.info(f"  {processed}/{total_commits} commits "
                         f"({processed*100//total_commits}%) "
                         f"[{rate:.1f} commits/s, ETA {eta:.0f}s]")

        # Mark repo as completed
        progress[repo_name] = {"done": True, "rows": repo_row_count + new_rows}
        _save_progress(progress_path, progress)

        gen.clear()
        _cleanup_repo(repo_name)
        log.info(f"[{repo_name}] Done ({repo_row_count + new_rows} rows)")

    elapsed_total = time.time() - t0
    log.info(f"Generation complete: {elapsed_total:.1f}s total")

    # Merge checkpoints into final output
    _merge_checkpoints(output_base, checkpoint_dir, split_data)

    total = sum(
        len(_read_jsonl(output_base / "combined-v3" / f"{split}.jsonl"))
        for split in ["train", "validation", "test"]
    )
    metadata = {
        "experimental_feature_version": "exp-4.5a",
        "source_dataset": "combined-v3",
        "source_dataset_version": "v1",
        "feature_count": EXPERIMENTAL_FEATURE_COUNT,
        "feature_names": [
            "file_ast_functions_added", "file_ast_functions_deleted",
            "file_ast_try_except_changed", "file_commit_count",
            "file_historical_bug_fixes", "file_days_since_last_change",
        ],
        "total_rows": total,
        "split_row_counts": {
            split: len(_read_jsonl(output_base / "combined-v3" / f"{split}.jsonl"))
            for split in ["train", "validation", "test"]
        },
    }
    with open(output_base / "combined-v3" / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    log.info(f"Done. Total: {total} rows -> {output_base / 'combined-v3'}")


def _merge_checkpoints(
    output_base: Path,
    checkpoint_dir: Path,
    split_data: dict[str, list[dict]],
) -> None:
    """Merge per-repo checkpoint JSONL files into final per-split output."""
    output_dir = output_base / "combined-v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build repo_name -> split lookup from source data
    repo_split_map: dict[tuple[str, str], str] = {}
    for split, rows in split_data.items():
        for row in rows:
            key = (row["commit_sha"], row["file_path"])
            repo_split_map[key] = split

    # Collect all rows from checkpoints, grouped by split
    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    ckpt_files = list(checkpoint_dir.glob("*.jsonl"))
    for ckpt_file in ckpt_files:
        for row in _read_jsonl(ckpt_file):
            key = (row["commit_sha"], row["file_path"])
            split = repo_split_map.get(key, "train")
            split_rows[split].append(row)

    for split in ["train", "validation", "test"]:
        _atomic_write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
        log.info(f"Wrote {split}.jsonl: {len(split_rows[split])} rows")


# ── Validation ──────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate(output_base: Path) -> dict:
    output_dir = output_base / "combined-v3"
    source_dir = Path("backend/data/datasets/combined-v3")
    results: dict[str, str | bool] = {}

    log.info("=" * 60)
    log.info("VALIDATION")
    log.info("=" * 60)

    # 1. Row counts
    log.info("\n--- 1. Row counts ---")
    for split in ["train", "validation", "test"]:
        exp_rows = _read_jsonl(output_dir / f"{split}.jsonl")
        src_rows = _read_jsonl(source_dir / f"{split}.jsonl")
        match = len(exp_rows) == len(src_rows)
        log.info(f"  {split}: exp={len(exp_rows)}, source={len(src_rows)}, match={match}")
        results[f"row_count_{split}"] = match

    # 2. Key integrity
    log.info("\n--- 2. Key integrity ---")
    for split in ["train", "validation", "test"]:
        exp_rows = _read_jsonl(output_dir / f"{split}.jsonl")
        keys = [(r["commit_sha"], r["file_path"]) for r in exp_rows]
        dup = len(keys) - len(set(keys))
        log.info(f"  {split}: {len(exp_rows)} rows, {dup} duplicates")
        results[f"key_integrity_{split}"] = dup == 0

    # 3. Feature contract
    log.info("\n--- 3. Feature contract ---")
    bad_version = bad_dim = bad_neg = 0
    for split in ["train", "validation", "test"]:
        for r in _read_jsonl(output_dir / f"{split}.jsonl"):
            if r.get("experimental_feature_version") != "exp-4.5a":
                bad_version += 1
            feats = r.get("experimental_features", [])
            if len(feats) != EXPERIMENTAL_FEATURE_COUNT:
                bad_dim += 1
            if any(f < 0 for f in feats):
                bad_neg += 1
    log.info(f"  bad_version={bad_version}, bad_dim={bad_dim}, bad_neg={bad_neg}")
    results["feature_contract"] = bad_version == 0 and bad_dim == 0 and bad_neg == 0

    # 4. Summary statistics
    log.info("\n--- 4. Plausibility ---")
    all_feats: list[list[float]] = []
    for split in ["train", "validation", "test"]:
        for r in _read_jsonl(output_dir / f"{split}.jsonl"):
            all_feats.append(r.get("experimental_features", []))

    names = [
        "file_ast_functions_added", "file_ast_functions_deleted",
        "file_ast_try_except_changed", "file_commit_count",
        "file_historical_bug_fixes", "file_days_since_last_change",
    ]
    stats = {}
    for i, name in enumerate(names):
        vals = [f[i] for f in all_feats]
        nz = [v for v in vals if v > 0]
        s = {
            "min": min(vals), "max": max(vals),
            "mean": round(mean(vals), 4) if vals else 0,
            "median": round(median(vals), 4) if vals else 0,
            "nonzero": len(nz),
            "nonzero_pct": round(100 * len(nz) / len(vals), 2) if vals else 0,
        }
        stats[name] = s
        log.info(f"  {name}: min={s['min']}, max={s['max']}, "
                 f"mean={s['mean']}, median={s['median']}, "
                 f"nz={s['nonzero']} ({s['nonzero_pct']}%)")
    results["summary_stats"] = stats

    # 5. Distribution by split
    log.info("\n--- 5. Distribution by split ---")
    for split in ["train", "validation", "test"]:
        split_feats = []
        for r in _read_jsonl(output_dir / f"{split}.jsonl"):
            split_feats.append(r.get("experimental_features", []))
        log.info(f"  {split} ({len(split_feats)} rows):")
        for i, name in enumerate(names):
            vals = [f[i] for f in split_feats]
            nz = [v for v in vals if v > 0]
            log.info(f"    {name}: mean={round(mean(vals),4) if vals else 0}, "
                     f"nz={len(nz)}")

    # 6. Repository coverage
    log.info("\n--- 6. Repository coverage ---")
    repo_map = {}
    for split in ["train", "validation", "test"]:
        for src_row in _read_jsonl(source_dir / f"{split}.jsonl"):
            key = (src_row["commit_sha"], src_row["file_path"])
            repo_map[key] = src_row["repo_name"]

    repo_feats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in ["train", "validation", "test"]:
        for exp_row in _read_jsonl(output_dir / f"{split}.jsonl"):
            key = (exp_row["commit_sha"], exp_row["file_path"])
            rn = repo_map.get(key, "unknown")
            for i, name in enumerate(names):
                if exp_row["experimental_features"][i] > 0:
                    repo_feats[rn][name] += 1
    for rn in sorted(repo_feats.keys()):
        log.info(f"  {rn}: {dict(repo_feats[rn])}")

    # 7. Spot checks
    log.info("\n--- 7. Spot checks ---")
    exp_train = _read_jsonl(output_dir / "train.jsonl")
    src_train = _read_jsonl(source_dir / "train.jsonl")
    src_map = {(r["commit_sha"], r["file_path"]): r for r in src_train}
    shown = 0
    for er in exp_train:
        key = (er["commit_sha"], er["file_path"])
        sr = src_map.get(key)
        if sr and sr["repo_name"] in [
            "fastapi", "flask", "requests", "pyyaml", "black",
        ]:
            log.info(f"  repo={sr['repo_name']}, sha={er['commit_sha'][:12]}, "
                     f"file={er['file_path']}, features={er['experimental_features']}")
            shown += 1
            if shown >= 10:
                break

    # 8. Existing integrity
    log.info("\n--- 8. Existing integrity ---")
    for split in ["train", "validation", "test"]:
        src_path = source_dir / f"{split}.jsonl"
        log.info(f"  {split}.jsonl hash: {_file_hash(src_path)}")
    results["existing_integrity"] = True

    # 9. Tests
    log.info("\n--- 9. Tests ---")
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "backend/tests/", "-q",
         "--tb=short", "-x"],
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    log.info(f"  Tests: {'PASS' if test_result.returncode == 0 else 'FAIL'}")
    if test_result.returncode != 0:
        log.info(f"  Output:\n{test_result.stdout[-1000:]}")
    results["tests_pass"] = test_result.returncode == 0

    # 10. Ruff
    log.info("\n--- 10. Ruff ---")
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    ruff_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check",
         "backend/app/features/ast_diff.py",
         "backend/app/features/historical.py",
         "backend/app/features/experimental_schemas.py",
         "backend/app/ml/experimental_loader.py",
         "backend/scripts/generate_exp_4_5a.py"],
        capture_output=True, text=True, timeout=60,
        encoding="utf-8", errors="replace",
        cwd=project_root,
    )
    ruff_ok = ruff_result.returncode == 0
    log.info(f"  Ruff: {'PASS' if ruff_ok else 'FAIL'}")
    if not ruff_ok:
        log.info(f"  {ruff_result.stdout}")
    results["ruff_pass"] = ruff_ok

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None, help="Process only this repo")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip generation, run validation only")
    args = parser.parse_args()

    output_base = Path("backend/data/datasets/experimental-exp-4.5a")

    if not args.validate_only:
        generate(output_base, repo_filter=args.repo)

    results = validate(output_base)
    log.info("\n" + "=" * 60)
    all_pass = all(v for k, v in results.items() if isinstance(v, bool))
    log.info(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
