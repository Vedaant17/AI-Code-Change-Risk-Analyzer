"""Defect attribution labeling (Phase 3).

Identifies bug-fix commits and attributes historical defect evidence to
earlier commits using a strict confidence hierarchy:

  High confidence:   explicit SHA reference, revert
  Medium confidence: line-level overlap with parent-content restoration
                     (only for small bug-fixes ≤ max_files_for_medium_attribution)
  Ambiguous:         all other partial evidence (excluded from supervised training)
  Negative:          no qualifying defect evidence within observation window

The revert commit itself is NOT automatically labeled positive.
It is labeled according to its own subsequent defect evidence.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import git

from backend.app.dataset.config import DatasetConfig
from backend.app.schemas.diff import CommitInfo, Hunk

logger = logging.getLogger(__name__)

# ── Bug-fix detection patterns ─────────────────────────────────────────────

_BUG_FIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bfix(?:es|ed)?\b", re.IGNORECASE),
    re.compile(r"\bbug\b", re.IGNORECASE),
    re.compile(r"\bresolve[sd]?\b", re.IGNORECASE),
    re.compile(r"\bpatch(?:es|ed)?\b", re.IGNORECASE),
    re.compile(r"\bhotfix\b", re.IGNORECASE),
    re.compile(r"\bregression\b", re.IGNORECASE),
]

# ── Revert detection pattern ───────────────────────────────────────────────

_REVERT_PATTERN = re.compile(r"^revert\s", re.IGNORECASE)

# ── SHA extraction pattern ─────────────────────────────────────────────────

_SHA_PATTERN = re.compile(r"[0-9a-f]{7,40}", re.IGNORECASE)


class DefectLabeler:
    """Identifies bug-fix commits and attributes defects to earlier commits.

    The labeler operates in two passes:
      1. Identify bug-fixes and reverts across all commits
      2. Attribute defects using the strict confidence hierarchy
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # ── Pass 1: Detection ──────────────────────────────────────────────────

    def identify_bug_fixes(self, commits: list[CommitInfo]) -> set[str]:
        """Return SHAs of commits whose messages match bug-fix patterns."""
        bug_fix_shas: set[str] = set()
        for c in commits:
            if self._is_bug_fix(c):
                bug_fix_shas.add(c.sha)
        return bug_fix_shas

    def identify_reverts(
        self, commits: list[CommitInfo]
    ) -> dict[str, str]:
        """Return {reverted_sha: revert_commit_sha} mapping.

        Tries two strategies:
          1. SHA reference in the revert message (e.g. "This reverts commit abc123")
          2. Message text match: extracts the quoted text from "Revert '...'" and
             matches it against known commit messages.
        """
        known_shas = {c.sha for c in commits}
        sha_index = {c.sha[:12]: c.sha for c in commits}
        revert_map: dict[str, str] = {}

        # Build message -> sha lookup for quoted-text matching
        msg_to_sha: dict[str, str] = {}
        for c in commits:
            normalized = c.message.strip()
            if normalized:
                msg_to_sha[normalized] = c.sha

        for c in commits:
            if not self._is_revert(c):
                continue
            # Strategy 1: SHA reference in the message
            referenced = self._extract_revert_sha(c.message, known_shas, sha_index)
            if referenced is not None:
                revert_map[referenced] = c.sha
                continue
            # Strategy 2: Quoted message text match
            referenced = self._extract_revert_message(c.message, msg_to_sha)
            if referenced is not None:
                revert_map[referenced] = c.sha
        return revert_map

    # ── Pass 2: Attribution ────────────────────────────────────────────────

    def attribute_defects(
        self,
        commits: list[CommitInfo],
        bug_fix_shas: set[str],
        revert_map: dict[str, str],
        repo: git.Repo | None = None,
    ) -> dict[str, tuple[str, int, float, str, str]]:
        """Attribute defect evidence to each commit.

        Returns dict mapping commit_sha ->
            (label_status, defect_label, confidence, source, evidence).

        Evaluation order for each candidate commit C:
          1. Is C referenced by a bug-fix via explicit SHA?  → positive (high)
          2. Is C reverted?  → positive (high)
          3. Does a small bug-fix overlap C's changes at line level
             with parent-content restoration?  → positive (medium) for
             the specific file, only if F.size ≤ max_files_for_medium_attribution
          4. Some evidence but insufficient?  → ambiguous
          5. No evidence?  → negative
        """
        known_shas = {c.sha for c in commits}
        sha_index = {c.sha[:12]: c.sha for c in commits}

        # Build lookup: sha -> CommitInfo
        commit_by_sha: dict[str, CommitInfo] = {c.sha: c for c in commits}

        # Build chronological index
        sha_order = [c.sha for c in commits]
        sha_position = {sha: i for i, sha in enumerate(sha_order)}

        # ── Pre-filter: exclude oversized bug-fixes (full exclusion) ──
        # Bug-fixes touching > max_files_for_exclusion files are excluded
        # from ALL attribution, including high-confidence SHA/revert.
        active_bug_fix_shas: set[str] = set()
        excluded_bug_fix_shas: set[str] = set()
        for bf_sha in bug_fix_shas:
            bf = commit_by_sha[bf_sha]
            if len(bf.files) > self.config.max_files_for_exclusion:
                excluded_bug_fix_shas.add(bf_sha)
            else:
                active_bug_fix_shas.add(bf_sha)

        # ── Step 1 & 2: High-confidence from SHA references and reverts ──
        results: dict[str, tuple[str, int, float, str, str]] = {}

        for c in commits:
            # Check if C is referenced by any (active) bug-fix via SHA
            for bf_sha in active_bug_fix_shas:
                if bf_sha == c.sha:
                    continue
                bf_msg = commit_by_sha[bf_sha].message
                resolved = self._resolve_sha_references(
                    bf_msg, known_shas - {bf_sha}, sha_index
                )
                if resolved == c.sha:
                    results[c.sha] = (
                        "positive", 1, 1.0,
                        "explicit_sha_reference",
                        f"Commit {c.sha[:12]} referenced by bug-fix {bf_sha[:12]} "
                        f"via explicit SHA — commit-level attribution propagated "
                        f"to all changed files",
                    )
                    break

            # Check if C is reverted
            if c.sha in revert_map:
                revert_sha = revert_map[c.sha]
                # Revert overrides only if not already high-confidence from SHA
                if c.sha not in results or results[c.sha][3] != "explicit_sha_reference":
                    results[c.sha] = (
                        "positive", 1, 1.0,
                        "revert",
                        f"Commit {c.sha[:12]} was reverted by {revert_sha[:12]} "
                        f"— commit-level attribution propagated to all changed files",
                    )

        # ── Step 3: Medium-confidence line-overlap (file-level) ──
        # Track file-level attributions separately
        file_attributions: dict[str, dict[str, tuple[str, int, float, str, str]]] = (
            defaultdict(dict)
        )  # commit_sha -> file_path -> result tuple

        for bf_sha in active_bug_fix_shas:
            bf = commit_by_sha[bf_sha]
            bf_position = sha_position.get(bf_sha, -1)

            # Full exclusion: bug-fix too large
            if len(bf.files) > self.config.max_files_for_exclusion:
                continue

            # Medium-confidence blocked: bug-fix exceeds medium threshold
            if len(bf.files) > self.config.max_files_for_medium_attribution:
                continue

            # Also check total lines
            bf_total_lines = sum(f.lines_added + f.lines_deleted for f in bf.files)
            if bf_total_lines > self.config.max_lines_for_medium_attribution:
                continue

            # Build file -> hunks lookup for the bug-fix
            bf_hunks_by_file: dict[str, list[Hunk]] = defaultdict(list)
            for f in bf.files:
                for h in f.hunks:
                    bf_hunks_by_file[f.path].append(h)

            # Scan backward through lookback window
            lookback_start = max(0, bf_position - self.config.lookback_commits)
            for pos in range(lookback_start, bf_position):
                candidate = commits[pos]
                candidate_sha = candidate.sha

                # Skip if already high-confidence
                if candidate_sha in results:
                    continue

                for c_file in candidate.files:
                    if c_file.path not in bf_hunks_by_file:
                        continue

                    # Check line-level overlap for this file
                    if repo is not None:
                        has_restoration = self._check_line_overlap_with_restoration(
                            repo, candidate, c_file.path,
                            bf, bf_hunks_by_file[c_file.path],
                        )
                    else:
                        has_restoration = self._check_line_overlap_basic(
                            candidate, c_file.path, bf,
                            bf_hunks_by_file[c_file.path],
                        )

                    if has_restoration:
                        evidence = (
                            f"Line-level overlap with parent-content restoration "
                            f"in bug-fix {bf_sha[:12]} for file {c_file.path}"
                        )
                        file_attributions[candidate_sha][c_file.path] = (
                            "positive", 1, 0.7,
                            "line_overlap", evidence,
                        )

        # ── Step 4 & 5: Merge file attributions and assign remaining ──

        # For commits with high-confidence: all files are positive
        for c in commits:
            if c.sha in results:
                continue  # already assigned

            # Check if any file-level medium attribution exists
            if c.sha in file_attributions:
                # At least one file has medium-confidence
                # For simplicity, mark the commit as having file-level positives
                # The builder will handle per-file assignment
                # Here we mark at commit level for tracking; builder expands
                file_attr = file_attributions[c.sha]
                any_positive = any(v[0] == "positive" for v in file_attr.values())
                if any_positive:
                    # Mark as having file-level attributions
                    # The builder will create per-file labels
                    results[c.sha] = (
                        "positive", 1, 0.7,
                        "line_overlap",
                        f"File-level line-overlap attribution in {len(file_attr)} "
                        f"file(s) via bug-fix evidence",
                    )
                    continue

            # Check for ambiguous evidence
            has_ambiguous_evidence = self._has_ambiguous_evidence(
                c, commits, active_bug_fix_shas, sha_position, commit_by_sha,
            )
            if has_ambiguous_evidence:
                results[c.sha] = (
                    "ambiguous", -1, 0.0,
                    "ambiguous",
                    "Some defect evidence exists but is insufficient for "
                    "supervised labeling",
                )
            else:
                results[c.sha] = (
                    "negative", 0, 0.0,
                    "none",
                    "No qualifying defect evidence observed within "
                    "the configured observation window",
                )

        # Store file-level attributions for the builder to use
        self._file_attributions = dict(file_attributions)

        return results

    def get_file_attributions(
        self,
    ) -> dict[str, dict[str, tuple[str, int, float, str, str]]]:
        """Return file-level attributions computed during attribute_defects.

        Returns {commit_sha: {file_path: (label_status, defect_label,
        confidence, source, evidence)}}.
        """
        return getattr(self, "_file_attributions", {})

    # ── Internal: Detection helpers ────────────────────────────────────────

    def _is_bug_fix(self, commit: CommitInfo) -> bool:
        """Check if a commit message indicates a bug-fix."""
        msg = commit.message.strip()
        if not msg:
            return False

        # Must match a bug-fix pattern
        matches_pattern = any(p.search(msg) for p in _BUG_FIX_PATTERNS)
        if not matches_pattern:
            return False

        # Must have a non-trivial diff
        if commit.stats.total_lines_changed == 0:
            return False

        return True

    def _is_revert(self, commit: CommitInfo) -> bool:
        """Check if a commit message indicates a revert."""
        return bool(_REVERT_PATTERN.match(commit.message.strip()))

    # ── Internal: SHA resolution ───────────────────────────────────────────

    def _extract_revert_sha(
        self,
        message: str,
        known_shas: set[str],
        sha_index: dict[str, str],
    ) -> str | None:
        """Extract the reverted commit SHA from a revert message."""
        return self._resolve_sha_references(message, known_shas, sha_index)

    _REVERT_QUOTED_PATTERN = re.compile(r"Revert\s+[\"'](.+?)[\"']", re.IGNORECASE)

    def _extract_revert_message(
        self,
        message: str,
        msg_to_sha: dict[str, str],
    ) -> str | None:
        """Extract the reverted commit by matching quoted message text.

        Handles standard git revert format: Revert "commit message"
        Matches the quoted text against known commit messages.
        """
        match = self._REVERT_QUOTED_PATTERN.search(message)
        if not match:
            return None
        quoted = match.group(1).strip()
        return msg_to_sha.get(quoted)

    def _resolve_sha_references(
        self,
        message: str,
        known_shas: set[str],
        sha_index: dict[str, str],
    ) -> str | None:
        """Find a valid, unambiguous SHA reference in a commit message.

        Returns the full SHA if exactly one known commit is referenced,
        or None if no valid reference is found.
        """
        candidates = _SHA_PATTERN.findall(message)
        resolved: str | None = None

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Try exact match first
            if candidate_lower in known_shas:
                if resolved is not None and resolved != candidate_lower:
                    return None  # Ambiguous: multiple valid SHAs
                resolved = candidate_lower
                continue

            # Try prefix match via index
            if candidate_lower in sha_index:
                full_sha = sha_index[candidate_lower]
                if resolved is not None and resolved != full_sha:
                    return None  # Ambiguous
                resolved = full_sha
                continue

            # Check if it's a prefix of any known SHA
            matches = [
                sha for sha in known_shas
                if sha.startswith(candidate_lower)
            ]
            if len(matches) == 1:
                if resolved is not None and resolved != matches[0]:
                    return None
                resolved = matches[0]
            elif len(matches) > 1:
                return None  # Ambiguous prefix

        return resolved

    # ── Internal: Line overlap checking ────────────────────────────────────

    def _check_line_overlap_basic(
        self,
        candidate: CommitInfo,
        file_path: str,
        bug_fix: CommitInfo,
        bug_fix_hunks: list[Hunk],
    ) -> bool:
        """Check line-level overlap without parent-content verification.

        Without a git.Repo, we cannot verify that the bug-fix actually
        restores lines to their pre-change state.  Therefore this method
        always returns False — medium-confidence attribution requires
        parent-content restoration, which is only possible with git access.
        """
        return False

    def _check_line_overlap_with_restoration(
        self,
        repo: git.Repo,
        candidate: CommitInfo,
        file_path: str,
        bug_fix: CommitInfo,
        bug_fix_hunks: list[Hunk],
    ) -> bool:
        """Check line-level overlap AND parent-content restoration.

        Uses GitPython to retrieve the candidate's parent file content
        and verify that the bug-fix restores lines to their pre-change
        state in the overlapping region.

        Returns True only if both overlap and restoration are confirmed.
        """
        candidate_file = None
        for f in candidate.files:
            if f.path == file_path:
                candidate_file = f
                break

        if candidate_file is None:
            return False

        # Get parent file content for the candidate commit
        try:
            commit_obj = repo.commit(candidate.sha)
            if not commit_obj.parents:
                return False
            parent = commit_obj.parents[0]
            parent_blob = parent.tree / file_path
            parent_lines = parent_blob.data_stream.read().decode(
                errors="replace",
            ).splitlines(keepends=True)
        except (KeyError, git.exc.GitCommandError, IndexError, AttributeError):
            return False

        for c_hunk in candidate_file.hunks:
            c_old_end = c_hunk.old_start + c_hunk.old_count
            for bf_hunk in bug_fix_hunks:
                bf_old_end = bf_hunk.old_start + bf_hunk.old_count
                # Check overlap in old-file coordinates
                if not (c_hunk.old_start < bf_old_end and
                        bf_hunk.old_start < c_old_end):
                    continue

                # Overlapping region in old-file coordinates
                overlap_start = max(c_hunk.old_start, bf_hunk.old_start)
                overlap_end = min(c_old_end, bf_old_end)

                # Extract bug-fix added lines in the overlapping region
                bf_added = self._extract_added_lines_in_range(
                    bf_hunk, overlap_start, overlap_end,
                )
                if not bf_added:
                    continue

                # Extract parent content in the same range
                parent_in_range = []
                for i in range(overlap_start - 1, min(overlap_end, len(parent_lines))):
                    if 0 <= i < len(parent_lines):
                        parent_in_range.append(parent_lines[i])

                # Check if bug-fix lines match parent content
                if self._lines_match(bf_added, parent_in_range):
                    return True

        return False

    def _extract_added_lines_in_range(
        self,
        hunk: Hunk,
        range_start: int,
        range_end: int,
    ) -> list[str]:
        """Extract added lines from a hunk that fall within an old-file line range."""
        added: list[str] = []
        current_old_line = hunk.old_start

        for line in hunk.content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                # Added line — doesn't advance old line counter
                if range_start <= current_old_line <= range_end:
                    added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                # Removed line — advances old line counter
                if range_start <= current_old_line <= range_end:
                    pass  # Removed lines don't contribute to "restored" content
                current_old_line += 1
            else:
                # Context line — advances old line counter
                current_old_line += 1

        return added

    def _lines_match(
        self,
        bug_fix_lines: list[str],
        parent_lines: list[str],
    ) -> bool:
        """Check if bug-fix lines match parent content.

        Normalizes whitespace and trailing newlines for comparison.
        """
        if not bug_fix_lines or not parent_lines:
            return False

        # Normalize: strip trailing whitespace and newlines
        bf_norm = [ln.rstrip("\n\r") for ln in bug_fix_lines]
        pr_norm = [ln.rstrip("\n\r") for ln in parent_lines]

        # Check if the bug-fix lines are a subsequence of parent lines
        # (allowing for surrounding context lines we may not have extracted)
        bf_set = set(bf_norm)
        pr_set = set(pr_norm)

        # All bug-fix lines must appear in the parent
        return bf_set.issubset(pr_set)

    # ── Internal: Ambiguous evidence check ─────────────────────────────────

    def _has_ambiguous_evidence(
        self,
        candidate: CommitInfo,
        all_commits: list[CommitInfo],
        bug_fix_shas: set[str],
        sha_position: dict[str, int],
        commit_by_sha: dict[str, CommitInfo],
    ) -> bool:
        """Check if a commit has some weak defect evidence (ambiguous).

        Returns True if any bug-fix within the observation window
        shares a changed file with the candidate, but the attribution
        does not meet medium-confidence thresholds.
        """
        candidate_position = sha_position.get(candidate.sha, -1)
        if candidate_position < 0:
            return False

        candidate_files = {f.path for f in candidate.files}
        if not candidate_files:
            return False

        # Scan forward within observation window
        for pos in range(
            candidate_position + 1,
            min(
                candidate_position + self.config.observation_window_commits + 1,
                len(all_commits),
            ),
        ):
            future_commit = all_commits[pos]

            # Check time window
            if (candidate.author_date is not None and
                    future_commit.author_date is not None):
                delta = future_commit.author_date - candidate.author_date
                if delta.days > self.config.observation_window_days:
                    break

            if future_commit.sha not in bug_fix_shas:
                continue

            # This is a bug-fix within the observation window
            future_files = {f.path for f in future_commit.files}
            if candidate_files & future_files:
                return True

        return False
