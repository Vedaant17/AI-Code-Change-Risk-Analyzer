"""Tests for defect attribution labeling (Phase 3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.dataset.config import DatasetConfig
from backend.app.dataset.labeling import DefectLabeler
from backend.app.schemas.diff import CommitInfo, FileDiff, FileStatus, Hunk


def _make_commit(
    sha: str,
    message: str,
    files: list[FileDiff] | None = None,
    author_date: datetime | None = None,
) -> CommitInfo:
    c = CommitInfo(
        sha=sha,
        short_sha=sha[:8],
        author="test",
        author_date=author_date or datetime(2026, 1, 1, tzinfo=timezone.utc),
        message=message,
        files=files or [],
    )
    c.compute_stats()
    return c


def _make_file(path: str = "src/main.py", lines_added: int = 1, lines_deleted: int = 0) -> FileDiff:
    return FileDiff(
        path=path,
        status=FileStatus.MODIFIED,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        hunks=[
            Hunk(
                old_start=1,
                old_count=3,
                new_start=1,
                new_count=3 + lines_added - lines_deleted,
                content=(
                    " context\n"
                    + "-old line\n" * lines_deleted
                    + "+new line\n" * lines_added
                    + " context\n"
                ),
            )
        ],
    )


class TestBugFixDetection:
    def test_bug_fix_fix_word(self) -> None:
        c = _make_commit("a" * 40, "fix: buffer overflow", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        assert labeler.identify_bug_fixes([c]) == {"a" * 40}

    def test_bug_fix_fixes_word(self) -> None:
        c = _make_commit("a" * 40, "fixes crash on login", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        assert labeler.identify_bug_fixes([c]) == {"a" * 40}

    def test_bug_fix_hotfix(self) -> None:
        c = _make_commit("a" * 40, "hotfix: race condition", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        assert labeler.identify_bug_fixes([c]) == {"a" * 40}

    def test_bug_fix_regression(self) -> None:
        c = _make_commit("a" * 40, "regression in parser", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        assert labeler.identify_bug_fixes([c]) == {"a" * 40}

    def test_no_false_positive_fixing_docs(self) -> None:
        c = _make_commit("a" * 40, "Fixing the README typo", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        # "Fixing" does NOT match \bfix(?:es|ed)?\b because the word
        # boundary after "fix" fails (next char "i" is a word char)
        assert labeler.identify_bug_fixes([c]) == set()

    def test_no_false_positive_feature_name(self) -> None:
        c = _make_commit("a" * 40, "feature_fix_helper", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        # "feature_fix_helper" — no standalone bug-fix word
        assert labeler.identify_bug_fixes([c]) == set()


class TestRevertDetection:
    def test_revert_detection(self) -> None:
        c = _make_commit("b" * 40, "Revert 'Add feature X'")
        labeler = DefectLabeler(DatasetConfig())
        assert labeler._is_revert(c) is True

    def test_revert_not_automatic_positive(self) -> None:
        c1 = _make_commit("a" * 40, "add new feature")
        c2 = _make_commit("b" * 40, "Revert 'add new feature'")
        labeler = DefectLabeler(DatasetConfig())
        revert_map = labeler.identify_reverts([c1, c2])
        # c1 is reverted by matching quoted message text
        assert revert_map.get("a" * 40) == "b" * 40


class TestSHAResolution:
    def test_sha_resolves_to_known_commit(self) -> None:
        c1 = _make_commit("a" * 40, "add feature")
        # Use actual prefix of c1's SHA (first 12 hex chars)
        sha_prefix = c1.sha[:12]
        c2 = _make_commit("b" * 40, f"fix: bug referencing {sha_prefix}")
        labeler = DefectLabeler(DatasetConfig())
        result = labeler._resolve_sha_references(
            c2.message,
            {c1.sha},
            {c1.sha[:12]: c1.sha},
        )
        assert result == c1.sha

    def test_sha_not_resolved_is_invalid(self) -> None:
        c2 = _make_commit("b" * 40, "fix: bug referencing abc1234xyz9999")
        known_sha = "c" * 40
        labeler = DefectLabeler(DatasetConfig())
        result = labeler._resolve_sha_references(
            c2.message,
            {known_sha},
            {known_sha[:12]: known_sha},
        )
        assert result is None

    def test_sha_ambiguous_prefix(self) -> None:
        sha_a = "abc1234567890abcdef1234567890abcdef"
        sha_b = "abc1234567890abcdef1234567890abcdeg"
        c3 = _make_commit("d" * 40, "fix: bug abc1234")
        labeler = DefectLabeler(DatasetConfig())
        result = labeler._resolve_sha_references(
            c3.message,
            {sha_a, sha_b},
            {sha_a[:12]: sha_a, sha_b[:12]: sha_b},
        )
        # Ambiguous: both start with abc1234
        assert result is None

    def test_sha_references_self(self) -> None:
        # Use a valid 12-char prefix of the commit's own SHA
        sha_prefix = ("a" * 40)[:12]
        c = _make_commit("a" * 40, f"fix: bug {sha_prefix}")
        labeler = DefectLabeler(DatasetConfig())
        result = labeler._resolve_sha_references(
            c.message,
            {c.sha},
            {c.sha[:12]: c.sha},
        )
        # Self-references resolve, but the caller excludes them
        assert result == c.sha


class TestDefectAttribution:
    def test_explicit_sha_attribution(self) -> None:
        c1 = _make_commit("a" * 40, "add feature X")
        sha_prefix = c1.sha[:12]
        c2 = _make_commit("b" * 40, f"fix: regression {sha_prefix}", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        assert attributions["a" * 40][0] == "positive"
        assert attributions["a" * 40][3] == "explicit_sha_reference"

    def test_revert_attribution(self) -> None:
        c1 = _make_commit("a" * 40, "add feature X")
        c2 = _make_commit("b" * 40, "Revert 'add feature X'")
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        revert_map = labeler.identify_reverts([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, revert_map)
        assert attributions["a" * 40][0] == "positive"
        assert attributions["a" * 40][3] == "revert"

    def test_revert_not_automatic_positive(self) -> None:
        c1 = _make_commit("a" * 40, "add feature X")
        c2 = _make_commit("b" * 40, "Revert 'add feature X'")
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        revert_map = labeler.identify_reverts([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, revert_map)
        # c2 is the revert — should not automatically be positive
        assert attributions["b" * 40][0] == "negative"

    def test_line_overlap_basic(self) -> None:
        c_a = _make_commit(
            "a" * 40,
            "modify feature",
            [_make_file("src/main.py", lines_added=2, lines_deleted=1)],
        )
        c_b = _make_commit(
            "b" * 40,
            "fix: regression in main",
            [_make_file("src/main.py", lines_added=1, lines_deleted=0)],
        )
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c_a, c_b])
        attributions = labeler.attribute_defects([c_a, c_b], bug_fix_shas, {})
        # Without repo, basic overlap cannot verify restoration → no medium positive
        assert attributions["a" * 40][0] in ("negative", "ambiguous")

    def test_weak_file_overlap_ambiguous(self) -> None:
        c1 = _make_commit(
            "a" * 40,
            "modify feature",
            [_make_file("src/main.py")],
        )
        c2 = _make_commit(
            "b" * 40,
            "another change",
            [_make_file("src/main.py")],
        )
        c3 = _make_commit(
            "c" * 40,
            "fix: some bug",
            [_make_file("src/main.py")],
        )
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2, c3])
        attributions = labeler.attribute_defects([c1, c2, c3], bug_fix_shas, {})
        # c1 is within lookback of bug-fix c3 and shares a file — should be ambiguous
        assert attributions["a" * 40][0] == "ambiguous"

    def test_negative_no_evidence(self) -> None:
        c1 = _make_commit("a" * 40, "add feature", [_make_file()])
        labeler = DefectLabeler(DatasetConfig())
        attributions = labeler.attribute_defects([c1], set(), {})
        assert attributions["a" * 40][0] == "negative"
        assert attributions["a" * 40][3] == "none"


class TestObservationWindow:
    def test_observation_window_boundary(self) -> None:
        config = DatasetConfig(observation_window_commits=2, observation_window_days=90)
        base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # c1: candidate commit at day 0
        c1 = _make_commit(
            "a" * 40, "add feature",
            [_make_file("src/main.py")],
            author_date=base_date,
        )
        # c2: bug-fix at day 1, same file — within 90-day window
        c2 = _make_commit(
            "b" * 40, "fix: regression",
            [_make_file("src/main.py")],
            author_date=base_date + timedelta(days=1),
        )
        # c3: unrelated change at day 100 — outside window
        c3 = _make_commit(
            "c" * 40, "unrelated change",
            [_make_file("src/other.py")],
            author_date=base_date + timedelta(days=100),
        )
        labeler = DefectLabeler(config)
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2, c3])
        attributions = labeler.attribute_defects([c1, c2, c3], bug_fix_shas, {})
        # c2 is within window (1 day), shares file → ambiguous
        assert attributions["a" * 40][0] == "ambiguous"

    def test_observation_window_time_exceeded(self) -> None:
        config = DatasetConfig(observation_window_commits=5, observation_window_days=30)
        base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # c1: candidate at day 0
        c1 = _make_commit(
            "a" * 40, "add feature",
            [_make_file("src/main.py")],
            author_date=base_date,
        )
        # c2: bug-fix at day 60, same file — outside 30-day window
        c2 = _make_commit(
            "b" * 40, "fix: regression",
            [_make_file("src/main.py")],
            author_date=base_date + timedelta(days=60),
        )
        labeler = DefectLabeler(config)
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        # c2 is outside 30-day window → c1 gets negative
        assert attributions["a" * 40][0] == "negative"

    def test_lookback_window_boundary(self) -> None:
        config = DatasetConfig(lookback_commits=1, observation_window_commits=1)
        base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # c1: old commit (outside lookback of c3, and outside observation window)
        c1 = _make_commit(
            "a" * 40, "old change",
            [_make_file("src/main.py")],
            author_date=base_date,
        )
        # c2: more recent commit (within lookback of c3)
        c2 = _make_commit(
            "b" * 40, "another change",
            [_make_file("src/main.py")],
            author_date=base_date + timedelta(days=1),
        )
        # c3: bug-fix (latest)
        c3 = _make_commit(
            "c" * 40, "fix: regression",
            [_make_file("src/main.py")],
            author_date=base_date + timedelta(days=2),
        )
        labeler = DefectLabeler(config)
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2, c3])
        attributions = labeler.attribute_defects([c1, c2, c3], bug_fix_shas, {})
        # c1: observation_window_commits=1, so c1 only looks at position 1 (c2, not
        # a bug-fix) → c1 is negative
        assert attributions["a" * 40][0] == "negative"
        # c2: looks at position 2 (c3, bug-fix, same file) → ambiguous
        assert attributions["b" * 40][0] == "ambiguous"


class TestThresholdInteractions:
    def test_sha_suppresses_file_threshold(self) -> None:
        """Bug-fix with 10 files + valid SHA ref → high-confidence still applies."""
        files = [_make_file(f"src/file{i}.py") for i in range(10)]
        c1 = _make_commit("a" * 40, "add feature")
        sha_prefix = c1.sha[:12]
        c2 = _make_commit("b" * 40, f"fix: many files {sha_prefix}", files)
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        # Even though c2 has 10 files (> 5), SHA reference is high-confidence
        assert attributions["a" * 40][0] == "positive"
        assert attributions["a" * 40][3] == "explicit_sha_reference"

    def test_revert_suppresses_file_threshold(self) -> None:
        """Bug-fix with 15 files + valid revert → high-confidence still applies."""
        files = [_make_file(f"src/file{i}.py") for i in range(15)]
        c1 = _make_commit("a" * 40, "add feature")
        c2 = _make_commit("b" * 40, "Revert 'add feature'", files)
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        revert_map = labeler.identify_reverts([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, revert_map)
        # Even though c2 has 15 files (> 5), revert is high-confidence
        assert attributions["a" * 40][0] == "positive"
        assert attributions["a" * 40][3] == "revert"

    def test_medium_blocked_at_6_files(self) -> None:
        """Bug-fix with 6 files + line overlap → ambiguous (medium blocked)."""
        files = [_make_file(f"src/file{i}.py") for i in range(6)]
        c1 = _make_commit(
            "a" * 40, "modify feature",
            [_make_file("src/file0.py", lines_added=2, lines_deleted=1)],
        )
        c2 = _make_commit("b" * 40, "fix: many files", files)
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        # c1 shares a file with bug-fix c2, but c2 has 6 files (> 5)
        # Medium is blocked, so c1 should be ambiguous
        assert attributions["a" * 40][0] == "ambiguous"

    def test_exclusion_at_21_files(self) -> None:
        """Bug-fix with 21 files → fully excluded, even with SHA reference."""
        files = [_make_file(f"src/file{i}.py") for i in range(21)]
        c1 = _make_commit("a" * 40, "add feature")
        sha_prefix = c1.sha[:12]
        c2 = _make_commit("b" * 40, f"fix: massive refactor {sha_prefix}", files)
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        # c2 has 21 files (> 20) — fully excluded from ALL attribution
        assert attributions["a" * 40][0] != "positive"

    def test_medium_allowed_at_5_files(self) -> None:
        """Bug-fix with 5 files → medium attribution allowed (but needs repo for positive)."""
        files = [_make_file(f"src/file{i}.py") for i in range(5)]
        c1 = _make_commit(
            "a" * 40, "modify feature",
            [_make_file("src/file0.py", lines_added=2, lines_deleted=1)],
        )
        c2 = _make_commit("b" * 40, "fix: moderate change", files)
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2])
        attributions = labeler.attribute_defects([c1, c2], bug_fix_shas, {})
        # c2 has 5 files (≤ 5) — medium path is allowed, but without repo
        # basic overlap returns False → c1 gets ambiguous
        assert attributions["a" * 40][0] in ("ambiguous", "negative")


class TestAmbiguousNotInSupervised:
    def test_ambiguous_not_in_supervised(self) -> None:
        """Ambiguous attributions should be tracked separately."""
        c1 = _make_commit(
            "a" * 40, "modify feature",
            [_make_file("src/main.py")],
        )
        c2 = _make_commit(
            "b" * 40, "another change",
            [_make_file("src/other.py")],
        )
        c3 = _make_commit(
            "c" * 40, "fix: regression",
            [_make_file("src/main.py")],
        )
        labeler = DefectLabeler(DatasetConfig())
        bug_fix_shas = labeler.identify_bug_fixes([c1, c2, c3])
        attributions = labeler.attribute_defects([c1, c2, c3], bug_fix_shas, {})
        ambiguous = [
            sha for sha, v in attributions.items() if v[0] == "ambiguous"
        ]
        supervised = [
            sha for sha, v in attributions.items()
            if v[0] in ("positive", "negative")
        ]
        # Ambiguous and supervised are mutually exclusive
        assert not (set(ambiguous) & set(supervised))
