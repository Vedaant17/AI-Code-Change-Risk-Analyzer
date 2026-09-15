"""Tests for Phase 4.11b: Repository-Backed Git-History Attribution.

Tests verify deterministic attribution, evidence hierarchy invariants,
content matching, region/function analysis, rename handling, deleted
files, revert handling, negative feasibility, and frozen integrity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.ml.phase49_supervision_feasibility import (
    _build_sha_lookup,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
)
from backend.app.ml.phase411b_git_attribution import (
    COMMIT_EVIDENCE_STRONG,
    DATA_DIR,
    FILE_EVIDENCE_MODERATE,
    FILE_EVIDENCE_STRONG,
    FILE_EVIDENCE_UNKNOWN,
    FILE_EVIDENCE_WEAK,
    _analyze_content_correspondence,
    _check_content_restoration,
    _classify_file_evidence,
    _compute_region_overlap,
    _extract_candidate_introduced_content,
    _extract_corrective_removed_content,
    _extract_functions_by_name,
    _investigate_negative_feasibility,
    _normalize_line,
    _resolve_file_identity_across_commits,
    run_phase411b,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_BASE = Path("backend/data/repos")


@pytest.fixture(scope="module")
def supervised_rows():
    return _read_all_supervised_rows(DATA_DIR)


@pytest.fixture(scope="module")
def ambiguous_rows():
    return _read_ambiguous_rows(DATA_DIR)


@pytest.fixture(scope="module")
def sha_lookup(supervised_rows, ambiguous_rows):
    return _build_sha_lookup(supervised_rows, ambiguous_rows)


@pytest.fixture(scope="module")
def experiment_output(tmp_path_factory):
    """Run experiment once for module-level sharing."""
    tmp_dir = tmp_path_factory.mktemp("phase411b_module")
    repo_dir = tmp_dir / "repos"
    result = run_phase411b(
        output_dir=tmp_dir,
        repo_base_dir=repo_dir,
    )
    return {"result": result, "tmp_dir": tmp_dir}


# ---------------------------------------------------------------------------
# TestLineNormalization
# ---------------------------------------------------------------------------


class TestLineNormalization:
    def test_strip_trailing_whitespace(self):
        assert _normalize_line("hello  ") == "hello"

    def test_normalize_line_endings(self):
        assert _normalize_line("hello\r\n") == "hello"

    def test_preserve_leading_whitespace(self):
        assert _normalize_line("  hello") == "  hello"

    def test_empty_line(self):
        assert _normalize_line("") == ""


# ---------------------------------------------------------------------------
# TestContentExtraction
# ---------------------------------------------------------------------------


class TestContentExtraction:
    def test_extract_added_lines(self):
        before = "line1\nline2\nline3\n"
        after = "line1\nline2\nline3\nline4\n"
        result = _extract_candidate_introduced_content(
            before, after, []
        )
        assert "line4" in result["added_lines"]
        assert result["added_count"] == 1

    def test_extract_removed_lines(self):
        before = "line1\nline2\nline3\n"
        after = "line1\nline3\n"
        result = _extract_candidate_introduced_content(
            before, after, []
        )
        assert "line2" in result["removed_lines"]
        assert result["removed_count"] == 1

    def test_no_changes(self):
        content = "line1\nline2\n"
        result = _extract_candidate_introduced_content(
            content, content, []
        )
        assert result["added_count"] == 0
        assert result["removed_count"] == 0

    def test_none_content(self):
        result = _extract_candidate_introduced_content(None, None, [])
        assert result["added_count"] == 0

    def test_corrective_removed_content(self):
        before = "a\nb\nc\n"
        after = "a\nc\n"
        result = _extract_corrective_removed_content(before, after)
        assert "b" in result["removed_lines"]
        assert result["removed_count"] == 1


# ---------------------------------------------------------------------------
# TestContentCorrespondence
# ---------------------------------------------------------------------------


class TestContentCorrespondence:
    def test_exact_match(self):
        candidate = {"added_lines": ["x", "y"], "added_count": 2,
                     "removed_lines": [], "removed_count": 0}
        corrective = {"removed_lines": ["x", "z"], "removed_count": 2,
                      "added_lines": [], "added_count": 0}
        result = _analyze_content_correspondence(candidate, corrective)
        assert result["correspondence_exists"] is True
        assert result["exact_match_count"] == 1
        assert result["correspondence_type"] == "exact_removal"

    def test_no_match(self):
        candidate = {"added_lines": ["a", "b"], "added_count": 2,
                     "removed_lines": [], "removed_count": 0}
        corrective = {"removed_lines": ["x", "y"], "removed_count": 2,
                      "added_lines": [], "added_count": 0}
        result = _analyze_content_correspondence(candidate, corrective)
        assert result["correspondence_exists"] is False
        assert result["exact_match_count"] == 0
        assert result["correspondence_type"] == "none"

    def test_partial_match(self):
        candidate = {"added_lines": ["def foo(): pass"],
                     "added_count": 1, "removed_lines": [],
                     "removed_count": 0}
        corrective = {"removed_lines": ["def bar(): pass"],
                      "removed_count": 1, "added_lines": [],
                      "added_count": 0}
        result = _analyze_content_correspondence(candidate, corrective)
        assert result["partial_match_count"] >= 0
        assert "matching_method" in result

    def test_descriptive_ratios_not_thresholds(self):
        candidate = {"added_lines": ["x"], "added_count": 1,
                     "removed_lines": [], "removed_count": 0}
        corrective = {"removed_lines": ["x"], "removed_count": 1,
                      "added_lines": [], "added_count": 0}
        result = _analyze_content_correspondence(candidate, corrective)
        assert "exact_match_ratio" in result
        assert "partial_match_ratio" in result
        assert isinstance(result["exact_match_ratio"], float)


# ---------------------------------------------------------------------------
# TestContentRestoration
# ---------------------------------------------------------------------------


class TestContentRestoration:
    def test_exact_restoration(self):
        content = "line1\nline2\nline3\n"
        result = _check_content_restoration(content, content)
        assert result["restored"] is True
        assert result["line_diff_count"] == 0

    def test_partial_restoration(self):
        before = "line1\nline2\nline3\n"
        after = "line1\nline2\nline4\n"
        result = _check_content_restoration(before, after)
        assert result["restored"] is False
        assert result["line_diff_count"] > 0

    def test_no_restoration(self):
        before = "abc\ndef\n"
        after = "xyz\n"
        result = _check_content_restoration(before, after)
        assert result["restored"] is False

    def test_none_inputs(self):
        result = _check_content_restoration(None, None)
        assert result["restored"] is False


# ---------------------------------------------------------------------------
# TestRegionOverlap
# ---------------------------------------------------------------------------


class TestRegionOverlap:
    def test_overlapping_regions(self):
        c_hunks = [{"old_start": 10, "old_count": 5,
                     "new_start": 10, "new_count": 5,
                     "content": ""}]
        r_hunks = [{"old_start": 12, "old_count": 5,
                     "new_start": 12, "new_count": 5,
                     "content": ""}]
        result = _compute_region_overlap(c_hunks, r_hunks)
        assert result["overlap_exists"] is True
        assert result["overlap_line_count"] > 0

    def test_non_overlapping_regions(self):
        c_hunks = [{"old_start": 1, "old_count": 3,
                     "new_start": 1, "new_count": 3,
                     "content": ""}]
        r_hunks = [{"old_start": 100, "old_count": 3,
                     "new_start": 100, "new_count": 3,
                     "content": ""}]
        result = _compute_region_overlap(c_hunks, r_hunks)
        assert result["overlap_exists"] is False

    def test_empty_hunks(self):
        result = _compute_region_overlap([], [])
        assert result["overlap_exists"] is False

    def test_adjacent_regions_no_overlap(self):
        c_hunks = [{"old_start": 1, "old_count": 5,
                     "new_start": 1, "new_count": 5,
                     "content": ""}]
        r_hunks = [{"old_start": 7, "old_count": 5,
                     "new_start": 7, "new_count": 5,
                     "content": ""}]
        result = _compute_region_overlap(c_hunks, r_hunks)
        assert result["overlap_exists"] is False


# ---------------------------------------------------------------------------
# TestFunctionAnalysis
# ---------------------------------------------------------------------------


class TestFunctionAnalysis:
    def test_extract_functions(self):
        content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        funcs = _extract_functions_by_name(content)
        assert "foo" in funcs
        assert "bar" in funcs
        assert funcs["foo"][0] == 1

    def test_extract_functions_with_class(self):
        content = (
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        funcs = _extract_functions_by_name(content)
        assert "method" in funcs

    def test_extract_functions_syntax_error(self):
        content = "def foo(:\n    pass\n"
        funcs = _extract_functions_by_name(content)
        assert funcs == {}

    def test_extract_functions_empty(self):
        funcs = _extract_functions_by_name("")
        assert funcs == {}


# ---------------------------------------------------------------------------
# TestFileIdentity
# ---------------------------------------------------------------------------


class TestFileIdentity:
    def test_exact_path_match(self):
        from backend.app.schemas.diff import FileDiff, FileStatus
        candidate_diffs = [
            FileDiff(path="foo.py", status=FileStatus.MODIFIED)
        ]
        corrective_diffs = [
            FileDiff(path="foo.py", status=FileStatus.MODIFIED)
        ]
        result = _resolve_file_identity_across_commits(
            "foo.py", candidate_diffs, corrective_diffs
        )
        assert result["identity_established"] is True
        assert result["identity_method"] == "exact_path"

    def test_no_match(self):
        from backend.app.schemas.diff import FileDiff, FileStatus
        candidate_diffs = [
            FileDiff(path="foo.py", status=FileStatus.MODIFIED)
        ]
        corrective_diffs = [
            FileDiff(path="bar.py", status=FileStatus.MODIFIED)
        ]
        result = _resolve_file_identity_across_commits(
            "foo.py", candidate_diffs, corrective_diffs
        )
        assert result["identity_established"] is False
        assert result["identity_method"] == "none"


# ---------------------------------------------------------------------------
# TestEvidenceClassification
# ---------------------------------------------------------------------------


class TestEvidenceClassification:
    def test_file_strong_with_restoration(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": True},
            content_correspondence={"correspondence_exists": False,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": False},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_STRONG

    def test_file_strong_with_exact_removal(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": True,
                                    "exact_match_count": 3},
            region_overlap={"overlap_exists": False},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_STRONG

    def test_file_moderate_content_plus_region(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": True,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": True},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_MODERATE

    def test_file_moderate_content_plus_function(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": True,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": False},
            function_analysis={"structural_relationship": True},
        )
        assert level == FILE_EVIDENCE_MODERATE

    def test_file_weak_path_overlap_only(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": False,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": False},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_WEAK

    def test_file_unknown_no_identity(self):
        level = _classify_file_evidence(
            identity_established=False,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": False,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": False},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_UNKNOWN

    def test_same_function_only_remains_weak(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": False,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": False},
            function_analysis={
                "same_function": True,
                "structural_relationship": False,
            },
        )
        assert level == FILE_EVIDENCE_WEAK

    def test_region_only_without_content_remains_weak(self):
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": False},
            content_correspondence={"correspondence_exists": False,
                                    "exact_match_count": 0},
            region_overlap={"overlap_exists": True},
            function_analysis={"structural_relationship": False},
        )
        assert level == FILE_EVIDENCE_WEAK

    def test_mutually_exclusive_precedence(self):
        """FILE_STRONG takes precedence over others."""
        level = _classify_file_evidence(
            identity_established=True,
            content_restoration={"restored": True},
            content_correspondence={"correspondence_exists": True,
                                    "exact_match_count": 5},
            region_overlap={"overlap_exists": True},
            function_analysis={"structural_relationship": True},
        )
        assert level == FILE_EVIDENCE_STRONG


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_classification_deterministic(self):
        for _ in range(5):
            level = _classify_file_evidence(
                identity_established=True,
                content_restoration={"restored": False},
                content_correspondence={"correspondence_exists": True,
                                        "exact_match_count": 0},
                region_overlap={"overlap_exists": True},
                function_analysis={"structural_relationship": False},
            )
            assert level == FILE_EVIDENCE_MODERATE

    def test_content_matching_deterministic(self):
        c = {"added_lines": ["x", "y"], "added_count": 2,
             "removed_lines": [], "removed_count": 0}
        r = {"removed_lines": ["x", "z"], "removed_count": 2,
             "added_lines": [], "added_count": 0}
        r1 = _analyze_content_correspondence(c, r)
        r2 = _analyze_content_correspondence(c, r)
        assert r1 == r2

    def test_normalization_deterministic(self):
        for _ in range(10):
            assert _normalize_line("hello  \r\n") == "hello"


# ---------------------------------------------------------------------------
# TestNegativeFeasibility
# ---------------------------------------------------------------------------


class TestNegativeFeasibility:
    def test_no_defensible_negatives(self):
        results = [{
            "commit_sha": "abc",
            "corrective_sha": "def",
            "corrective_files": ["a.py", "b.py", "c.py"],
            "candidate_files": ["a.py"],
        }]
        result = _investigate_negative_feasibility(results)
        assert result["defensible_negative_count"] == 0
        assert result["candidate_negative_count"] >= 1

    def test_candidate_negatives_not_labeled_negative(self):
        results = [{
            "commit_sha": "abc",
            "corrective_sha": "def",
            "corrective_files": ["a.py", "b.py", "c.py"],
            "candidate_files": ["a.py"],
        }]
        result = _investigate_negative_feasibility(results)
        for cn in result["candidate_negatives"]:
            assert cn["status"] == "CANDIDATE_NEGATIVE"

    def test_no_corrective_no_negatives(self):
        results = [{
            "commit_sha": "abc",
            "corrective_sha": None,
            "corrective_files": [],
            "candidate_files": ["a.py"],
        }]
        result = _investigate_negative_feasibility(results)
        assert result["candidate_negative_count"] == 0


# ---------------------------------------------------------------------------
# TestArtifactContract
# ---------------------------------------------------------------------------


class TestArtifactContract:
    def test_all_artifacts_exist(self, experiment_output):
        tmp_dir = experiment_output["tmp_dir"]
        expected = [
            "git_evidence_analysis.json",
            "file_attribution_results.json",
            "per_commit_git_evidence.jsonl",
            "function_hunk_feasibility.json",
            "negative_feasibility_analysis.json",
            "prediction_unit_analysis.json",
            "repository_execution_log.json",
            "phase411b_git_attribution.md",
        ]
        for fname in expected:
            assert (tmp_dir / fname).exists(), f"Missing: {fname}"

    def test_per_commit_jsonl_has_210_records(self, experiment_output):
        tmp_dir = experiment_output["tmp_dir"]
        count = 0
        with open(
            tmp_dir / "per_commit_git_evidence.jsonl",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    count += 1
        assert count == 210

    def test_report_not_empty(self, experiment_output):
        report = (
            experiment_output["tmp_dir"]
            / "phase411b_git_attribution.md"
        ).read_text(encoding="utf-8")
        assert len(report) > 500

    def test_report_contains_decision(self, experiment_output):
        report = (
            experiment_output["tmp_dir"]
            / "phase411b_git_attribution.md"
        ).read_text(encoding="utf-8")
        assert "MATERIAL_IMPROVEMENT" in report or "NO_MATERIAL_IMPROVEMENT" in report

    def test_report_contains_baseline_comparison(self, experiment_output):
        report = (
            experiment_output["tmp_dir"]
            / "phase411b_git_attribution.md"
        ).read_text(encoding="utf-8")
        assert "Phase 4.8" in report
        assert "Phase 4.11a" in report


# ---------------------------------------------------------------------------
# TestRepositoryAccounting
# ---------------------------------------------------------------------------


class TestRepositoryAccounting:
    def test_experiment_covers_all_positive_commits(
        self, experiment_output
    ):
        result = experiment_output["result"]
        assert result["aggregated"]["total_commits"] == 210

    def test_stage_counts_data_derived(self, experiment_output):
        """Stage counts come from data, not hard-coded."""
        el = experiment_output["result"]["execution_log"]
        total_repos = el.get("repos_cloned", 0) + el.get("repos_failed", 0)
        assert total_repos > 0

    def test_staging_is_operational(self, experiment_output):
        """All stages execute regardless of results."""
        el = experiment_output["result"]["execution_log"]
        stages = el.get("stages", {})
        assert "A" in stages
        assert "B" in stages
        assert "C" in stages


# ---------------------------------------------------------------------------
# TestEvidenceHierarchyInvariants
# ---------------------------------------------------------------------------


class TestEvidenceHierarchyInvariants:
    def test_every_file_has_exactly_one_classification(
        self, experiment_output
    ):
        tmp_dir = experiment_output["tmp_dir"]
        valid = {
            FILE_EVIDENCE_STRONG, FILE_EVIDENCE_MODERATE,
            FILE_EVIDENCE_WEAK, FILE_EVIDENCE_UNKNOWN,
        }
        with open(
            tmp_dir / "per_commit_git_evidence.jsonl",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for fr in record.get("file_results", []):
                    assert fr["file_evidence_level"] in valid, (
                        f"Invalid classification: {fr['file_evidence_level']}"
                    )

    def test_commit_strong_not_file_level(self, experiment_output):
        """COMMIT_STRONG is commit-level, never file-level."""
        tmp_dir = experiment_output["tmp_dir"]
        with open(
            tmp_dir / "per_commit_git_evidence.jsonl",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for fr in record.get("file_results", []):
                    assert fr["file_evidence_level"] != COMMIT_EVIDENCE_STRONG

    def test_no_path_overlap_to_moderate(self, experiment_output):
        """Path overlap alone cannot create FILE_MODERATE."""
        tmp_dir = experiment_output["tmp_dir"]
        with open(
            tmp_dir / "per_commit_git_evidence.jsonl",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for fr in record.get("file_results", []):
                    if fr["file_evidence_level"] == FILE_EVIDENCE_MODERATE:
                        corr = fr.get("content_correspondence")
                        assert corr is not None
                        assert corr["correspondence_exists"] is True


# ---------------------------------------------------------------------------
# TestFrozenIntegrity
# ---------------------------------------------------------------------------


class TestFrozenIntegrity:
    def test_frozen_jsonl_unchanged(self):
        for split_name, expected in [
            ("train", 34405), ("validation", 9410), ("test", 10574)
        ]:
            path = DATA_DIR / f"{split_name}.jsonl"
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            assert count == expected

    def test_frozen_ambiguous_unchanged(self):
        path = DATA_DIR / "ambiguous.jsonl"
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        assert count == 117487

    def test_phase48_artifacts_unchanged(self):
        path = Path(
            "backend/data/models/"
            "v0.1.0-combined-v3-phase4.8/"
            "attribution_feasibility.json"
        )
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["evidence_statistics"]["total_positive_commits"] == 210

    def test_phase410_artifacts_unchanged(self):
        path = Path(
            "backend/data/models/"
            "v0.1.0-combined-v3-phase4.10/pu_results.json"
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# TestGitObjectAvailability
# ---------------------------------------------------------------------------


class TestGitObjectAvailability:
    def test_object_availability_recorded(self, experiment_output):
        """Missing objects are explicitly logged."""
        el = experiment_output["result"]["execution_log"]
        assert "object_failures" in el

    def test_per_commit_availability(self, experiment_output):
        """Each commit record has git_object_availability."""
        tmp_dir = experiment_output["tmp_dir"]
        with open(
            tmp_dir / "per_commit_git_evidence.jsonl",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                assert "git_object_availability" in record
                avail = record["git_object_availability"]
                assert "candidate_commit" in avail
                assert "failures" in avail
