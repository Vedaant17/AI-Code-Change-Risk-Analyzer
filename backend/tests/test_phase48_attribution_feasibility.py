"""Tests for Phase 4.8: File-Level Attribution Feasibility Study.

All tests use deterministic mock data that exercises the evidence
classification logic without relying on frozen dataset files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.ml.phase48_attribution_feasibility import (
    _detect_path_transform_candidates,
    _extract_fix_sha_prefix,
    _resolve_full_sha,
    classify_commit_evidence,
    compute_evidence_statistics,
    compute_repository_analysis,
    compute_split_analysis,
    derive_recommendation,
    investigate_negative_labels,
    run_feasibility_study,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_row(
    repo_name: str,
    commit_sha: str,
    file_path: str,
    defect_label: int,
    label_source: str = "explicit_sha_reference",
    label_evidence: str = "",
    commit_message: str = "",
    split: str = "train",
) -> dict[str, Any]:
    """Create a minimal JSONL row dict."""
    return {
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "file_path": file_path,
        "defect_label": defect_label,
        "label_source": label_source,
        "label_evidence": label_evidence,
        "commit_message": commit_message,
        "split": split,
        "files_changed": 3,
        "insertions": 10,
        "deletions": 5,
        "net_changes": 5,
        "semantic_entropy": 0.5,
        "changed_test_files": 0,
        "test_ratio": 0.0,
        "sklearn_class_weight": 1.0,
        "max_file_complexity": 15.0,
        "mean_file_complexity": 10.0,
        "total_hunks": 3,
        "max_hunk_size": 8,
        "mean_hunk_size": 5.0,
        "large_hunk_ratio": 0.0,
        "std_hunk_size": 2.0,
        "max_difficulty": 5.0,
        "mean_difficulty": 4.0,
        "semantic_signal_count": 1,
        "semantic_signal_ratio": 0.33,
        "unique_semantic_signals": 1,
        "has_high_impact_bug_fix": 1,
        "has_revert": 0,
        "has_line_overlap": 0,
        "line_overlap_ratio": 0.0,
        "max_overlap_ratio": 0.0,
        "corrective_files_count": 0,
        "corrective_hunks_count": 0,
        "overlapping_hunks_count": 0,
        "referenced_file_path": "",
        "has_repair_kws": 1,
        "repair_kw_ratio": 0.2,
        "has_code_review": 0,
        "has_architecture_review": 0,
        "has_security_review": 0,
        "review_ratio": 0.0,
        "doc_change_ratio": 0.0,
        "test_in_change_ratio": 0.0,
        "meta_has_core_module_changes": 0,
        "meta_has_critical_path_changes": 0,
        "meta_has_release_blocker_changes": 0,
        "meta_multi_component": 0,
        "meta_change_scope": 0,
        "meta_architecture_footprint": 0,
        "meta_priority_escalation": 0,
        "meta_recent_commits_in_file": 0,
        "meta_repeated_reverts_in_component": 0,
        "meta_churn_volatility": 0,
        "pr_number": None,
        "bug_fix_commits": [],
        "bug_fix_count": 0,
        "reverted_by_commits": [],
        "revert_count": 0,
        "revert_sources": [],
        "bug_fix_sources": [],
        "additional_repair_keywords": [],
        "context_files": [],
        "context_files_by_label": {},
    }


def _mock_fix_sha() -> str:
    """Return a deterministic 12-char fix SHA prefix."""
    return "abcdef123456"


def _mock_label_evidence(fix_prefix: str) -> str:
    """Return label_evidence containing a bug-fix SHA reference."""
    return f"bug-fix {fix_prefix}"


# ---------------------------------------------------------------------------
# Unit tests: SHA extraction
# ---------------------------------------------------------------------------

class TestExtractFixShaPrefix:
    """Tests for _extract_fix_sha_prefix."""

    def test_bug_fix_sha_found(self) -> None:
        evidence = "bug-fix abcdef123456"
        result = _extract_fix_sha_prefix(evidence)
        assert result == "abcdef123456"

    def test_revert_sha_found(self) -> None:
        evidence = "reverted by abcdef123456"
        result = _extract_fix_sha_prefix(evidence)
        assert result == "abcdef123456"

    def test_no_sha_found(self) -> None:
        evidence = "no sha here"
        result = _extract_fix_sha_prefix(evidence)
        assert result is None

    def test_empty_string(self) -> None:
        assert _extract_fix_sha_prefix("") is None

    def test_partial_sha(self) -> None:
        evidence = "bug-fix abcdef12"
        result = _extract_fix_sha_prefix(evidence)
        assert result == "abcdef12"

    def test_full_40_char_sha(self) -> None:
        evidence = "bug-fix abcdef1234567890abcdef1234567890abcdef"
        result = _extract_fix_sha_prefix(evidence)
        assert result == "abcdef1234567890abcdef1234567890abcdef"


# ---------------------------------------------------------------------------
# Unit tests: SHA resolution
# ---------------------------------------------------------------------------

class TestResolveFullSha:
    """Tests for _resolve_full_sha."""

    def test_exact_match(self) -> None:
        full = "abcdef1234567890"
        lookup = {"abcdef1234567890": []}
        result = _resolve_full_sha("abcdef123456", lookup)
        assert result == full

    def test_no_match(self) -> None:
        lookup = {"1234567890abcdef": []}
        result = _resolve_full_sha("abcdef123456", lookup)
        assert result is None

    def test_ambiguous_match(self) -> None:
        lookup = {
            "abcdef1234560000": [],
            "abcdef1234560001": [],
        }
        result = _resolve_full_sha("abcdef123456", lookup)
        assert result is None

    def test_empty_lookup(self) -> None:
        result = _resolve_full_sha("abcdef123456", {})
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: Path transform detection
# ---------------------------------------------------------------------------

class TestDetectPathTransformCandidates:
    """Tests for _detect_path_transform_candidates."""

    def test_same_paths_no_transform(self) -> None:
        result = _detect_path_transform_candidates(
            ["src/foo.py"], ["src/foo.py"]
        )
        assert result == []

    def test_different_paths_same_basename(self) -> None:
        result = _detect_path_transform_candidates(
            ["src/foo.py"], ["lib/foo.py"]
        )
        assert len(result) == 1
        assert result[0]["basename"] == "foo.py"

    def test_completely_different(self) -> None:
        result = _detect_path_transform_candidates(
            ["src/foo.py"], ["src/bar.py"]
        )
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests: Evidence classification
# ---------------------------------------------------------------------------

class TestClassifyCommitEvidence:
    """Tests for classify_commit_evidence."""

    def test_commit_only_full_overlap(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [_make_row("repo", fix_sha, "src/foo.py", 0)],
        }
        result = classify_commit_evidence(
            [candidate], sha_lookup
        )
        assert result["evidence_category"] == "COMMIT_ONLY"
        assert result["path_overlap_files"] == ["src/foo.py"]
        assert result["unknown_files"] == []
        assert result["attributed_files"] == []

    def test_commit_only_zero_overlap(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [_make_row("repo", fix_sha, "src/bar.py", 0)],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["evidence_category"] == "COMMIT_ONLY"
        assert result["path_overlap_files"] == []
        assert result["unknown_files"] == ["src/foo.py"]

    def test_file_partial(self) -> None:
        rows = [
            _make_row("repo", "aaa" + "0" * 37, "src/foo.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_label_evidence(_mock_fix_sha())),
            _make_row("repo", "aaa" + "0" * 37, "src/bar.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_label_evidence(_mock_fix_sha())),
        ]
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: rows,
            fix_sha: [_make_row("repo", fix_sha, "src/foo.py", 0)],
        }
        result = classify_commit_evidence(rows, sha_lookup)
        assert result["evidence_category"] == "FILE_PARTIAL"
        assert result["path_associated_files"] == ["src/foo.py"]
        assert result["unknown_files"] == ["src/bar.py"]

    def test_no_fix_sha_found(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence="no sha",
        )
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["evidence_category"] == "NO_ATTRIBUTION"

    def test_unresolvable_fix_sha(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence("zzzzzzzzzzzz"),
        )
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["evidence_category"] == "NO_ATTRIBUTION"

    def test_fix_commit_empty_rows(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            "abcdef1234567890": [],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["evidence_category"] == "NO_ATTRIBUTION"

    def test_no_path_transform(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [_make_row("repo", fix_sha, "src/foo.py", 0)],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["path_transform_candidates"] == []

    def test_path_transform_detected(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [_make_row("repo", fix_sha, "lib/foo.py", 0)],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["evidence_category"] == "COMMIT_ONLY"
        assert len(result["path_transform_candidates"]) == 1

    def test_corrective_files_count(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [
                _make_row("repo", fix_sha, "src/foo.py", 0),
                _make_row("repo", fix_sha, "src/bar.py", 0),
            ],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["corrective_file_count"] == 2

    def test_stores_candidate_metadata(self) -> None:
        candidate = _make_row(
            "repo", "aaa" + "0" * 37,
            "src/foo.py", 1,
            label_source="explicit_sha_reference",
            label_evidence=_mock_label_evidence(_mock_fix_sha()),
            split="test",
        )
        fix_sha = "abcdef1234567890"
        sha_lookup = {
            "aaa" + "0" * 37: [candidate],
            fix_sha: [_make_row("repo", fix_sha, "src/foo.py", 0)],
        }
        result = classify_commit_evidence([candidate], sha_lookup)
        assert result["repo_name"] == "repo"
        assert result["commit_sha"] == "aaa" + "0" * 37
        assert result["split"] == "test"


# ---------------------------------------------------------------------------
# Unit tests: Statistics
# ---------------------------------------------------------------------------

class TestComputeEvidenceStatistics:
    """Tests for compute_evidence_statistics."""

    def test_empty_results(self) -> None:
        result = compute_evidence_statistics([])
        assert result["total_positive_commits"] == 0
        assert result["by_category"] == {}

    def test_all_file_partial(self) -> None:
        results = [
            {
                "repo_name": "r", "commit_sha": "a" + "0" * 39,
                "evidence_category": "FILE_PARTIAL",
                "label_source": "explicit_sha_reference",
                "candidate_file_count": 3,
                "path_overlap_files": [],
                "path_associated_files": ["a.py"],
                "unknown_files": ["b.py", "c.py"],
                "corrective_files": ["a.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = compute_evidence_statistics(results)
        assert result["total_positive_commits"] == 1
        assert result["by_category"]["FILE_PARTIAL"] == 1
        assert result["file_partial_details"]["commits"] == 1
        assert result["file_partial_details"]["positive_file_rows"] == 1
        assert result["file_partial_details"]["unknown_file_rows"] == 2

    def test_all_commit_only(self) -> None:
        results = [
            {
                "repo_name": "r", "commit_sha": "a" + "0" * 39,
                "evidence_category": "COMMIT_ONLY",
                "label_source": "explicit_sha_reference",
                "candidate_file_count": 2,
                "path_overlap_files": ["a.py"],
                "path_associated_files": [],
                "unknown_files": ["b.py"],
                "corrective_files": ["a.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = compute_evidence_statistics(results)
        assert result["commit_only_details"]["commits"] == 1
        assert result["commit_only_details"]["full_overlap_commits"] == 0
        assert result["commit_only_details"]["zero_overlap_commits"] == 0

    def test_zero_overlap(self) -> None:
        results = [
            {
                "repo_name": "r", "commit_sha": "a" + "0" * 39,
                "evidence_category": "COMMIT_ONLY",
                "label_source": "explicit_sha_reference",
                "candidate_file_count": 2,
                "path_overlap_files": [],
                "path_associated_files": [],
                "unknown_files": ["a.py", "b.py"],
                "corrective_files": [],
                "corrective_file_count": 0,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = compute_evidence_statistics(results)
        assert result["commit_only_details"]["zero_overlap_commits"] == 1


# ---------------------------------------------------------------------------
# Unit tests: Split analysis
# ---------------------------------------------------------------------------

class TestComputeSplitAnalysis:
    """Tests for compute_split_analysis."""

    def test_all_three_splits(self) -> None:
        results = [
            {
                "repo_name": "r", "commit_sha": "a" + "0" * 39,
                "evidence_category": "FILE_PARTIAL",
                "label_source": "explicit_sha_reference",
                "split": "train",
                "candidate_file_count": 2,
                "path_overlap_files": [],
                "path_associated_files": ["a.py"],
                "unknown_files": ["b.py"],
                "corrective_files": ["a.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
            {
                "repo_name": "r", "commit_sha": "b" + "0" * 39,
                "evidence_category": "COMMIT_ONLY",
                "label_source": "revert",
                "split": "validation",
                "candidate_file_count": 1,
                "path_overlap_files": [],
                "path_associated_files": [],
                "unknown_files": ["a.py"],
                "corrective_files": ["other.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = compute_split_analysis(results)
        assert "train" in result
        assert "validation" in result
        assert "test" in result
        assert result["train"]["total_positive_commits"] == 1
        assert result["validation"]["total_positive_commits"] == 1
        assert result["test"]["total_positive_commits"] == 0


# ---------------------------------------------------------------------------
# Unit tests: Repository analysis
# ---------------------------------------------------------------------------

class TestComputeRepositoryAnalysis:
    """Tests for compute_repository_analysis."""

    def test_multiple_repos(self) -> None:
        results = [
            {
                "repo_name": "repo_a", "commit_sha": "a" + "0" * 39,
                "evidence_category": "FILE_PARTIAL",
                "label_source": "explicit_sha_reference",
                "candidate_file_count": 2,
                "path_overlap_files": [],
                "path_associated_files": ["a.py"],
                "unknown_files": ["b.py"],
                "corrective_files": ["a.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
            {
                "repo_name": "repo_b", "commit_sha": "c" + "0" * 39,
                "evidence_category": "COMMIT_ONLY",
                "label_source": "revert",
                "candidate_file_count": 1,
                "path_overlap_files": [],
                "path_associated_files": [],
                "unknown_files": ["c.py"],
                "corrective_files": ["d.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = compute_repository_analysis(results)
        repos = result["repos"]
        assert len(repos) == 2
        repo_names = {r["repo_name"] for r in repos}
        assert repo_names == {"repo_a", "repo_b"}


# ---------------------------------------------------------------------------
# Unit tests: Negative label investigation
# ---------------------------------------------------------------------------

class TestInvestigateNegativeLabels:
    """Tests for investigate_negative_labels."""

    def test_defensible_negatives_always_zero(self) -> None:
        results = [
            {
                "repo_name": "r", "commit_sha": "a" + "0" * 39,
                "evidence_category": "FILE_PARTIAL",
                "label_source": "explicit_sha_reference",
                "candidate_file_count": 2,
                "path_overlap_files": [],
                "path_associated_files": ["a.py"],
                "unknown_files": ["b.py"],
                "corrective_files": ["a.py"],
                "corrective_file_count": 1,
                "corrective_sha": None,
                "corrective_sha_prefix": "abcdef123456",
                "path_transform_candidates": [],
            },
        ]
        result = investigate_negative_labels(results)
        assert result["defensible_negative_files"] == 0
        assert result["conclusion"] != ""


# ---------------------------------------------------------------------------
# Unit tests: Recommendation
# ---------------------------------------------------------------------------

class TestDeriveRecommendation:
    """Tests for derive_recommendation."""

    def test_no_file_partial(self) -> None:
        evidence_stats = {
            "total_positive_commits": 100,
            "by_category": {"COMMIT_ONLY": 80, "NO_ATTRIBUTION": 20},
            "file_partial_details": {
                "commits": 0,
                "positive_file_rows": 0,
                "unknown_file_rows": 0,
            },
            "file_yield": {
                "total_candidate_files": 100,
                "total_unknown_files": 100,
            },
        }
        neg_inv = {"defensible_negative_files": 0}
        rec = derive_recommendation(evidence_stats, neg_inv)
        assert rec["recommendation"] == "FILE_LEVEL_ATTRIBUTION_NOT_FEASIBLE"

    def test_high_unknown_rate(self) -> None:
        evidence_stats = {
            "total_positive_commits": 100,
            "by_category": {"FILE_PARTIAL": 20, "COMMIT_ONLY": 80},
            "file_partial_details": {
                "commits": 20,
                "positive_file_rows": 10,
                "unknown_file_rows": 90,
            },
            "file_yield": {
                "total_candidate_files": 100,
                "total_unknown_files": 85,
            },
        }
        neg_inv = {"defensible_negative_files": 0}
        rec = derive_recommendation(evidence_stats, neg_inv)
        assert rec["recommendation"] == "PATH_LEVEL_EVIDENCE_ONLY"

    def test_no_defensible_negatives(self) -> None:
        evidence_stats = {
            "total_positive_commits": 100,
            "by_category": {"FILE_PARTIAL": 30, "COMMIT_ONLY": 70},
            "file_partial_details": {
                "commits": 30,
                "positive_file_rows": 30,
                "unknown_file_rows": 30,
            },
            "file_yield": {
                "total_candidate_files": 100,
                "total_unknown_files": 50,
            },
        }
        neg_inv = {"defensible_negative_files": 0}
        rec = derive_recommendation(evidence_stats, neg_inv)
        assert rec["recommendation"] == "PATH_LEVEL_EVIDENCE_ONLY"


# ---------------------------------------------------------------------------
# Integration test: Full run with temp directory
# ---------------------------------------------------------------------------

class TestRunFeasibilityStudy:
    """Integration test with a synthetic dataset."""

    def test_full_run_produces_artifacts(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "datasets"
        output_dir = tmp_path / "artifacts"

        split_dir = data_dir
        split_dir.mkdir(parents=True)

        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("repo", "aaa" + "0" * 37, "src/foo.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_label_evidence(_mock_fix_sha()),
                       split="train"),
            _make_row("repo", "aaa" + "0" * 37, "src/bar.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_label_evidence(_mock_fix_sha()),
                       split="train"),
            _make_row("repo", fix_sha, "src/foo.py", 0, split="train"),
            _make_row("repo", fix_sha, "src/other.py", 0, split="train"),
            _make_row("repo", "bbb" + "0" * 37, "lib/a.py", 1,
                       label_source="revert",
                       label_evidence=f"reverted by {_mock_fix_sha()}",
                       split="validation"),
            _make_row("repo", fix_sha, "lib/a.py", 0, split="validation"),
            _make_row("repo", "ccc" + "0" * 37, "test/t.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence="no sha",
                       split="test"),
        ]

        for split_name in ("train", "validation", "test"):
            split_rows = [r for r in rows if r["split"] == split_name]
            with open(data_dir / f"{split_name}.jsonl", "w") as f:
                for row in split_rows:
                    f.write(json.dumps(row) + "\n")

        result = run_feasibility_study(data_dir, output_dir)

        assert output_dir.exists()
        assert (output_dir / "attribution_feasibility.json").exists()
        assert (output_dir / "file_attribution_cases.jsonl").exists()
        assert (output_dir / "repository_analysis.json").exists()
        assert (output_dir / "split_analysis.json").exists()
        assert (output_dir / "phase48_attribution_feasibility.md").exists()

        assert result["evidence_stats"]["total_positive_commits"] == 3

        cats = result["evidence_stats"]["by_category"]
        total_categorized = (
            cats.get("FILE_PARTIAL", 0)
            + cats.get("COMMIT_ONLY", 0)
            + cats.get("NO_ATTRIBUTION", 0)
        )
        assert total_categorized == 3

        with open(output_dir / "file_attribution_cases.jsonl") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        assert len(cases) == 3

        for case in cases:
            if case["evidence_category"] == "COMMIT_ONLY":
                assert case.get("attributed_files") == []

    def test_empty_data_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "empty_datasets"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        result = run_feasibility_study(data_dir, output_dir)
        assert result["evidence_stats"]["total_positive_commits"] == 0
        assert (output_dir / "phase48_attribution_feasibility.md").exists()


# ---------------------------------------------------------------------------
# Frozen data integrity test
# ---------------------------------------------------------------------------

class TestFrozenDataIntegrity:
    """Verify that frozen dataset files are not modified."""

    def test_jsonl_files_intact(self) -> None:
        frozen_dir = Path("backend/data/datasets/combined-v3")
        if not frozen_dir.exists():
            pytest.skip("Frozen dataset not available")

        metadata_path = frozen_dir / "metadata.json"
        if not metadata_path.exists():
            pytest.skip("metadata.json not found")

        with open(metadata_path) as f:
            _metadata = json.load(f)

        expected_row_counts = {
            "train": 34405,
            "validation": 9410,
            "test": 10574,
        }

        for split_name, expected in expected_row_counts.items():
            jsonl_path = frozen_dir / f"{split_name}.jsonl"
            if not jsonl_path.exists():
                continue
            with open(jsonl_path, encoding="utf-8") as f:
                actual = sum(1 for line in f if line.strip())
            assert actual == expected, (
                f"{split_name}.jsonl has {actual} rows, expected {expected}"
            )
