"""Tests for Phase 4.7 Label & Attribution Distribution Audit.

Verifies:
- JSONL loading
- Exact label counts
- All-positive detection
- Mixed-commit detection
- Positive-file ratio
- Attribution-source aggregation
- Repository aggregation
- Commit-size buckets
- Deterministic output
- Source non-mutation
- Synthetic mixed commit
- Synthetic all-positive commit
- Zero-positive commit
- Phase 4.6 manifest consumption
- Valid label-source values
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.ml.phase47_label_audit import (
    VALID_LABEL_SOURCES,
    analyze_attribution_sources,
    analyze_by_repo,
    analyze_commit_size,
    analyze_positive_commit_composition,
    classify_commit,
    run_audit,
)

V3_DATA_DIR = Path("backend/data/datasets/combined-v3")


class TestJsonlLoading:
    """Tests for JSONL loading."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_load_all_rows_returns_rows(self):
        """Loading returns non-empty list."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        assert len(rows) > 0

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_load_all_rows_expected_count(self):
        """Loading returns expected row count: train + val + test examples."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        metadata_path = V3_DATA_DIR / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        expected = (
            meta["train_examples"]
            + meta["validation_examples"]
            + meta["test_examples"]
        )
        assert len(rows) == expected

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_load_all_rows_has_required_fields(self):
        """Each row has required fields for audit."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        required = {"repo_name", "commit_sha", "defect_label", "split",
                    "label_source"}
        for row in rows[:100]:
            assert required.issubset(row.keys())


class TestExactLabelCounts:
    """Tests for exact label counts."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_positive_count_matches_metadata(self):
        """Positive row count matches metadata."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        pos = sum(1 for r in rows if r["defect_label"] == 1)
        metadata_path = V3_DATA_DIR / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert pos == meta["positive_examples"]

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_negative_count_matches_metadata(self):
        """Negative row count matches metadata."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        neg = sum(1 for r in rows if r["defect_label"] == 0)
        metadata_path = V3_DATA_DIR / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert neg == meta["negative_examples"]


class TestAllPositiveDetection:
    """Tests for all-positive commit detection."""

    def test_synthetic_all_positive_commit(self):
        """A commit where every file is positive is classified as all_positive."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
        ]
        info = classify_commit(rows)
        assert info["classification"] == "all_positive"
        assert info["positive_files"] == 3
        assert info["negative_files"] == 0
        assert info["positive_ratio"] == 1.0


class TestMixedCommitDetection:
    """Tests for mixed commit detection."""

    def test_synthetic_mixed_commit(self):
        """A commit with both positive and negative files is mixed."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
        ]
        info = classify_commit(rows)
        assert info["classification"] == "mixed"
        assert info["positive_files"] == 1
        assert info["negative_files"] == 2
        assert abs(info["positive_ratio"] - 1 / 3) < 1e-10


class TestZeroPositiveCommit:
    """Tests for zero-positive commit detection."""

    def test_synthetic_all_negative_commit(self):
        """A commit with no positive files."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
        ]
        info = classify_commit(rows)
        assert info["classification"] == "all_negative"
        assert info["positive_files"] == 0
        assert info["positive_ratio"] == 0.0


class TestPositiveFileRatio:
    """Tests for positive-file ratio."""

    def test_ratio_calculation(self):
        """Positive ratio is correctly calculated."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
        ]
        info = classify_commit(rows)
        assert abs(info["positive_ratio"] - 2 / 3) < 1e-10


class TestAttributionSourceAggregation:
    """Tests for attribution-source aggregation."""

    def test_sources_collected(self):
        """Attribution sources are correctly aggregated."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
            {"repo_name": "r", "commit_sha": "c2", "defect_label": 1,
             "split": "train", "label_source": "revert"},
        ]
        info_c1 = classify_commit([rows[0], rows[1]])
        info_c2 = classify_commit([rows[2]])
        assert "explicit_sha_reference" in info_c1["attribution_sources"]
        assert "revert" in info_c2["attribution_sources"]


class TestRepositoryAggregation:
    """Tests for repository aggregation."""

    def test_per_repo_counts(self):
        """Per-repo positive commit counts are correct."""
        rows = [
            {"repo_name": "r1", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r1", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
            {"repo_name": "r1", "commit_sha": "c2", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r1", "commit_sha": "c2", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r2", "commit_sha": "c3", "defect_label": 0,
             "split": "train", "label_source": "none"},
            {"repo_name": "r2", "commit_sha": "c4", "defect_label": 1,
             "split": "train", "label_source": "revert"},
        ]
        result = analyze_by_repo(rows)
        repos = {r["repo_name"]: r for r in result["repos"]}
        # r1: c1 is mixed (1 pos, 1 neg), c2 is all-positive -> 2 pos commits, 1 mixed
        assert repos["r1"]["positive_commits"] == 2
        assert repos["r1"]["mixed_commits"] == 1
        assert repos["r1"]["all_positive_commits"] == 1
        # r2: c3 all-negative, c4 all-positive -> 1 pos commit, 0 mixed
        assert repos["r2"]["positive_commits"] == 1
        assert repos["r2"]["mixed_commits"] == 0


class TestCommitSizeBuckets:
    """Tests for commit-size buckets."""

    def test_bucket_assignment(self):
        """Commits are assigned to correct size buckets."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c2", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c2", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c2", "defect_label": 0,
             "split": "train", "label_source": "none"},
        ]
        result = analyze_commit_size(rows)
        # c1: 1 file -> bucket "1"
        # c2: 3 files -> bucket "2-3"
        assert result["buckets"]["1"]["count"] == 1
        assert result["buckets"]["2-3"]["count"] == 1


class TestDeterministicOutput:
    """Tests for deterministic output."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_same_results_twice(self):
        """Running audit twice produces same results."""
        result1 = run_audit(V3_DATA_DIR)
        result2 = run_audit(V3_DATA_DIR)

        assert result1["composition"]["overall"]["total_commits"] == \
            result2["composition"]["overall"]["total_commits"]
        assert result1["composition"]["overall"]["positive_commits"] == \
            result2["composition"]["overall"]["positive_commits"]
        assert result1["composition"]["overall"]["mixed_commits"] == \
            result2["composition"]["overall"]["mixed_commits"]


class TestSourceNonMutation:
    """Tests for source non-mutation."""

    def test_read_only_operations(self):
        """Audit does not modify any input data."""
        rows = [
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 1,
             "split": "train", "label_source": "explicit_sha_reference"},
            {"repo_name": "r", "commit_sha": "c1", "defect_label": 0,
             "split": "train", "label_source": "none"},
        ]
        original = json.dumps(rows, sort_keys=True)
        analyze_positive_commit_composition(rows)
        analyze_attribution_sources(rows)
        analyze_by_repo(rows)
        analyze_commit_size(rows)
        assert json.dumps(rows, sort_keys=True) == original


class TestValidLabelSources:
    """Tests for valid label-source values."""

    def test_all_sources_valid(self):
        """All label_source values in dataset are valid."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        if not V3_DATA_DIR.exists():
            pytest.skip("Dataset not available")
        rows = _read_all_rows(V3_DATA_DIR)
        sources = {r["label_source"] for r in rows}
        assert sources.issubset(VALID_LABEL_SOURCES)


class TestMixedCommitExists:
    """Tests for mixed commit existence."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_mixed_commit_count(self):
        """Report the number of mixed commits (may be zero)."""
        from backend.app.ml.phase47_label_audit import _read_all_rows

        rows = _read_all_rows(V3_DATA_DIR)
        composition = analyze_positive_commit_composition(rows)
        # We expect 0 mixed commits based on preliminary analysis
        # This test documents the finding
        mixed = composition["overall"]["mixed_commits"]
        all_pos = composition["overall"]["all_positive_commits"]
        pos = composition["overall"]["positive_commits"]
        assert mixed + all_pos == pos


class TestPhase46ManifestConsumption:
    """Tests for Phase 4.6 manifest consumption."""

    def test_repo_manifest_exists(self):
        """Phase 4.6 REPO_MANIFEST is accessible."""
        from backend.app.ml.repo_split import REPO_MANIFEST

        assert len(REPO_MANIFEST) == 50

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_all_repos_in_manifest(self):
        """All repos in dataset are in REPO_MANIFEST."""
        from backend.app.ml.phase47_label_audit import _read_all_rows
        from backend.app.ml.repo_split import REPO_MANIFEST

        rows = _read_all_rows(V3_DATA_DIR)
        repos = {r["repo_name"] for r in rows}
        assert repos.issubset(set(REPO_MANIFEST.keys()))
