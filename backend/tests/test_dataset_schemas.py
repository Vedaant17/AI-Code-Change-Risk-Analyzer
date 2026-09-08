"""Tests for dataset schemas (Phase 3)."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.dataset.config import DatasetConfig
from backend.app.dataset.schemas import DatasetManifest, DatasetRow


class TestDatasetRow:
    def test_default_values(self) -> None:
        row = DatasetRow()
        assert row.dataset_version == "v1"
        assert row.label_status == "negative"
        assert row.defect_label == 0
        assert row.split == "train"
        assert row.commit_features == []
        assert row.file_features == []

    def test_positive_label(self) -> None:
        row = DatasetRow(
            label_status="positive",
            defect_label=1,
            label_confidence=1.0,
            label_source="explicit_sha_reference",
        )
        assert row.label_status == "positive"
        assert row.defect_label == 1

    def test_negative_label(self) -> None:
        row = DatasetRow(
            label_status="negative",
            defect_label=0,
            label_confidence=0.0,
            label_source="none",
        )
        assert row.label_status == "negative"
        assert row.defect_label == 0

    def test_ambiguous_label(self) -> None:
        row = DatasetRow(
            label_status="ambiguous",
            defect_label=-1,
            label_confidence=0.0,
            label_source="ambiguous",
        )
        assert row.label_status == "ambiguous"
        assert row.defect_label == -1

    def test_feature_vectors_stored(self) -> None:
        commit_vec = [0.1] * 29
        file_vec = [0.2] * 16
        row = DatasetRow(
            commit_features=commit_vec,
            file_features=file_vec,
        )
        assert len(row.commit_features) == 29
        assert len(row.file_features) == 16

    def test_label_confidence_bounds(self) -> None:
        row = DatasetRow(label_confidence=0.0)
        assert row.label_confidence == 0.0
        row2 = DatasetRow(label_confidence=1.0)
        assert row2.label_confidence == 1.0


class TestDatasetManifest:
    def test_manifest_creation(self) -> None:
        manifest = DatasetManifest(
            dataset_version="v1",
            feature_version="v1",
            labeling_version="v1",
            config=DatasetConfig().model_dump(),
            repo_url="https://github.com/org/repo",
            repo_name="repo",
            repo_revision="abc123",
            total_commits_processed=10,
            total_commits_included=10,
            total_commits_excluded=0,
            total_file_examples=50,
            positive_examples=5,
            negative_examples=40,
            ambiguous_examples=5,
            positive_rate=0.1,
            train_commits=7,
            validation_commits=1,
            test_commits=2,
            train_examples=35,
            validation_examples=5,
            test_examples=10,
            commits_per_language={".py": 10},
            examples_per_language={".py": 50},
            time_range_start="2026-01-01T00:00:00+00:00",
            time_range_end="2026-01-10T00:00:00+00:00",
            bug_fix_commits_found=3,
            revert_commits_found=1,
            explicit_sha_attributions=2,
            revert_attributions=1,
            line_overlap_attributions=0,
            ambiguous_attributions=0,
            exclusion_counts={},
        )
        assert manifest.total_commits_processed == 10
        assert manifest.positive_rate == 0.1
        assert manifest.repo_name == "repo"
