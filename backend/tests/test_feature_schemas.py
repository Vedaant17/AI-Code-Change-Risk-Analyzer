"""Tests for feature schema serialization, versioning, and vector ordering."""

from __future__ import annotations

import pytest

from backend.app.features.schemas import (
    COMMIT_FEATURE_NAMES,
    FEATURE_VERSION,
    FILE_FEATURE_NAMES,
    CommitFeatures,
    FeatureExtractionResult,
    FileFeatures,
)


class TestFeatureVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(FEATURE_VERSION, str)

    def test_version_format(self) -> None:
        assert FEATURE_VERSION.startswith("v")


class TestCommitFeaturesSchema:
    def test_default_values(self) -> None:
        cf = CommitFeatures()
        assert cf.lines_added == 0
        assert cf.files_changed == 0
        assert cf.change_entropy == 0.0
        assert cf.to_feature_vector() == [0.0] * len(COMMIT_FEATURE_NAMES)

    def test_feature_names_unique(self) -> None:
        assert len(COMMIT_FEATURE_NAMES) == len(set(COMMIT_FEATURE_NAMES))

    def test_feature_names_all_exist(self) -> None:
        cf = CommitFeatures()
        d = cf.model_dump()
        for name in COMMIT_FEATURE_NAMES:
            assert name in d, f"Field {name!r} not in CommitFeatures"

    def test_feature_names_count_matches_model(self) -> None:
        # CommitFeatures has these feature fields plus commit_sha
        feature_fields = [f for f in CommitFeatures.model_fields if f != "commit_sha"]
        assert len(COMMIT_FEATURE_NAMES) == len(feature_fields)


class TestFileFeaturesSchema:
    def test_default_values(self) -> None:
        ff = FileFeatures()
        assert ff.lines_added == 0
        assert ff.to_feature_vector() == [0.0] * len(FILE_FEATURE_NAMES)

    def test_feature_names_unique(self) -> None:
        assert len(FILE_FEATURE_NAMES) == len(set(FILE_FEATURE_NAMES))

    def test_feature_names_all_exist(self) -> None:
        ff = FileFeatures()
        d = ff.model_dump()
        for name in FILE_FEATURE_NAMES:
            assert name in d, f"Field {name!r} not in FileFeatures"

    def test_feature_names_count_matches_model(self) -> None:
        feature_fields = [f for f in FileFeatures.model_fields if f != "file_path"]
        assert len(FILE_FEATURE_NAMES) == len(feature_fields)


class TestSerializationRoundTrip:
    def test_commit_features_round_trip(self) -> None:
        cf = CommitFeatures(lines_added=10, lines_deleted=5, files_changed=3)
        d = cf.model_dump()
        cf2 = CommitFeatures.model_validate(d)
        assert cf == cf2

    def test_file_features_round_trip(self) -> None:
        ff = FileFeatures(file_path="test.py", lines_added=3, lines_deleted=1)
        d = ff.model_dump()
        ff2 = FileFeatures.model_validate(d)
        assert ff == ff2

    def test_feature_extraction_result_round_trip(self) -> None:
        result = FeatureExtractionResult(
            commit_features=CommitFeatures(lines_added=1),
            file_features=[FileFeatures(file_path="a.py")],
            feature_version=FEATURE_VERSION,
        )
        d = result.model_dump()
        result2 = FeatureExtractionResult.model_validate(d)
        assert result == result2


class TestVectorOrderingContract:
    """Verify that the vector ordering is stable and matches field declarations."""

    def test_commit_vector_index_0_is_lines_added(self) -> None:
        cf = CommitFeatures(lines_added=42)
        vec = cf.to_feature_vector()
        assert vec[0] == 42.0

    def test_commit_vector_last_is_change_entropy(self) -> None:
        cf = CommitFeatures(change_entropy=1.5)
        vec = cf.to_feature_vector()
        assert vec[-1] == 1.5

    def test_file_vector_index_0_is_language(self) -> None:
        ff = FileFeatures(language=0.5)
        vec = ff.to_feature_vector()
        assert vec[0] == 0.5

    def test_file_vector_last_is_avg_hunk_size(self) -> None:
        ff = FileFeatures(avg_hunk_size=7.5)
        vec = ff.to_feature_vector()
        assert vec[-1] == 7.5

    def test_vector_length_stable(self) -> None:
        """If someone adds a field, this test will fail until FEATURE_NAMES is updated."""
        assert len(CommitFeatures().to_feature_vector()) == len(COMMIT_FEATURE_NAMES)
        assert len(FileFeatures().to_feature_vector()) == len(FILE_FEATURE_NAMES)
