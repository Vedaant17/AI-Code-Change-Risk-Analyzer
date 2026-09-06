"""End-to-end integration test for the feature extraction pipeline."""

from __future__ import annotations

from backend.app.features.extractor import FeatureExtractor
from backend.app.features.schemas import (
    COMMIT_FEATURE_NAMES,
    FEATURE_VERSION,
    FILE_FEATURE_NAMES,
    FeatureExtractionResult,
)
from backend.app.schemas.diff import CommitInfo


class TestFeatureExtractor:
    def test_single_file_commit(self, single_python_commit_info: CommitInfo) -> None:
        ext = FeatureExtractor()
        result = ext.extract(single_python_commit_info)

        assert isinstance(result, FeatureExtractionResult)
        assert result.feature_version == FEATURE_VERSION
        assert len(result.file_features) == 1
        assert result.commit_features.files_changed == 1

    def test_multi_file_commit(self, multi_language_commit_info: CommitInfo) -> None:
        ext = FeatureExtractor()
        result = ext.extract(multi_language_commit_info)

        assert len(result.file_features) == 2
        assert result.commit_features.files_changed == 2
        assert result.commit_features.languages_touched == 2

    def test_test_prod_coupling(
        self, test_prod_coupling_commit_info: CommitInfo
    ) -> None:
        ext = FeatureExtractor()
        result = ext.extract(test_prod_coupling_commit_info)

        assert result.commit_features.test_prod_coupling is True
        assert result.commit_features.test_files_changed == 1
        assert result.commit_features.prod_files_changed == 1

    def test_empty_commit(self, empty_commit_info: CommitInfo) -> None:
        ext = FeatureExtractor()
        result = ext.extract(empty_commit_info)

        assert result.commit_features.files_changed == 0
        assert len(result.file_features) == 0
        assert result.commit_features.to_feature_vector() == [0.0] * len(COMMIT_FEATURE_NAMES)

    def test_binary_only(self, binary_only_commit_info: CommitInfo) -> None:
        ext = FeatureExtractor()
        result = ext.extract(binary_only_commit_info)

        assert len(result.file_features) == 1
        assert result.file_features[0].is_binary is True
        assert result.commit_features.files_binary == 1

    def test_vectors_are_correct_length(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        ext = FeatureExtractor()
        result = ext.extract(multi_language_commit_info)

        commit_vec = result.commit_features.to_feature_vector()
        assert len(commit_vec) == len(COMMIT_FEATURE_NAMES)

        for ff in result.file_features:
            file_vec = ff.to_feature_vector()
            assert len(file_vec) == len(FILE_FEATURE_NAMES)

    def test_deterministic(self, multi_language_commit_info: CommitInfo) -> None:
        ext = FeatureExtractor()
        r1 = ext.extract(multi_language_commit_info)
        r2 = ext.extract(multi_language_commit_info)

        assert r1.commit_features.to_feature_vector() == r2.commit_features.to_feature_vector()
        assert len(r1.file_features) == len(r2.file_features)
        for f1, f2 in zip(r1.file_features, r2.file_features):
            assert f1.to_feature_vector() == f2.to_feature_vector()

    def test_feature_version_present(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        ext = FeatureExtractor()
        result = ext.extract(multi_language_commit_info)
        assert result.feature_version == FEATURE_VERSION
