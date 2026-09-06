"""Tests for commit-level feature extraction."""

from __future__ import annotations

from backend.app.features.commit_features import extract_commit_features
from backend.app.features.schemas import COMMIT_FEATURE_NAMES, CommitFeatures
from backend.app.schemas.diff import CommitInfo


class TestCommitFeatures:
    def test_single_python_commit(
        self, single_python_commit_info: CommitFeatures
    ) -> None:
        # single_python_commit_info is actually a CommitInfo fixture
        from backend.app.schemas.diff import CommitInfo

        assert isinstance(single_python_commit_info, CommitInfo)
        cf = extract_commit_features(single_python_commit_info)

        assert cf.files_changed == 1
        assert cf.lines_added > 0
        assert cf.total_lines_changed > 0
        assert cf.has_prod_changes is True
        assert cf.has_test_changes is False
        assert cf.test_prod_coupling is False
        assert cf.languages_touched >= 1

    def test_multi_language(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        assert cf.files_changed == 2
        assert cf.languages_touched == 2

    def test_test_prod_coupling(
        self, test_prod_coupling_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(test_prod_coupling_commit_info)
        assert cf.files_changed == 2
        assert cf.test_files_changed == 1
        assert cf.prod_files_changed == 1
        assert cf.has_test_changes is True
        assert cf.has_prod_changes is True
        assert cf.test_prod_coupling is True
        assert cf.test_ratio == 0.5

    def test_empty_commit(self, empty_commit_info: CommitInfo) -> None:
        cf = extract_commit_features(empty_commit_info)
        assert cf.files_changed == 0
        assert cf.lines_added == 0
        assert cf.total_lines_changed == 0
        assert cf.avg_hunks_per_file == 0.0
        assert cf.change_entropy == 0.0
        assert cf.test_prod_coupling is False

    def test_binary_only(self, binary_only_commit_info: CommitInfo) -> None:
        cf = extract_commit_features(binary_only_commit_info)
        assert cf.files_changed == 1
        assert cf.files_binary == 1
        assert cf.total_function_declarations_changed == 0
        assert cf.total_imports_changed == 0

    def test_ratios_sum_to_one(self, multi_language_commit_info: CommitInfo) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        assert abs(cf.additions_ratio + cf.deletions_ratio - 1.0) < 1e-9

    def test_test_ratio_bounds(self, test_prod_coupling_commit_info: CommitInfo) -> None:
        cf = extract_commit_features(test_prod_coupling_commit_info)
        assert 0.0 <= cf.test_ratio <= 1.0

    def test_change_entropy_non_negative(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        assert cf.change_entropy >= 0.0

    def test_indent_stats_non_negative(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        assert cf.avg_changed_line_indent >= 0.0
        assert cf.max_changed_line_indent >= 0

    def test_commit_sha_preserved(
        self, single_python_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(single_python_commit_info)
        assert cf.commit_sha == single_python_commit_info.sha


class TestCommitFeatureVectorOrdering:
    def test_vector_length(self, multi_language_commit_info: CommitInfo) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        vec = cf.to_feature_vector()
        assert len(vec) == len(COMMIT_FEATURE_NAMES)

    def test_vector_matches_field_order(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        cf = extract_commit_features(multi_language_commit_info)
        vec = cf.to_feature_vector()
        d = cf.model_dump()
        for i, name in enumerate(COMMIT_FEATURE_NAMES):
            assert vec[i] == float(d[name]), f"Mismatch at index {i} ({name})"

    def test_deterministic(
        self, multi_language_commit_info: CommitInfo
    ) -> None:
        vec1 = extract_commit_features(multi_language_commit_info).to_feature_vector()
        vec2 = extract_commit_features(multi_language_commit_info).to_feature_vector()
        assert vec1 == vec2
