"""Tests for experimental feature schema (Phase 4.5a).

Verifies schema correctness, vector ordering, and serialization.
"""

from __future__ import annotations

import pytest

from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
    EXPERIMENTAL_FEATURE_VERSION,
    ExperimentalFileFeatures,
)


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(EXPERIMENTAL_FEATURE_VERSION, str)

    def test_version_starts_with_exp(self):
        assert EXPERIMENTAL_FEATURE_VERSION.startswith("exp")


class TestFeatureContract:
    def test_feature_count(self):
        assert EXPERIMENTAL_FEATURE_COUNT == 6

    def test_feature_names_length(self):
        assert len(EXPERIMENTAL_FEATURE_NAMES) == 6

    def test_feature_names_unique(self):
        assert len(set(EXPERIMENTAL_FEATURE_NAMES)) == 6

    def test_all_names_are_strings(self):
        for name in EXPERIMENTAL_FEATURE_NAMES:
            assert isinstance(name, str)


class TestExperimentalFileFeatures:
    def test_defaults_all_zero(self):
        f = ExperimentalFileFeatures()
        vec = f.to_feature_vector()
        assert vec == [0.0] * 6

    def test_vector_length(self):
        f = ExperimentalFileFeatures()
        assert len(f.to_feature_vector()) == 6

    def test_vector_ordering(self):
        f = ExperimentalFileFeatures(
            file_ast_functions_added=1.0,
            file_ast_functions_deleted=2.0,
            file_ast_try_except_changed=3.0,
            file_commit_count=4.0,
            file_historical_bug_fixes=5.0,
            file_days_since_last_change=6.0,
        )
        vec = f.to_feature_vector()
        assert vec == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_deterministic(self):
        f = ExperimentalFileFeatures(file_ast_functions_added=1.0)
        v1 = f.to_feature_vector()
        v2 = f.to_feature_vector()
        assert v1 == v2

    def test_file_path_not_in_vector(self):
        f = ExperimentalFileFeatures(file_path="src/main.py")
        vec = f.to_feature_vector()
        assert len(vec) == 6

    def test_serialization_roundtrip(self):
        f = ExperimentalFileFeatures(
            file_path="test.py",
            file_ast_functions_added=1.0,
            file_commit_count=10.0,
        )
        dumped = f.model_dump()
        restored = ExperimentalFileFeatures.model_validate(dumped)
        assert f.to_feature_vector() == restored.to_feature_vector()

    def test_vector_matches_names_order(self):
        f = ExperimentalFileFeatures(
            file_ast_functions_added=1.0,
            file_ast_functions_deleted=2.0,
            file_ast_try_except_changed=3.0,
            file_commit_count=4.0,
            file_historical_bug_fixes=5.0,
            file_days_since_last_change=6.0,
        )
        d = f.model_dump()
        for i, name in enumerate(EXPERIMENTAL_FEATURE_NAMES):
            assert f.to_feature_vector()[i] == float(d[name])

    def test_ge_constraints(self):
        """Negative values should be rejected."""
        with pytest.raises(Exception):
            ExperimentalFileFeatures(file_ast_functions_added=-1.0)
