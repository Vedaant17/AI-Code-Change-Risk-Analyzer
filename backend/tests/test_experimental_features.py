"""Tests for Phase 4.5 experimental feature loading and model training.

Verifies:
- Experimental features load correctly for all modes (E0/E1/E2/E4)
- Feature dimensions match expectations
- Baseline vs experimental comparison is reproducible
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.app.ml.dataset_loader import FEATURE_NAMES
from backend.app.ml.experimental_loader import (
    _build_experimental_lookup,
    _get_mode_feature_names,
    _get_mode_indices,
    load_experimental_dataset,
    load_experimental_split,
)
from backend.app.ml.metrics import compute_baseline, compute_metrics
from backend.app.ml.models import (
    MajorityClassBaseline,
    WeightedLogisticRegression,
)

# Test data paths
V3_DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DATA_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")


class TestExperimentalLoader:
    """Tests for the experimental loader module."""

    def test_mode_indices(self):
        """Verify mode indices are correct."""
        assert _get_mode_indices("E0") == []
        assert _get_mode_indices("E1") == [0, 1, 2]  # AST features
        assert _get_mode_indices("E2") == [3, 4, 5]  # Historical features
        assert _get_mode_indices("E4") == [0, 1, 2, 3, 4, 5]  # All 6

    def test_mode_feature_names(self):
        """Verify feature names for each mode."""
        e0_names = _get_mode_feature_names("E0")
        assert e0_names == FEATURE_NAMES
        assert len(e0_names) == 45

        e1_names = _get_mode_feature_names("E1")
        assert len(e1_names) == 48  # 45 + 3 AST
        assert "file_ast_functions_added" in e1_names

        e2_names = _get_mode_feature_names("E2")
        assert len(e2_names) == 48  # 45 + 3 historical
        assert "file_commit_count" in e2_names

        e4_names = _get_mode_feature_names("E4")
        assert len(e4_names) == 51  # 45 + 6
        assert "file_ast_functions_added" in e4_names
        assert "file_commit_count" in e4_names

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_load_experimental_split(self):
        """Test loading a single split with experimental features."""
        v1_path = V3_DATA_DIR / "train.jsonl"
        exp_path = EXP_DATA_DIR / "experimental_features.jsonl"

        if not v1_path.exists() or not exp_path.exists():
            pytest.skip("Dataset files not available")

        dataset = load_experimental_split(v1_path, exp_path, mode="E4")
        assert len(dataset) > 0
        assert dataset.X.shape[1] == 51  # 45 + 6

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_load_experimental_dataset(self):
        """Test loading all splits with experimental features."""
        if not EXP_DATA_DIR.exists():
            pytest.skip("Experimental dataset not available")

        train, val, test = load_experimental_dataset(
            V3_DATA_DIR, EXP_DATA_DIR, mode="E4"
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert train.X.shape[1] == 51
        assert val.X.shape[1] == 51
        assert test.X.shape[1] == 51

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_e0_matches_baseline(self):
        """Verify E0 mode produces identical features to baseline."""
        from backend.app.ml.dataset_loader import load_dataset

        v1_train, v1_val, v1_test = load_dataset(V3_DATA_DIR)
        e0_train, e0_val, e0_test = load_experimental_dataset(
            V3_DATA_DIR, EXP_DATA_DIR, mode="E0"
        )

        # E0 should have same dimensions as baseline
        assert e0_train.X.shape == v1_train.X.shape
        assert e0_val.X.shape == v1_val.X.shape
        assert e0_test.X.shape == v1_test.X.shape

        # Features should be identical
        np.testing.assert_array_equal(e0_train.X, v1_train.X)
        np.testing.assert_array_equal(e0_val.X, v1_val.X)
        np.testing.assert_array_equal(e0_test.X, v1_test.X)

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_experimental_features_are_numeric(self):
        """Verify all experimental features are numeric values."""
        if not EXP_DATA_DIR.exists():
            pytest.skip("Experimental dataset not available")

        train, val, test = load_experimental_dataset(
            V3_DATA_DIR, EXP_DATA_DIR, mode="E4"
        )

        # Check no NaN or Inf
        assert np.all(np.isfinite(train.X)), "Training data contains NaN/Inf"
        assert np.all(np.isfinite(val.X)), "Validation data contains NaN/Inf"
        assert np.all(np.isfinite(test.X)), "Test data contains NaN/Inf"


class TestModelTraining:
    """Tests for model training with experimental features."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_baseline_trains(self):
        """Verify baseline model trains successfully."""
        from backend.app.ml.dataset_loader import load_dataset

        train, val, test = load_dataset(V3_DATA_DIR)
        model = MajorityClassBaseline()
        model.fit(train.X, train.y)

        probs = model.predict_proba(test.X)
        assert probs.shape == (len(test),)
        assert np.all((probs == 0) | (probs == 1))  # Binary predictions

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_experimental_model_trains(self):
        """Verify model with experimental features trains successfully."""
        if not EXP_DATA_DIR.exists():
            pytest.skip("Experimental dataset not available")

        train, val, test = load_experimental_dataset(
            V3_DATA_DIR, EXP_DATA_DIR, mode="E4"
        )
        model = WeightedLogisticRegression()
        model.fit(train.X, train.y)

        probs = model.predict_proba(test.X)
        assert probs.shape == (len(test),)
        assert np.all(np.isfinite(probs))

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="V3 dataset not available"
    )
    def test_metrics_computation(self):
        """Verify metrics computation works correctly."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.75, 0.85])

        metrics = compute_metrics(y_true, y_prob)
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert metrics.total == 10

    def test_baseline_metrics(self):
        """Verify baseline metrics computation."""
        y_true = np.array([0, 0, 0, 1, 1])
        baseline = compute_baseline(y_true)

        assert baseline.majority_class == 0
        assert baseline.majority_count == 3
        assert baseline.accuracy == 0.6
        assert baseline.total == 5


class TestExperimentalLookup:
    """Tests for the experimental lookup builder."""

    def test_lookup_builds_from_file(self):
        """Test lookup construction from a JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "commit_sha": "abc123",
                "file_path": "test.py",
                "experimental_features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            }) + "\n")
            f.write(json.dumps({
                "commit_sha": "def456",
                "file_path": "other.py",
                "experimental_features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }) + "\n")
            temp_path = Path(f.name)

        try:
            lookup = _build_experimental_lookup(temp_path)
            assert len(lookup) == 2
            assert lookup[("abc123", "test.py")] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            assert lookup[("def456", "other.py")] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        finally:
            temp_path.unlink()

    def test_lookup_empty_file(self):
        """Test lookup with non-existent file."""
        lookup = _build_experimental_lookup(Path("nonexistent.jsonl"))
        assert len(lookup) == 0
