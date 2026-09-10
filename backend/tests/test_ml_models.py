"""Tests for Phase 4 ML models."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.models import (
    HAS_XGBOOST,
    MajorityClassBaseline,
    WeightedLogisticRegression,
    XGBoostDefectModel,
)


@pytest.fixture
def simple_data():
    """Small synthetic dataset: 100 samples, 45 features, 10% positive."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 45)
    y = np.zeros(100, dtype=int)
    y[:10] = 1  # 10% positive
    return X, y


@pytest.fixture
def balanced_data():
    """Balanced synthetic dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 45)
    y = np.array([0] * 30 + [1] * 30, dtype=int)
    return X, y


class TestMajorityClassBaseline:
    def test_predicts_majority_class(self, simple_data):
        X, y = simple_data
        model = MajorityClassBaseline()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (100,)
        # Majority class is 0, so all predictions should be 0.0
        assert np.all(preds == 0.0)

    def test_positive_majority(self):
        y = np.array([1, 1, 1, 0])
        X = np.random.RandomState(42).randn(4, 45)
        model = MajorityClassBaseline()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert np.all(preds == 1.0)

    def test_name(self):
        assert MajorityClassBaseline().name == "majority_class_baseline"

    def test_get_params(self):
        m = MajorityClassBaseline()
        m.fit(np.zeros((3, 45)), np.array([0, 0, 1]))
        p = m.get_params()
        assert "majority_class" in p


class TestWeightedLogisticRegression:
    def test_fits_and_predicts(self, simple_data):
        X, y = simple_data
        model = WeightedLogisticRegression()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (100,)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_name(self):
        assert WeightedLogisticRegression().name == "weighted_logistic_regression"

    def test_get_params(self):
        m = WeightedLogisticRegression()
        p = m.get_params()
        assert p["class_weight"] == "balanced"


class TestXGBoostDefectModel:
    def test_fits_and_predicts(self, simple_data):
        X, y = simple_data
        model = XGBoostDefectModel(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (100,)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_uses_xgboost_when_available(self, simple_data):
        X, y = simple_data
        model = XGBoostDefectModel(n_estimators=5)
        model.fit(X, y)
        if HAS_XGBOOST:
            assert model._uses_xgboost is True
        assert model.get_params()["backend"] in ("xgboost", "sklearn")

    def test_with_validation_set(self, simple_data):
        X, y = simple_data
        model = XGBoostDefectModel(n_estimators=5)
        model.fit(X[:80], y[:80], X_val=X[80:], y_val=y[80:])
        preds = model.predict_proba(X)
        assert preds.shape == (100,)

    def test_custom_scale_pos_weight(self, simple_data):
        X, y = simple_data
        model = XGBoostDefectModel(scale_pos_weight=5.0)
        model.fit(X, y)
        params = model.get_params()
        assert params["scale_pos_weight"] == 5.0

    def test_name(self):
        assert XGBoostDefectModel().name == "xgboost_defect_model"

    def test_predict_before_fit_raises(self):
        model = XGBoostDefectModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros((1, 45)))

    def test_training_reproducible(self, simple_data):
        X, y = simple_data
        m1 = XGBoostDefectModel(n_estimators=5, random_state=42)
        m1.fit(X, y)
        p1 = m1.predict_proba(X[:5])

        m2 = XGBoostDefectModel(n_estimators=5, random_state=42)
        m2.fit(X, y)
        p2 = m2.predict_proba(X[:5])

        np.testing.assert_array_almost_equal(p1, p2)
