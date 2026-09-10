"""Tests for Phase 4 metric calculation."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.metrics import (
    compute_baseline,
    compute_metrics,
)


class TestBaseline:
    def test_majority_positive(self):
        y = np.array([1, 1, 1, 0, 0])
        b = compute_baseline(y)
        assert b.majority_class == 1
        assert b.majority_count == 3
        assert b.accuracy == pytest.approx(0.6)

    def test_majority_negative(self):
        y = np.array([0, 0, 0, 1])
        b = compute_baseline(y)
        assert b.majority_class == 0
        assert b.majority_count == 3
        assert b.accuracy == pytest.approx(0.75)

    def test_empty(self):
        b = compute_baseline(np.array([], dtype=int))
        assert b.accuracy == 0.0
        assert b.total == 0

    def test_balanced(self):
        y = np.array([0, 0, 1, 1])
        b = compute_baseline(y)
        assert b.majority_count == 2
        assert b.accuracy == pytest.approx(0.5)

    def test_to_dict(self):
        b = compute_baseline(np.array([0, 0, 1]))
        d = b.to_dict()
        assert "accuracy" in d
        assert "majority_class" in d


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.roc_auc == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.true_positives == 2
        assert m.true_negatives == 2
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.roc_auc == pytest.approx(0.0)
        assert m.true_positives == 0
        assert m.false_positives == 2
        assert m.false_negatives == 2

    def test_no_positive_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.positive_predictions == 0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_no_positives_in_ground_truth(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m.roc_auc == 0.0
        assert m.actual_positives == 0
        assert m.actual_negatives == 4

    def test_empty(self):
        m = compute_metrics(np.array([], dtype=int), np.array([], dtype=float))
        assert m.total == 0
        assert m.roc_auc == 0.0

    def test_threshold_affects_predictions(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        m_low = compute_metrics(y_true, y_prob, threshold=0.2)
        m_high = compute_metrics(y_true, y_prob, threshold=0.8)
        assert m_low.positive_predictions >= m_high.positive_predictions

    def test_pr_auc_computed(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m.pr_auc <= 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        m = compute_metrics(y_true, y_prob)
        assert len(m.confusion_matrix) == 2
        assert len(m.confusion_matrix[0]) == 2
        assert len(m.confusion_matrix[1]) == 2

    def test_to_dict(self):
        m = compute_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
        d = m.to_dict()
        assert "roc_auc" in d
        assert "pr_auc" in d
        assert "confusion_matrix" in d
