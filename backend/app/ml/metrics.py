"""Comprehensive metric calculation for Phase 4 model evaluation.

Reports ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix,
and positive prediction counts. Designed for imbalanced defect datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalMetrics:
    """Container for all evaluation metrics."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    positive_predictions: int
    actual_positives: int
    actual_negatives: int
    total: int

    def to_dict(self) -> dict:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "positive_predictions": self.positive_predictions,
            "actual_positives": self.actual_positives,
            "actual_negatives": self.actual_negatives,
            "total": self.total,
        }


@dataclass
class BaselineMetrics:
    """Majority-class baseline metrics."""

    accuracy: float
    majority_class: int
    majority_count: int
    total: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "majority_class": self.majority_class,
            "majority_count": self.majority_count,
            "total": self.total,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def compute_baseline(y_true: np.ndarray) -> BaselineMetrics:
    """Compute majority-class baseline metrics.

    The majority class baseline always predicts the most frequent class.
    """
    total = len(y_true)
    if total == 0:
        return BaselineMetrics(
            accuracy=0.0, majority_class=0, majority_count=0,
            total=0, positive_count=0, negative_count=0,
        )

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    majority_class = 1 if positive_count > negative_count else 0
    majority_count = max(positive_count, negative_count)
    accuracy = majority_count / total

    return BaselineMetrics(
        accuracy=accuracy,
        majority_class=majority_class,
        majority_count=majority_count,
        total=total,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold for converting probabilities to labels.

    Returns
    -------
    EvalMetrics
        All computed metrics.
    """
    total = len(y_true)
    if total == 0:
        return EvalMetrics(
            roc_auc=0.0, pr_auc=0.0, precision=0.0, recall=0.0, f1=0.0,
            confusion_matrix=[[0, 0], [0, 0]],
            true_positives=0, false_positives=0,
            true_negatives=0, false_negatives=0,
            positive_predictions=0, actual_positives=0,
            actual_negatives=0, total=0,
        )

    actual_positives = int(np.sum(y_true == 1))
    actual_negatives = int(np.sum(y_true == 0))

    # Handle edge case: no positives in ground truth
    if actual_positives == 0:
        roc_auc = 0.0
        pr_auc = 0.0
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = 0.0
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = 0.0

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Handle edge case: no positive predictions
    positive_predictions = int(np.sum(y_pred == 1))
    if positive_predictions == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return EvalMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=[[tn, fp], [fn, tp]],
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        positive_predictions=positive_predictions,
        actual_positives=actual_positives,
        actual_negatives=actual_negatives,
        total=total,
    )
