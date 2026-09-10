"""Feature importance extraction for Phase 4 defect models.

Provides global feature importance for tree-based models (XGBoost)
and coefficient-based importance for linear models (LogisticRegression).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """A single feature's importance ranking."""

    rank: int
    feature_name: str
    importance: float
    feature_index: int


def extract_importance(
    model,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract global feature importance from a fitted model.

    Parameters
    ----------
    model : BaseModel from backend.app.ml.models
        A fitted model instance.
    feature_names : list[str]
        Ordered names for the 45 features.

    Returns
    -------
    list[FeatureImportance]
        Features ranked by importance (descending).
    """
    inner = getattr(model, "_model", None)
    if inner is None:
        logger.warning("Model has no internal _model; returning empty importance")
        return []

    # XGBoost / GradientBoosting: feature_importances_
    if hasattr(inner, "feature_importances_"):
        importances = np.array(inner.feature_importances_, dtype=np.float64)
    # LogisticRegression: absolute coefficients
    elif hasattr(inner, "coef_"):
        importances = np.abs(inner.coef_[0]).astype(np.float64)
    else:
        logger.warning("Model type does not support feature importance extraction")
        return []

    # Normalize to [0, 1] for comparability
    max_val = importances.max()
    if max_val > 0:
        importances = importances / max_val

    # Sort descending
    indices = np.argsort(importances)[::-1]

    results: list[FeatureImportance] = []
    for rank, idx in enumerate(indices, 1):
        results.append(FeatureImportance(
            rank=rank,
            feature_name=feature_names[idx],
            importance=float(importances[idx]),
            feature_index=int(idx),
        ))

    return results


def format_importance_report(importances: list[FeatureImportance], top_n: int = 20) -> str:
    """Format a human-readable feature importance report."""
    lines = ["Feature Importance (ranked):", ""]
    lines.append(f"{'Rank':<6}{'Feature':<40}{'Importance':<12}")
    lines.append("-" * 58)
    for fi in importances[:top_n]:
        lines.append(f"{fi.rank:<6}{fi.feature_name:<40}{fi.importance:<12.4f}")
    if len(importances) > top_n:
        lines.append(f"... and {len(importances) - top_n} more features")
    return "\n".join(lines)
