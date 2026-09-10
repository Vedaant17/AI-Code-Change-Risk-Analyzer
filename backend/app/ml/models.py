"""ML models for Phase 4 defect prediction.

Implements:
  1. MajorityClassBaseline — always predicts the most frequent class
  2. WeightedLogisticRegression — simple linear baseline with class weighting
  3. XGBoostClassifier — primary tree-based model with scale_pos_weight
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Try XGBoost; fall back gracefully if unavailable
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available; using GradientBoostingClassifier fallback")


class BaseModel(ABC):
    """Abstract base for all defect prediction models."""

    @abstractmethod
    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> None:
        """Train the model."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class for each sample."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return model configuration as a serializable dict."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""


class MajorityClassBaseline(BaseModel):
    """Always predicts the majority class from training data."""

    def __init__(self) -> None:
        self._majority_class: int = 0
        self._fitted = False

    @property
    def name(self) -> str:
        return "majority_class_baseline"

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> None:
        counts = np.bincount(y_train.astype(int))
        self._majority_class = int(np.argmax(counts))
        self._fitted = True
        logger.info("MajorityClassBaseline: majority class = %d", self._majority_class)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1.0 for majority class, 0.0 for minority."""
        result = np.zeros(len(X), dtype=np.float64)
        if self._majority_class == 1:
            result[:] = 1.0
        return result

    def get_params(self) -> dict:
        return {"majority_class": self._majority_class}


class WeightedLogisticRegression(BaseModel):
    """Logistic regression with class_weight='balanced'."""

    def __init__(self, max_iter: int = 1000) -> None:
        self._max_iter = max_iter
        self._model: LogisticRegression | None = None

    @property
    def name(self) -> str:
        return "weighted_logistic_regression"

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> None:
        self._model = LogisticRegression(
            class_weight="balanced",
            max_iter=self._max_iter,
            random_state=42,
            solver="lbfgs",
        )
        self._model.fit(X_train, y_train)
        logger.info("WeightedLogisticRegression trained on %d samples", len(y_train))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        return {
            "max_iter": self._max_iter,
            "class_weight": "balanced",
            "solver": "lbfgs",
        }


class XGBoostDefectModel(BaseModel):
    """XGBoost classifier with scale_pos_weight for class imbalance.

    Falls back to sklearn GradientBoostingClassifier if xgboost is unavailable.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float | None = None,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._scale_pos_weight = scale_pos_weight
        self._random_state = random_state
        self._model = None
        self._uses_xgboost = False

    @property
    def name(self) -> str:
        return "xgboost_defect_model"

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> None:
        # Compute scale_pos_weight if not provided
        spw = self._scale_pos_weight
        if spw is None:
            neg_count = int(np.sum(y_train == 0))
            pos_count = int(np.sum(y_train == 1))
            spw = neg_count / pos_count if pos_count > 0 else 1.0

        if HAS_XGBOOST:
            self._model = XGBClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                scale_pos_weight=spw,
                random_state=self._random_state,
                eval_metric="aucpr",
                use_label_encoder=False,
                verbosity=0,
            )
            self._uses_xgboost = True
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            self._model = GradientBoostingClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                random_state=self._random_state,
            )
            self._uses_xgboost = False

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        if self._uses_xgboost and eval_set:
            self._model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self._model.fit(X_train, y_train)

        logger.info(
            "XGBoostDefectModel trained: backend=%s, spw=%.2f, samples=%d",
            "xgboost" if self._uses_xgboost else "sklearn",
            spw, len(y_train),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "scale_pos_weight": self._scale_pos_weight,
            "random_state": self._random_state,
            "backend": "xgboost" if self._uses_xgboost else "sklearn",
        }
