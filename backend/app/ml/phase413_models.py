"""Phase 4.13 Models: Baselines and model definitions.

Implements:
    B0: Random ranking (within-commit)
    B1: Change-size heuristic (within-commit)
    B2: Biased P-vs-U XGBoost (file-level)
    B3: Evidence-ranking pairwise logistic (pairwise)
    B4: Commit-level fallback XGBoost (commit-level)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from backend.app.ml.models import XGBoostDefectModel

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# B0: Random baseline
# ---------------------------------------------------------------------------

def random_within_commit_scores(
    metadata: list[dict],
    seed: int = RANDOM_SEED,
) -> list[float]:
    """Assign random scores independently within each commit.

    Returns a list of scores aligned with metadata.
    """
    rng = np.random.RandomState(seed)
    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    scores = np.zeros(len(metadata), dtype=np.float64)
    for indices in by_commit.values():
        n = len(indices)
        scores[indices] = rng.uniform(0.0, 1.0, size=n)
    return scores.tolist()


# ---------------------------------------------------------------------------
# B1: Change-size heuristic
# ---------------------------------------------------------------------------

def change_size_scores(
    metadata: list[dict],
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Rank files by file_total_lines_changed (descending) within each commit.

    Falls back to random if the feature is not found.
    """
    idx = None
    for i, name in enumerate(feature_names):
        if name == "file_total_lines_changed":
            idx = i
            break

    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    scores = np.zeros(len(metadata), dtype=np.float64)
    if idx is None:
        rng = np.random.RandomState(RANDOM_SEED)
        for indices in by_commit.values():
            scores[indices] = rng.uniform(0.0, 1.0, size=len(indices))
        return scores.tolist()

    for indices in by_commit.values():
        vals = X[indices, idx]
        ranks = np.argsort(np.argsort(-vals)).astype(np.float64)
        max_rank = max(len(indices) - 1, 1)
        scores[indices] = 1.0 - ranks / max_rank
    return scores.tolist()


# ---------------------------------------------------------------------------
# B2: Biased P-vs-U XGBoost
# ---------------------------------------------------------------------------

def train_biased_pu_model(
    X_P: np.ndarray,
    X_U: np.ndarray,
    metadata_P: list[dict],
    metadata_U: list[dict],
    feature_names: list[str],
) -> tuple[XGBoostDefectModel, list[float], list[dict]]:
    """Train biased P-vs-U XGBoost model.

    P = OBSERVED_POSITIVE (label=1)
    U = UNLABELED (label=0)

    This is NOT legitimate PU learning. It is a biased baseline.

    Returns (model, scores, metadata) where scores are aligned with
    the concatenation of metadata_P + metadata_U.
    """
    X_train = np.vstack([X_P, X_U])
    y_train = np.concatenate([
        np.ones(len(X_P), dtype=np.int64),
        np.zeros(len(X_U), dtype=np.int64),
    ])
    all_metadata = list(metadata_P) + list(metadata_U)

    model = XGBoostDefectModel(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)

    return model, proba.tolist(), all_metadata


def predict_biased_pu_scores(
    model: XGBoostDefectModel,
    X: np.ndarray,
) -> list[float]:
    """Score test files using the trained biased P-vs-U model."""
    return model.predict_proba(X).tolist()


# ---------------------------------------------------------------------------
# B3: Evidence-ranking pairwise
# ---------------------------------------------------------------------------

@dataclass
class PairwiseRankingModel:
    """Pairwise logistic ranking model.

    Learns score(FILE_STRONG) > score(FILE_WEAK) using logistic loss
    on the score difference.
    """
    learning_rate: float = 0.1
    n_iterations: int = 500
    l2_reg: float = 0.01
    random_state: int = RANDOM_SEED
    weights: np.ndarray | None = field(default=None, repr=False)
    bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(
        self,
        X_strong: np.ndarray,
        X_weak: np.ndarray,
    ) -> None:
        """Fit pairwise ranking from strong vs weak feature pairs."""
        n_features = X_strong.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        rng = np.random.RandomState(self.random_state)
        n_pairs = len(X_strong)

        for _ in range(self.n_iterations):
            indices = rng.permutation(n_pairs)
            for idx in indices:
                diff = X_strong[idx] - X_weak[idx]
                margin = np.dot(self.weights, diff) + self.bias
                prob = self._sigmoid(margin)
                gradient = (1.0 - prob) * diff - self.l2_reg * self.weights
                self.weights += self.learning_rate * gradient
                self.bias += self.learning_rate * (1.0 - prob)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score feature vectors. Higher = stronger evidence."""
        if self.weights is None:
            raise RuntimeError("Model not fitted")
        return X @ self.weights + self.bias

    def predict_pairwise_accuracy(
        self,
        X_strong: np.ndarray,
        X_weak: np.ndarray,
    ) -> float:
        """Fraction of pairs where score(strong) > score(weak)."""
        s_strong = self.score(X_strong)
        s_weak = self.score(X_weak)
        correct = np.sum(s_strong > s_weak)
        return float(correct / len(X_strong)) if len(X_strong) > 0 else 0.5


# ---------------------------------------------------------------------------
# B4: Commit-level fallback
# ---------------------------------------------------------------------------

def train_commit_level_model(
    X_train: np.ndarray,
    metadata_train: list[dict],
) -> tuple[XGBoostDefectModel, list[float], list[dict]]:
    """Train commit-level XGBoost model.

    Labels:
        observed-positive commit: >= 1 FILE_STRONG file
        observed-unlabeled commit: 0 FILE_STRONG files

    All files in a commit share the commit-level label.

    Returns (model, commit_scores, unique_commit_metadata).
    """
    commit_data: dict[tuple[str, str], dict] = {}
    for i, m in enumerate(metadata_train):
        key = (m["repo_name"], m["commit_sha"])
        if key not in commit_data:
            commit_data[key] = {"indices": [], "repo_name": m["repo_name"],
                                "commit_sha": m["commit_sha"]}
        commit_data[key]["indices"].append(i)

    commits = sorted(commit_data.keys())
    X_commit = np.zeros((len(commits), X_train.shape[1]), dtype=np.float64)
    y_commit = np.zeros(len(commits), dtype=np.int64)
    commit_meta: list[dict] = []

    for c_idx, c_key in enumerate(commits):
        c_data = commit_data[c_key]
        file_indices = c_data["indices"]
        X_commit[c_idx] = X_train[file_indices].mean(axis=0)
        commit_meta.append({
            "repo_name": c_data["repo_name"],
            "commit_sha": c_data["commit_sha"],
            "n_files": len(file_indices),
        })

    has_file_strong = set()
    for i, m in enumerate(metadata_train):
        if m.get("observed_positive", False):
            has_file_strong.add((m["repo_name"], m["commit_sha"]))

    for c_idx, c_key in enumerate(commits):
        if c_key in has_file_strong:
            y_commit[c_idx] = 1

    model = XGBoostDefectModel(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_commit, y_commit)
    commit_scores = model.predict_proba(X_commit).tolist()

    return model, commit_scores, commit_meta


def predict_commit_level_scores(
    model: XGBoostDefectModel,
    X: np.ndarray,
    metadata: list[dict],
) -> list[float]:
    """Predict commit-level scores for test files.

    All files in a commit receive the same commit-level score.
    """
    commit_features: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        commit_features[(m["repo_name"], m["commit_sha"])].append(i)

    commit_X = {}
    for c_key, indices in commit_features.items():
        commit_X[c_key] = X[indices].mean(axis=0).reshape(1, -1)

    commit_scores_map = {}
    for c_key, c_X in commit_X.items():
        commit_scores_map[c_key] = float(model.predict_proba(c_X)[0])

    scores = np.zeros(len(metadata), dtype=np.float64)
    for i, m in enumerate(metadata):
        c_key = (m["repo_name"], m["commit_sha"])
        scores[i] = commit_scores_map.get(c_key, 0.0)
    return scores.tolist()
