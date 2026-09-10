"""Dataset loading layer for Phase 4 ML training.

Loads JSONL files produced by Phase 3, validates rows against DatasetRow,
combines 29 commit features + 16 file features into a 45-dimensional input
vector, and preserves feature names/order in explicit metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from backend.app.dataset.schemas import DatasetRow
from backend.app.features.schemas import COMMIT_FEATURE_NAMES, FILE_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Canonical 45-feature ordering: 29 commit + 16 file
# File features are prefixed with "file_" to avoid name collisions
# (e.g. lines_added appears in both commit and file features).
_RAW_COMMIT_FEATURES: list[str] = list(COMMIT_FEATURE_NAMES)
_RAW_FILE_FEATURES: list[str] = list(FILE_FEATURE_NAMES)

FEATURE_NAMES: list[str] = _RAW_COMMIT_FEATURES + [f"file_{f}" for f in _RAW_FILE_FEATURES]
NUM_FEATURES: int = len(FEATURE_NAMES)  # 45
COMMIT_FEATURE_COUNT: int = len(COMMIT_FEATURE_NAMES)  # 29
FILE_FEATURE_COUNT: int = len(FILE_FEATURE_NAMES)  # 16

assert NUM_FEATURES == 45, f"Expected 45 features, got {NUM_FEATURES}"


class SupervisedDataset:
    """Container for a single supervised split (train, validation, or test).

    Attributes
    ----------
    X : np.ndarray of shape (n_samples, 45)
        Combined feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary labels: 1 = positive (defect), 0 = negative.
    metadata : list[dict]
        Per-row metadata (commit_sha, file_path, split, etc.).
    feature_names : list[str]
        Ordered feature names matching the columns of X.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: list[dict],
        feature_names: list[str],
    ) -> None:
        self.X = X
        self.y = y
        self.metadata = metadata
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.y)

    @property
    def positive_count(self) -> int:
        return int(np.sum(self.y == 1))

    @property
    def negative_count(self) -> int:
        return int(np.sum(self.y == 0))

    @property
    def positive_rate(self) -> float:
        return self.positive_count / len(self.y) if len(self.y) > 0 else 0.0


def load_split(path: Path) -> SupervisedDataset:
    """Load a single JSONL split file into a SupervisedDataset.

    Parameters
    ----------
    path : Path
        Path to a JSONL file (train.jsonl, validation.jsonl, or test.jsonl).

    Returns
    -------
    SupervisedDataset
        Validated features, labels, and metadata.

    Raises
    ------
    ValueError
        If any row has defect_label == -1 (ambiguous) or feature
        dimensions are incorrect.
    """
    rows: list[DatasetRow] = []
    raw_lines: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw_lines.append(line)
            try:
                obj = json.loads(line)
                row = DatasetRow.model_validate(obj)
            except Exception as e:
                raise ValueError(f"Invalid row at line {line_no} in {path}: {e}") from e
            rows.append(row)

    if not rows:
        return SupervisedDataset(
            X=np.empty((0, NUM_FEATURES), dtype=np.float64),
            y=np.empty(0, dtype=np.int64),
            metadata=[],
            feature_names=list(FEATURE_NAMES),
        )

    # Validate: no ambiguous rows in supervised splits
    ambiguous_rows = [r for r in rows if r.defect_label == -1]
    if ambiguous_rows:
        raise ValueError(
            f"Found {len(ambiguous_rows)} ambiguous rows (defect_label=-1) "
            f"in supervised split {path.name}. This should never happen."
        )

    # Validate feature dimensions
    X_rows: list[list[float]] = []
    metadata: list[dict] = []
    y_labels: list[int] = []

    for i, row in enumerate(rows):
        cf = row.commit_features
        ff = row.file_features

        if len(cf) != COMMIT_FEATURE_COUNT:
            raise ValueError(
                f"Row {i}: commit_features has {len(cf)} elements, "
                f"expected {COMMIT_FEATURE_COUNT}"
            )
        if len(ff) != FILE_FEATURE_COUNT:
            raise ValueError(
                f"Row {i}: file_features has {len(ff)} elements, "
                f"expected {FILE_FEATURE_COUNT}"
            )

        # Combine into 45-dimensional vector (commit || file)
        X_rows.append(cf + ff)
        y_labels.append(row.defect_label)
        metadata.append({
            "commit_sha": row.commit_sha,
            "file_path": row.file_path,
            "split": row.split,
            "label_status": row.label_status,
            "label_source": row.label_source,
            "label_confidence": row.label_confidence,
            "label_evidence": row.label_evidence,
            "commit_message": row.commit_message,
            "author": row.author,
            "file_status": row.file_status,
            "repo_name": row.repo_name,
        })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)

    return SupervisedDataset(
        X=X,
        y=y,
        metadata=metadata,
        feature_names=list(FEATURE_NAMES),
    )


def load_dataset(
    data_dir: Path,
) -> tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    """Load train, validation, and test splits from a dataset directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.jsonl, validation.jsonl, test.jsonl.

    Returns
    -------
    tuple of (train, val, test) SupervisedDataset instances.
    """
    train = load_split(data_dir / "train.jsonl")
    val = load_split(data_dir / "validation.jsonl")
    test = load_split(data_dir / "test.jsonl")

    logger.info(
        "Loaded dataset: train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
        len(train), train.positive_count,
        len(val), val.positive_count,
        len(test), test.positive_count,
    )

    return train, val, test
