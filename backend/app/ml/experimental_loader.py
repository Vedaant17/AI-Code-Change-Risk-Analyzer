"""Experimental dataset loader for Phase 4.5a ablation experiments.

Loads the frozen v1 45-feature JSONL data and combines it with
experimental features to construct feature vectors for ablation
experiments (E0/E1/E2/E4).  The frozen ``DatasetRow`` schema and
the existing ``load_split()`` function are not modified.

Ablation modes
--------------
E0 : v1 45 features only (identical to ``load_split()``)
E1 : E0 + 3 AST features (indices 0-2 of experimental)
E2 : E0 + 3 historical features (indices 3-5 of experimental)
E4 : E0 + all 6 experimental features

Feature ordering
----------------
v1 features are never reordered.  Experimental features are appended.
This guarantees E0 reproduces the exact 45-feature vector from existing
combined-v3 data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from backend.app.dataset.schemas import DatasetRow
from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
)
from backend.app.ml.dataset_loader import (
    COMMIT_FEATURE_COUNT,
    FEATURE_NAMES,
    FILE_FEATURE_COUNT,
    NUM_FEATURES,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)

# Experimental feature subsets for each ablation mode
_AST_INDICES = [0, 1, 2]         # file_ast_functions_added/deleted/try_except
_HISTORICAL_INDICES = [3, 4, 5]  # file_commit_count/bug_fixes/days_since


def _build_experimental_lookup(
    exp_path: Path,
) -> dict[tuple[str, str], list[float]]:
    """Build a lookup dict from (commit_sha, file_path) -> experimental vector.

    Parameters
    ----------
    exp_path:
        Path to a JSONL file where each line has keys:
        ``commit_sha``, ``file_path``, ``experimental_features``.

    Returns
    -------
    dict
        Mapping from (commit_sha, file_path) to list of 6 floats.
    """
    lookup: dict[tuple[str, str], list[float]] = {}
    if not exp_path.exists():
        return lookup

    with open(exp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["commit_sha"], obj["file_path"])
            lookup[key] = obj["experimental_features"]
    return lookup


def _get_mode_indices(mode: str) -> list[int]:
    """Return the indices into the experimental vector for the given mode."""
    if mode == "E0":
        return []
    elif mode == "E1":
        return _AST_INDICES
    elif mode == "E2":
        return _HISTORICAL_INDICES
    elif mode == "E4":
        return list(range(EXPERIMENTAL_FEATURE_COUNT))
    else:
        raise ValueError(f"Unknown ablation mode: {mode!r}. Expected E0, E1, E2, or E4.")


def _get_mode_feature_names(mode: str) -> list[str]:
    """Return the full feature name list for the given mode."""
    indices = _get_mode_indices(mode)
    exp_names = [EXPERIMENTAL_FEATURE_NAMES[i] for i in indices]
    return FEATURE_NAMES + exp_names


def load_experimental_split(
    v1_path: Path,
    exp_path: Path,
    mode: str = "E1",
) -> SupervisedDataset:
    """Load a single split with experimental features.

    Parameters
    ----------
    v1_path:
        Path to the frozen v1 JSONL file (train.jsonl, etc.).
    exp_path:
        Path to the experimental features JSONL file.
    mode:
        Ablation mode: E0, E1, E2, or E4.

    Returns
    -------
    SupervisedDataset
        Feature matrix with v1 features + selected experimental features.
    """
    exp_lookup = _build_experimental_lookup(exp_path)
    mode_indices = _get_mode_indices(mode)
    mode_feature_names = _get_mode_feature_names(mode)
    mode_dim = NUM_FEATURES + len(mode_indices)

    rows: list[DatasetRow] = []
    with open(v1_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row = DatasetRow.model_validate(obj)
            except Exception as e:
                raise ValueError(
                    f"Invalid row at line {line_no} in {v1_path}: {e}"
                ) from e
            rows.append(row)

    if not rows:
        return SupervisedDataset(
            X=np.empty((0, mode_dim), dtype=np.float64),
            y=np.empty(0, dtype=np.int64),
            metadata=[],
            feature_names=mode_feature_names,
        )

    # Validate no ambiguous rows
    ambiguous_rows = [r for r in rows if r.defect_label == -1]
    if ambiguous_rows:
        raise ValueError(
            f"Found {len(ambiguous_rows)} ambiguous rows in {v1_path.name}"
        )

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

        # v1 45-dimensional vector (commit || file)
        base_vec = cf + ff

        # Append experimental features for the selected mode
        key = (row.commit_sha, row.file_path)
        exp_vec = exp_lookup.get(key, [0.0] * EXPERIMENTAL_FEATURE_COUNT)
        selected_exp = [exp_vec[i] for i in mode_indices]

        X_rows.append(base_vec + selected_exp)
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
        feature_names=mode_feature_names,
    )


def load_experimental_dataset(
    data_dir: Path,
    exp_dir: Path,
    mode: str = "E1",
) -> tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    """Load train, validation, and test splits with experimental features.

    Parameters
    ----------
    data_dir:
        Directory containing v1 train.jsonl, validation.jsonl, test.jsonl.
    exp_dir:
        Directory containing experimental_features.jsonl.
    mode:
        Ablation mode: E0, E1, E2, or E4.

    Returns
    -------
    tuple of (train, val, test) SupervisedDataset instances.
    """
    exp_path = exp_dir / "experimental_features.jsonl"

    train = load_experimental_split(data_dir / "train.jsonl", exp_path, mode)
    val = load_experimental_split(data_dir / "validation.jsonl", exp_path, mode)
    test = load_experimental_split(data_dir / "test.jsonl", exp_path, mode)

    logger.info(
        "Loaded experimental dataset (mode=%s): "
        "train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d), dim=%d",
        mode,
        len(train), train.positive_count,
        len(val), val.positive_count,
        len(test), test.positive_count,
        train.X.shape[1],
    )

    return train, val, test
