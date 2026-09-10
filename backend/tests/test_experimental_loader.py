"""Tests for experimental dataset loader (Phase 4.5a).

Verifies E0/E1/E2/E4 vector construction, backward compatibility,
and ablation mode correctness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
)
from backend.app.ml.dataset_loader import (
    COMMIT_FEATURE_COUNT,
    FEATURE_NAMES,
    FILE_FEATURE_COUNT,
    NUM_FEATURES,
    load_split,
)
from backend.app.ml.experimental_loader import (
    _get_mode_feature_names,
    _get_mode_indices,
    load_experimental_split,
)

# -- Helpers -----------------------------------------------------------------


def _make_v1_row(
    commit_sha: str = "a" * 40,
    file_path: str = "src/main.py",
    defect_label: int = 0,
    label_status: str = "negative",
    split: str = "train",
) -> dict:
    """Create a minimal v1 DatasetRow as a dict."""
    return {
        "dataset_version": "v1",
        "repo_url": "",
        "repo_name": "test-repo",
        "commit_sha": commit_sha,
        "commit_timestamp": "2025-01-01T00:00:00+00:00",
        "commit_message": "test commit",
        "author": "tester",
        "file_path": file_path,
        "file_status": "modified",
        "feature_version": "v1",
        "commit_features": [0.0] * COMMIT_FEATURE_COUNT,
        "file_features": [0.0] * FILE_FEATURE_COUNT,
        "label_status": label_status,
        "defect_label": defect_label,
        "label_confidence": 0.0,
        "label_source": "none",
        "label_evidence": "",
        "split": split,
    }


def _make_exp_row(
    commit_sha: str = "a" * 40,
    file_path: str = "src/main.py",
    features: list[float] | None = None,
) -> dict:
    """Create an experimental features row as a dict."""
    return {
        "commit_sha": commit_sha,
        "file_path": file_path,
        "experimental_features": features or [0.0] * EXPERIMENTAL_FEATURE_COUNT,
        "experimental_feature_version": "exp-4.5a",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def v1_dir():
    """Create a temporary directory with v1 JSONL files."""
    tmpdir = tempfile.mkdtemp()
    data_dir = Path(tmpdir) / "data"
    data_dir.mkdir()

    rows = [
        _make_v1_row(commit_sha="a" * 40, file_path="a.py", split="train"),
        _make_v1_row(commit_sha="b" * 40, file_path="b.py", split="train"),
        _make_v1_row(commit_sha="c" * 40, file_path="c.py", split="validation"),
        _make_v1_row(commit_sha="d" * 40, file_path="d.py", split="test"),
    ]
    _write_jsonl(data_dir / "train.jsonl", rows[:2])
    _write_jsonl(data_dir / "validation.jsonl", rows[2:3])
    _write_jsonl(data_dir / "test.jsonl", rows[3:4])

    yield data_dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def exp_dir():
    """Create a temporary directory with experimental features."""
    tmpdir = tempfile.mkdtemp()
    exp_path = Path(tmpdir)

    rows = [
        _make_exp_row(
            commit_sha="a" * 40, file_path="a.py",
            features=[1.0, 0.0, 1.0, 5.0, 2.0, 30.0],
        ),
        _make_exp_row(
            commit_sha="b" * 40, file_path="b.py",
            features=[0.0, 1.0, 0.0, 10.0, 0.0, 0.0],
        ),
        _make_exp_row(
            commit_sha="c" * 40, file_path="c.py",
            features=[2.0, 0.0, 0.0, 3.0, 1.0, 15.0],
        ),
        _make_exp_row(
            commit_sha="d" * 40, file_path="d.py",
            features=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
    ]
    _write_jsonl(exp_path / "experimental_features.jsonl", rows)

    yield exp_path
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# -- Mode index tests --------------------------------------------------------


class TestModeIndices:
    def test_e0_indices(self):
        assert _get_mode_indices("E0") == []

    def test_e1_indices(self):
        assert _get_mode_indices("E1") == [0, 1, 2]

    def test_e2_indices(self):
        assert _get_mode_indices("E2") == [3, 4, 5]

    def test_e4_indices(self):
        assert _get_mode_indices("E4") == [0, 1, 2, 3, 4, 5]

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown ablation mode"):
            _get_mode_indices("E99")


class TestModeFeatureNames:
    def test_e0_names(self):
        names = _get_mode_feature_names("E0")
        assert len(names) == NUM_FEATURES
        assert names == FEATURE_NAMES

    def test_e1_names(self):
        names = _get_mode_feature_names("E1")
        assert len(names) == NUM_FEATURES + 3
        assert names[:NUM_FEATURES] == FEATURE_NAMES
        assert names[NUM_FEATURES:] == [
            "file_ast_functions_added",
            "file_ast_functions_deleted",
            "file_ast_try_except_changed",
        ]

    def test_e4_names(self):
        names = _get_mode_feature_names("E4")
        assert len(names) == NUM_FEATURES + 6
        assert names[NUM_FEATURES:] == EXPERIMENTAL_FEATURE_NAMES


# -- Loading tests -----------------------------------------------------------


class TestLoadExperimentalSplit:
    def test_e0_reproduces_v1(self, v1_dir):
        """E0 produces identical vectors to the frozen v1 loader."""
        v1_ds = load_split(v1_dir / "train.jsonl")
        exp_ds = load_experimental_split(
            v1_dir / "train.jsonl",
            v1_dir / "nonexistent.jsonl",  # no experimental file
            mode="E0",
        )
        assert v1_ds.X.shape == exp_ds.X.shape
        assert np.array_equal(v1_ds.X, exp_ds.X)
        assert v1_ds.y.tolist() == exp_ds.y.tolist()

    def test_e1_dimension(self, v1_dir, exp_dir):
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        ds = load_experimental_split(train, exp, mode="E1")
        assert ds.X.shape[1] == NUM_FEATURES + 3

    def test_e2_dimension(self, v1_dir, exp_dir):
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        ds = load_experimental_split(train, exp, mode="E2")
        assert ds.X.shape[1] == NUM_FEATURES + 3

    def test_e4_dimension(self, v1_dir, exp_dir):
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        ds = load_experimental_split(train, exp, mode="E4")
        assert ds.X.shape[1] == NUM_FEATURES + 6

    def test_v1_features_unchanged_in_all_modes(self, v1_dir, exp_dir):
        """First 45 features are identical across all modes."""
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        e0 = load_experimental_split(train, exp, mode="E0")
        e1 = load_experimental_split(train, exp, mode="E1")
        e4 = load_experimental_split(train, exp, mode="E4")

        assert np.array_equal(e0.X, e1.X[:, :NUM_FEATURES])
        assert np.array_equal(e0.X, e4.X[:, :NUM_FEATURES])

    def test_missing_experimental_row_gets_zeros(self, v1_dir):
        """Rows without experimental features get zeros in E1 mode."""
        exp_path = v1_dir / "empty_exp.jsonl"
        _write_jsonl(exp_path, [])  # no experimental data

        ds = load_experimental_split(v1_dir / "train.jsonl", exp_path, mode="E1")
        # Last 3 columns should all be zero
        assert np.all(ds.X[:, NUM_FEATURES:] == 0.0)

    def test_feature_names_accessible(self, v1_dir, exp_dir):
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        ds = load_experimental_split(train, exp, mode="E4")
        assert len(ds.feature_names) == NUM_FEATURES + 6

    def test_metadata_preserved(self, v1_dir, exp_dir):
        train = v1_dir / "train.jsonl"
        exp = exp_dir / "experimental_features.jsonl"
        ds = load_experimental_split(train, exp, mode="E1")
        assert len(ds.metadata) == 2
        assert ds.metadata[0]["commit_sha"] == "a" * 40

    def test_positive_count(self, v1_dir, exp_dir):
        rows = [
            _make_v1_row(
                commit_sha="a" * 40, file_path="a.py",
                defect_label=1, label_status="positive",
                split="train",
            ),
        ]
        _write_jsonl(v1_dir / "train_pos.jsonl", rows)

        exp_rows = [
            _make_exp_row(commit_sha="a" * 40, file_path="a.py", features=[1.0] * 6),
        ]
        exp_pos = v1_dir / "exp_pos.jsonl"
        _write_jsonl(exp_pos, exp_rows)

        ds = load_experimental_split(v1_dir / "train_pos.jsonl", exp_pos, mode="E1")
        assert ds.positive_count == 1

    def test_empty_split(self, v1_dir):
        _write_jsonl(v1_dir / "empty.jsonl", [])
        ds = load_experimental_split(
            v1_dir / "empty.jsonl",
            v1_dir / "nonexistent.jsonl",
            mode="E1",
        )
        assert len(ds) == 0
        assert ds.X.shape == (0, NUM_FEATURES + 3)


# -- Backward compatibility -------------------------------------------------


class TestBackwardCompatibility:
    def test_v1_load_split_unchanged(self, v1_dir):
        """The existing load_split() function works on v1 data."""
        ds = load_split(v1_dir / "train.jsonl")
        assert ds.X.shape == (2, NUM_FEATURES)
        assert len(ds.feature_names) == NUM_FEATURES

    def test_v1_feature_names_unchanged(self):
        """v1 FEATURE_NAMES are unchanged."""
        assert len(FEATURE_NAMES) == 45
        assert FEATURE_NAMES[0] == "lines_added"
        assert FEATURE_NAMES[28] == "change_entropy"
        assert FEATURE_NAMES[29] == "file_language"
        assert FEATURE_NAMES[44] == "file_avg_hunk_size"
