"""Tests for Phase 4 dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.features.schemas import COMMIT_FEATURE_NAMES, FILE_FEATURE_NAMES
from backend.app.ml.dataset_loader import (
    FEATURE_NAMES,
    NUM_FEATURES,
    load_dataset,
    load_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    commit_sha: str = "a" * 40,
    file_path: str = "src/main.py",
    defect_label: int = 0,
    label_status: str = "negative",
    split: str = "train",
    commit_features: list[float] | None = None,
    file_features: list[float] | None = None,
) -> dict:
    """Build a minimal DatasetRow-compatible dict."""
    if commit_features is None:
        commit_features = [0.0] * len(COMMIT_FEATURE_NAMES)
    if file_features is None:
        file_features = [0.0] * len(FILE_FEATURE_NAMES)
    return {
        "commit_sha": commit_sha,
        "file_path": file_path,
        "defect_label": defect_label,
        "label_status": label_status,
        "label_confidence": 1.0 if defect_label == 1 else 0.0,
        "label_source": "explicit_sha_reference" if defect_label == 1 else "none",
        "split": split,
        "commit_features": commit_features,
        "file_features": file_features,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureContract:
    def test_feature_count_is_45(self):
        assert NUM_FEATURES == 45

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 45

    def test_commit_features_first(self):
        assert FEATURE_NAMES[:29] == list(COMMIT_FEATURE_NAMES)

    def test_file_features_last(self):
        assert FEATURE_NAMES[29:] == [f"file_{f}" for f in FILE_FEATURE_NAMES]

    def test_no_duplicates(self):
        assert len(set(FEATURE_NAMES)) == 45


class TestLoadSplit:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        rows = [
            _make_row(commit_sha="a" * 40, file_path="a.py", defect_label=0),
            _make_row(commit_sha="b" * 40, file_path="b.py", defect_label=1),
        ]
        _write_jsonl(tmp_path / "train.jsonl", rows)

        ds = load_split(tmp_path / "train.jsonl")
        assert len(ds) == 2
        assert ds.X.shape == (2, 45)
        assert ds.y.tolist() == [0, 1]

    def test_feature_vector_concatenation(self, tmp_path: Path):
        cf = [float(i) for i in range(29)]
        ff = [float(i + 100) for i in range(16)]
        rows = [_make_row(commit_features=cf, file_features=ff)]
        _write_jsonl(tmp_path / "train.jsonl", rows)

        ds = load_split(tmp_path / "train.jsonl")
        expected = cf + ff
        np.testing.assert_array_almost_equal(ds.X[0], expected)

    def test_metadata_preserved(self, tmp_path: Path):
        rows = [_make_row(commit_sha="abc123", file_path="test.py", split="train")]
        _write_jsonl(tmp_path / "train.jsonl", rows)

        ds = load_split(tmp_path / "train.jsonl")
        assert ds.metadata[0]["commit_sha"] == "abc123"
        assert ds.metadata[0]["file_path"] == "test.py"
        assert ds.metadata[0]["split"] == "train"

    def test_feature_names_accessible(self, tmp_path: Path):
        _write_jsonl(tmp_path / "train.jsonl", [_make_row()])
        ds = load_split(tmp_path / "train.jsonl")
        assert ds.feature_names == FEATURE_NAMES

    def test_positive_negative_counts(self, tmp_path: Path):
        rows = [
            _make_row(defect_label=0),
            _make_row(defect_label=0),
            _make_row(defect_label=1),
        ]
        _write_jsonl(tmp_path / "train.jsonl", rows)
        ds = load_split(tmp_path / "train.jsonl")
        assert ds.positive_count == 1
        assert ds.negative_count == 2
        assert ds.positive_rate == pytest.approx(1 / 3)


class TestAmbiguousRejection:
    def test_rejects_ambiguous_rows(self, tmp_path: Path):
        rows = [_make_row(defect_label=-1, label_status="ambiguous")]
        _write_jsonl(tmp_path / "train.jsonl", rows)
        with pytest.raises(ValueError, match="ambiguous"):
            load_split(tmp_path / "train.jsonl")

    def test_rejects_ambiguous_among_valid(self, tmp_path: Path):
        rows = [
            _make_row(defect_label=0),
            _make_row(defect_label=-1, label_status="ambiguous"),
            _make_row(defect_label=1),
        ]
        _write_jsonl(tmp_path / "train.jsonl", rows)
        with pytest.raises(ValueError, match="1 ambiguous"):
            load_split(tmp_path / "train.jsonl")


class TestDimensionValidation:
    def test_rejects_wrong_commit_feature_count(self, tmp_path: Path):
        rows = [_make_row(commit_features=[1.0] * 28)]  # wrong: 28 not 29
        _write_jsonl(tmp_path / "train.jsonl", rows)
        with pytest.raises(ValueError, match="commit_features has 28"):
            load_split(tmp_path / "train.jsonl")

    def test_rejects_wrong_file_feature_count(self, tmp_path: Path):
        rows = [_make_row(file_features=[1.0] * 15)]  # wrong: 15 not 16
        _write_jsonl(tmp_path / "train.jsonl", rows)
        with pytest.raises(ValueError, match="file_features has 15"):
            load_split(tmp_path / "train.jsonl")


class TestEmptySplit:
    def test_empty_file_returns_empty_dataset(self, tmp_path: Path):
        _write_jsonl(tmp_path / "train.jsonl", [])
        ds = load_split(tmp_path / "train.jsonl")
        assert len(ds) == 0
        assert ds.X.shape == (0, 45)
        assert ds.y.shape == (0,)
        assert ds.positive_count == 0
        assert ds.negative_count == 0


class TestLoadDataset:
    def test_loads_all_three_splits(self, tmp_path: Path):
        _write_jsonl(tmp_path / "train.jsonl", [_make_row(defect_label=0)])
        _write_jsonl(tmp_path / "validation.jsonl", [_make_row(defect_label=0)])
        _write_jsonl(tmp_path / "test.jsonl", [_make_row(defect_label=1)])

        train, val, test = load_dataset(tmp_path)
        assert len(train) == 1
        assert len(val) == 1
        assert len(test) == 1
        assert test.y[0] == 1

    def test_preserves_chronological_isolation(self, tmp_path: Path):
        rows_train = [_make_row(commit_sha=f"{i:040d}", defect_label=0) for i in range(5)]
        rows_val = [_make_row(commit_sha=f"{i+100:040d}", defect_label=0) for i in range(3)]
        rows_test = [_make_row(commit_sha=f"{i+200:040d}", defect_label=1) for i in range(2)]

        _write_jsonl(tmp_path / "train.jsonl", rows_train)
        _write_jsonl(tmp_path / "validation.jsonl", rows_val)
        _write_jsonl(tmp_path / "test.jsonl", rows_test)

        train, val, test = load_dataset(tmp_path)
        train_shas = {m["commit_sha"] for m in train.metadata}
        val_shas = {m["commit_sha"] for m in val.metadata}
        test_shas = {m["commit_sha"] for m in test.metadata}
        # No overlap
        assert len(train_shas & val_shas) == 0
        assert len(train_shas & test_shas) == 0
        assert len(val_shas & test_shas) == 0
