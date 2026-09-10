"""Tests for Phase 4 feature importance and artifact I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.features.schemas import COMMIT_FEATURE_NAMES, FILE_FEATURE_NAMES
from backend.app.ml.artifacts import save_artifacts
from backend.app.ml.dataset_loader import FEATURE_NAMES
from backend.app.ml.importance import (
    extract_importance,
    format_importance_report,
)
from backend.app.ml.metrics import compute_metrics
from backend.app.ml.models import (
    MajorityClassBaseline,
    WeightedLogisticRegression,
    XGBoostDefectModel,
)


@pytest.fixture
def fitted_xgb():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 45)
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    model = XGBoostDefectModel(n_estimators=10, max_depth=3)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_lr():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 45)
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    model = WeightedLogisticRegression()
    model.fit(X, y)
    return model, X, y


class TestFeatureImportance:
    def test_xgb_importance_length(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        assert len(imp) == 45

    def test_lr_importance_length(self, fitted_lr):
        model, _, _ = fitted_lr
        imp = extract_importance(model, FEATURE_NAMES)
        assert len(imp) == 45

    def test_sorted_descending(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        values = [i.importance for i in imp]
        assert values == sorted(values, reverse=True)

    def test_ranks_sequential(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        ranks = [i.rank for i in imp]
        assert ranks == list(range(1, 46))

    def test_feature_names_match(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        names = [i.feature_name for i in imp]
        assert set(names) == set(FEATURE_NAMES)

    def test_all_importances_non_negative(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        for fi in imp:
            assert fi.importance >= 0.0

    def test_baseline_returns_empty(self):
        m = MajorityClassBaseline()
        m.fit(np.zeros((5, 45)), np.array([0, 0, 0, 1, 1]))
        imp = extract_importance(m, FEATURE_NAMES)
        assert imp == []

    def test_format_report(self, fitted_xgb):
        model, _, _ = fitted_xgb
        imp = extract_importance(model, FEATURE_NAMES)
        report = format_importance_report(imp, top_n=5)
        assert "Feature Importance" in report
        assert "Rank" in report


class TestArtifactSaving:
    def test_saves_all_files(self, tmp_path: Path, fitted_xgb):
        model, X, y = fitted_xgb
        m = compute_metrics(y, model.predict_proba(X))

        save_artifacts(
            output_dir=tmp_path,
            model_version="v0.1.0-test",
            model=model,
            feature_names=FEATURE_NAMES,
            dataset_version="v1",
            feature_version="v1",
            train_count=80, val_count=10, test_count=10,
            train_pos=8, train_neg=72,
            val_pos=2, val_neg=8,
            test_pos=0, test_neg=10,
            metrics=m.to_dict(),
        )

        model_dir = tmp_path / "v0.1.0-test"
        assert model_dir.exists()
        assert (model_dir / "model.json").exists()
        assert (model_dir / "feature_schema.json").exists()
        assert (model_dir / "metrics.json").exists()
        assert (model_dir / "metadata.json").exists()

    def test_feature_schema_integrity(self, tmp_path: Path, fitted_xgb):
        model, X, y = fitted_xgb
        m = compute_metrics(y, model.predict_proba(X))

        save_artifacts(
            output_dir=tmp_path,
            model_version="v0.1.0",
            model=model,
            feature_names=FEATURE_NAMES,
            dataset_version="v1",
            feature_version="v1",
            train_count=80, val_count=10, test_count=10,
            train_pos=8, train_neg=72,
            val_pos=2, val_neg=8,
            test_pos=0, test_neg=10,
            metrics=m.to_dict(),
        )

        fs = json.loads((tmp_path / "v0.1.0" / "feature_schema.json").read_text())
        assert fs["num_features"] == 45
        assert len(fs["feature_names"]) == 45
        assert fs["commit_features"] == list(COMMIT_FEATURE_NAMES)
        assert fs["file_features"] == [f"file_{f}" for f in FILE_FEATURE_NAMES]

    def test_metadata_contains_counts(self, tmp_path: Path, fitted_xgb):
        model, X, y = fitted_xgb
        m = compute_metrics(y, model.predict_proba(X))

        save_artifacts(
            output_dir=tmp_path,
            model_version="v0.1.0",
            model=model,
            feature_names=FEATURE_NAMES,
            dataset_version="v1",
            feature_version="v1",
            train_count=80, val_count=10, test_count=10,
            train_pos=8, train_neg=72,
            val_pos=2, val_neg=8,
            test_pos=0, test_neg=10,
            metrics=m.to_dict(),
        )

        meta = json.loads((tmp_path / "v0.1.0" / "metadata.json").read_text())
        assert meta["train_examples"] == 80
        assert meta["validation_examples"] == 10
        assert meta["test_examples"] == 10
        assert meta["feature_count"] == 45
        assert meta["model_type"] == "xgboost_defect_model"

    def test_predictions_jsonl_sorted(self, tmp_path: Path, fitted_xgb):
        model, X, y = fitted_xgb
        preds = [
            {"commit_sha": "b" * 40, "file_path": "z.py", "predicted_probability": 0.9},
            {"commit_sha": "a" * 40, "file_path": "a.py", "predicted_probability": 0.1},
            {"commit_sha": "a" * 40, "file_path": "b.py", "predicted_probability": 0.5},
        ]

        save_artifacts(
            output_dir=tmp_path,
            model_version="v0.1.0",
            model=model,
            feature_names=FEATURE_NAMES,
            dataset_version="v1",
            feature_version="v1",
            train_count=0, val_count=0, test_count=3,
            train_pos=0, train_neg=0,
            val_pos=0, val_neg=0,
            test_pos=0, test_neg=3,
            metrics={},
            predictions=preds,
        )

        lines = (tmp_path / "v0.1.0" / "predictions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["commit_sha"] < second["commit_sha"] or (
            first["commit_sha"] == second["commit_sha"]
            and first["file_path"] <= second["file_path"]
        )
