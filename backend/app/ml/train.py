"""Phase 4 training script.

Trains and evaluates defect prediction models using the full ML pipeline.
If a real Phase 3 dataset is available, uses it; otherwise uses a synthetic
dataset with realistic class imbalance (2% positive rate) for pipeline validation.

Outputs:
  data/models/<model_version>/  — all artifacts
  Console output — metrics, feature importance, limitations
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.ml.artifacts import save_artifacts  # noqa: E402
from backend.app.ml.dataset_loader import (  # noqa: E402
    FEATURE_NAMES,
    NUM_FEATURES,
    SupervisedDataset,
    load_dataset,
)
from backend.app.ml.importance import extract_importance, format_importance_report  # noqa: E402
from backend.app.ml.metrics import compute_baseline, compute_metrics  # noqa: E402
from backend.app.ml.models import (  # noqa: E402
    MajorityClassBaseline,
    WeightedLogisticRegression,
    XGBoostDefectModel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    n_train: int = 500, n_val: int = 100, n_test: int = 100,
    positive_rate: float = 0.02,
) -> tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    """Generate a synthetic dataset mimicking real defect distribution.

    Uses realistic feature distributions with signal injected so the model
    can learn meaningful patterns (not random noise).
    """
    rng = np.random.RandomState(42)

    def _make_split(n: int) -> SupervisedDataset:
        X = rng.randn(n, NUM_FEATURES).astype(np.float64)
        # Inject signal: a few features correlate with defect label
        # Feature 3 (files_changed), Feature 9 (total_hunks), Feature 14 (test_ratio)
        n_pos = max(1, int(n * positive_rate))
        y = np.zeros(n, dtype=np.int64)
        pos_idx = rng.choice(n, n_pos, replace=False)
        y[pos_idx] = 1

        # Make positive examples have higher values in signal features
        for idx in pos_idx:
            X[idx, 3] += 2.0   # files_changed
            X[idx, 9] += 1.5   # total_hunks
            X[idx, 14] -= 1.0  # test_ratio (lower = more risky)

        metadata = [
            {
                "commit_sha": f"{i:040d}",
                "file_path": f"src/file_{i}.py",
                "split": "synthetic",
                "label_status": "positive" if y[i] == 1 else "negative",
                "label_source": "explicit_sha_reference" if y[i] == 1 else "none",
            }
            for i in range(n)
        ]

        return SupervisedDataset(
            X=X, y=y, metadata=metadata, feature_names=list(FEATURE_NAMES),
        )

    return _make_split(n_train), _make_split(n_val), _make_split(n_test)


def main() -> None:
    """Run the full Phase 4 training pipeline."""
    data_dir = project_root / "data"
    models_dir = data_dir / "models"

    # Attempt to load real dataset; fall back to synthetic
    real_data_dir = data_dir / "processed"
    use_synthetic = True

    if (real_data_dir / "train.jsonl").exists():
        logger.info("Loading real Phase 3 dataset from %s", real_data_dir)
        try:
            train_ds, val_ds, test_ds = load_dataset(real_data_dir)
            if train_ds.positive_count > 0:
                use_synthetic = False
                logger.info(
                    "Real dataset loaded: train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
                    len(train_ds), train_ds.positive_count,
                    len(val_ds), val_ds.positive_count,
                    len(test_ds), test_ds.positive_count,
                )
        except Exception as e:
            logger.warning("Failed to load real dataset: %s. Using synthetic.", e)

    if use_synthetic:
        logger.info("Generating synthetic dataset for pipeline validation")
        train_ds, val_ds, test_ds = generate_synthetic_dataset()
        logger.info(
            "Synthetic dataset: train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
            len(train_ds), train_ds.positive_count,
            len(val_ds), val_ds.positive_count,
            len(test_ds), test_ds.positive_count,
        )

    model_version = "v0.1.0-local"

    # ── Baselines ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    baseline = compute_baseline(train_ds.y)
    print(f"Majority-class baseline (train): accuracy={baseline.accuracy:.4f}")
    mc = baseline.majority_class
    cnt = baseline.majority_count
    tot = baseline.total
    print(f"  majority_class={mc}, count={cnt}/{tot}")
    if tot > 0:
        pr = baseline.positive_count / tot
        print(f"  positive_rate={pr:.4f}")

    # ── Models ──────────────────────────────────────────────────────────────
    models = [
        ("majority_class", MajorityClassBaseline()),
        ("weighted_logistic_regression", WeightedLogisticRegression()),
        ("xgboost", XGBoostDefectModel(n_estimators=200, max_depth=6, learning_rate=0.1)),
    ]

    all_results = {}

    for name, model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {name}")
        print(f"{'=' * 70}")

        model.fit(train_ds.X, train_ds.y, X_val=val_ds.X, y_val=val_ds.y)

        # Evaluate on all splits
        for split_name, ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
            y_prob = model.predict_proba(ds.X)
            metrics = compute_metrics(ds.y, y_prob)

            print(f"\n--- {split_name.upper()} ---")
            print(f"  ROC-AUC:           {metrics.roc_auc:.4f}")
            print(f"  PR-AUC:            {metrics.pr_auc:.4f}")
            print(f"  Precision:         {metrics.precision:.4f}")
            print(f"  Recall:            {metrics.recall:.4f}")
            print(f"  F1:                {metrics.f1:.4f}")
            print(f"  Positive predictions: {metrics.positive_predictions}/{metrics.total}")
            print(f"  Actual positives:  {metrics.actual_positives}/{metrics.total}")
            print(f"  Confusion matrix:  {metrics.confusion_matrix}")

            if split_name == "test":
                all_results[name] = metrics.to_dict()

    # ── Feature Importance (XGBoost) ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE (XGBoost)")
    print("=" * 70)

    xgb_model = [m for n, m in models if n == "xgboost"][0]
    importances = extract_importance(xgb_model, FEATURE_NAMES)
    print(format_importance_report(importances, top_n=20))

    # ── File-level Predictions (XGBoost, test set) ─────────────────────────
    print(f"\n{'=' * 70}")
    print("FILE-LEVEL PREDICTIONS (XGBoost, test set)")
    print("=" * 70)

    test_probs = xgb_model.predict_proba(test_ds.X)
    test_preds = []
    for i in range(len(test_ds)):
        test_preds.append({
            "commit_sha": test_ds.metadata[i]["commit_sha"],
            "file_path": test_ds.metadata[i]["file_path"],
            "actual_label": int(test_ds.y[i]),
            "predicted_probability": float(test_probs[i]),
            "predicted_label": int(test_probs[i] >= 0.5),
            "split": "test",
        })

    # Show first 10
    for p in test_preds[:10]:
        print(f"  {p['commit_sha'][:12]}... {p['file_path']:30s} "
              f"actual={p['actual_label']} prob={p['predicted_probability']:.4f} "
              f"pred={p['predicted_label']}")

    # ── Save Artifacts ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING ARTIFACTS")
    print("=" * 70)

    model_dir = save_artifacts(
        output_dir=models_dir,
        model_version=model_version,
        model=xgb_model,
        feature_names=FEATURE_NAMES,
        dataset_version="synthetic-v1" if use_synthetic else "v1",
        feature_version="v1",
        train_count=len(train_ds),
        val_count=len(val_ds),
        test_count=len(test_ds),
        train_pos=train_ds.positive_count,
        train_neg=train_ds.negative_count,
        val_pos=val_ds.positive_count,
        val_neg=val_ds.negative_count,
        test_pos=test_ds.positive_count,
        test_neg=test_ds.negative_count,
        metrics={
            "baseline": baseline.to_dict(),
            "models": all_results,
            "feature_importance_top20": [
                {"rank": fi.rank, "feature": fi.feature_name, "importance": fi.importance}
                for fi in importances[:20]
            ],
        },
        predictions=test_preds,
    )
    print(f"Artifacts saved to: {model_dir}")

    # ── Limitations ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("LIMITATIONS")
    print("=" * 70)
    if use_synthetic:
        print("  - Using SYNTHETIC data for pipeline validation only")
        print("  - Results are NOT statistically meaningful")
        print("  - Run with real Phase 3 dataset for actual evaluation")
    else:
        if test_ds.positive_count < 30:
            print(f"  - Test set has only {test_ds.positive_count} positive examples")
            print("  - Results may not be statistically significant")
            print("  - Consider collecting more data before drawing conclusions")
        if train_ds.positive_count < 100:
            print(f"  - Training set has only {train_ds.positive_count} positive examples")
            print("  - Model performance may improve with more training data")
    print("  - Chronological split means temporal distribution shift is possible")
    print("  - File-level attribution uses deterministic rules, not human review")
    print("  - Feature set (v1) is intentionally conservative")

    print(f"\n{'=' * 70}")
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
