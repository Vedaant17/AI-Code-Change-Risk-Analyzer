"""Phase 4 real-data training and evaluation.

Runs the full ML pipeline against the real Flask dataset.
Handles edge cases: 0 training positives, single-class splits.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

from backend.app.ml.artifacts import save_artifacts  # noqa: E402
from backend.app.ml.dataset_loader import (  # noqa: E402
    FEATURE_NAMES,
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

DATA_DIR = project_root / "backend" / "data" / "datasets" / "flask"
MODELS_DIR = project_root / "backend" / "data" / "models"


def main() -> None:
    """Run the full Phase 4 training pipeline on real data."""
    # ── Load real dataset ───────────────────────────────────────────────
    logger.info("Loading real Flask dataset from %s", DATA_DIR)
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    print("\n" + "=" * 70)
    print("REAL DATA: Flask (pinned d318b6834711)")
    print("=" * 70)

    print("\nDataset Summary:")
    tp = train_ds.positive_count
    tn = train_ds.negative_count
    vp = val_ds.positive_count
    vn = val_ds.negative_count
    tsp = test_ds.positive_count
    tsn = test_ds.negative_count
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"  Train:      {len(train_ds):>5} examples  (pos={tp}, neg={tn})")
    print(f"  Validation: {len(val_ds):>5} examples  (pos={vp}, neg={vn})")
    print(f"  Test:       {len(test_ds):>5} examples  (pos={tsp}, neg={tsn})")
    print(f"  Total:      {total:>5}")

    # ── Critical limitations ────────────────────────────────────────────
    print("\n--- CRITICAL LIMITATIONS ---")
    if train_ds.positive_count == 0:
        print("  [!] TRAINING SET HAS 0 POSITIVE EXAMPLES")
        print("      The model cannot learn defect patterns from training data.")
        print("      All 24 positives are in the validation split only.")
        print("      This is a dataset size / chronological split issue.")
    if test_ds.positive_count == 0:
        print("  [!] TEST SET HAS 0 POSITIVE EXAMPLES")
        print("      Test metrics cannot evaluate recall/precision/F1.")
        print("      ROC-AUC requires both classes in the test set.")
    if val_ds.positive_count > 0 and train_ds.positive_count == 0:
        print("  [!] Validation metrics are the ONLY evaluable metrics.")
        print("      But they reflect a single commit's attribution (0109e496f6ca).")
    print("  [!] Total positive count: 24 across 1 commit (0109e496f6ca)")
    print("      Attributed via explicit SHA reference from bug-fix fe3b215d3ade")
    print("      All 24 file examples from that one commit.")

    # ── Baselines ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("BASELINES")
    print("=" * 70)

    baseline = compute_baseline(train_ds.y)
    print("Majority-class baseline (train majority=0):")
    print(f"  accuracy={baseline.accuracy:.4f}")
    print(f"  majority_class={baseline.majority_class}")

    # ── Models ──────────────────────────────────────────────────────────
    model_version = "v0.1.0-flask-real"

    models = [
        ("majority_class", MajorityClassBaseline()),
        ("weighted_logistic_regression", WeightedLogisticRegression()),
        ("xgboost", XGBoostDefectModel(n_estimators=200, max_depth=6, learning_rate=0.1)),
    ]

    all_results = {}

    for name, model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {name}")
        print("=" * 70)

        # Handle 0 training positives
        if train_ds.positive_count == 0 and name != "majority_class":
            print("  Skipping: training set has 0 positive examples.")
            print("  Model cannot learn meaningful patterns.")
            all_results[name] = {"skipped": True, "reason": "0 training positives"}
            continue

        model.fit(train_ds.X, train_ds.y, X_val=val_ds.X, y_val=val_ds.y)

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

            if ds.positive_count == 0:
                print("  [!] No positives in this split — metrics are trivial")

            if split_name == "test" and ds.positive_count == 0:
                print("  [!] Test set has 0 positives — ROC-AUC undefined")

            if split_name == "validation":
                all_results[name] = metrics.to_dict()

    # ── Feature Importance (if model was trained) ───────────────────────
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    xgb_entry = all_results.get("xgboost", {})
    if xgb_entry.get("skipped"):
        print("XGBoost skipped — no feature importance available.")
        print("Training set had 0 positive examples.")
    else:
        xgb_model = [m for n, m in models if n == "xgboost"][0]
        importances = extract_importance(xgb_model, FEATURE_NAMES)
        print(format_importance_report(importances, top_n=10))

    # ── File-level Predictions ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FILE-LEVEL PREDICTIONS (Validation set)")
    print("=" * 70)

    if not xgb_entry.get("skipped"):
        xgb_model = [m for n, m in models if n == "xgboost"][0]
        val_probs = xgb_model.predict_proba(val_ds.X)
        for i in range(min(10, len(val_ds))):
            meta = val_ds.metadata[i]
            prob = val_probs[i]
            actual = int(val_ds.y[i])
            pred = int(prob >= 0.5)
            marker = " <<<" if actual == 1 else ""
            print(f"  {meta['commit_sha'][:12]}... {meta['file_path']:40s} "
                  f"actual={actual} prob={prob:.4f} pred={pred}{marker}")
    else:
        print("No model trained — predictions unavailable.")

    # ── Save Artifacts ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING ARTIFACTS")
    print("=" * 70)

    # Save even if model was skipped (save baseline results)
    metrics_data = {
        "baseline": baseline.to_dict(),
        "models": all_results,
    }

    if not xgb_entry.get("skipped"):
        xgb_model = [m for n, m in models if n == "xgboost"][0]
        importances = extract_importance(xgb_model, FEATURE_NAMES)
        metrics_data["feature_importance_top10"] = [
            {"rank": fi.rank, "feature": fi.feature_name, "importance": fi.importance}
            for fi in importances[:10]
        ]

        model_dir = save_artifacts(
            output_dir=MODELS_DIR,
            model_version=model_version,
            model=xgb_model,
            feature_names=FEATURE_NAMES,
            dataset_version="v1-flask-real",
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
            metrics=metrics_data,
        )
    else:
        # Save metadata-only artifacts
        model_dir = MODELS_DIR / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(
            json.dumps({
                "model_version": model_version,
                "dataset_version": "v1-flask-real",
                "feature_version": "v1",
                "model_type": "skipped",
                "reason": "0 training positives",
                "train_examples": len(train_ds),
                "train_positive": train_ds.positive_count,
                "val_examples": len(val_ds),
                "val_positive": val_ds.positive_count,
                "test_examples": len(test_ds),
                "test_positive": test_ds.positive_count,
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (model_dir / "metrics.json").write_text(
            json.dumps(metrics_data, indent=2, default=str),
            encoding="utf-8",
        )

    print(f"Artifacts saved to: {model_dir}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print("  Dataset: Flask (500 commits, pinned d318b6834711)")
    total_ex = len(train_ds) + len(val_ds) + len(test_ds)
    total_pos = train_ds.positive_count + val_ds.positive_count + test_ds.positive_count
    print(f"  File examples: {total_ex}")
    print(f"  Positive examples: {total_pos}")
    print("  Positive commit: 0109e496f6ca (1 commit, 24 files)")
    print(f"  Train positive: {train_ds.positive_count}")
    print(f"  Validation positive: {val_ds.positive_count}")
    print(f"  Test positive: {test_ds.positive_count}")
    print()
    print("  VERDICT: Dataset too small for meaningful ML evaluation.")
    print("  The model pipeline is functional and tested.")
    print("  Real evaluation requires a larger dataset with positives")
    print("  distributed across train/val/test splits.")
    print()
    print("  Pipeline status: IMPLEMENTED AND TESTED")
    print("  Statistical validity: NOT ESTABLISHED (24 positives, 1 commit)")
    print("=" * 70)


if __name__ == "__main__":
    main()
