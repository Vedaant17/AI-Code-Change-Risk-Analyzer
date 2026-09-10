"""Reproducibility check for combined-v3: run training twice and compare.

Trains WeightedLogisticRegression + XGBoost on combined-v3 dataset twice
with deterministic seeds, then compares metrics, predictions, and artifact hashes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

from backend.app.ml.dataset_loader import load_dataset  # noqa: E402
from backend.app.ml.metrics import compute_metrics  # noqa: E402
from backend.app.ml.models import WeightedLogisticRegression, XGBoostDefectModel  # noqa: E402

DATA_DIR = project_root / "backend" / "data" / "datasets" / "combined-v3"
MODEL_DIR = project_root / "backend" / "data" / "models" / "v0.1.0-combined-v3-real"


def run_once() -> dict:
    """Train both models once and return key metrics + predictions."""
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    results = {}
    predictions = {}

    for name, model in [
        ("xgboost", XGBoostDefectModel(n_estimators=200, max_depth=6, learning_rate=0.1)),
        ("lr", WeightedLogisticRegression()),
    ]:
        model.fit(train_ds.X, train_ds.y, X_val=val_ds.X, y_val=val_ds.y)

        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            y_prob = model.predict_proba(ds.X)
            metrics = compute_metrics(ds.y, y_prob)
            key = f"{name}_{split_name}"
            results[key] = {
                "roc_auc": round(metrics.roc_auc, 6),
                "pr_auc": round(metrics.pr_auc, 6),
                "precision": round(metrics.precision, 6),
                "recall": round(metrics.recall, 6),
                "f1": round(metrics.f1, 6),
                "pos_pred": metrics.positive_predictions,
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "tn": metrics.true_negatives,
                "fn": metrics.false_negatives,
            }

        # Store test predictions for comparison
        if name == "xgboost":
            test_prob = model.predict_proba(test_ds.X)
            predictions = {
                "y_true": test_ds.y.tolist(),
                "y_prob": [round(float(p), 6) for p in test_prob],
            }

    return {"metrics": results, "predictions": predictions}


def _compare_dicts(d1: dict, d2: dict, path: str = "") -> list[str]:
    """Recursively compare two dicts, return list of mismatches."""
    mismatches = []
    all_keys = set(d1.keys()) | set(d2.keys())
    for key in sorted(all_keys):
        full_path = f"{path}.{key}" if path else key
        if key not in d1:
            mismatches.append(f"  MISSING in run1: {full_path}")
        elif key not in d2:
            mismatches.append(f"  MISSING in run2: {full_path}")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            mismatches.extend(_compare_dicts(d1[key], d2[key], full_path))
        elif d1[key] != d2[key]:
            mismatches.append(f"  MISMATCH {full_path}: {d1[key]} vs {d2[key]}")
    return mismatches


def main() -> None:
    print("=" * 70)
    print("REPRODUCIBILITY CHECK -- combined-v3")
    print("=" * 70)

    print("\nRun 1...")
    r1 = run_once()
    print("Run 2...")
    r2 = run_once()

    # Compare metrics
    print("\n--- Metrics Comparison ---")
    mismatches = _compare_dicts(r1["metrics"], r2["metrics"])
    if mismatches:
        print("REPRODUCIBILITY: FAIL (metric mismatches)")
        for m in mismatches:
            print(m)
    else:
        print("All metrics identical across both runs.")
        print("REPRODUCIBILITY: PASS (metrics)")

    # Compare predictions
    print("\n--- Prediction Comparison ---")
    pred_match = r1["predictions"] == r2["predictions"]
    if pred_match:
        print("Test predictions identical across both runs.")
        print("REPRODUCIBILITY: PASS (predictions)")
    else:
        print("REPRODUCIBILITY: FAIL (prediction mismatch)")

    # Hash artifact files
    print("\n--- Artifact Hashes ---")
    if MODEL_DIR.exists():
        for fname in ["feature_schema.json", "model.json", "metadata.json"]:
            fp = MODEL_DIR / fname
            if fp.exists():
                h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
                print(f"  {fname}: sha256={h}")
            else:
                print(f"  {fname}: NOT FOUND")
    else:
        print(f"  Model directory not found: {MODEL_DIR}")

    # Summary
    print("\n--- Summary ---")
    all_pass = not mismatches and pred_match
    if all_pass:
        print("REPRODUCIBILITY: PASS")
        print("Both runs produce identical metrics and predictions.")
    else:
        print("REPRODUCIBILITY: FAIL")
        print("Runs produced different results -- check for non-determinism.")


if __name__ == "__main__":
    main()
