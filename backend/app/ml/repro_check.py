"""Reproducibility check: run training twice and compare metrics/artifacts."""

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

DATA_DIR = project_root / "backend" / "data" / "datasets" / "combined-v2"


def run_once(seed: int = 42) -> dict:
    """Train both models once and return key metrics."""
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    results = {}
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
            }
    return results


def main() -> None:
    print("Run 1...")
    r1 = run_once()
    print("Run 2...")
    r2 = run_once()

    all_match = True
    for key in r1:
        if r1[key] != r2[key]:
            all_match = False
            print(f"  MISMATCH {key}: {r1[key]} vs {r2[key]}")

    if all_match:
        print("\nAll metrics identical across both runs.")
        print("REPRODUCIBILITY: PASS")
    else:
        print("\nREPRODUCIBILITY: FAIL")

    # Check artifact file hashes
    model_dir = project_root / "backend" / "data" / "models" / "v0.1.0-combined-v2-real"
    for fname in ["feature_schema.json", "model.json", "metadata.json"]:
        fp = model_dir / fname
        if fp.exists():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
            print(f"  {fname}: sha256={h}")


if __name__ == "__main__":
    main()
