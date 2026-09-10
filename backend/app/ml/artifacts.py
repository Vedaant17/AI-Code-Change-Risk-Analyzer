"""Model artifact I/O for Phase 4.

Creates deterministic artifact structures under data/models/<model_version>/:
  - model.json  (serialized model params + feature schema)
  - feature_schema.json  (feature names, ordering, count)
  - metrics.json  (train/val/test metrics)
  - metadata.json  (dataset info, model config, counts)
  - predictions.jsonl  (per-example test predictions)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON serializer fallback for numpy types."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_artifacts(
    output_dir: Path,
    model_version: str,
    model,
    feature_names: list[str],
    dataset_version: str,
    feature_version: str,
    train_count: int,
    val_count: int,
    test_count: int,
    train_pos: int,
    train_neg: int,
    val_pos: int,
    val_neg: int,
    test_pos: int,
    test_neg: int,
    metrics: dict,
    predictions: list[dict] | None = None,
) -> Path:
    """Save all model artifacts to disk.

    Returns the model directory path.
    """
    model_dir = output_dir / model_version
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. feature_schema.json
    feature_schema = {
        "feature_version": feature_version,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "commit_features": feature_names[:29],
        "file_features": feature_names[29:],
    }
    _write_json(model_dir / "feature_schema.json", feature_schema)

    # 2. model.json
    model_data = {
        "model_name": model.name,
        "model_params": model.get_params(),
    }
    _write_json(model_dir / "model.json", model_data)

    # 3. metrics.json
    _write_json(model_dir / "metrics.json", metrics)

    # 4. metadata.json
    metadata = {
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "model_version": model_version,
        "model_type": model.name,
        "model_config": model.get_params(),
        "feature_count": len(feature_names),
        "feature_ordering": feature_names,
        "train_examples": train_count,
        "validation_examples": val_count,
        "test_examples": test_count,
        "train_positive": train_pos,
        "train_negative": train_neg,
        "validation_positive": val_pos,
        "validation_negative": val_neg,
        "test_positive": test_pos,
        "test_negative": test_neg,
    }
    _write_json(model_dir / "metadata.json", metadata)

    # 5. predictions.jsonl
    if predictions:
        with open(model_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for pred in sorted(predictions, key=lambda p: (p["commit_sha"], p["file_path"])):
                f.write(json.dumps(pred, default=_json_default, separators=(",", ":")) + "\n")

    logger.info("Artifacts saved to %s", model_dir)
    return model_dir


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as compact JSON with deterministic key ordering."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default, sort_keys=True)
        f.write("\n")
