#!/usr/bin/env python3
"""Merge Phase 4.5a experimental features into a single JSONL file.

The experimental loader expects a single experimental_features.jsonl file.
This script merges the per-split experimental features into one file.
"""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENTAL_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")
OUTPUT_PATH = EXPERIMENTAL_DIR / "experimental_features.jsonl"


def merge_experimental_features() -> int:
    """Merge train/val/test experimental features into a single file.

    Returns the total number of rows written.
    """
    total_rows = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for split in ["train", "validation", "test"]:
            split_path = EXPERIMENTAL_DIR / f"{split}.jsonl"
            if not split_path.exists():
                print(f"Warning: {split_path} not found, skipping")
                continue

            with open(split_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # Write only the keys needed by experimental_loader
                    out_obj = {
                        "commit_sha": obj["commit_sha"],
                        "file_path": obj["file_path"],
                        "experimental_features": obj["experimental_features"],
                    }
                    out.write(json.dumps(out_obj, separators=(",", ":")) + "\n")
                    total_rows += 1

            print(f"Processed {split}: wrote {total_rows} total rows")

    print(f"\nMerged experimental features written to {OUTPUT_PATH}")
    print(f"Total rows: {total_rows}")
    return total_rows


if __name__ == "__main__":
    merge_experimental_features()
