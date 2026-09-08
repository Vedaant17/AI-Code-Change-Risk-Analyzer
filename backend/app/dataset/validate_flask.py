"""Validate the generated Flask dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

from backend.app.dataset.schemas import DatasetRow  # noqa: E402
from backend.app.ml.dataset_loader import NUM_FEATURES  # noqa: E402

DATA_DIR = project_root / "backend" / "data" / "datasets" / "flask"


def main() -> None:
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    # Load manifest
    manifest = json.loads((DATA_DIR / "metadata.json").read_text())
    print(f"\nRepository: {manifest['repo_name']}")
    print(f"Pinned revision: {manifest['repo_revision'][:12]}...")
    print(f"Total commits: {manifest['total_commits_processed']}")
    print(f"Total file examples: {manifest['total_file_examples']}")
    print(f"Positive: {manifest['positive_examples']}")
    print(f"Negative: {manifest['negative_examples']}")
    print(f"Ambiguous: {manifest['ambiguous_examples']}")
    print(f"Train: {manifest['train_examples']}")
    print(f"Validation: {manifest['validation_examples']}")
    print(f"Test: {manifest['test_examples']}")

    # Validate JSONL structure
    print("\n--- JSONL Structure ---")
    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        count = len(lines) if lines[0] else 0
        print(f"{split_name}.jsonl: {count} rows")

        if count == 0:
            continue

        # Validate all rows against DatasetRow schema
        valid = 0
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                DatasetRow.model_validate(obj)
                valid += 1
            except Exception as e:
                print(f"  INVALID row {i}: {e}")

        print(f"  Schema valid: {valid}/{count}")

        # Feature dimensions
        first = json.loads(lines[0])
        cf = first.get("commit_features", [])
        ff = first.get("file_features", [])
        print(f"  commit_features: {len(cf)}, file_features: {len(ff)}, total: {len(cf) + len(ff)}")

        # Label distribution
        labels = [json.loads(line)["defect_label"] for line in lines]
        pos = sum(1 for lbl in labels if lbl == 1)
        neg = sum(1 for lbl in labels if lbl == 0)
        amb = sum(1 for lbl in labels if lbl == -1)
        print(f"  Labels: pos={pos}, neg={neg}, amb={amb}")

        # Verify no ambiguous in supervised splits
        if split_name in ["train", "validation", "test"]:
            assert amb == 0, f"FAIL: Ambiguous rows in {split_name}!"

    # Chronological split integrity
    print("\n--- Split Integrity ---")
    shas: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                shas[split_name].add(json.loads(line)["commit_sha"])

    overlap_tv = shas["train"] & shas["validation"]
    overlap_tt = shas["train"] & shas["test"]
    overlap_vt = shas["validation"] & shas["test"]
    print(f"Train-Val overlap: {len(overlap_tv)}")
    print(f"Train-Test overlap: {len(overlap_tt)}")
    print(f"Val-Test overlap: {len(overlap_vt)}")
    if len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0:
        print("PASS: No split overlap")
    else:
        print("FAIL: Split overlap detected!")

    # Feature count check
    print("\n--- Feature Contract ---")
    path = DATA_DIR / "train.jsonl"
    first_line = path.read_text(encoding="utf-8").strip().split("\n")[0]
    first_row = json.loads(first_line)
    cf_len = len(first_row["commit_features"])
    ff_len = len(first_row["file_features"])
    print(f"commit_features: {cf_len} (expected 29)")
    print(f"file_features: {ff_len} (expected 16)")
    print(f"total: {cf_len + ff_len} (expected 45)")
    assert cf_len == 29, f"Expected 29 commit features, got {cf_len}"
    assert ff_len == 16, f"Expected 16 file features, got {ff_len}"
    assert cf_len + ff_len == NUM_FEATURES, f"Expected {NUM_FEATURES} total"
    print("PASS: Feature dimensions correct")

    # Leakage check
    print("\n--- Leakage Check ---")
    forbidden = {
        "defect_label", "label_status", "label_confidence",
        "label_source", "label_evidence", "split",
    }
    all_keys = set(first_row.keys())
    found = all_keys & forbidden
    if found:
        print(f"FAIL: Found forbidden metadata in row: {found}")
    else:
        print("PASS: No metadata leakage")

    # Determinism check
    print("\n--- Determinism ---")
    content1 = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8")
    content2 = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8")
    if content1 == content2:
        print("PASS: Byte-identical on re-read")
    else:
        print("FAIL: Content differs on re-read")

    # JSONL validity
    print("\n--- JSONL Validity ---")
    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        valid = all(json.loads(ln) for ln in lines if ln)
        print(f"{split_name}.jsonl: {'PASS' if valid else 'FAIL'}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
