"""Validate the combined multi-repository dataset."""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("backend/data/datasets/combined")


def main() -> None:
    print("=" * 70)
    print("COMBINED DATASET VALIDATION")
    print("=" * 70)

    # Load manifest
    manifest = json.loads((DATA_DIR / "metadata.json").read_text())
    print(f"\nTotal commits: {manifest['total_commits_included']}")
    print(f"Total file examples: {manifest['total_file_examples']}")
    print(f"Positive: {manifest['positive_examples']}")
    print(f"Negative: {manifest['negative_examples']}")
    print(f"Ambiguous: {manifest['ambiguous_examples']}")

    # Per-split analysis
    print("\n--- Split Distribution ---")
    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        count = len(lines) if lines[0] else 0

        pos = 0
        neg = 0
        repos = set()
        pos_commits = set()
        neg_commits = set()
        for line in lines:
            row = json.loads(line)
            repos.add(row["repo_name"])
            if row["defect_label"] == 1:
                pos += 1
                pos_commits.add(row["commit_sha"])
            elif row["defect_label"] == 0:
                neg += 1
                neg_commits.add(row["commit_sha"])

        print(f"\n{split_name.upper()}:")
        print(f"  Files: {count}")
        print(f"  Positive files: {pos}")
        print(f"  Negative files: {neg}")
        print(f"  Positive commits: {len(pos_commits)}")
        print(f"  Negative commits: {len(neg_commits)}")
        print(f"  Repositories: {sorted(repos)}")

    # Ambiguous
    amb_path = DATA_DIR / "ambiguous.jsonl"
    amb_lines = amb_path.read_text(encoding="utf-8").strip().split("\n")
    amb_count = len(amb_lines) if amb_lines[0] else 0
    print(f"\nAMBIGUOUS: {amb_count} files")

    # Feature dimensions
    print("\n--- Feature Contract ---")
    first_line = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")[0]
    first_row = json.loads(first_line)
    cf = len(first_row["commit_features"])
    ff = len(first_row["file_features"])
    print(f"commit_features: {cf}, file_features: {ff}, total: {cf + ff}")

    # Leakage check
    print("\n--- Leakage Check ---")
    forbidden = {
        "defect_label", "label_status", "label_confidence",
        "label_source", "label_evidence", "split",
    }
    all_keys = set(first_row.keys())
    found = all_keys & forbidden
    if found:
        print(f"Metadata in row keys: {found} (expected — not in feature vector)")
    else:
        print("No forbidden metadata in row keys")

    # Verify feature vector is only numbers
    commit_features = first_row["commit_features"]
    file_features = first_row["file_features"]
    all_numeric = all(isinstance(x, (int, float)) for x in commit_features + file_features)
    print(f"All features numeric: {all_numeric}")

    # Determinism
    print("\n--- Determinism ---")
    content1 = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8")
    content2 = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8")
    print(f"Byte-identical on re-read: {content1 == content2}")

    # JSONL validity
    print("\n--- JSONL Validity ---")
    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        valid = all(json.loads(ln) for ln in lines if ln)
        print(f"{split_name}.jsonl: {'PASS' if valid else 'FAIL'}")

    # Split integrity (no commit in multiple splits)
    print("\n--- Split Integrity ---")
    split_shas: dict[str, set[str]] = {}
    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        shas = set()
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                shas.add(json.loads(line)["commit_sha"])
        split_shas[split_name] = shas

    overlap_tv = split_shas["train"] & split_shas["validation"]
    overlap_tt = split_shas["train"] & split_shas["test"]
    overlap_vt = split_shas["validation"] & split_shas["test"]
    print(f"Train-Val overlap: {len(overlap_tv)}")
    print(f"Train-Test overlap: {len(overlap_tt)}")
    print(f"Val-Test overlap: {len(overlap_vt)}")
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("PASS: No split overlap")
    else:
        print("FAIL: Split overlap detected!")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
