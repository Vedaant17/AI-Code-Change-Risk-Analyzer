"""Manual label quality audit for the combined dataset."""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("backend/data/datasets/combined")


def audit_split(split_name: str, n_pos: int = 15, n_neg: int = 10, n_amb: int = 10) -> None:
    """Audit examples from a split."""
    path = DATA_DIR / f"{split_name}.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")

    positives = []
    negatives = []
    ambig = []

    for line in lines:
        row = json.loads(line)
        if row["defect_label"] == 1:
            positives.append(row)
        elif row["defect_label"] == 0:
            negatives.append(row)
        elif row["defect_label"] == -1:
            ambig.append(row)

    print(f"\n{'=' * 70}")
    print(f"AUDIT: {split_name.upper()} SPLIT")
    print(f"{'=' * 70}")

    # Positives
    print(f"\n--- POSITIVE EXAMPLES ({min(n_pos, len(positives))} of {len(positives)}) ---")
    for row in positives[:n_pos]:
        sha = row["commit_sha"][:12]
        fp = row["file_path"]
        repo = row["repo_name"]
        src = row["label_source"]
        conf = row["label_confidence"]
        ev = row["label_evidence"][:120]
        msg = row["commit_message"][:80]
        print(f"  Repo: {repo}")
        print(f"  SHA: {sha}...  File: {fp}")
        print(f"  Source: {src}  Confidence: {conf}")
        print(f"  Evidence: {ev}")
        print(f"  Commit msg: {msg}")
        print()

    # Negatives
    print(f"\n--- NEGATIVE EXAMPLES ({min(n_neg, len(negatives))} of {len(negatives)}) ---")
    for row in negatives[:n_neg]:
        sha = row["commit_sha"][:12]
        fp = row["file_path"]
        repo = row["repo_name"]
        msg = row["commit_message"][:80]
        print(f"  Repo: {repo}  SHA: {sha}...  File: {fp}")
        print(f"  Commit msg: {msg}")

    # Ambiguous
    print(f"\n--- AMBIGUOUS EXAMPLES ({min(n_amb, len(ambig))} of {len(ambig)}) ---")
    for row in ambig[:n_amb]:
        sha = row["commit_sha"][:12]
        fp = row["file_path"]
        repo = row["repo_name"]
        src = row["label_source"]
        ev = row["label_evidence"][:120]
        msg = row["commit_message"][:80]
        print(f"  Repo: {repo}  SHA: {sha}...  File: {fp}")
        print(f"  Source: {src}")
        print(f"  Evidence: {ev}")
        print(f"  Commit msg: {msg}")


def main() -> None:
    print("=" * 70)
    print("LABEL QUALITY AUDIT")
    print("=" * 70)

    # Audit from train (most positives)
    audit_split("train", n_pos=15, n_neg=10, n_amb=10)

    # Audit from test
    audit_split("test", n_pos=13, n_neg=5, n_amb=5)

    # Summary
    print(f"\n{'=' * 70}")
    print("AUDIT SUMMARY")
    print("=" * 70)

    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        pos_commits = set()
        for line in lines:
            row = json.loads(line)
            if row["defect_label"] == 1:
                pos_commits.add(row["commit_sha"])

        print(f"\n{split_name.upper()}:")
        print(f"  Positive commits: {len(pos_commits)}")
        for sha in sorted(pos_commits):
            # Find the commit message
            for line in lines:
                row = json.loads(line)
                if row["commit_sha"] == sha and row["defect_label"] == 1:
                    print(f"    {sha[:12]}... ({row['repo_name']}) {row['commit_message'][:60]}")
                    break


if __name__ == "__main__":
    main()
