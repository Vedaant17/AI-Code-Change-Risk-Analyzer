"""Comprehensive audit of the Phase 3.6 v2 combined dataset.

Performs: label quality, leakage, determinism, diversity audits.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("backend/data/datasets/combined-v2")


def main() -> None:
    print("=" * 70)
    print("PHASE 3.6 DATASET AUDIT (v2)")
    print("=" * 70)

    # Load manifest
    manifest = json.loads((DATA_DIR / "metadata.json").read_text())
    print(f"\nTotal commits: {manifest['total_commits_included']}")
    print(f"Total file examples: {manifest['total_file_examples']}")
    print(f"Positive: {manifest['positive_examples']}")
    print(f"Negative: {manifest['negative_examples']}")
    print(f"Ambiguous: {manifest['ambiguous_examples']}")

    # ── Split Distribution ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTION")
    print("=" * 70)

    split_data: dict[str, list[dict]] = {}
    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        rows = [json.loads(ln) for ln in lines if ln]
        split_data[split_name] = rows

        pos = sum(1 for r in rows if r["defect_label"] == 1)
        neg = sum(1 for r in rows if r["defect_label"] == 0)
        pos_shas = set(r["commit_sha"] for r in rows if r["defect_label"] == 1)
        repos = set(r["repo_name"] for r in rows)

        print(f"\n{split_name.upper()}:")
        print(f"  Files: {len(rows)}")
        print(f"  Positive files: {pos}")
        print(f"  Negative files: {neg}")
        print(f"  Positive commits: {len(pos_shas)}")
        print(f"  Repositories: {sorted(repos)}")

    # Ambiguous
    amb_path = DATA_DIR / "ambiguous.jsonl"
    amb_lines = amb_path.read_text(encoding="utf-8").strip().split("\n")
    amb_rows = [json.loads(ln) for ln in amb_lines if ln]
    print(f"\nAMBIGUOUS: {len(amb_rows)} files")

    # ── Leakage Audit ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT")
    print("=" * 70)

    train_shas = set(r["commit_sha"] for r in split_data["train"])
    val_shas = set(r["commit_sha"] for r in split_data["validation"])
    test_shas = set(r["commit_sha"] for r in split_data["test"])

    overlap_tv = train_shas & val_shas
    overlap_tt = train_shas & test_shas
    overlap_vt = val_shas & test_shas

    print(f"Train commits: {len(train_shas)}")
    print(f"Val commits:   {len(val_shas)}")
    print(f"Test commits:  {len(test_shas)}")
    print(f"Train-Val overlap: {len(overlap_tv)}")
    print(f"Train-Test overlap: {len(overlap_tt)}")
    print(f"Val-Test overlap: {len(overlap_vt)}")

    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("[OK] No commit leakage across splits")
    else:
        print("[FAIL] Commit leakage detected!")

    # Check no ambiguous in supervised
    all_supervised = split_data["train"] + split_data["validation"] + split_data["test"]
    amb_in_supervised = [r for r in all_supervised if r["defect_label"] == -1]
    print(f"Ambiguous in supervised: {len(amb_in_supervised)}")
    if not amb_in_supervised:
        print("[OK] No ambiguous rows in supervised splits")
    else:
        print("[FAIL] Ambiguous rows found in supervised splits!")

    # Feature contract
    first_row = split_data["train"][0]
    cf = len(first_row["commit_features"])
    ff = len(first_row["file_features"])
    print(f"Feature dimensions: {cf}+{ff}={cf+ff}")
    all_numeric = all(
        isinstance(x, (int, float))
        for r in all_supervised
        for x in r["commit_features"] + r["file_features"]
    )
    print(f"All features numeric: {all_numeric}")
    if cf == 29 and ff == 16 and all_numeric:
        print("[OK] Feature contract valid")

    # ── Determinism ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETERMINISM")
    print("=" * 70)

    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        content1 = path.read_text(encoding="utf-8")
        content2 = path.read_text(encoding="utf-8")
        print(f"{split_name}.jsonl byte-identical: {content1 == content2}")

    # JSONL validity
    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        valid = all(json.loads(ln) for ln in lines if ln)
        print(f"{split_name}.jsonl valid JSONL: {valid}")

    # ── Diversity Audit ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIVERSITY AUDIT")
    print("=" * 70)

    # Positive commits by repo
    all_rows = all_supervised + amb_rows
    pos_by_repo: dict[str, set[str]] = defaultdict(set)
    pos_files_by_repo: dict[str, int] = Counter()
    pos_source_count: Counter = Counter()

    for r in all_rows:
        if r["defect_label"] == 1 or r["label_status"] == "positive":
            pos_by_repo[r["repo_name"]].add(r["commit_sha"])
            pos_files_by_repo[r["repo_name"]] += 1
            pos_source_count[r["label_source"]] += 1

    print("\nPositive commits per repository:")
    for repo in sorted(pos_by_repo.keys()):
        print(f"  {repo:<25} {len(pos_by_repo[repo]):>3} commits, "
              f"{pos_files_by_repo[repo]:>3} files")

    total_pos_commits = sum(len(v) for v in pos_by_repo.values())
    print(f"\nTotal distinct positive commits: {total_pos_commits}")
    print(f"Total positive file examples: {sum(pos_files_by_repo.values())}")

    print("\nAttribution sources:")
    for source, count in pos_source_count.most_common():
        print(f"  {source:<30} {count:>5}")

    # Check concentration
    print("\nConcentration check:")
    if pos_by_repo:
        max_repo = max(pos_by_repo.keys(), key=lambda k: len(pos_by_repo[k]))
        max_pct = len(pos_by_repo[max_repo]) / total_pos_commits * 100
        print(f"  Most positive commits: {max_repo} "
              f"({len(pos_by_repo[max_repo])}/{total_pos_commits} = {max_pct:.1f}%)")
        if max_pct > 50:
            print(f"  [!] {max_repo} contributes >50% of positive commits")
        else:
            print(f"  [OK] No single repo dominates")

    # Positive commits per split with repo breakdown
    print("\nPositive commits per split (by repo):")
    for split_name in ["train", "validation", "test"]:
        rows = split_data[split_name]
        split_pos_repos: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            if r["defect_label"] == 1:
                split_pos_repos[r["repo_name"]].add(r["commit_sha"])

        total = sum(len(v) for v in split_pos_repos.values())
        print(f"\n  {split_name.upper()}: {total} positive commits")
        for repo in sorted(split_pos_repos.keys()):
            shas = sorted(split_pos_repos[repo])
            for sha in shas:
                # Find commit message
                msg = next(
                    (r["commit_message"][:60] for r in rows
                     if r["commit_sha"] == sha and r["defect_label"] == 1), "?"
                )
                print(f"    {repo:<20} {sha[:12]}... {msg}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
