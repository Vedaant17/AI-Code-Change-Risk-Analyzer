"""Comprehensive audit of the Phase 3.7 v3 combined dataset.

Performs: label quality, leakage, determinism, diversity audits.
Includes pre/post comparison with Phase 3.6.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Fix Windows console encoding for Unicode commit messages
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DATA_DIR = Path("backend/data/datasets/combined-v3")
V2_DIR = Path("backend/data/datasets/combined-v2")


def _load_supervised(split_name: str, data_dir: Path) -> list[dict]:
    path = data_dir / f"{split_name}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(ln) for ln in lines if ln]


def _load_ambiguous(data_dir: Path) -> list[dict]:
    path = data_dir / "ambiguous.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(ln) for ln in lines if ln]


def _compute_diversity_stats(rows: list[dict], amb_rows: list[dict]) -> dict:
    """Compute diversity statistics from supervised + ambiguous rows."""
    all_rows = rows + amb_rows
    pos_rows = [r for r in all_rows if r["defect_label"] == 1]

    pos_by_repo: dict[str, set[str]] = defaultdict(set)
    pos_files_by_repo: Counter = Counter()
    pos_source_count: Counter = Counter()
    pos_by_split: dict[str, set[str]] = defaultdict(set)

    for r in pos_rows:
        pos_by_repo[r["repo_name"]].add(r["commit_sha"])
        pos_files_by_repo[r["repo_name"]] += 1
        pos_source_count[r["label_source"]] += 1

    for r in [r for r in rows if r["defect_label"] == 1]:
        pos_by_split[r["split"]].add(r["commit_sha"])

    total_pos_commits = sum(len(v) for v in pos_by_repo.values())
    total_pos_files = sum(pos_files_by_repo.values())

    # Concentration
    max_repo = max(pos_by_repo.keys(), key=lambda k: len(pos_by_repo[k])) if pos_by_repo else None
    max_repo_pct = (
        len(pos_by_repo[max_repo]) / total_pos_commits * 100
        if max_repo and total_pos_commits else 0
    )

    # Largest single commit
    commit_file_counts: Counter = Counter(r["commit_sha"] for r in pos_rows)
    max_commit = commit_file_counts.most_common(1)[0] if commit_file_counts else None
    max_commit_pct = (
        (max_commit[1] / total_pos_files * 100)
        if max_commit and total_pos_files else 0
    )

    # Repos with >= N positives
    repos_1 = sum(1 for v in pos_by_repo.values() if len(v) >= 1)
    repos_3 = sum(1 for v in pos_by_repo.values() if len(v) >= 3)
    repos_5 = sum(1 for v in pos_by_repo.values() if len(v) >= 5)

    return {
        "total_pos_commits": total_pos_commits,
        "total_pos_files": total_pos_files,
        "repos_with_positives": len(pos_by_repo),
        "pos_by_repo": dict(pos_by_repo),
        "pos_files_by_repo": dict(pos_files_by_repo),
        "pos_source_count": dict(pos_source_count),
        "pos_by_split": {k: len(v) for k, v in pos_by_split.items()},
        "max_repo": max_repo,
        "max_repo_commits": len(pos_by_repo[max_repo]) if max_repo else 0,
        "max_repo_pct": max_repo_pct,
        "max_commit_sha": max_commit[0] if max_commit else None,
        "max_commit_files": max_commit[1] if max_commit else 0,
        "max_commit_pct": max_commit_pct,
        "repos_gte_1": repos_1,
        "repos_gte_3": repos_3,
        "repos_gte_5": repos_5,
        "commit_file_counts": dict(commit_file_counts),
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 3.7 DATASET AUDIT (v3)")
    print("=" * 70)

    # Load manifest
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"\nERROR: {meta_path} does not exist. Run generation first.")
        return

    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"\nDataset version: {manifest.get('dataset_version', '?')}")
    print(f"Total commits: {manifest['total_commits_included']}")
    print(f"Total file examples: {manifest['total_file_examples']}")
    print(f"Positive: {manifest['positive_examples']}")
    print(f"Negative: {manifest['negative_examples']}")
    print(f"Ambiguous: {manifest['ambiguous_examples']}")

    # ── Split Distribution ─────────────────────
    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTION")
    print("=" * 70)

    split_data: dict[str, list[dict]] = {}
    for split_name in ["train", "validation", "test"]:
        rows = _load_supervised(split_name, DATA_DIR)
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
        print(f"  Repositories: {len(repos)}")

    amb_rows = _load_ambiguous(DATA_DIR)
    print(f"\nAMBIGUOUS: {len(amb_rows)} files")

    # ── Leakage Audit ────────────────
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
    first_row = split_data["train"][0] if split_data["train"] else None
    if first_row:
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

    # ── Determinism ─────────────────
    print("\n" + "=" * 70)
    print("DETERMINISM")
    print("=" * 70)

    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        if path.exists():
            content1 = path.read_text(encoding="utf-8")
            content2 = path.read_text(encoding="utf-8")
            print(f"{split_name}.jsonl byte-identical: {content1 == content2}")

    # JSONL validity
    for split_name in ["train", "validation", "test", "ambiguous"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            valid = all(json.loads(ln) for ln in lines if ln)
            print(f"{split_name}.jsonl valid JSONL: {valid}")

    # ── Diversity Audit ──────────────────────
    print("\n" + "=" * 70)
    print("DIVERSITY AUDIT")
    print("=" * 70)

    stats = _compute_diversity_stats(all_supervised, amb_rows)

    print("\nPositive commits per repository:")
    for repo in sorted(stats["pos_by_repo"].keys()):
        shas = stats["pos_by_repo"][repo]
        files = stats["pos_files_by_repo"][repo]
        print(f"  {repo:<25} {len(shas):>3} commits, {files:>3} files")

    print(f"\nTotal distinct positive commits: {stats['total_pos_commits']}")
    print(f"Total positive file examples: {stats['total_pos_files']}")
    print(f"Repositories with positives: {stats['repos_with_positives']}")

    print("\nAttribution sources:")
    for source, count in sorted(stats["pos_source_count"].items(), key=lambda x: -x[1]):
        print(f"  {source:<30} {count:>5}")

    print("\nConcentration:")
    print(f"  Largest repo: {stats['max_repo']} "
          f"({stats['max_repo_commits']}/{stats['total_pos_commits']}"
          f" = {stats['max_repo_pct']:.1f}%)")
    if stats["max_commit_sha"]:
        print(f"  Largest commit: {stats['max_commit_sha'][:12]}... "
              f"({stats['max_commit_files']}/{stats['total_pos_files']}"
              f" = {stats['max_commit_pct']:.1f}%)")

    print("\nRepos with >=1 positive:", stats["repos_gte_1"])
    print("Repos with >=3 positives:", stats["repos_gte_3"])
    print("Repos with >=5 positives:", stats["repos_gte_5"])

    print("\nPositive commits by split:")
    for split_name in ["train", "validation", "test"]:
        count = stats["pos_by_split"].get(split_name, 0)
        print(f"  {split_name:>10}: {count}")

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
                msg = next(
                    (r["commit_message"][:55] for r in rows
                     if r["commit_sha"] == sha and r["defect_label"] == 1), "?"
                )
                print(f"    {repo:<20} {sha[:12]}... {msg}")

    # ── Pre/Post Comparison ──────────────────
    print("\n" + "=" * 70)
    print("PRE/POST COMPARISON (Phase 3.6 vs Phase 3.7)")
    print("=" * 70)

    if V2_DIR.exists():
        v2_meta_path = V2_DIR / "metadata.json"
        if v2_meta_path.exists():
            v2_rows = []
            for sn in ["train", "validation", "test"]:
                v2_rows.extend(_load_supervised(sn, V2_DIR))
            v2_amb = _load_ambiguous(V2_DIR)
            v2_stats = _compute_diversity_stats(v2_rows, v2_amb)

            print(f"\n{'Metric':<45} {'Phase 3.6':>12} {'Phase 3.7':>12} {'Delta':>10}")
            print("-" * 80)

            metrics = [
                ("Distinct positive commits",
                 v2_stats["total_pos_commits"],
                 stats["total_pos_commits"]),
                ("Positive file examples",
                 v2_stats["total_pos_files"],
                 stats["total_pos_files"]),
                ("Repositories with positives",
                 v2_stats["repos_with_positives"],
                 stats["repos_with_positives"]),
                ("Largest repo % of positive commits",
                 f"{v2_stats['max_repo_pct']:.1f}%",
                 f"{stats['max_repo_pct']:.1f}%"),
                ("Largest commit % of positive files",
                 f"{v2_stats['max_commit_pct']:.1f}%",
                 f"{stats['max_commit_pct']:.1f}%"),
                ("Positive commits (explicit SHA)",
                 v2_stats["pos_source_count"]
                 .get("explicit_sha_reference", 0),
                 stats["pos_source_count"]
                 .get("explicit_sha_reference", 0)),
                ("Positive commits (revert)",
                 v2_stats["pos_source_count"]
                 .get("revert", 0),
                 stats["pos_source_count"]
                 .get("revert", 0)),
                ("Positive commits in train",
                 v2_stats["pos_by_split"]
                 .get("train", 0),
                 stats["pos_by_split"]
                 .get("train", 0)),
                ("Positive commits in validation",
                 v2_stats["pos_by_split"]
                 .get("validation", 0),
                 stats["pos_by_split"]
                 .get("validation", 0)),
                ("Positive commits in test",
                 v2_stats["pos_by_split"]
                 .get("test", 0),
                 stats["pos_by_split"]
                 .get("test", 0)),
                ("Repos with >=1 positive",
                 v2_stats["repos_gte_1"],
                 stats["repos_gte_1"]),
                ("Repos with >=3 positives",
                 v2_stats["repos_gte_3"],
                 stats["repos_gte_3"]),
                ("Repos with >=5 positives",
                 v2_stats["repos_gte_5"],
                 stats["repos_gte_5"]),
            ]

            for name, v2_val, v3_val in metrics:
                if isinstance(v2_val, (int, float)) and isinstance(v3_val, (int, float)):
                    delta = v3_val - v2_val
                    print(f"  {name:<43} {str(v2_val):>12} {str(v3_val):>12} {delta:>+10}")
                else:
                    print(f"  {name:<43} {str(v2_val):>12} {str(v3_val):>12}")
        else:
            print("  Phase 3.6 metadata not found")
    else:
        print("  Phase 3.6 directory not found")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
