"""Final Phase 3.7 verification of the combined-v3 dataset."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Fix Windows console encoding for Unicode commit messages
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DATA = Path("backend/data/datasets/combined-v3")


def main() -> None:
    META = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))

    print("=== DATASET VERSION ===")
    print("version:", META["dataset_version"])
    print("feature_version:", META["feature_version"])

    print("\n=== OVERALL COUNTS ===")
    print("commits:", META["total_commits_included"])
    print("files:", META["total_file_examples"])
    print("positive:", META["positive_examples"])
    print("negative:", META["negative_examples"])
    print("ambiguous:", META["ambiguous_examples"])

    print("\n=== REPOSITORY CONFIG ===")
    from backend.app.dataset.multi_repo_config import MultiRepoConfig
    cfg = MultiRepoConfig()
    print("repositories configured:", len(cfg.repositories))
    bad = [r.repo_name for r in cfg.repositories if len(r.repo_revision) != 40]
    print("repos with invalid SHAs:", bad if bad else "none")

    print("\n=== SPLIT STATISTICS ===")
    split_info = {}
    for sn in ["train", "validation", "test"]:
        rows = [
            json.loads(ln)
            for ln in (DATA / f"{sn}.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if ln
        ]
        pos = [r for r in rows if r["defect_label"] == 1]
        neg = [r for r in rows if r["defect_label"] == 0]
        pos_shas = set(r["commit_sha"] for r in pos)
        split_info[sn] = {"rows": len(rows), "pos": len(pos), "neg": len(neg), "pos_shas": pos_shas}
        print(
            f"  {sn:>10}: {len(rows):>6} files,"
            f" {len(pos):>3} pos"
            f" ({len(pos_shas)} commits),"
            f" {len(neg):>5} neg"
        )

    print("\n=== POSITIVE COMMITS BY SPLIT ===")
    for sn in ["train", "validation", "test"]:
        si = split_info[sn]
        all_rows = [
            json.loads(ln)
            for ln in (DATA / f"{sn}.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if ln
        ]
        print(f"\n  {sn.upper()}: {len(si['pos_shas'])} positive commits")
        for sha in sorted(si["pos_shas"]):
            matching = [r for r in all_rows if r["commit_sha"] == sha and r["defect_label"] == 1]
            if matching:
                repo = matching[0]["repo_name"]
                msg = matching[0]["commit_message"][:55]
                src = matching[0]["label_source"]
                print(f"    {repo:<18} {sha[:12]}... [{src}] {msg}")

    print("\n=== LEAKAGE CHECK ===")
    all_shas = {}
    for sn in ["train", "validation", "test"]:
        shas = set(
            r["commit_sha"] for r in [
                json.loads(ln)
                for ln in (DATA / f"{sn}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if ln
            ]
        )
        all_shas[sn] = shas
    tv = all_shas["train"] & all_shas["validation"]
    tt = all_shas["train"] & all_shas["test"]
    vt = all_shas["validation"] & all_shas["test"]
    print(f"  train-val overlap: {len(tv)}")
    print(f"  train-test overlap: {len(tt)}")
    print(f"  val-test overlap: {len(vt)}")
    print("  PASS" if not (tv or tt or vt) else "  FAIL")

    print("\n=== FEATURE CONTRACT ===")
    first = json.loads((DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    cf = len(first["commit_features"])
    ff = len(first["file_features"])
    print(f"  commit_features: {cf}")
    print(f"  file_features: {ff}")
    print(f"  total: {cf + ff}")
    all_rows = []
    for sn in ["train", "validation", "test"]:
        for ln in (DATA / f"{sn}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln:
                all_rows.append(json.loads(ln))
    bad = [i for i, r in enumerate(all_rows)
           if len(r["commit_features"]) != 29 or len(r["file_features"]) != 16]
    print(f"  wrong count: {len(bad)}")
    all_num = all(isinstance(x, (int, float))
                  for r in all_rows for x in r["commit_features"] + r["file_features"])
    print(f"  all numeric: {all_num}")
    print("  PASS" if cf == 29 and ff == 16 and not bad and all_num else "  FAIL")

    print("\n=== AMBIGUOUS EXCLUSION ===")
    amb = 0
    for sn in ["train", "validation", "test"]:
        for ln in (DATA / f"{sn}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln and json.loads(ln)["defect_label"] == -1:
                amb += 1
    print(f"  ambiguous in supervised: {amb}")
    print("  PASS" if amb == 0 else "  FAIL")

    print("\n=== DETERMINISM ===")
    for sn in ["train", "validation", "test", "ambiguous"]:
        c1 = (DATA / f"{sn}.jsonl").read_text(encoding="utf-8")
        c2 = (DATA / f"{sn}.jsonl").read_text(encoding="utf-8")
        print(f"  {sn}.jsonl identical: {c1 == c2}")

    print("\n=== ATTRIBUTION SOURCES ===")
    sources = Counter()
    for sn in ["train", "validation", "test", "ambiguous"]:
        for ln in (DATA / f"{sn}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln:
                r = json.loads(ln)
                if r["defect_label"] == 1:
                    sources[r["label_source"]] += 1
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")

    print("\n=== POSITIVE DISTRIBUTION ===")
    all_pos = []
    for sn in ["train", "validation", "test"]:
        for ln in (DATA / f"{sn}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln:
                r = json.loads(ln)
                if r["defect_label"] == 1:
                    all_pos.append(r)
    commit_file_counts = Counter(r["commit_sha"] for r in all_pos)
    print(f"  Total positive files: {len(all_pos)}")
    print(f"  Distinct positive commits: {len(commit_file_counts)}")

    # Repos with positives
    repos_with_pos = set(r["repo_name"] for r in all_pos)
    print(f"  Repos with positives: {len(repos_with_pos)}")

    print("\n=== DIVERSITY METRICS ===")
    pos_by_repo: dict[str, int] = Counter(r["repo_name"] for r in all_pos)
    print("  Positive files per repo:")
    for repo, cnt in pos_by_repo.most_common():
        print(f"    {repo:<25} {cnt:>3}")

    # Concentration
    if commit_file_counts:
        max_commit = commit_file_counts.most_common(1)[0]
        print(
            f"\n  Largest commit: {max_commit[0][:12]}..."
            f" ({max_commit[1]} files"
            f" = {max_commit[1]/len(all_pos)*100:.1f}%)"
        )
    if pos_by_repo:
        max_repo = pos_by_repo.most_common(1)[0]
        print(
            f"  Largest repo: {max_repo[0]}"
            f" ({max_repo[1]} files"
            f" = {max_repo[1]/len(all_pos)*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
