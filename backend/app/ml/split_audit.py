"""Read-only audit of the original chronological combined-v3 split.

Documents repository overlap, positive prevalence, split sizes, and
limitations of the original split.  Does not modify any files.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from backend.app.dataset.schemas import DatasetRow

logger = logging.getLogger(__name__)


def audit_original_split(data_dir: Path) -> dict:
    """Audit the original chronological combined-v3 split.

    Reads train.jsonl, validation.jsonl, test.jsonl and computes:
    - Per-split sizes and positive counts
    - Repository overlap between splits
    - Positive commit distribution
    - Any commits appearing in multiple splits

    Returns a dict suitable for JSON serialization.
    """
    splits: dict[str, list[DatasetRow]] = {"train": [], "validation": [], "test": []}
    for split_name in ("train", "validation", "test"):
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    row = DatasetRow.model_validate(obj)
                except Exception as e:
                    raise ValueError(
                        f"Invalid row at line {line_no} in {path}: {e}"
                    ) from e
                splits[split_name].append(row)

    # Per-split stats
    split_stats = {}
    for split_name, rows in splits.items():
        total = len(rows)
        pos = sum(1 for r in rows if r.defect_label == 1)
        neg = sum(1 for r in rows if r.defect_label == 0)
        ambiguous = sum(1 for r in rows if r.defect_label == -1)
        repos = set(r.repo_name for r in rows)
        pos_commits = set(
            r.commit_sha for r in rows if r.defect_label == 1
        )
        split_stats[split_name] = {
            "total_rows": total,
            "positive_rows": pos,
            "negative_rows": neg,
            "ambiguous_rows": ambiguous,
            "prevalence": pos / total if total > 0 else 0.0,
            "num_repos": len(repos),
            "repos": sorted(repos),
            "num_positive_commits": len(pos_commits),
        }

    # Cross-split repository overlap
    train_repos = set(r.repo_name for r in splits["train"])
    val_repos = set(r.repo_name for r in splits["validation"])
    test_repos = set(r.repo_name for r in splits["test"])

    repo_overlap = {
        "train_val": sorted(train_repos & val_repos),
        "train_test": sorted(train_repos & test_repos),
        "val_test": sorted(val_repos & test_repos),
        "all_three": sorted(train_repos & val_repos & test_repos),
    }

    # Cross-split commit overlap
    train_commits = set(r.commit_sha for r in splits["train"])
    val_commits = set(r.commit_sha for r in splits["validation"])
    test_commits = set(r.commit_sha for r in splits["test"])

    commit_overlap = {
        "train_val": len(train_commits & val_commits),
        "train_test": len(train_commits & test_commits),
        "val_test": len(val_commits & test_commits),
    }

    # Positive repo concentration across splits
    pos_repo_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, rows in splits.items():
        for r in rows:
            if r.defect_label == 1:
                pos_repo_split[r.repo_name][split_name] += 1

    repos_in_multiple_splits = {
        repo: dict(counts)
        for repo, counts in pos_repo_split.items()
        if len(counts) > 1
    }

    return {
        "split_stats": split_stats,
        "repo_overlap": repo_overlap,
        "commit_overlap": commit_overlap,
        "repos_with_positives_in_multiple_splits": repos_in_multiple_splits,
        "total_repos": len(
            train_repos | val_repos | test_repos
        ),
    }


def save_split_audit(output_dir: Path, data_dir: Path) -> Path:
    """Run the audit and save to split_audit.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split_audit.json"
    result = audit_original_split(data_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    logger.info("Split audit saved to %s", path)
    return path


def _json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")