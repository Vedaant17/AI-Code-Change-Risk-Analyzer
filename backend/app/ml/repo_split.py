"""Unseen-repository evaluation split for Phase 4.6.

Reads the frozen combined-v3 JSONL files and reassigns rows based on
a deterministic repository-level manifest.  The original chronological
split is NOT modified.

The manifest assigns each of the 50 repositories to exactly one of
TRAIN (17), VALIDATION (17), or TEST (16) using round-robin assignment
sorted by (positive_commit_count DESC, repo_name ASC).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from backend.app.dataset.schemas import DatasetRow
from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
)
from backend.app.ml.dataset_loader import (
    COMMIT_FEATURE_COUNT,
    FEATURE_NAMES,
    FILE_FEATURE_COUNT,
    NUM_FEATURES,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen manifest — the authoritative 50-repository split assignment.
# Round-robin by (positive_commit_count DESC, repo_name ASC).
# This MUST NOT be modified.
# ---------------------------------------------------------------------------

REPO_MANIFEST: dict[str, str] = {
    # TRAIN (17 repos)
    "aiohttp": "train",
    "bandit": "train",
    "falcon": "train",
    "flask-migrate": "train",
    "flask-sqlalchemy": "train",
    "flask-testing": "train",
    "invoke": "train",
    "jinja2": "train",
    "marshmallow": "train",
    "networkx": "train",
    "paramiko": "train",
    "pycodestyle": "train",
    "pymongo": "train",
    "rich": "train",
    "sqlalchemy": "train",
    "tornado": "train",
    "tox": "train",
    # VALIDATION (17 repos)
    "bottle": "validation",
    "fabric": "validation",
    "fastapi": "validation",
    "flask-wtf": "validation",
    "httpie": "validation",
    "isort": "validation",
    "peewee": "validation",
    "pre-commit": "validation",
    "pymemcache": "validation",
    "pytest": "validation",
    "pyyaml": "validation",
    "requests": "validation",
    "starlette": "validation",
    "sympy": "validation",
    "twisted": "validation",
    "urllib3": "validation",
    "uvicorn": "validation",
    # TEST (16 repos)
    "Pillow": "test",
    "black": "test",
    "celery": "test",
    "cherrypy": "test",
    "click": "test",
    "flake8": "test",
    "flask": "test",
    "httpx": "test",
    "itsdangerous": "test",
    "mypy": "test",
    "nox": "test",
    "psycopg2": "test",
    "pydantic": "test",
    "redis": "test",
    "treq": "test",
    "werkzeug": "test",
}

TRAIN_REPOS = sorted(k for k, v in REPO_MANIFEST.items() if v == "train")
VALIDATION_REPOS = sorted(k for k, v in REPO_MANIFEST.items() if v == "validation")
TEST_REPOS = sorted(k for k, v in REPO_MANIFEST.items() if v == "test")

assert len(REPO_MANIFEST) == 50, f"Expected 50 repos, got {len(REPO_MANIFEST)}"
assert len(TRAIN_REPOS) == 17, f"Expected 17 train repos, got {len(TRAIN_REPOS)}"
assert len(VALIDATION_REPOS) == 17, f"Expected 17 validation repos, got {len(VALIDATION_REPOS)}"
assert len(TEST_REPOS) == 16, f"Expected 16 test repos, got {len(TEST_REPOS)}"
assert len(set(TRAIN_REPOS) & set(VALIDATION_REPOS)) == 0, "Train-Val overlap"
assert len(set(TRAIN_REPOS) & set(TEST_REPOS)) == 0, "Train-Test overlap"
assert len(set(VALIDATION_REPOS) & set(TEST_REPOS)) == 0, "Val-Test overlap"


def load_all_rows(
    data_dir: Path,
) -> list[DatasetRow]:
    """Read every row from the frozen combined-v3 JSONL files.

    Reads train.jsonl, validation.jsonl, test.jsonl and returns a flat
    list of DatasetRow objects.  The original ``split`` field is preserved
    but is NOT used for the unseen-repository evaluation.
    """
    all_rows: list[DatasetRow] = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = data_dir / name
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
                all_rows.append(row)
    logger.info("Loaded %d total rows from %s", len(all_rows), data_dir)
    return all_rows


def reassign_splits(
    rows: list[DatasetRow],
) -> dict[str, list[DatasetRow]]:
    """Reassign rows to train/validation/test based on REPO_MANIFEST.

    Returns a dict with keys "train", "validation", "test", each
    containing the list of DatasetRow objects whose repo_name maps to
    that split in REPO_MANIFEST.

    Raises ValueError if a repo_name is not in REPO_MANIFEST.
    """
    result: dict[str, list[DatasetRow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    unknown_repos: set[str] = set()
    for row in rows:
        split = REPO_MANIFEST.get(row.repo_name)
        if split is None:
            unknown_repos.add(row.repo_name)
            continue
        result[split].append(row)
    if unknown_repos:
        raise ValueError(
            f"Unknown repositories in dataset: {sorted(unknown_repos)}. "
            "These are not in REPO_MANIFEST."
        )
    logger.info(
        "Reassigned splits: train=%d, validation=%d, test=%d",
        len(result["train"]),
        len(result["validation"]),
        len(result["test"]),
    )
    return result


def build_experimental_lookup(
    exp_path: Path,
) -> dict[tuple[str, str], list[float]]:
    """Build a lookup dict from (commit_sha, file_path) to 6-float vector.

    Reads the single experimental_features.jsonl file produced by Phase 4.5.
    """
    lookup: dict[tuple[str, str], list[float]] = {}
    if not exp_path.exists():
        logger.warning("Experimental features file not found: %s", exp_path)
        return lookup
    with open(exp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["commit_sha"], obj["file_path"])
            lookup[key] = obj["experimental_features"]
    logger.info("Built experimental lookup with %d entries", len(lookup))
    return lookup


def _rows_to_dataset(
    rows: list[DatasetRow],
    feature_names: list[str],
    mode_indices: list[int],
    exp_lookup: dict[tuple[str, str], list[float]],
    split_name: str,
) -> SupervisedDataset:
    """Convert a list of DatasetRow objects to a SupervisedDataset.

    Constructs feature vectors by concatenating commit_features (29) +
    file_features (16) + selected experimental features (len(mode_indices)).
    """
    mode_dim = NUM_FEATURES + len(mode_indices)
    if not rows:
        return SupervisedDataset(
            X=np.empty((0, mode_dim), dtype=np.float64),
            y=np.empty(0, dtype=np.int64),
            metadata=[],
            feature_names=list(feature_names),
        )

    X_rows: list[list[float]] = []
    y_labels: list[int] = []
    metadata: list[dict] = []

    for row in rows:
        cf = row.commit_features
        ff = row.file_features
        if len(cf) != COMMIT_FEATURE_COUNT:
            raise ValueError(
                f"Row commit_features has {len(cf)} elements, "
                f"expected {COMMIT_FEATURE_COUNT}"
            )
        if len(ff) != FILE_FEATURE_COUNT:
            raise ValueError(
                f"Row file_features has {len(ff)} elements, "
                f"expected {FILE_FEATURE_COUNT}"
            )
        base_vec = cf + ff
        key = (row.commit_sha, row.file_path)
        exp_vec = exp_lookup.get(key, [0.0] * EXPERIMENTAL_FEATURE_COUNT)
        selected_exp = [exp_vec[i] for i in mode_indices]
        X_rows.append(base_vec + selected_exp)
        y_labels.append(row.defect_label)
        metadata.append({
            "commit_sha": row.commit_sha,
            "file_path": row.file_path,
            "split": split_name,
            "label_status": row.label_status,
            "label_source": row.label_source,
            "label_confidence": row.label_confidence,
            "label_evidence": row.label_evidence,
            "commit_message": row.commit_message,
            "author": row.author,
            "file_status": row.file_status,
            "repo_name": row.repo_name,
        })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return SupervisedDataset(
        X=X,
        y=y,
        metadata=metadata,
        feature_names=list(feature_names),
    )


def build_repo_split_dataset(
    data_dir: Path,
    exp_dir: Path,
    mode: str = "E2",
) -> tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    """Build train/validation/test SupervisedDatasets under the unseen-repo split.

    Parameters
    ----------
    data_dir:
        Path to the frozen combined-v3 directory containing train.jsonl,
        validation.jsonl, test.jsonl.
    exp_dir:
        Path to the experimental-exp-4.5a/combined-v3 directory containing
        experimental_features.jsonl.
    mode:
        Ablation mode for experimental feature selection.  Must be E0, E1,
        E2, or E4.  Default E2 (baseline + 3 historical features).

    Returns
    -------
    (train, val, test) SupervisedDataset instances with feature dimensions:
        E0: 45, E1: 48, E2: 48, E4: 51
    """
    if mode == "E0":
        mode_indices: list[int] = []
    elif mode == "E1":
        mode_indices = [0, 1, 2]
    elif mode == "E2":
        mode_indices = [3, 4, 5]
    elif mode == "E4":
        mode_indices = list(range(EXPERIMENTAL_FEATURE_COUNT))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected E0, E1, E2, or E4.")

    exp_names = [EXPERIMENTAL_FEATURE_NAMES[i] for i in mode_indices]
    feature_names = FEATURE_NAMES + exp_names

    # Load all rows from frozen JSONL files
    all_rows = load_all_rows(data_dir)

    # Reassign splits based on repo manifest
    split_rows = reassign_splits(all_rows)

    # Build experimental feature lookup
    exp_path = exp_dir / "experimental_features.jsonl"
    exp_lookup = build_experimental_lookup(exp_path)

    # Build SupervisedDataset for each split
    train = _rows_to_dataset(
        split_rows["train"], feature_names, mode_indices, exp_lookup, "train"
    )
    val = _rows_to_dataset(
        split_rows["validation"], feature_names, mode_indices, exp_lookup, "validation"
    )
    test = _rows_to_dataset(
        split_rows["test"], feature_names, mode_indices, exp_lookup, "test"
    )

    logger.info(
        "Built repo-split dataset (mode=%s, dim=%d): "
        "train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
        mode, train.X.shape[1],
        len(train), train.positive_count,
        len(val), val.positive_count,
        len(test), test.positive_count,
    )
    return train, val, test


def save_manifest(output_dir: Path) -> Path:
    """Write repo_split_manifest.json with the full manifest and statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "repo_split_manifest.json"

    # Compute per-repo stats from the frozen data
    data_dir = output_dir.parent.parent / "data" / "datasets" / "combined-v3"
    all_rows = load_all_rows(data_dir)

    repo_stats: dict[str, dict] = {}
    for row in all_rows:
        repo = row.repo_name
        if repo not in repo_stats:
            repo_stats[repo] = {
                "total_rows": 0,
                "positive_rows": 0,
                "positive_commits": set(),
            }
        repo_stats[repo]["total_rows"] += 1
        if row.defect_label == 1:
            repo_stats[repo]["positive_rows"] += 1
            repo_stats[repo]["positive_commits"].add(row.commit_sha)

    # Build manifest entries
    repos_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for repo, split in sorted(REPO_MANIFEST.items()):
        default_stats = {"total_rows": 0, "positive_rows": 0, "positive_commits": set()}
        stats = repo_stats.get(repo, default_stats)
        repos_by_split[split].append({
            "repo_name": repo,
            "total_rows": stats["total_rows"],
            "positive_rows": stats["positive_rows"],
            "positive_commits": len(stats["positive_commits"]),
        })

    # Split summaries
    split_summaries = {}
    for split_name, repo_list in repos_by_split.items():
        total = sum(r["total_rows"] for r in repo_list)
        pos = sum(r["positive_rows"] for r in repo_list)
        pos_commits = sum(r["positive_commits"] for r in repo_list)
        split_summaries[split_name] = {
            "repo_count": len(repo_list),
            "total_rows": total,
            "positive_rows": pos,
            "positive_commits": pos_commits,
            "prevalence": pos / total if total > 0 else 0.0,
        }

    manifest = {
        "phase": "4.6",
        "method": "round-robin by (positive_commit_count DESC, repo_name ASC)",
        "repos": repos_by_split,
        "summaries": split_summaries,
        "total_repos": len(REPO_MANIFEST),
        "total_rows": sum(s["total_rows"] for s in split_summaries.values()),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_default, sort_keys=True)
        f.write("\n")

    logger.info("Manifest saved to %s", path)
    return path


def _json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
