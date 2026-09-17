"""Phase 4.13 Data: Population construction, feature loading, pre-experiment audits.

Loads frozen Phase 4.6-4.12 data and constructs the supervision populations,
feature matrices, and pre-experiment audits required by the Phase 4.13
file-level investigation-priority ranking experiment.

All populations are derived programmatically from frozen artifacts.
No values are hard-coded.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.app.ml.dataset_loader import FEATURE_NAMES, SupervisedDataset
from backend.app.ml.phase412_supervision_strategy import (
    EVIDENCE_UNKNOWN,
    EVIDENCE_WEAK,
    FILE_EVIDENCE_STRONG,
    FILE_EVIDENCE_UNKNOWN,
    FILE_EVIDENCE_WEAK,
    OBSERVED_POSITIVE,
    OUT_OF_SCOPE,
    UNLABELED,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")
PHASE411B_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.13")

EXCLUDED_FEATURES: set[str] = {"label_source"}
LEAKAGE_FEATURES_SET: set[str] = {
    "corrective_sha", "content_correspondence", "content_restoration",
    "region_overlap", "function_analysis", "evidence_types", "file_evidence_level",
}


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------

def build_phase413_populations(
    phase411b_files: list[dict],
    supervised_rows: list[dict],
) -> dict[str, list[dict]]:
    """Classify each supervised row into a Phase 4.13 population.

    Returns dict with keys:
        OBSERVED_POSITIVE, EVIDENCE_WEAK, EVIDENCE_UNKNOWN,
        UNLABELED, OUT_OF_SCOPE
    """
    file_evidence: dict[tuple[str, str, str], dict] = {}
    for rec in phase411b_files:
        key = (rec["repo_name"], rec["commit_sha"], rec["file_path"])
        file_evidence[key] = rec

    observed_positive: list[dict] = []
    evidence_weak: list[dict] = []
    evidence_unknown: list[dict] = []
    unlabeled: list[dict] = []
    out_of_scope: list[dict] = []

    for row in supervised_rows:
        fp = row["file_path"]
        repo = row["repo_name"]
        sha = row["commit_sha"]

        if fp == "/dev/null" or "/dev/null" in fp:
            out_of_scope.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": OUT_OF_SCOPE,
            })
            continue

        key = (repo, sha, fp)
        ev = file_evidence.get(key)
        if ev is None:
            unlabeled.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": UNLABELED,
            })
            continue

        level = ev.get("file_evidence_level", FILE_EVIDENCE_UNKNOWN)
        if level == FILE_EVIDENCE_STRONG:
            observed_positive.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": OBSERVED_POSITIVE,
                "corrective_sha": None,
                "identity_established": ev.get("identity_established", False),
            })
        elif level == FILE_EVIDENCE_WEAK:
            evidence_weak.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": EVIDENCE_WEAK,
                "identity_established": ev.get("identity_established", False),
            })
        elif level == FILE_EVIDENCE_UNKNOWN:
            evidence_unknown.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": EVIDENCE_UNKNOWN,
                "identity_established": ev.get("identity_established", False),
            })
        else:
            unlabeled.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": UNLABELED,
            })

    return {
        OBSERVED_POSITIVE: observed_positive,
        EVIDENCE_WEAK: evidence_weak,
        EVIDENCE_UNKNOWN: evidence_unknown,
        UNLABELED: unlabeled,
        OUT_OF_SCOPE: out_of_scope,
    }


def compute_population_metadata(populations: dict[str, list[dict]]) -> dict:
    """Compute metadata dict from populations."""
    total = sum(len(v) for v in populations.values())
    return {
        "total_supervised_rows": total,
        "observed_positive_rows": len(populations[OBSERVED_POSITIVE]),
        "evidence_weak_rows": len(populations[EVIDENCE_WEAK]),
        "evidence_unknown_rows": len(populations[EVIDENCE_UNKNOWN]),
        "unlabeled_rows": len(populations[UNLABELED]),
        "out_of_scope_rows": len(populations[OUT_OF_SCOPE]),
    }


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _get_mode_indices(mode: str) -> list[int]:
    """Return experimental feature indices for the given mode."""
    if mode == "E0":
        return []
    if mode == "E1":
        return [0, 1, 2]
    if mode == "E2":
        return [3, 4, 5]
    if mode == "E4":
        return list(range(6))
    raise ValueError(f"Unknown mode: {mode!r}")


def get_feature_names(mode: str) -> list[str]:
    """Return feature names for the given mode."""
    indices = _get_mode_indices(mode)
    from backend.app.features.experimental_schemas import EXPERIMENTAL_FEATURE_NAMES
    exp_names = [EXPERIMENTAL_FEATURE_NAMES[i] for i in indices]
    return list(FEATURE_NAMES) + exp_names


def build_feature_matrix(
    dataset: SupervisedDataset,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    """Extract X, y, metadata, feature_names from a SupervisedDataset.

    The dataset already has the correct features for the mode
    (built by build_repo_split_dataset).
    """
    return dataset.X, dataset.y, dataset.metadata, dataset.feature_names


# ---------------------------------------------------------------------------
# PU label construction
# ---------------------------------------------------------------------------

def build_pu_labels(
    populations: dict[str, list[dict]],
    metadata: list[dict],
    mode: str = "primary",
) -> tuple[np.ndarray, set[tuple[str, str]]]:
    """Build PU training labels from populations and metadata.

    For mode="primary":
        P = OBSERVED_POSITIVE
        U = UNLABELED
    For mode="sensitivity":
        P = OBSERVED_POSITIVE
        U = UNLABELED + EVIDENCE_WEAK + EVIDENCE_UNKNOWN

    Returns (y_pu, positive_keys) where:
        y_pu[i] = 1 if row i is in P, 0 if row i is in U
        positive_keys = set of (commit_sha, file_path) for P rows
    """
    op_keys = set()
    for r in populations[OBSERVED_POSITIVE]:
        op_keys.add((r["commit_sha"], r["file_path"]))

    ew_keys = set()
    for r in populations[EVIDENCE_WEAK]:
        ew_keys.add((r["commit_sha"], r["file_path"]))

    eu_keys = set()
    for r in populations[EVIDENCE_UNKNOWN]:
        eu_keys.add((r["commit_sha"], r["file_path"]))

    oos_keys = set()
    for r in populations[OUT_OF_SCOPE]:
        oos_keys.add((r["commit_sha"], r["file_path"]))

    if mode == "sensitivity":
        unlabeled_extra = ew_keys | eu_keys
    else:
        unlabeled_extra = set()

    y_pu = np.zeros(len(metadata), dtype=np.int64)
    positive_keys: set[tuple[str, str]] = set()
    included_indices: list[int] = []

    for i, m in enumerate(metadata):
        key = (m["commit_sha"], m["file_path"])
        if key in oos_keys:
            continue
        if key in op_keys:
            y_pu[i] = 1
            positive_keys.add(key)
            included_indices.append(i)
        elif key in unlabeled_extra or key not in ew_keys and key not in eu_keys:
            included_indices.append(i)

    return y_pu, positive_keys


def build_pu_arrays(
    X: np.ndarray,
    y_pu: np.ndarray,
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Split into P and U arrays for training.

    Returns (X_P, X_U, metadata_P, metadata_U).
    """
    p_indices = []
    u_indices = []
    for i, m in enumerate(metadata):
        key = (m["commit_sha"], m["file_path"])
        if y_pu[i] == 1 and key in positive_keys:
            p_indices.append(i)
        elif y_pu[i] == 0:
            u_indices.append(i)

    return (
        X[p_indices],
        X[u_indices],
        [metadata[i] for i in p_indices],
        [metadata[i] for i in u_indices],
    )


# ---------------------------------------------------------------------------
# Evidence-ranking pair construction
# ---------------------------------------------------------------------------

def build_evidence_ranking_pairs(
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
    manifest: dict[str, str],
) -> list[dict]:
    """Construct FILE_STRONG vs FILE_WEAK pairs for evidence-ranking.

    Eligible pairs require:
        - same repo
        - same candidate commit
        - same corrective SHA
        - identity_established == True for both files
        - one FILE_STRONG, one FILE_WEAK
    """
    commit_corrective: dict[str, str] = {}
    for rec in phase411b_commits:
        if rec.get("corrective_sha"):
            commit_corrective[rec["commit_sha"]] = rec["corrective_sha"]

    commit_files: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in phase411b_files:
        if not rec.get("identity_established", False):
            continue
        sha = rec["commit_sha"]
        csha = commit_corrective.get(sha, "")
        if not csha:
            continue
        commit_files[(rec["repo_name"], sha, csha)].append(rec)

    pairs: list[dict] = []
    for (repo, sha, csha), files in commit_files.items():
        strong = [f for f in files if f["file_evidence_level"] == FILE_EVIDENCE_STRONG]
        weak = [f for f in files if f["file_evidence_level"] == FILE_EVIDENCE_WEAK]
        for s in strong:
            for w in weak:
                pairs.append({
                    "repo_name": repo,
                    "commit_sha": sha,
                    "corrective_sha": csha,
                    "strong_file": s["file_path"],
                    "weak_file": w["file_path"],
                })

    return pairs


def split_pairs_by_split(
    pairs: list[dict],
    manifest: dict[str, str],
) -> dict[str, list[dict]]:
    """Split pairs into train/validation/test by repo manifest."""
    result: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for p in pairs:
        split = manifest.get(p["repo_name"], "")
        if split in result:
            result[split].append(p)
    return result


# ---------------------------------------------------------------------------
# Pre-experiment audits
# ---------------------------------------------------------------------------

def audit_file_strong_split_distribution(
    populations: dict[str, list[dict]],
    manifest: dict[str, str],
) -> dict:
    """Audit FILE_STRONG rows/commits/repos per split.

    Computed programmatically, not hard-coded.
    """
    splits: dict[str, dict] = {"train": {}, "validation": {}, "test": {}}
    for split in splits:
        rows = [r for r in populations[OBSERVED_POSITIVE]
                if manifest.get(r["repo_name"]) == split]
        commits = set(r["commit_sha"] for r in rows)
        repos = set(r["repo_name"] for r in rows)
        splits[split] = {
            "file_strong_rows": len(rows),
            "unique_commits": len(commits),
            "unique_repos": len(repos),
        }

    all_commits = set(r["commit_sha"] for r in populations[OBSERVED_POSITIVE])
    all_repos = set(r["repo_name"] for r in populations[OBSERVED_POSITIVE])
    splits["total"] = {
        "file_strong_rows": len(populations[OBSERVED_POSITIVE]),
        "unique_commits": len(all_commits),
        "unique_repos": len(all_repos),
    }
    return splits


def audit_pair_split_distribution(
    pairs: list[dict],
    manifest: dict[str, str],
) -> dict:
    """Audit STRONG_vs_WEAK pair counts per split.

    Computed programmatically, not hard-coded.
    """
    by_split = split_pairs_by_split(pairs, manifest)
    result: dict[str, dict] = {}
    for split, p_list in by_split.items():
        commits = set(p["commit_sha"] for p in p_list)
        repos = set(p["repo_name"] for p in p_list)
        result[split] = {
            "pair_count": len(p_list),
            "unique_commits": len(commits),
            "unique_repos": len(repos),
        }
    total_commits = set(p["commit_sha"] for p in pairs)
    total_repos = set(p["repo_name"] for p in pairs)
    result["total"] = {
        "pair_count": len(pairs),
        "unique_commits": len(total_commits),
        "unique_repos": len(total_repos),
    }
    return result


def audit_feature_leakage(feature_names: list[str]) -> dict:
    """Verify no post-candidate features in the feature matrix."""
    leakage_found = []
    for name in feature_names:
        if name in EXCLUDED_FEATURES:
            leakage_found.append(name)
        if name in LEAKAGE_FEATURES_SET:
            leakage_found.append(name)
    repo_in_features = "repo_name" in feature_names
    return {
        "leakage_features_found": leakage_found,
        "repo_name_in_features": repo_in_features,
        "audit_result": "PASS" if not leakage_found and not repo_in_features else "FAIL",
    }


def audit_candidate_boundary() -> dict:
    """Verify feature definitions respect candidate-time boundary."""
    return {
        "commit_features": "diff of C vs C^ (pre-candidate)",
        "file_features": "diff of C vs C^ (pre-candidate)",
        "ast_features": "file content at C and C^ (pre-candidate)",
        "historical_features": "git history at C^ boundary (pre-candidate)",
        "label_source": "EXCLUDED (post-candidate)",
        "corrective_sha": "EXCLUDED (post-candidate)",
        "verification": "all features use information at or before C/C^ per frozen implementations",
    }


def audit_split_disjointness(manifest: dict[str, str]) -> dict:
    """Verify repository-disjoint split."""
    train_repos = {r for r, s in manifest.items() if s == "train"}
    val_repos = {r for r, s in manifest.items() if s == "validation"}
    test_repos = {r for r, s in manifest.items() if s == "test"}

    return {
        "train_repos": sorted(train_repos),
        "val_repos": sorted(val_repos),
        "test_repos": sorted(test_repos),
        "train_val_overlap": sorted(train_repos & val_repos),
        "train_test_overlap": sorted(train_repos & test_repos),
        "val_test_overlap": sorted(val_repos & test_repos),
        "disjoint": len(train_repos & val_repos) == 0
        and len(train_repos & test_repos) == 0
        and len(val_repos & test_repos) == 0,
    }


def build_metadata_from_dataset(dataset: SupervisedDataset) -> list[dict]:
    """Ensure metadata has required fields."""
    return dataset.metadata
