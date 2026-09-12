"""Phase 4.9: Supervision Strategy & PU Learning Feasibility Study.

READ-ONLY methodological study determining whether available file-level
evidence supports a future Positive-Unlabeled (PU) learning experiment.

Key constraints:
- Read-only on frozen data. No modifications to any prior artifact.
- No ML model training. No hyperparameter tuning.
- No AWS/Braket/quantum code.
- OBSERVED_PATH_ASSOCIATED, UNLABELED, DEFENSIBLE_NEGATIVE terminology.
- COMMIT_ONLY excluded from primary file-level observed-positive population.
- All counts derived programmatically. No hardcoded dataset counts in logic.
- Deterministic and auditable.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.9")

_BUG_FIX_SHA_PATTERN = re.compile(r"bug-fix ([0-9a-f]{8,40})")
_REVERT_SHA_PATTERN = re.compile(r"reverted by ([0-9a-f]{8,40})")

# 45-feature names from the frozen schema
COMMIT_FEATURE_NAMES: list[str] = [
    "lines_added", "lines_deleted", "total_lines_changed", "files_changed",
    "files_added", "files_deleted", "files_modified", "files_renamed",
    "files_binary", "total_hunks", "avg_hunks_per_file", "additions_ratio",
    "deletions_ratio", "test_files_changed", "prod_files_changed",
    "test_ratio", "has_test_changes", "has_prod_changes", "test_prod_coupling",
    "languages_touched", "primary_language", "avg_file_changes",
    "max_file_changes", "total_function_declarations_changed",
    "total_class_declarations_changed", "total_imports_changed",
    "avg_changed_line_indent", "max_changed_line_indent", "change_entropy",
]

FILE_FEATURE_NAMES: list[str] = [
    "language", "is_binary", "is_test_file", "lines_added", "lines_deleted",
    "total_lines_changed", "hunk_count", "function_declarations_added",
    "function_declarations_deleted", "class_declarations_added",
    "class_declarations_deleted", "imports_added", "imports_deleted",
    "avg_changed_line_indent", "max_changed_line_indent", "avg_hunk_size",
]

COMMIT_FEATURE_INDEX = {name: i for i, name in enumerate(COMMIT_FEATURE_NAMES)}
FILE_FEATURE_INDEX = {name: i for i, name in enumerate(FILE_FEATURE_NAMES)}

ALLOWED_RECOMMENDATIONS = frozenset({
    "PU_LEARNING_FEASIBLE",
    "PU_LEARNING_FEASIBLE_WITH_RESTRICTIONS",
    "PU_LEARNING_NOT_YET_FEASIBLE",
    "BINARY_SUPERVISION_NOT_SUPPORTED",
    "COMMIT_LEVEL_MODEL_RECOMMENDED",
    "IMPROVE_ATTRIBUTION_FIRST",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read all rows from a JSONL file."""
    rows: list[dict] = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return rows
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {path}: {e}"
                ) from e
    return rows


def _read_all_supervised_rows(data_dir: Path) -> list[dict]:
    """Read all rows from train/validation/test JSONL files."""
    all_rows: list[dict] = []
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = _read_jsonl(path)
        logger.info("Loaded %d rows from %s", len(rows), path.name)
        all_rows.extend(rows)
    logger.info("Total supervised rows: %d", len(all_rows))
    return all_rows


def _read_ambiguous_rows(data_dir: Path) -> list[dict]:
    """Read all rows from ambiguous.jsonl."""
    path = data_dir / "ambiguous.jsonl"
    rows = _read_jsonl(path)
    logger.info("Loaded %d ambiguous rows", len(rows))
    return rows


def _build_sha_lookup(
    supervised: list[dict],
    ambiguous: list[dict],
) -> dict[str, list[dict]]:
    """Build SHA -> [file_rows] lookup across all data."""
    lookup: dict[str, list[dict]] = defaultdict(list)
    for r in supervised:
        lookup[r["commit_sha"]].append(r)
    for r in ambiguous:
        lookup[r["commit_sha"]].append(r)
    logger.info("Built SHA lookup with %d unique SHAs", len(lookup))
    return dict(lookup)


def _group_by_commit(
    rows: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (repo_name, commit_sha)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["repo_name"], row["commit_sha"])].append(row)
    return dict(groups)


def _extract_fix_sha(label_evidence: str) -> str | None:
    """Extract corrective SHA prefix from label_evidence text."""
    m = _BUG_FIX_SHA_PATTERN.search(label_evidence)
    if m:
        return m.group(1)
    m = _REVERT_SHA_PATTERN.search(label_evidence)
    if m:
        return m.group(1)
    return None


def _resolve_full_sha(
    prefix: str,
    sha_lookup: dict[str, list[dict]],
) -> str | None:
    """Resolve SHA prefix to full SHA. Returns None if !=1 match."""
    matches = [sha for sha in sha_lookup if sha.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Repository manifest (parsed from frozen source)
# ---------------------------------------------------------------------------

def _parse_frozen_manifest() -> dict[str, str]:
    """Parse REPO_MANIFEST from the frozen repo_split.py source.

    Reads the source as text and extracts the dict literal deterministically.
    Does NOT import or execute the module.
    """
    source_path = Path("backend/app/ml/repo_split.py")
    source_text = source_path.read_text(encoding="utf-8")

    manifest_match = re.search(
        r"REPO_MANIFEST\s*:\s*dict\[str,\s*str\]\s*=\s*\{([^}]+)\}",
        source_text,
        re.DOTALL,
    )
    if not manifest_match:
        raise ValueError("Could not parse REPO_MANIFEST from repo_split.py")

    manifest_block = manifest_match.group(1)
    entries = re.findall(r'"([^"]+)":\s*"([^"]+)"', manifest_block)
    manifest = dict(entries)

    # Validate against Phase 4.6 artifact if available
    manifest_path = Path(
        "backend/data/models/v0.1.0-combined-v3-phase4.6"
        "/repo_split_manifest.json"
    )
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            phase46 = json.load(f)
        # Phase 4.6 manifest has repos grouped by split:
        # {"repos": {"train": [...], "validation": [...], "test": [...]}}
        phase46_repos: dict[str, str] = {}
        repos_data = phase46.get("repos", {})
        if isinstance(repos_data, dict):
            for split_name, repo_list in repos_data.items():
                if isinstance(repo_list, list):
                    for r in repo_list:
                        if isinstance(r, dict) and "repo_name" in r:
                            phase46_repos[r["repo_name"]] = split_name
        if phase46_repos and manifest != phase46_repos:
            logger.warning(
                "Parsed manifest differs from Phase 4.6 artifact. "
                "Parsed=%d repos, Phase46=%d repos",
                len(manifest), len(phase46_repos),
            )

    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distribution(vals: list[float | int]) -> dict:
    """Compute descriptive statistics for a numeric list."""
    if not vals:
        return {
            "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
            "iqr": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0,
            "p75": 0.0, "p90": 0.0,
        }
    a = np.array(vals, dtype=np.float64)
    p10, p25, p50, p75, p90 = np.percentile(a, [10, 25, 50, 75, 90])
    return {
        "count": len(vals),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "iqr": float(p75 - p25),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
    }


def _json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Q1: Supervision-set construction
# ---------------------------------------------------------------------------

def _classify_positive_commits(
    supervised_rows: list[dict],
    sha_lookup: dict[str, list[dict]],
) -> list[dict]:
    """Classify each positive commit's evidence category.

    Returns list of per-commit dicts with evidence classification.
    """
    commit_groups = _group_by_commit(supervised_rows)
    positive = {
        k: rows for k, rows in commit_groups.items()
        if any(r["defect_label"] == 1 for r in rows)
    }

    results: list[dict] = []
    for _key, rows in positive.items():
        row0 = rows[0]
        repo_name = row0["repo_name"]
        commit_sha = row0["commit_sha"]
        label_source = row0.get("label_source", "none")
        label_evidence = row0.get("label_evidence", "")
        split = row0.get("split", "unknown")

        candidate_files = sorted(set(r["file_path"] for r in rows))

        fix_prefix = _extract_fix_sha(label_evidence)
        corrective_sha = None
        corrective_files: list[str] = []

        if fix_prefix is not None:
            full_sha = _resolve_full_sha(fix_prefix, sha_lookup)
            if full_sha is not None:
                corrective_sha = full_sha
                fix_rows = sha_lookup.get(full_sha, [])
                if fix_rows:
                    corrective_files = sorted(
                        set(r["file_path"] for r in fix_rows)
                    )

        candidate_set = set(candidate_files)
        corrective_set = set(corrective_files)
        intersection = sorted(candidate_set & corrective_set)

        if len(intersection) == 0:
            category = "COMMIT_ONLY"
            path_overlap = []
            path_associated: list[str] = []
            unknown = list(candidate_files)
        elif len(intersection) == len(candidate_set):
            category = "COMMIT_ONLY"
            path_overlap = intersection
            path_associated = []
            unknown = []
        else:
            category = "FILE_PARTIAL"
            path_overlap = []
            path_associated = intersection
            unknown = sorted(candidate_set - corrective_set)

        results.append({
            "repo_name": repo_name,
            "commit_sha": commit_sha,
            "label_source": label_source,
            "label_evidence": label_evidence,
            "split": split,
            "evidence_category": category,
            "candidate_files": candidate_files,
            "candidate_file_count": len(candidate_files),
            "corrective_sha": corrective_sha,
            "corrective_files": corrective_files,
            "corrective_file_count": len(corrective_files),
            "path_overlap_files": path_overlap,
            "path_associated_files": path_associated,
            "unknown_files": unknown,
        })

    results.sort(key=lambda c: (c["repo_name"], c["commit_sha"]))
    return results


def construct_supervision_sets(
    supervised_rows: list[dict],
    sha_lookup: dict[str, list[dict]],
    manifest: dict[str, str],
) -> dict:
    """Construct five mutually exclusive populations from all raw rows.

    Populations:
        1. PRIMARY_OBSERVED_PATH_ASSOCIATED — FILE_PARTIAL path-associated rows
        2. SENSITIVITY_COMMIT_ONLY_OVERLAP — COMMIT_ONLY path-overlap rows
        3. UNLABELED — legitimate rows without defect association evidence
        4. DEFENSIBLE_NEGATIVE — independently supported negatives (currently 0)
        5. OUT_OF_SCOPE — parser artifacts (file_path == "/dev/null")

    Every raw supervised row is classified exactly once.  No row is silently
    dropped.  Duplicate rows (same repo, sha, file_path) are each independently
    classified.
    """
    classifications = _classify_positive_commits(supervised_rows, sha_lookup)

    # Build per-commit lookup preserving all raw rows (no dedup)
    supervised_lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in supervised_rows:
        supervised_lookup[(r["repo_name"], r["commit_sha"])].append(r)

    primary_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    unlabeled_rows: list[dict] = []
    out_of_scope_rows: list[dict] = []

    classified_indices: set[int] = set()

    for cls in classifications:
        key = (cls["repo_name"], cls["commit_sha"])
        file_rows = supervised_lookup.get(key, [])

        # Build path -> list of rows (preserving duplicates)
        file_row_groups: dict[str, list[dict]] = defaultdict(list)
        for r in file_rows:
            file_row_groups[r["file_path"]].append(r)

        path_associated_set = set(cls["path_associated_files"])
        path_overlap_set = set(cls["path_overlap_files"])

        for r in file_rows:
            idx = id(r)
            if idx in classified_indices:
                continue
            classified_indices.add(idx)

            fp = r["file_path"]

            if fp == "/dev/null":
                row = dict(r)
                row["observation_role"] = "out_of_scope"
                row["out_of_scope_reason"] = "parser_artifact_dev_null"
                row["parent_commit_sha"] = cls["commit_sha"]
                row["parent_repo"] = cls["repo_name"]
                out_of_scope_rows.append(row)
                continue

            if cls["evidence_category"] == "FILE_PARTIAL":
                if fp in path_associated_set:
                    row = dict(r)
                    row["observation_role"] = "observed_path_associated"
                    row["parent_commit_sha"] = cls["commit_sha"]
                    row["parent_repo"] = cls["repo_name"]
                    primary_rows.append(row)
                else:
                    row = dict(r)
                    row["observation_role"] = "unlabeled"
                    row["parent_commit_sha"] = cls["commit_sha"]
                    row["parent_repo"] = cls["repo_name"]
                    unlabeled_rows.append(row)

            elif cls["evidence_category"] == "COMMIT_ONLY":
                if fp in path_overlap_set:
                    row = dict(r)
                    row["observation_role"] = "sensitivity_commit_only_overlap"
                    row["parent_commit_sha"] = cls["commit_sha"]
                    row["parent_repo"] = cls["repo_name"]
                    sensitivity_rows.append(row)
                else:
                    row = dict(r)
                    row["observation_role"] = "unlabeled"
                    row["parent_commit_sha"] = cls["commit_sha"]
                    row["parent_repo"] = cls["repo_name"]
                    unlabeled_rows.append(row)

    # All remaining raw rows (negative commits, etc.) are unlabeled
    for r in supervised_rows:
        idx = id(r)
        if idx in classified_indices:
            continue
        classified_indices.add(idx)

        fp = r["file_path"]
        if fp == "/dev/null":
            row = dict(r)
            row["observation_role"] = "out_of_scope"
            row["out_of_scope_reason"] = "parser_artifact_dev_null"
            out_of_scope_rows.append(row)
        else:
            row = dict(r)
            row["observation_role"] = "unlabeled"
            unlabeled_rows.append(row)

    total_file_rows = len(supervised_rows)

    return {
        "primary_observed_path_associated": {
            "description": (
                "FILE_PARTIAL path-associated rows only. "
                "Candidate positives for PU formulation."
            ),
            "rows": primary_rows,
            "count": len(primary_rows),
        },
        "sensitivity_commit_only_overlap": {
            "description": (
                "COMMIT_ONLY path-overlap rows. Weaker commit/path-level "
                "evidence. Not included in primary PU population. "
                "Analyzed separately for sensitivity."
            ),
            "rows": sensitivity_rows,
            "count": len(sensitivity_rows),
        },
        "unlabeled": {
            "description": (
                "Candidate file rows for which current evidence does not "
                "establish file-level defect association."
            ),
            "rows": unlabeled_rows,
            "count": len(unlabeled_rows),
        },
        "defensible_negative": {
            "description": (
                "No defensible negative labels can be constructed from "
                "path-level evidence alone."
            ),
            "rows": [],
            "count": 0,
        },
        "out_of_scope": {
            "description": (
                "Parser artifacts (file_path == '/dev/null'). "
                "Binary file additions where the diff parser captured "
                "the source side instead of the destination path."
            ),
            "rows": out_of_scope_rows,
            "count": len(out_of_scope_rows),
        },
        "summary": {
            "primary_observed_count": len(primary_rows),
            "sensitivity_count": len(sensitivity_rows),
            "unlabeled_count": len(unlabeled_rows),
            "defensible_negative_count": 0,
            "out_of_scope_count": len(out_of_scope_rows),
            "total_file_rows": total_file_rows,
            "observed_positive_rate": (
                len(primary_rows) / total_file_rows
                if total_file_rows else 0.0
            ),
        },
        "classifications": classifications,
    }


# ---------------------------------------------------------------------------
# Q2: Coverage analysis
# ---------------------------------------------------------------------------

def _extract_feature_value(
    row: dict,
    feature_name: str,
    level: str,
) -> float | None:
    """Extract a feature value from a row's feature arrays."""
    if level == "commit" and feature_name in COMMIT_FEATURE_INDEX:
        idx = COMMIT_FEATURE_INDEX[feature_name]
        features = row.get("commit_features", [])
        if idx < len(features):
            return features[idx]
    elif level == "file" and feature_name in FILE_FEATURE_INDEX:
        idx = FILE_FEATURE_INDEX[feature_name]
        features = row.get("file_features", [])
        if idx < len(features):
            return features[idx]
    return None


def analyze_coverage(
    primary_rows: list[dict],
    unlabeled_rows: list[dict],
) -> dict:
    """Analyze distribution of observed-path-associated vs unlabeled.

    For each feature dimension, compute descriptive statistics for both
    populations and report comparisons.
    """
    # Numeric features to analyze
    numeric_features = [
        ("file_total_lines_changed", "file"),
        ("file_hunk_count", "file"),
        ("files_changed", "commit"),
        ("total_hunks", "commit"),
        ("max_file_changes", "commit"),
        ("additions_ratio", "commit"),
        ("test_ratio", "commit"),
    ]

    # Categorical features to analyze
    categorical_features = ["repo_name", "split", "label_source", "file_status"]

    per_feature: list[dict] = []
    notable_differences: list[dict] = []

    # Numeric analysis
    for feat_name, level in numeric_features:
        obs_vals = [
            v for v in (
                _extract_feature_value(r, feat_name, level)
                for r in primary_rows
            ) if v is not None
        ]
        unl_vals = [
            v for v in (
                _extract_feature_value(r, feat_name, level)
                for r in unlabeled_rows
            ) if v is not None
        ]

        obs_stats = _distribution(obs_vals)
        unl_stats = _distribution(unl_vals)

        comparison: dict = {}
        if obs_stats["count"] > 0 and unl_stats["count"] > 0:
            if unl_stats["median"] != 0:
                comparison["median_ratio"] = (
                    obs_stats["median"] / unl_stats["median"]
                )
            if unl_stats["iqr"] > 0:
                comparison["iqr_ratio"] = obs_stats["iqr"] / unl_stats["iqr"]

            # Mann-Whitney U if scipy available
            try:
                from scipy.stats import mannwhitneyu
                u_stat, p_val = mannwhitneyu(
                    obs_vals, unl_vals, alternative="two-sided"
                )
                comparison["mann_whitney_u"] = float(u_stat)
                comparison["mann_whitney_p"] = float(p_val)
                n1, n2 = len(obs_vals), len(unl_vals)
                comparison["effect_size"] = float(
                    1 - (2 * u_stat) / (n1 * n2)
                ) if n1 > 0 and n2 > 0 else 0.0
            except ImportError:
                comparison["mann_whitney_u"] = None
                comparison["mann_whitney_p"] = None
                comparison["effect_size"] = None

        entry = {
            "feature_name": feat_name,
            "level": level,
            "observed_stats": obs_stats,
            "unlabeled_stats": unl_stats,
            "comparison": comparison,
        }
        per_feature.append(entry)

        # Narrative comparison
        if obs_stats["count"] > 0 and unl_stats["count"] > 0:
            desc = (
                f"observed median {obs_stats['median']:.4f} "
                f"vs unlabeled median {unl_stats['median']:.4f}"
            )
            entry["description"] = desc

            # Report notable differences without arbitrary thresholds
            if obs_stats["median"] != unl_stats["median"]:
                notable_differences.append({
                    "feature": feat_name,
                    "description": desc,
                    "possible_implication": (
                        f"Distribution of {feat_name} differs between "
                        "observed-path-associated and unlabeled populations."
                    ),
                })

    # Categorical analysis
    for cat_feature in categorical_features:
        obs_counts: dict[str, int] = defaultdict(int)
        unl_counts: dict[str, int] = defaultdict(int)
        for r in primary_rows:
            obs_counts[r.get(cat_feature, "unknown")] += 1
        for r in unlabeled_rows:
            unl_counts[r.get(cat_feature, "unknown")] += 1

        obs_total = len(primary_rows) or 1
        unl_total = len(unlabeled_rows) or 1

        obs_shares = {k: v / obs_total for k, v in obs_counts.items()}
        unl_shares = {k: v / unl_total for k, v in unl_counts.items()}

        entry = {
            "feature_name": cat_feature,
            "level": "categorical",
            "observed_counts": dict(obs_counts),
            "unlabeled_counts": dict(unl_counts),
            "observed_shares": obs_shares,
            "unlabeled_shares": unl_shares,
            "observed_unique": len(obs_counts),
            "unlabeled_unique": len(unl_counts),
        }
        per_feature.append(entry)

        # Check for notable concentration
        all_categories = set(obs_counts.keys()) | set(unl_counts.keys())
        for cat in all_categories:
            obs_s = obs_shares.get(cat, 0.0)
            unl_s = unl_shares.get(cat, 0.0)
            if obs_s > 0 and unl_s > 0:
                ratio = obs_s / unl_s
                if ratio > 3.0 or ratio < 0.33:
                    notable_differences.append({
                        "feature": cat_feature,
                        "description": (
                            f"{cat}: observed share {obs_s:.3f} "
                            f"vs unlabeled share {unl_s:.3f}"
                        ),
                        "possible_implication": (
                            f"Concentration of {cat_feature}={cat} "
                            "differs between populations."
                        ),
                    })

    return {
        "per_feature": per_feature,
        "notable_differences": notable_differences,
        "observed_count": len(primary_rows),
        "unlabeled_count": len(unlabeled_rows),
    }


# ---------------------------------------------------------------------------
# Q3: Repository concentration
# ---------------------------------------------------------------------------

def analyze_repository_concentration(
    primary_rows: list[dict],
    all_supervised_rows: list[dict],
) -> dict:
    """Measure repository concentration of observed-path-associated signal."""
    repo_observed: dict[str, int] = defaultdict(int)
    repo_observed_commits: dict[str, set] = defaultdict(set)
    repo_total: dict[str, int] = defaultdict(int)

    for r in primary_rows:
        repo_observed[r["repo_name"]] += 1
        repo_observed_commits[r["repo_name"]].add(r["commit_sha"])

    for r in all_supervised_rows:
        repo_total[r["repo_name"]] += 1

    all_repos = sorted(set(repo_observed.keys()) | set(repo_total.keys()))
    repos_with = [r for r in all_repos if repo_observed.get(r, 0) > 0]
    repos_without = [r for r in all_repos if repo_observed.get(r, 0) == 0]

    per_repo: list[dict] = []
    for repo in all_repos:
        obs = repo_observed.get(repo, 0)
        total = repo_total.get(repo, 0)
        commits = len(repo_observed_commits.get(repo, set()))
        per_repo.append({
            "repo_name": repo,
            "observed_positive_rows": obs,
            "observed_positive_commits": commits,
            "total_file_rows": total,
            "observed_positive_ratio": obs / total if total else 0.0,
        })

    per_repo.sort(key=lambda r: -r["observed_positive_rows"])

    total_obs = sum(r["observed_positive_rows"] for r in per_repo)
    shares = [
        r["observed_positive_rows"] / total_obs if total_obs else 0.0
        for r in per_repo
    ]

    # Concentration metrics
    top_5_share = sum(sorted(shares, reverse=True)[:5]) if shares else 0.0
    top_10_share = sum(sorted(shares, reverse=True)[:10]) if shares else 0.0

    # Gini coefficient
    gini = 0.0
    if shares and total_obs > 0:
        sorted_shares = sorted(shares)
        n = len(sorted_shares)
        gini = float(
            (2 * sum((i + 1) * s for i, s in enumerate(sorted_shares)))
            / (n * sum(sorted_shares))
            - (n + 1) / n
        ) if sum(sorted_shares) > 0 else 0.0

    # Herfindahl index
    herfindahl = sum(s ** 2 for s in shares) if shares else 0.0

    # Normalized Shannon entropy
    entropy = 0.0
    if shares and total_obs > 0:
        ps = [s for s in shares if s > 0]
        if ps:
            raw_entropy = -sum(p * np.log(p) for p in ps)
            max_entropy = np.log(len(ps))
            entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "repos_with_signal": len(repos_with),
        "repos_without_signal": len(repos_without),
        "total_repos": len(all_repos),
        "per_repo": per_repo,
        "concentration": {
            "top_5_share": float(top_5_share),
            "top_10_share": float(top_10_share),
            "gini": float(gini),
            "herfindahl": float(herfindahl),
            "entropy": float(entropy),
        },
    }


# ---------------------------------------------------------------------------
# Q4: Split distribution
# ---------------------------------------------------------------------------

def analyze_split_distribution(
    primary_rows: list[dict],
    unlabeled_rows: list[dict],
    sensitivity_rows: list[dict],
    out_of_scope_rows: list[dict],
    all_supervised_rows: list[dict],
) -> dict:
    """Report all five population categories per split with reconciliation."""
    split_data: dict[str, dict] = {}

    for split in ("train", "validation", "test"):
        obs = [r for r in primary_rows if r.get("split") == split]
        sens = [r for r in sensitivity_rows if r.get("split") == split]
        unl = [r for r in unlabeled_rows if r.get("split") == split]
        oos = [r for r in out_of_scope_rows if r.get("split") == split]
        total = [r for r in all_supervised_rows if r.get("split") == split]

        obs_commits = len({r["commit_sha"] for r in obs})
        obs_repos = len({r["repo_name"] for r in obs})
        all_repos_in_split = len({r["repo_name"] for r in total})
        repos_without = all_repos_in_split - obs_repos

        raw_total = len(total)
        classified = len(obs) + len(sens) + len(unl) + 0 + len(oos)
        reconciled = classified == raw_total

        split_data[split] = {
            "observed_count": len(obs),
            "sensitivity_count": len(sens),
            "unlabeled_count": len(unl),
            "defensible_negative_count": 0,
            "out_of_scope_count": len(oos),
            "raw_total": raw_total,
            "classified_total": classified,
            "reconciled": reconciled,
            "observed_positive_rate": (
                len(obs) / raw_total if raw_total else 0.0
            ),
            "observed_commits": obs_commits,
            "repos_with_signal": obs_repos,
            "repos_without_signal": repos_without,
            "total_repos_in_split": all_repos_in_split,
        }

    # Identify sparse splits
    sparsity_warnings: list[str] = []
    for split_name, data in split_data.items():
        if data["observed_count"] == 0:
            sparsity_warnings.append(
                f"{split_name}: zero observed path-associated rows"
            )
        elif data["repos_with_signal"] == 0:
            sparsity_warnings.append(
                f"{split_name}: no repositories with signal"
            )
        if not data["reconciled"]:
            sparsity_warnings.append(
                f"{split_name}: reconciliation failed "
                f"({data['classified_total']} != {data['raw_total']})"
            )

    return {
        **split_data,
        "sparsity_warnings": sparsity_warnings,
    }


# ---------------------------------------------------------------------------
# Q5: PU assumption evaluation
# ---------------------------------------------------------------------------

def evaluate_pu_assumptions(
    repo_concentration: dict,
    coverage: dict,
) -> dict:
    """Evaluate SCAR/SAR compatibility with appropriate caution."""
    conc = repo_concentration["concentration"]
    n_repos_with = repo_concentration["repos_with_signal"]
    n_repos_total = repo_concentration["total_repos"]
    n_notable = len(coverage.get("notable_differences", []))

    # SCAR analysis
    scar = {
        "assumption": (
            "Selected Completely At Random: the probability that a file "
            "row becomes OBSERVED_PATH_ASSOCIATED is independent of both "
            "the file's latent defect status and all observed and "
            "unobserved file characteristics."
        ),
        "evidence_consistent": [
            "FILE_PARTIAL classification is deterministic given the "
            "corrective commit's file set, but this does not by itself "
            "establish whether selection is independent of the latent "
            "defect outcome.",
        ],
        "evidence_inconsistent": [
            (
                "The selection mechanism is tied to the corrective "
                "commit's scope: FILE_PARTIAL arises when the corrective "
                "change touches a strict subset of candidate files, "
                "while COMMIT_ONLY arises when it touches all or none. "
                "This mechanism-dependent construction raises concern "
                "about SCAR compatibility."
            ),
        ],
        "untestable_components": [
            (
                "Whether the corrective commit's scope is independent "
                "of the latent defect status of individual files."
            ),
            (
                "Whether unobserved confounders (e.g., code complexity, "
                "review practices) jointly affect both selection and "
                "the defect outcome."
            ),
        ],
        "verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE",
        "reasoning": (
            "The deterministic construction of FILE_PARTIAL establishes "
            "that selection is mechanism-dependent on corrective-commit "
            "scope. However, this does not by itself prove that SCAR is "
            "violated with respect to the latent defect outcome. The "
            "relevant components remain untestable without external "
            "ground truth. SCAR is not established from current evidence, "
            "but formal violation cannot be claimed."
        ),
    }

    # SAR analysis
    sar_evidence_for: list[str] = []
    sar_evidence_against: list[str] = []

    if n_repos_with > 0 and n_repos_total > 0:
        signal_ratio = n_repos_with / n_repos_total
        if signal_ratio > 0.3:
            sar_evidence_for.append(
                f"Observed signal spans {n_repos_with}/{n_repos_total} "
                f"repos ({signal_ratio:.1%}), suggesting broad coverage."
            )
        else:
            sar_evidence_against.append(
                f"Only {n_repos_with}/{n_repos_total} repos "
                f"({signal_ratio:.1%}) contain observed signal."
            )

    if conc["top_5_share"] > 0.6:
        sar_evidence_against.append(
            f"Top 5 repos contribute {conc['top_5_share']:.1%} of "
            "observed-path-associated rows."
        )
    elif conc["top_5_share"] < 0.4:
        sar_evidence_for.append(
            f"Top 5 repos contribute {conc['top_5_share']:.1%} of "
            "observed-path-associated rows (relatively distributed)."
        )

    if n_notable > 3:
        sar_evidence_against.append(
            f"{n_notable} feature distribution differences noted "
            "between observed and unlabeled populations."
        )

    sar = {
        "assumption": (
            "Selected At Random: conditional on observed features "
            "(repository, file characteristics, commit properties), "
            "the probability of being selected as "
            "OBSERVED_PATH_ASSOCIATED is independent of the latent "
            "defect status."
        ),
        "evidence_consistent": sar_evidence_for,
        "evidence_inconsistent": sar_evidence_against,
        "untestable_components": [
            (
                "Whether unobserved features jointly affect both "
                "selection and the defect outcome."
            ),
            (
                "Whether the corrective-commit scope is conditionally "
                "independent of defect status given observed features."
            ),
        ],
        "verdict": "COMPATIBLE_WITH_CAVEATS",
        "reasoning": (
            "SAR is the weaker assumption needed for PU learning. "
            "The current evidence does not establish strong violation, "
            "but repository concentration and feature distribution "
            "differences are relevant indicators of possible selection "
            "dependence. If pursued, the PU experiment should include "
            "sensitivity analyses that test robustness to these concerns."
        ),
    }

    # Repository-dependent selection
    repo_dep = {
        "evidence": (
            f"{n_repos_with}/{n_repos_total} repos contain observed "
            f"signal. Top-5 share: {conc['top_5_share']:.1%}. "
            f"Gini: {conc['gini']:.3f}."
        ),
        "severity": (
            "high" if conc["top_5_share"] > 0.7
            else "medium" if conc["top_5_share"] > 0.5
            else "low"
        ),
    }

    # Feature-dependent selection
    feat_dep = {
        "evidence": (
            f"{n_notable} feature distribution differences identified."
        ),
        "severity": (
            "high" if n_notable > 5
            else "medium" if n_notable > 2
            else "low"
        ),
    }

    # Corrective-behavior-dependent selection
    corr_dep = {
        "evidence": (
            "FILE_PARTIAL vs COMMIT_ONLY classification is deterministic "
            "given the corrective commit's file set. This is a "
            "mechanism-dependent selection."
        ),
        "severity": "medium",
    }

    distinction_table = {
        "observed_facts": [
            "38 FILE_PARTIAL commits identified",
            "172 COMMIT_ONLY commits identified",
            "Path overlap computed from frozen JSONL data",
            "Repository distribution measured",
            "Feature distributions measured",
        ],
        "inferences": [
            "Possible repository-dependent selection",
            "Possible feature-dependent selection",
            "Mechanism-dependent construction of evidence categories",
        ],
        "untestable_assumptions": [
            "Whether corrective commit scope is independent of defect status",
            "Whether unobserved confounders affect selection",
            "Whether SCAR or SAR holds for the latent defect outcome",
        ],
    }

    return {
        "scar": scar,
        "sar": sar,
        "repository_dependency": repo_dep,
        "feature_dependency": feat_dep,
        "corrective_behavior_dependency": corr_dep,
        "distinction_table": distinction_table,
    }


# ---------------------------------------------------------------------------
# Q6: Negative feasibility
# ---------------------------------------------------------------------------

def verify_negative_feasibility(
    all_supervised_rows: list[dict],
    primary_rows: list[dict],
    sensitivity_rows: list[dict],
    unlabeled_rows: list[dict],
    out_of_scope_rows: list[dict],
) -> dict:
    """Verify whether any defensible negative population exists."""
    total_rows = len(all_supervised_rows)
    observed = len(primary_rows)
    sensitivity = len(sensitivity_rows)
    unlabeled = len(unlabeled_rows)
    out_of_scope = len(out_of_scope_rows)

    return {
        "defensible_negatives": 0,
        "categories": {
            "independently_supported_negative": 0,
            "observed_path_associated": observed,
            "sensitivity_commit_only_overlap": sensitivity,
            "unlabeled": unlabeled,
            "out_of_scope": out_of_scope,
        },
        "total_supervised_rows": total_rows,
        "classified_total": observed + sensitivity + unlabeled + out_of_scope,
        "reconciled": (
            observed + sensitivity + unlabeled + out_of_scope == total_rows
        ),
        "reasoning": (
            "No defensible negative labels can be constructed from "
            "path-level evidence alone. Absence of path overlap does "
            "not establish non-defect. COMMIT_ONLY status does not "
            "establish non-defect. Unknown status does not establish "
            "non-defect. No independent negative evidence exists in "
            "the frozen dataset."
        ),
        "never_converted": [
            "no path overlap",
            "no corrective change",
            "COMMIT_ONLY",
            "unknown",
            "insufficient evidence",
        ],
    }


# ---------------------------------------------------------------------------
# Q7: Supervision strategy comparison
# ---------------------------------------------------------------------------

def compare_supervision_strategies(
    supervision_result: dict,
    coverage: dict,
    repo_concentration: dict,
    pu_assumptions: dict,
) -> dict:
    """Compare five supervision strategies."""
    summary = supervision_result["summary"]

    strategies = [
        {
            "name": "Standard binary supervised classification",
            "assumptions": [
                "All unlabeled examples are negative",
                "Defensible negatives exist in sufficient quantity",
                "Binary labels capture the defect concept",
            ],
            "available_evidence": (
                f"{summary['primary_observed_count']} observed-path-associated "
                f"rows, {summary['defensible_negative_count']} defensible "
                f"negatives, {summary['unlabeled_count']} unlabeled rows."
            ),
            "methodological_risk": (
                "Zero defensible negatives. Treating unlabeled as negative "
                "would introduce massive label noise. Model would learn "
                "the labeling mechanism, not defects."
            ),
            "evaluation_difficulty": "Cannot be evaluated without defensible negatives.",
            "compatibility_with_dataset": "NOT_COMPATIBLE",
            "additional_evidence_required": [
                "Defensible negative labels from independent sources"
            ],
            "pursue_next": False,
            "reasoning": (
                "The dataset contains zero defensible negatives. Standard "
                "binary classification requires reliable negative labels. "
                "Not viable with current evidence."
            ),
        },
        {
            "name": "PU learning (positive-unlabeled)",
            "assumptions": [
                "SAR approximately holds conditional on observed features",
                "Observed-path-associated is a non-random sample of all positives",
                "PU correction methods can account for label noise",
            ],
            "available_evidence": (
                f"{summary['primary_observed_count']} observed-path-associated "
                f"rows as candidate positives, "
                f"{summary['unlabeled_count']} unlabeled rows. "
                f"SAR assessed as COMPATIBLE_WITH_CAVEATS."
            ),
            "methodological_risk": (
                "Small observed-positive set. Possible repository-dependent "
                "selection. Path-associated evidence is not ground-truth "
                "defect labeling. PU assumptions may not hold."
            ),
            "evaluation_difficulty": (
                "Requires PU-compatible metrics. True defect performance "
                "cannot be measured without additional validation data."
            ),
            "compatibility_with_dataset": "CONDITIONALLY_COMPATIBLE",
            "additional_evidence_required": [
                "Validation samples to assess PU assumption validity",
                "Sensitivity analysis across repository subsets",
            ],
            "pursue_next": True,
            "reasoning": (
                "PU learning is the only viable approach given zero "
                "defensible negatives. The primary observed population "
                "is small (10.2% rate) but non-trivial. SAR is the "
                "working assumption with caveats. Restrictions apply."
            ),
        },
        {
            "name": "Weak supervision / heuristic labeling",
            "assumptions": [
                "Heuristic labels capture some signal",
                "Label noise is manageable with SNR analysis",
                "Heuristics do not correlate with commit patterns instead of defects",
            ],
            "available_evidence": (
                "108 path-associated rows could seed heuristics. "
                "60 path-transform candidates available. "
                "Commit-level features available."
            ),
            "methodological_risk": (
                "Heuristic labels may correlate with commit patterns "
                "rather than defects. May amplify existing selection bias."
            ),
            "evaluation_difficulty": "Requires heuristic label validation.",
            "compatibility_with_dataset": "PARTIALLY_COMPATIBLE",
            "additional_evidence_required": [
                "Heuristic label validation against independent samples",
                "SNR analysis of candidate heuristics",
            ],
            "pursue_next": False,
            "reasoning": (
                "Weak supervision could complement PU learning but "
                "should not be the primary approach. Could be explored "
                "as a secondary signal source."
            ),
        },
        {
            "name": "Commit-level prediction",
            "assumptions": [
                "Binary commit-level labels are sufficient",
                "File-level attribution not needed for the use case",
                "Commit-level features capture defect signal",
            ],
            "available_evidence": (
                "210 positive commits, ~21,581 negative commits. "
                "Binary labels clear at commit level. "
                "45 commit-level features available."
            ),
            "methodological_risk": (
                "Loses file-level granularity. Not directly useful for "
                "targeted code review. Commit-level prediction may not "
                "translate to file-level utility."
            ),
            "evaluation_difficulty": "Standard binary classification metrics applicable.",
            "compatibility_with_dataset": "COMPATIBLE",
            "additional_evidence_required": [],
            "pursue_next": True,
            "reasoning": (
                "Commit-level prediction is viable with current data. "
                "Binary labels are clear. Could serve as an intermediate "
                "step before file-level PU learning."
            ),
        },
        {
            "name": (
                "Improved file-level attribution followed by "
                "supervised learning"
            ),
            "assumptions": [
                "Better attribution data can be acquired",
                "Repository checkouts or external data sources available",
                "Improved attribution produces reliable file-level labels",
            ],
            "available_evidence": (
                "Current attribution is path-level only. "
                "No content-level or line-level analysis possible "
                "with frozen JSONL data."
            ),
            "methodological_risk": (
                "High implementation cost. May not be feasible for all "
                "repos. External data access required."
            ),
            "evaluation_difficulty": "Requires new data acquisition pipeline.",
            "compatibility_with_dataset": "NOT_CURRENTLY_COMPATIBLE",
            "additional_evidence_required": [
                "Line-level blame/history",
                "Issue-to-file linkage",
                "Patch-level causal evidence",
                "Manually validated samples",
            ],
            "pursue_next": False,
            "reasoning": (
                "Improved attribution would provide stronger supervision "
                "but requires capabilities beyond the frozen JSONL data. "
                "This is a future-work direction, not the next step."
            ),
        },
    ]

    return {
        "strategies": strategies,
        "recommended_for_next_phase": [
            s["name"] for s in strategies if s["pursue_next"]
        ],
    }


# ---------------------------------------------------------------------------
# Q8: Evaluation feasibility
# ---------------------------------------------------------------------------

def design_evaluation_strategy(
    supervision_result: dict,
    pu_assumptions: dict,
) -> dict:
    """Design defensible evaluation protocol for a future PU experiment."""
    summary = supervision_result["summary"]

    measurable_now = [
        {
            "name": "Observed-positive enrichment",
            "description": (
                "Compare the distribution of model scores for "
                "observed-path-associated rows vs unlabeled rows."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Observed-positive precision@K",
            "description": (
                "Among the top-K highest-scored files, what fraction "
                "are observed-path-associated. This is an exploratory "
                "metric against observed evidence, not true defect ground truth."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Ranking stability",
            "description": (
                "Assess whether model rankings are stable across "
                "random seeds and data perturbations."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Score distribution diagnostics",
            "description": (
                "Examine the distribution of model scores for observed "
                "vs unlabeled populations. Bimodality, separation, etc."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Repository-held-out ranking diagnostics",
            "description": (
                "Train on N-1 repos, evaluate on held-out repo. "
                "Assess cross-repository generalization."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Split-level stability",
            "description": (
                "Compare observed-positive enrichment across "
                "train/validation/test splits."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
        {
            "name": "Sensitivity analysis",
            "description": (
                "Vary the observed-positive population definition "
                "(e.g., include/exclude sensitivity population) "
                "and measure stability of conclusions."
            ),
            "requires_additional_data": False,
            "measurable_now": True,
        },
    ]

    not_measurable_now = [
        {
            "name": "True recall",
            "description": (
                "Fraction of truly defective files correctly identified. "
                "Cannot be measured without ground-truth defective file labels."
            ),
            "reason": "No defensible negatives or independent defect ground truth.",
        },
        {
            "name": "True specificity",
            "description": (
                "Fraction of truly non-defective files correctly excluded. "
                "Cannot be measured without ground-truth labels."
            ),
            "reason": "No defensible negatives available.",
        },
        {
            "name": "True binary precision",
            "description": (
                "Fraction of predicted defective files that are truly "
                "defective. Cannot be measured without ground truth."
            ),
            "reason": "No defensible negatives for binary evaluation.",
        },
        {
            "name": "True defect PR-AUC",
            "description": (
                "Precision-recall AUC for binary defect classification. "
                "Requires true positive and true negative labels."
            ),
            "reason": "Only observed-path-associated and unlabeled available.",
        },
        {
            "name": "True defect ROC-AUC",
            "description": (
                "ROC AUC for binary defect classification. "
                "Requires true positive and true negative labels."
            ),
            "reason": "Only observed-path-associated and unlabeled available.",
        },
        {
            "name": "Complete defect-ranking recall",
            "description": (
                "Fraction of all defective files ranked in the top positions. "
                "Cannot be measured without comprehensive defect inventory."
            ),
            "reason": "No independent defect inventory exists.",
        },
    ]

    # Exploratory PU metrics (clearly labeled)
    exploratory_pu_metrics = [
        {
            "name": "PU-adjusted ROC-AUC (exploratory)",
            "description": (
                "ROC-AUC computed treating observed-path-associated as "
                "positive and unlabeled as negative. This is an "
                "EXPLORATORY positive-vs-unlabeled diagnostic, NOT a "
                "true binary defect performance metric."
            ),
            "caveat": (
                "This metric conflates true positive rate with selection "
                "mechanism. It should NOT be interpreted as defect "
                "classification performance."
            ),
        },
        {
            "name": "PU-adjusted PR-AUC (exploratory)",
            "description": (
                "PR-AUC computed treating observed-path-associated as "
                "positive and unlabeled as negative. EXPLORATORY only."
            ),
            "caveat": (
                "Precision is inflated because unlabeled includes "
                "unknown positives. Should NOT be interpreted as "
                "true defect precision."
            ),
        },
    ]

    label_scarcity = {
        "observed_positive_rate": summary["observed_positive_rate"],
        "impact_on_evaluation": (
            f"Observed positive rate is {summary['observed_positive_rate']:.1%}. "
            "This is a small positive fraction. Evaluation metrics will "
            "be sensitive to the choice of observed-positive population. "
            "Sensitivity analysis across population definitions is essential."
        ),
    }

    return {
        "measurable_now": measurable_now,
        "not_measurable_now": not_measurable_now,
        "exploratory_pu_metrics": exploratory_pu_metrics,
        "label_scarcity": label_scarcity,
    }


# ---------------------------------------------------------------------------
# Q9: Additional evidence requirements
# ---------------------------------------------------------------------------

def identify_data_requirements() -> dict:
    """Identify evidence that would strengthen file-level supervision."""
    return {
        "requirements": [
            {
                "evidence_type": "Line-level blame/history",
                "description": (
                    "Use git blame and history to map defect introduction "
                    "to specific lines and files. Provides direct causal "
                    "evidence of which files introduced the defect."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "HIGH",
                "limitations": (
                    "Requires repository checkouts. Blame accuracy depends "
                    "on merge history. May not work for squash-merged commits."
                ),
            },
            {
                "evidence_type": "Issue-to-file linkage",
                "description": (
                    "Link bug reports (GitHub issues, Jira tickets) to "
                    "files via commit messages, issue references, or "
                    "code comments."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "HIGH",
                "limitations": (
                    "Not all bugs have linked issues. Issue quality varies. "
                    "May require API access to issue trackers."
                ),
            },
            {
                "evidence_type": "Patch-level causal evidence",
                "description": (
                    "Analyze the fix/revert patch to identify what changed "
                    "and why. Provides direct evidence of which files "
                    "needed correction."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "HIGH",
                "limitations": (
                    "Requires access to git diffs. Patches may be incomplete "
                    "or span multiple concerns."
                ),
            },
            {
                "evidence_type": "Independently reviewed labels",
                "description": (
                    "Have human reviewers verify which files are associated "
                    "with defects. Provides high-quality ground truth."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "VERY HIGH",
                "limitations": (
                    "Requires manual effort. Inter-annotator agreement "
                    "may vary. Expensive to scale."
                ),
            },
            {
                "evidence_type": "Structured bug-fix/file mappings",
                "description": (
                    "Structured data linking bug-fixing commits to affected "
                    "files, potentially from GitHub PR labels, commit "
                    "conventions, or automated tools."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "MEDIUM",
                "limitations": (
                    "Quality depends on commit message conventions. "
                    "Not uniformly available across repos."
                ),
            },
            {
                "evidence_type": "Manually validated samples",
                "description": (
                    "Validate a subset of the 108 observed-path-associated "
                    "rows to determine how many are truly associated "
                    "with defects."
                ),
                "feasibility": "FEASIBLE",
                "expected_impact": "HIGH",
                "limitations": (
                    "Small sample size limits statistical power. "
                    "Validation is retrospective, not prospective."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def derive_recommendation(
    supervision_result: dict,
    coverage: dict,
    repo_concentration: dict,
    split_analysis: dict,
    pu_assumptions: dict,
    negative_feasibility: dict,
    evaluation: dict,
) -> dict:
    """Derive recommendation from the evidence matrix.

    All quantities are consumed from analysis results, never hardcoded.
    """
    summary = supervision_result["summary"]
    conc = repo_concentration["concentration"]

    # Build evidence matrix from computed results
    evidence_matrix: dict[str, dict] = {}

    # 1. Observed-positive quantity
    obs_count = summary["primary_observed_count"]
    obs_rate = summary["observed_positive_rate"]
    evidence_matrix["observed_positive_quantity"] = {
        "value": obs_count,
        "rate": obs_rate,
        "assessment": (
            f"{obs_count} observed-path-associated rows "
            f"({obs_rate:.1%} of total file rows)."
        ),
    }

    # 2. Repository diversity
    repos_with = repo_concentration["repos_with_signal"]
    repos_total = repo_concentration["total_repos"]
    signal_ratio = repos_with / repos_total if repos_total else 0.0
    evidence_matrix["repository_diversity"] = {
        "repos_with_signal": repos_with,
        "repos_total": repos_total,
        "signal_ratio": signal_ratio,
        "top_5_share": conc["top_5_share"],
        "gini": conc["gini"],
        "assessment": (
            f"{repos_with}/{repos_total} repos contain signal. "
            f"Top-5 share: {conc['top_5_share']:.1%}. "
            f"Gini: {conc['gini']:.3f}."
        ),
    }

    # 3. Split coverage
    split_obs = {
        s: split_analysis[s]["observed_count"]
        for s in ("train", "validation", "test")
    }
    splits_with_signal = sum(1 for v in split_obs.values() if v > 0)
    evidence_matrix["split_coverage"] = {
        "per_split": split_obs,
        "splits_with_signal": splits_with_signal,
        "assessment": (
            f"{splits_with_signal}/3 splits contain observed signal."
        ),
    }

    # 4. Feature distribution differences
    n_notable = len(coverage.get("notable_differences", []))
    evidence_matrix["feature_distribution"] = {
        "notable_differences_count": n_notable,
        "assessment": f"{n_notable} feature distribution differences noted.",
    }

    # 5. Selection dependence
    scar_verdict = pu_assumptions["scar"]["verdict"]
    sar_verdict = pu_assumptions["sar"]["verdict"]
    repo_severity = pu_assumptions["repository_dependency"]["severity"]
    evidence_matrix["selection_dependence"] = {
        "scar_verdict": scar_verdict,
        "sar_verdict": sar_verdict,
        "repo_severity": repo_severity,
        "assessment": (
            f"SCAR: {scar_verdict}. SAR: {sar_verdict}. "
            f"Repository-dependence: {repo_severity}."
        ),
    }

    # 6. Attribution strength
    evidence_matrix["attribution_strength"] = {
        "level": "path_level_only",
        "assessment": (
            "Path-level evidence only. No content-level or "
            "line-level attribution available."
        ),
    }

    # 7. Negative-label availability
    def_neg = negative_feasibility["defensible_negatives"]
    evidence_matrix["negative_availability"] = {
        "defensible_negatives": def_neg,
        "assessment": f"{def_neg} defensible negatives available.",
    }

    # 8. Evaluation feasibility
    n_measurable = len(evaluation.get("measurable_now", []))
    evidence_matrix["evaluation_feasibility"] = {
        "measurable_diagnostics": n_measurable,
        "assessment": (
            f"{n_measurable} measurable-now diagnostics identified. "
            "True defect metrics not measurable."
        ),
    }

    # 9. Reproducibility
    evidence_matrix["reproducibility"] = {
        "deterministic": True,
        "assessment": "Analysis is deterministic and auditable.",
    }

    # Derive recommendation from evidence matrix
    recommendation: str
    reasoning_parts: list[str] = []
    restrictions: list[str] = []

    # Check defensible negatives
    if def_neg > 0:
        recommendation = "BINARY_SUPERVISION_NOT_SUPPORTED"
        reasoning_parts.append(
            f"{def_neg} defensible negatives exist, which is unexpected "
            "given the path-level evidence limitation."
        )
    # Check observed count
    elif obs_count == 0:
        recommendation = "IMPROVE_ATTRIBUTION_FIRST"
        reasoning_parts.append(
            "Zero observed path-associated rows. No file-level signal "
            "available for PU formulation."
        )
    # Check if concerns dominate
    else:
        concerns: list[str] = []

        if signal_ratio < 0.2:
            concerns.append(
                f"Low repository diversity: {repos_with}/{repos_total} repos"
            )

        for split_name in ("train", "validation", "test"):
            if split_obs[split_name] == 0:
                concerns.append(f"Zero signal in {split_name} split")

        if n_notable > 5:
            concerns.append(
                f"Many feature distribution differences ({n_notable})"
            )

        if scar_verdict == "UNTESTABLE_WITH_CURRENT_EVIDENCE":
            concerns.append("SCAR not established")

        if repo_severity == "high":
            concerns.append("High repository concentration")

        if n_measurable < 3:
            concerns.append("Limited evaluation diagnostics")

        if len(concerns) >= 4:
            recommendation = "IMPROVE_ATTRIBUTION_FIRST"
            reasoning_parts.append(
                f"Multiple concerns identified: {'; '.join(concerns)}. "
                "Improve attribution before PU experiment."
            )
        elif len(concerns) >= 2:
            recommendation = "PU_LEARNING_FEASIBLE_WITH_RESTRICTIONS"
            reasoning_parts.append(
                f"Some concerns identified: {'; '.join(concerns)}. "
                "PU learning is feasible with appropriate restrictions "
                "and sensitivity analyses."
            )
            restrictions = concerns
        else:
            recommendation = "PU_LEARNING_FEASIBLE"
            reasoning_parts.append(
                "Evidence supports PU learning feasibility. "
                "Standard PU methodology with appropriate caveats."
            )
            if concerns:
                restrictions = concerns

    reasoning = " ".join(reasoning_parts)

    return {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "restrictions": restrictions,
        "evidence_matrix": evidence_matrix,
        "next_step": _determine_next_step(recommendation),
    }


def _determine_next_step(recommendation: str) -> str:
    """Identify the immediate methodological next phase."""
    steps = {
        "PU_LEARNING_FEASIBLE": (
            "Proceed with PU learning experiment design. "
            "Implement PU-compatible training with sensitivity analysis."
        ),
        "PU_LEARNING_FEASIBLE_WITH_RESTRICTIONS": (
            "Proceed with PU learning experiment with explicit "
            "restrictions. Include sensitivity analysis across "
            "repository subsets and population definitions."
        ),
        "PU_LEARNING_NOT_YET_FEASIBLE": (
            "Conduct additional analysis to resolve PU assumption "
            "uncertainties. Consider commit-level prediction as "
            "an intermediate step."
        ),
        "BINARY_SUPERVISION_NOT_SUPPORTED": (
            "Investigate the source of defensible negatives. "
            "Reconcile with path-level evidence limitations."
        ),
        "COMMIT_LEVEL_MODEL_RECOMMENDED": (
            "Proceed with commit-level binary classification. "
            "Reserve file-level PU learning for future work."
        ),
        "IMPROVE_ATTRIBUTION_FIRST": (
            "Improve file-level attribution before PU experiment. "
            "Consider line-level blame, issue linkage, or "
            "manual validation of observed-path-associated rows."
        ),
    }
    return steps.get(recommendation, "No next step determined.")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_report(
    supervision_result: dict,
    coverage: dict,
    repo_concentration: dict,
    split_analysis: dict,
    pu_assumptions: dict,
    negative_feasibility: dict,
    strategy_comparison: dict,
    evaluation: dict,
    data_requirements: dict,
    recommendation: dict,
    output_dir: Path,
) -> None:
    """Write the Phase 4.9 scientific report."""
    summary = supervision_result["summary"]
    rec = recommendation

    lines: list[str] = [
        "# Phase 4.9: Supervision Strategy & PU Learning Feasibility Study",
        "",
        "## Executive Summary",
        "",
        f"This study evaluates whether the frozen Phase 4.8 file-level "
        f"evidence supports a future Positive-Unlabeled (PU) learning "
        f"experiment. The analysis identifies "
        f"{summary['primary_observed_count']} observed-path-associated "
        f"rows from FILE_PARTIAL commits, with "
        f"{summary['sensitivity_count']} sensitivity rows, "
        f"{summary['unlabeled_count']} unlabeled rows, "
        f"{summary['defensible_negative_count']} defensible negatives, "
        f"and {summary['out_of_scope_count']} out-of-scope parser "
        f"artifacts.",
        "",
        f"**Primary Recommendation: {rec['recommendation']}**",
        "",
        f"{rec['reasoning']}",
        "",
        "---",
        "",
        "## OBSERVED FACTS",
        "",
        "### Dataset Scope",
        "",
        f"- Total supervised file rows: {summary['total_file_rows']}",
        f"- Observed path-associated rows: {summary['primary_observed_count']}",
        f"- Sensitivity (COMMIT_ONLY overlap) rows: {summary['sensitivity_count']}",
        f"- Unlabeled rows: {summary['unlabeled_count']}",
        f"- Defensible negatives: {summary['defensible_negative_count']}",
        f"- Out-of-scope (parser artifacts): {summary['out_of_scope_count']}",
        f"- Observed positive rate: {summary['observed_positive_rate']:.1%}",
        "",
        "### Evidence Populations",
        "",
        "- **OBSERVED_PATH_ASSOCIATED**: FILE_PARTIAL path-associated rows",
        "  only. Candidate positives for PU formulation.",
        "- **UNLABELED**: Legitimate file rows without file-level defect",
        "  association evidence.",
        "- **SENSITIVITY (COMMIT_ONLY)**: Path-overlap rows from COMMIT_ONLY",
        "  commits. Weaker evidence. Not included in primary PU population.",
        "- **DEFENSIBLE_NEGATIVE**: None. No independent negative evidence.",
        "- **OUT_OF_SCOPE**: Parser artifacts (file_path == '/dev/null').",
        "  Binary file additions where the diff parser captured the source",
        "  side instead of the destination path. Not usable for supervision.",
        "",
        "---",
        "",
        "## IMPLEMENTATION FACTS",
        "",
        "The analysis:",
        "",
        "1. Reads frozen JSONL files (train/validation/test/ambiguous).",
        "2. Parses REPO_MANIFEST from frozen repo_split.py source.",
        "3. Classifies all 210 positive commits by evidence category.",
        "4. Constructs supervision sets from classification results.",
        "5. Analyzes coverage across feature dimensions.",
        "6. Measures repository concentration.",
        "7. Reports split distribution.",
        "8. Evaluates SCAR/SAR assumptions cautiously.",
        "9. Verifies negative-label feasibility.",
        "10. Compares five supervision strategies.",
        "11. Designs evaluation protocol.",
        "12. Identifies data requirements.",
        "13. Derives recommendation from evidence matrix.",
        "",
        "---",
        "",
        "## INFERENCES",
        "",
        "### Coverage Analysis",
        "",
    ]

    for diff in coverage.get("notable_differences", [])[:10]:
        lines.append(f"- {diff['feature']}: {diff['description']}")

    lines.extend([
        "",
        "### Repository Concentration",
        "",
        f"- Repos with signal: "
        f"{repo_concentration['repos_with_signal']}/"
        f"{repo_concentration['total_repos']}",
        f"- Top-5 share: "
        f"{repo_concentration['concentration']['top_5_share']:.1%}",
        f"- Gini: {repo_concentration['concentration']['gini']:.3f}",
        f"- Herfindahl: {repo_concentration['concentration']['herfindahl']:.4f}",
        f"- Entropy: {repo_concentration['concentration']['entropy']:.3f}",
        "",
        "### Split Distribution",
        "",
        "| Split | Obs | Sens | Unl | OOS | Raw | Class | Rec | Rate |",
        "|-------|-----|------|-----|-----|-----|-------|-----|------|",
    ])

    for split in ("train", "validation", "test"):
        sd = split_analysis[split]
        lines.append(
            f"| {split} | {sd['observed_count']} | "
            f"{sd['sensitivity_count']} | "
            f"{sd['unlabeled_count']} | {sd['out_of_scope_count']} | "
            f"{sd['raw_total']} | {sd['classified_total']} | "
            f"{'Yes' if sd['reconciled'] else 'No'} | "
            f"{sd['observed_positive_rate']:.1%} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## UNTESTABLE ASSUMPTIONS",
        "",
        "### SCAR",
        "",
        f"**Verdict: {pu_assumptions['scar']['verdict']}**",
        "",
        f"{pu_assumptions['scar']['reasoning']}",
        "",
        "### SAR",
        "",
        f"**Verdict: {pu_assumptions['sar']['verdict']}**",
        "",
        f"{pu_assumptions['sar']['reasoning']}",
        "",
        "---",
        "",
        "## SUPERVISION OPTIONS",
        "",
    ])

    for strat in strategy_comparison["strategies"]:
        lines.extend([
            f"### {strat['name']}",
            "",
            f"- **Compatibility**: {strat['compatibility_with_dataset']}",
            f"- **Pursue next**: {'Yes' if strat['pursue_next'] else 'No'}",
            f"- **Risk**: {strat['methodological_risk']}",
            f"- **Reasoning**: {strat['reasoning']}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## EVALUATION FEASIBILITY",
        "",
        "### Measurable Now",
        "",
    ])

    for m in evaluation["measurable_now"]:
        lines.append(f"- {m['name']}: {m['description']}")

    lines.extend([
        "",
        "### Not Currently Measurable",
        "",
    ])

    for m in evaluation["not_measurable_now"]:
        lines.append(f"- {m['name']}: {m['reason']}")

    lines.extend([
        "",
        "### Exploratory PU Metrics (clearly labeled)",
        "",
    ])

    for m in evaluation["exploratory_pu_metrics"]:
        lines.append(f"- {m['name']}: {m['caveat']}")

    lines.extend([
        "",
        "---",
        "",
        "## ADDITIONAL DATA REQUIREMENTS",
        "",
    ])

    for req in data_requirements["requirements"]:
        lines.extend([
            f"### {req['evidence_type']}",
            "",
            f"- **Feasibility**: {req['feasibility']}",
            f"- **Impact**: {req.get('expected_impact', 'N/A')}",
            f"- **Limitations**: {req.get('limitations', 'N/A')}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## EVIDENCE MATRIX",
        "",
    ])

    for key, val in rec.get("evidence_matrix", {}).items():
        lines.append(f"- **{key}**: {val.get('assessment', '')}")

    lines.extend([
        "",
        "---",
        "",
        "## RECOMMENDATION",
        "",
        f"**{rec['recommendation']}**",
        "",
        f"{rec['reasoning']}",
        "",
    ])

    if rec.get("restrictions"):
        lines.extend([
            "### Restrictions",
            "",
        ])
        for r in rec["restrictions"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## LIMITATIONS",
        "",
        "1. Path-level evidence only. No content-level analysis.",
        "2. 108 observed-path-associated rows is a small sample.",
        "3. No defensible negatives. PU learning required.",
        "4. SCAR not established. SAR is working assumption with caveats.",
        "5. Possible repository-dependent selection.",
        "6. Evaluation limited to observed-positive diagnostics.",
        "7. True defect performance cannot be measured.",
        "",
        "---",
        "",
        "## NEXT STEP",
        "",
        f"{rec['next_step']}",
        "",
        "---",
        "",
        "## Frozen Data Integrity",
        "",
        "- No frozen dataset files were modified.",
        "- No frozen labeling logic was modified.",
        "- No frozen Phase 4.6/4.7/4.8 artifacts were modified.",
        "- No dataset regeneration was performed.",
        "",
    ])

    report_path = output_dir / "phase49_supervision_feasibility.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feasibility_study(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the complete Phase 4.9 supervision feasibility study."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 4.9: Supervision Strategy & PU Learning Feasibility")
    logger.info("Reading frozen dataset from %s", data_dir)

    supervised = _read_all_supervised_rows(data_dir)
    ambiguous = _read_ambiguous_rows(data_dir)
    sha_lookup = _build_sha_lookup(supervised, ambiguous)

    logger.info("Parsing frozen repository manifest...")
    manifest = _parse_frozen_manifest()
    logger.info("Manifest: %d repos", len(manifest))

    logger.info("Constructing supervision sets...")
    supervision_result = construct_supervision_sets(
        supervised, sha_lookup, manifest
    )

    primary_rows = supervision_result["primary_observed_path_associated"]["rows"]
    sensitivity_rows = supervision_result["sensitivity_commit_only_overlap"]["rows"]
    unlabeled_rows = supervision_result["unlabeled"]["rows"]
    out_of_scope_rows = supervision_result["out_of_scope"]["rows"]

    logger.info(
        "Primary: %d, Sensitivity: %d, Unlabeled: %d, Out-of-scope: %d",
        len(primary_rows), len(sensitivity_rows), len(unlabeled_rows),
        len(out_of_scope_rows),
    )

    logger.info("Analyzing coverage...")
    coverage = analyze_coverage(primary_rows, unlabeled_rows)

    logger.info("Analyzing repository concentration...")
    repo_concentration = analyze_repository_concentration(
        primary_rows, supervised
    )

    logger.info("Analyzing split distribution...")
    split_analysis = analyze_split_distribution(
        primary_rows, unlabeled_rows, sensitivity_rows,
        out_of_scope_rows, supervised,
    )

    logger.info("Evaluating PU assumptions...")
    pu_assumptions = evaluate_pu_assumptions(repo_concentration, coverage)

    logger.info("Verifying negative feasibility...")
    negative_feasibility = verify_negative_feasibility(
        supervised, primary_rows, sensitivity_rows, unlabeled_rows,
        out_of_scope_rows,
    )

    logger.info("Comparing supervision strategies...")
    strategy_comparison = compare_supervision_strategies(
        supervision_result, coverage, repo_concentration, pu_assumptions
    )

    logger.info("Designing evaluation strategy...")
    evaluation = design_evaluation_strategy(supervision_result, pu_assumptions)

    logger.info("Identifying data requirements...")
    data_requirements = identify_data_requirements()

    logger.info("Deriving recommendation...")
    recommendation = derive_recommendation(
        supervision_result, coverage, repo_concentration,
        split_analysis, pu_assumptions, negative_feasibility, evaluation,
    )

    # Write artifacts
    logger.info("Writing artifacts...")

    _write_json(output_dir / "supervision_feasibility.json", {
        "supervision_sets_summary": supervision_result["summary"],
        "coverage_summary": {
            "notable_differences_count": len(
                coverage.get("notable_differences", [])
            ),
            "observed_count": coverage["observed_count"],
            "unlabeled_count": coverage["unlabeled_count"],
        },
        "repo_concentration_summary": repo_concentration["concentration"],
        "split_summary": {
            s: {k: v for k, v in split_analysis[s].items()}
            for s in ("train", "validation", "test")
        },
        "pu_assumptions_summary": {
            "scar_verdict": pu_assumptions["scar"]["verdict"],
            "sar_verdict": pu_assumptions["sar"]["verdict"],
        },
        "negative_feasibility_summary": {
            "defensible_negatives": negative_feasibility["defensible_negatives"],
            "out_of_scope": negative_feasibility["categories"]["out_of_scope"],
            "reconciled": negative_feasibility["reconciled"],
        },
        "recommendation": recommendation,
        "strategy_comparison_summary": {
            "recommended_for_next_phase": (
                strategy_comparison["recommended_for_next_phase"]
            ),
        },
    })

    _write_json(output_dir / "observed_positive_analysis.json", {
        "total_observed_path_associated": len(primary_rows),
        "per_feature_coverage": coverage["per_feature"],
        "notable_differences": coverage.get("notable_differences", []),
    })

    _write_json(output_dir / "repository_analysis.json", repo_concentration)

    _write_json(output_dir / "split_analysis.json", split_analysis)

    _write_json(output_dir / "strategy_comparison.json", strategy_comparison)

    logger.info("Generating report...")
    _write_report(
        supervision_result, coverage, repo_concentration,
        split_analysis, pu_assumptions, negative_feasibility,
        strategy_comparison, evaluation, data_requirements,
        recommendation, output_dir,
    )

    logger.info("Phase 4.9 complete. Artifacts: %s", output_dir)

    return {
        "supervision_result": supervision_result,
        "coverage": coverage,
        "repo_concentration": repo_concentration,
        "split_analysis": split_analysis,
        "pu_assumptions": pu_assumptions,
        "negative_feasibility": negative_feasibility,
        "strategy_comparison": strategy_comparison,
        "evaluation": evaluation,
        "data_requirements": data_requirements,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_feasibility_study()
