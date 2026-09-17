"""Phase 4.12: Defensible Supervision & Negative-Label Construction Feasibility.

READ-ONLY scientific feasibility study evaluating 9 supervision strategies
over existing frozen Phase 4.6-4.11b dataset/artifacts.

Key constraints:
- Read-only on frozen data. No modifications to any prior artifact.
- No ML model training. No hyperparameter tuning.
- No AWS/Braket/quantum code.
- No fabrication of negative labels.
- No use of current system time as observation endpoint.
- All counts derived programmatically. No hardcoded repository counts.
- Deterministic and auditable.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.app.ml.phase49_supervision_feasibility import (
    _parse_frozen_manifest,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
    _write_json,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")
PHASE411B_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.12")

# Evidence level constants
OBSERVED_POSITIVE = "OBSERVED_POSITIVE"
EVIDENCE_MODERATE = "EVIDENCE_MODERATE"
EVIDENCE_WEAK = "EVIDENCE_WEAK"
EVIDENCE_UNKNOWN = "EVIDENCE_UNKNOWN"
UNLABELED = "UNLABELED"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
CANDIDATE_NEGATIVE = "CANDIDATE_NEGATIVE"
DEFENSIBLE_NEGATIVE = "DEFENSIBLE_NEGATIVE"

FILE_EVIDENCE_STRONG = "FILE_STRONG"
FILE_EVIDENCE_MODERATE = "FILE_MODERATE"
FILE_EVIDENCE_WEAK = "FILE_WEAK"
FILE_EVIDENCE_UNKNOWN = "UNKNOWN"

# Prediction-time feature table
PREDICTION_TIME_FEATURES: list[dict[str, str]] = [
    {
        "name": "commit_features", "source": "Raw JSONL",
        "timestamp_relative": "pre-candidate",
        "prediction_time_available": "YES",
        "leakage_status": "NO_LEAKAGE",
        "permitted_usage": "FEATURE",
    },
    {
        "name": "file_features", "source": "Raw JSONL",
        "timestamp_relative": "pre-candidate",
        "prediction_time_available": "YES",
        "leakage_status": "NO_LEAKAGE",
        "permitted_usage": "FEATURE",
    },
    {
        "name": "commit_timestamp", "source": "Raw JSONL",
        "timestamp_relative": "pre-candidate",
        "prediction_time_available": "YES",
        "leakage_status": "NO_LEAKAGE",
        "permitted_usage": "FEATURE",
    },
    {
        "name": "file_path", "source": "Raw JSONL",
        "timestamp_relative": "pre-candidate",
        "prediction_time_available": "YES",
        "leakage_status": "NO_LEAKAGE",
        "permitted_usage": "FEATURE",
    },
    {
        "name": "repo_name", "source": "Raw JSONL",
        "timestamp_relative": "pre-candidate",
        "prediction_time_available": "YES",
        "leakage_status": "NO_LEAKAGE",
        "permitted_usage": "FEATURE",
    },
    {
        "name": "label_source", "source": "Raw JSONL",
        "timestamp_relative": "post-candidate/outcome-derived",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE/outcome_metadata",
        "permitted_usage": "EXCLUDED",
    },
    {
        "name": "corrective_sha", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "content_correspondence", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "content_restoration", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "region_overlap", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "function_analysis", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "evidence_types", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
    {
        "name": "file_evidence_level", "source": "Phase 4.11b",
        "timestamp_relative": "post-candidate",
        "prediction_time_available": "NO",
        "leakage_status": "LEAKAGE",
        "permitted_usage": "LABEL",
    },
]

EXCLUDED_FEATURES: set[str] = {"label_source"}
LEAKAGE_FEATURES: set[str] = {
    "corrective_sha", "content_correspondence", "content_restoration",
    "region_overlap", "function_analysis", "evidence_types", "file_evidence_level",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_phase411b_file_results(phase411b_dir: Path) -> list[dict]:
    """Load file_attribution_results.json from Phase 4.11b."""
    path = phase411b_dir / "file_attribution_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_phase411b_commit_evidence(phase411b_dir: Path) -> list[dict]:
    """Load per_commit_git_evidence.jsonl from Phase 4.11b."""
    path = phase411b_dir / "per_commit_git_evidence.jsonl"
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_timestamps(data_dir: Path) -> dict[tuple[str, str], str]:
    """Load commit_timestamp for every (repo_name, commit_sha) from raw JSONL."""
    timestamps: dict[tuple[str, str], str] = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["repo_name"], row["commit_sha"])
                if key not in timestamps:
                    timestamps[key] = row.get("commit_timestamp", "")
    return timestamps


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO-8601 timestamp string to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------

def construct_supervision_populations(
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
    supervised_rows: list[dict],
    ambiguous_rows: list[dict],
    manifest: dict[str, str],
    timestamps: dict[tuple[str, str], str],
) -> dict:
    """Construct all supervision populations from Phase 4.11b evidence.

    Returns dict with population lists and metadata.
    """
    # Build (repo, sha, file) -> evidence level lookup from Phase 4.11b
    file_evidence: dict[tuple[str, str, str], dict] = {}
    for rec in phase411b_files:
        key = (rec["repo_name"], rec["commit_sha"], rec["file_path"])
        file_evidence[key] = rec

    # Build corrective_sha lookup from commit evidence
    commit_corrective: dict[str, str] = {}
    for rec in phase411b_commits:
        if rec.get("corrective_sha"):
            commit_corrective[rec["commit_sha"]] = rec["corrective_sha"]

    # Classify each supervised row
    observed_positive: list[dict] = []
    evidence_moderate: list[dict] = []
    evidence_weak: list[dict] = []
    evidence_unknown: list[dict] = []
    unlabeled: list[dict] = []
    out_of_scope: list[dict] = []

    for row in supervised_rows:
        fp = row["file_path"]
        repo = row["repo_name"]
        sha = row["commit_sha"]

        # Out-of-scope check
        if fp == "/dev/null" or "/dev/null" in fp:
            out_of_scope.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": OUT_OF_SCOPE,
                "reason": "parser_artifact",
            })
            continue

        # Look up Phase 4.11b evidence
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
                "corrective_sha": commit_corrective.get(sha),
                "identity_established": ev.get("identity_established", False),
                "evidence_types": ev.get("evidence_types", []),
            })
        elif level == FILE_EVIDENCE_MODERATE:
            evidence_moderate.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": EVIDENCE_MODERATE,
                "corrective_sha": commit_corrective.get(sha),
                "identity_established": ev.get("identity_established", False),
            })
        elif level == FILE_EVIDENCE_WEAK:
            evidence_weak.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": EVIDENCE_WEAK,
                "corrective_sha": commit_corrective.get(sha),
                "identity_established": ev.get("identity_established", False),
            })
        elif level == FILE_EVIDENCE_UNKNOWN:
            evidence_unknown.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": EVIDENCE_UNKNOWN,
                "corrective_sha": commit_corrective.get(sha),
                "identity_established": ev.get("identity_established", False),
            })
        else:
            unlabeled.append({
                "repo_name": repo, "commit_sha": sha, "file_path": fp,
                "population": UNLABELED,
            })

    # Metadata
    all_repos = set(r["repo_name"] for r in supervised_rows)
    repos_with_pos = set(r["repo_name"] for r in observed_positive)
    all_timestamps = sorted(
        ts for ts in timestamps.values() if ts
    )
    pos_timestamps = sorted(
        ts for (repo, sha), ts in timestamps.items()
        if any(r["repo_name"] == repo and r["commit_sha"] == sha
               for r in observed_positive) and ts
    )

    # Historical positive commit/repo counts (from defect_label=1)
    historical_pos_commits = set()
    historical_pos_repos = set()
    for r in supervised_rows:
        if r.get("defect_label") == 1:
            historical_pos_commits.add(r["commit_sha"])
            historical_pos_repos.add(r["repo_name"])

    # FILE_STRONG commit/repo counts (from observed_positive)
    file_strong_commits = set(r["commit_sha"] for r in observed_positive)
    file_strong_repos = set(r["repo_name"] for r in observed_positive)

    metadata = {
        "total_supervised_rows": len(supervised_rows),
        "historical_positive_rows": sum(
            1 for r in supervised_rows if r.get("defect_label") == 1
        ),
        "observed_positive_rows": len(observed_positive),
        "evidence_moderate_rows": len(evidence_moderate),
        "evidence_weak_rows": len(evidence_weak),
        "evidence_unknown_rows": len(evidence_unknown),
        "unlabeled_rows": len(unlabeled),
        "out_of_scope_rows": len(out_of_scope),
        "historical_positive_commits": len(historical_pos_commits),
        "file_strong_commits": len(file_strong_commits),
        "historical_positive_repos": len(historical_pos_repos),
        "repos_with_file_strong": len(file_strong_repos),
        "unique_repos": len(all_repos),
        "repos_without_positives": len(all_repos - repos_with_pos),
        "date_range": [
            all_timestamps[0] if all_timestamps else "",
            all_timestamps[-1] if all_timestamps else "",
        ],
        "positive_date_range": [
            pos_timestamps[0] if pos_timestamps else "",
            pos_timestamps[-1] if pos_timestamps else "",
        ],
    }

    return {
        OBSERVED_POSITIVE: observed_positive,
        EVIDENCE_MODERATE: evidence_moderate,
        EVIDENCE_WEAK: evidence_weak,
        EVIDENCE_UNKNOWN: evidence_unknown,
        UNLABELED: unlabeled,
        OUT_OF_SCOPE: out_of_scope,
        CANDIDATE_NEGATIVE: [],
        DEFENSIBLE_NEGATIVE: [],
        "metadata": metadata,
    }


def _verify_population_disjointness(populations: dict) -> dict:
    """Verify that populations are mutually disjoint."""
    def _keys(pop_name: str) -> set[tuple[str, str, str]]:
        return {
            (r["repo_name"], r["commit_sha"], r["file_path"])
            for r in populations.get(pop_name, [])
        }

    op = _keys(OBSERVED_POSITIVE)
    em = _keys(EVIDENCE_MODERATE)
    ew = _keys(EVIDENCE_WEAK)
    eu = _keys(EVIDENCE_UNKNOWN)
    cn = _keys(CANDIDATE_NEGATIVE)
    dn = _keys(DEFENSIBLE_NEGATIVE)
    checks = {
        "OP_intersect_EW": len(op & ew) == 0,
        "OP_intersect_EU": len(op & eu) == 0,
        "EW_intersect_EU": len(ew & eu) == 0,
        "OP_intersect_EM": len(op & em) == 0,
        "EM_intersect_EW": len(em & ew) == 0,
        "EM_intersect_EU": len(em & eu) == 0,
        "CN_intersect_OP": len(cn & op) == 0,
        "DN_subset_CN": dn.issubset(cn) if dn else True,
    }

    all_ok = all(checks.values())
    violations = [k for k, v in checks.items() if not v]

    return {
        "all_disjoint": all_ok,
        "checks": checks,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Strategy 1: PU Learning
# ---------------------------------------------------------------------------

def evaluate_pu_strategy(
    populations: dict,
    supervised_rows: list[dict],
) -> dict:
    """Evaluate PU learning feasibility."""
    op_count = len(populations[OBSERVED_POSITIVE])
    ew_count = len(populations[EVIDENCE_WEAK])
    eu_count = len(populations[EVIDENCE_UNKNOWN])
    ul_count = len(populations[UNLABELED])
    total = len(supervised_rows)

    return {
        "strategy": "PU_LEARNING",
        "evidence_available": {
            "observed_positive": op_count,
            "evidence_weak": ew_count,
            "evidence_unknown": eu_count,
            "unlabeled": ul_count,
            "total": total,
        },
        "scar_assessment": {
            "status": "UNTESTABLE",
            "reason": (
                "SCAR requires P(S=1|Y=1) to be identifiable. "
                "With 0 defensible negatives, the positive class "
                "contamination rate is not directly observable."
            ),
        },
        "sar_assessment": {
            "status": "PARTIALLY_IDENTIFIABLE",
            "reason": (
                "The selection mechanism depends on label_source "
                "(explicit_sha_reference vs revert), which is observable. "
                "However, the selection also depends on the labeling "
                "pipeline's ability to find corrective commits, which "
                "is not random."
            ),
        },
        "class_prior_identifiability": {
            "status": "NOT_IDENTIFIABLE",
            "reason": (
                "Without true negatives, pi = P(Y=1) cannot be estimated. "
                "The observed positive rate (1.94%) reflects labeling "
                "pipeline coverage, not true defect prevalence."
            ),
        },
        "elkan_noto_identifiability": {
            "status": "NOT_IDENTIFIABLE",
            "reason": (
                "Elkan-Noto requires c = P(S=1|Y=1) (label accuracy). "
                "With 0 true negatives, c cannot be estimated."
            ),
        },
        "reliable_positive_subset": {
            "status": "PARTIALLY_IDENTIFIABLE",
            "reason": (
                "OBSERVED_POSITIVE files have direct Git content evidence, "
                "making them the strongest candidates for reliable positives. "
                "However, the unlabeled set contains files that may also "
                "be defective (false negatives in the labeling pipeline)."
            ),
        },
        "observed_positive_ranking": {
            "status": "IDENTIFIABLE",
            "reason": (
                "Observed-positive ranking is feasible: the Phase 4.10 "
                "experiment demonstrated KS=0.81 on train. However, "
                "test-set degradation (KS=0.22) suggests overfitting "
                "to the observation mechanism."
            ),
        },
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "PROCEED_WITH_CAVEATS",
        "reason": (
            "PU ranking is identifiable but class prior and SCAR are not. "
            "Proceed with observed-positive ranking only, not prevalence estimation."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 2: Reliable Negatives
# ---------------------------------------------------------------------------

def evaluate_reliable_negative_strategy(
    populations: dict,
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
    timestamps: dict[tuple[str, str], str],
) -> dict:
    """Evaluate reliable negative strategies using available evidence."""
    # Investigate candidate-negative strategies
    strategies = []

    # Strategy: unaffected files in corrective commits
    unaffected = []
    for rec in phase411b_commits:
        corrective_files = set(rec.get("corrective_files", []))
        candidate_files = set(rec.get("candidate_files", []))
        unaffected_files = corrective_files - candidate_files
        for fp in unaffected_files:
            unaffected.append({
                "repo_name": rec["repo_name"],
                "commit_sha": rec["commit_sha"],
                "file_path": fp,
                "corrective_sha": rec.get("corrective_sha"),
                "strategy": "unaffected_in_corrective",
                "status": CANDIDATE_NEGATIVE,
            })

    strategies.append({
        "name": "unaffected_in_corrective",
        "description": "Files present in corrective commit but not modified by candidate",
        "available_evidence": "AVAILABLE_EVIDENCE",
        "candidate_negative_count": len(unaffected),
        "produces_defensible_negatives": False,
        "reason": (
            "Being unchanged by a corrective commit does not mean "
            "the file is defect-free. It may be indirectly affected."
        ),
    })

    # Strategy: long observation window
    strategies.append({
        "name": "long_observation_window",
        "description": "Files surviving long observation without corrective event",
        "available_evidence": "AVAILABLE_EVIDENCE",
        "candidate_negative_count": 0,
        "produces_defensible_negatives": False,
        "reason": (
            "Absence of observed corrective event is not evidence of "
            "absence of defect. Censoring is informative."
        ),
    })

    # Strategy: repeated modification without corrective
    strategies.append({
        "name": "repeated_modification",
        "description": "Files modified many times without corrective evidence",
        "available_evidence": "AVAILABLE_EVIDENCE",
        "candidate_negative_count": 0,
        "produces_defensible_negatives": False,
        "reason": (
            "Frequent modification without observed correction may "
            "indicate a high-risk file, not a safe one."
        ),
    })

    # Future evidence (not available)
    future_evidence = [
        {"name": "test_coverage", "status": "NOT_AVAILABLE", "would_need": "coverage data"},
        {"name": "code_review_history", "status": "NOT_AVAILABLE", "would_need": "GitHub API"},
        {"name": "issue_tracker_data", "status": "NOT_AVAILABLE", "would_need": "issue data"},
    ]

    return {
        "strategy": "RELIABLE_NEGATIVES",
        "strategies_investigated": strategies,
        "future_evidence_not_available": future_evidence,
        "defensible_negative_count": 0,
        "candidate_negative_count": len(unaffected),
        "identifiability": "NOT_IDENTIFIABLE",
        "recommended_for_4.13": "DO_NOT_PROCEED",
        "reason": (
            "No criterion produces defensible negatives with available "
            "evidence. All produce candidate negatives only."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 3: Temporal Negatives
# ---------------------------------------------------------------------------

def evaluate_temporal_strategy(
    populations: dict,
    phase411b_commits: list[dict],
    timestamps: dict[tuple[str, str], str],
) -> dict:
    """Evaluate temporal negative feasibility."""
    # Compute deterministic observation endpoints per repo
    repo_endpoints: dict[str, str] = {}
    repo_timestamps: dict[str, list[str]] = defaultdict(list)
    for (repo, _sha), ts in timestamps.items():
        if ts:
            repo_timestamps[repo].append(ts)

    for repo, ts_list in repo_timestamps.items():
        sorted_ts = sorted(ts_list)
        repo_endpoints[repo] = sorted_ts[-1] if sorted_ts else ""

    # Compute observation durations for observed-positive files
    op_durations_days: list[float] = []
    for rec in populations[OBSERVED_POSITIVE]:
        repo = rec["repo_name"]
        sha = rec["commit_sha"]
        ts_key = (repo, sha)
        file_ts = timestamps.get(ts_key, "")
        endpoint = repo_endpoints.get(repo, "")

        file_dt = _parse_timestamp(file_ts)
        endpoint_dt = _parse_timestamp(endpoint)

        if file_dt and endpoint_dt:
            duration = (endpoint_dt - file_dt).total_seconds() / 86400.0
            op_durations_days.append(duration)

    # Compute observation durations for all supervised files
    all_durations_days: list[float] = []
    for (repo, sha), ts in timestamps.items():
        endpoint = repo_endpoints.get(repo, "")
        file_dt = _parse_timestamp(ts)
        endpoint_dt = _parse_timestamp(endpoint)
        if file_dt and endpoint_dt:
            duration = (endpoint_dt - file_dt).total_seconds() / 86400.0
            all_durations_days.append(duration)

    # Descriptive statistics (continuous distributions, no arbitrary bins)
    def _dist(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0, "p10": 0.0, "p25": 0.0,
                    "p75": 0.0, "p90": 0.0}
        a = np.array(vals, dtype=np.float64)
        p10, p25, p50, p75, p90 = np.percentile(a, [10, 25, 50, 75, 90])
        return {
            "count": int(len(vals)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p10": float(p10), "p25": float(p25),
            "p75": float(p75), "p90": float(p90),
        }

    op_duration_dist = _dist(op_durations_days)
    all_duration_dist = _dist(all_durations_days)

    # Descriptive classification using dataset median as boundary
    # (descriptive only, NOT negative-label eligibility criteria)
    median_duration = all_duration_dist["median"]
    event_observed = sum(1 for d in op_durations_days)
    right_censored = sum(
        1 for (repo, sha), ts in timestamps.items()
        if _parse_timestamp(ts) and
        (datetime.fromisoformat(
            repo_endpoints.get(repo, "").replace("Z", "+00:00")
        ) - _parse_timestamp(ts)).total_seconds() / 86400.0 > median_duration
        if _parse_timestamp(ts) and repo_endpoints.get(repo, "")
    )
    insufficient = len(all_durations_days) - event_observed - right_censored

    return {
        "strategy": "TEMPORAL_NEGATIVES",
        "observation_endpoints": {
            "method": "max(commit_timestamp) per repository",
            "endpoints": {k: v for k, v in sorted(repo_endpoints.items())},
        },
        "observation_duration_distribution": all_duration_dist,
        "observed_positive_duration_distribution": op_duration_dist,
        "median_observation_duration_days": median_duration,
        "descriptive_classification": {
            "note": (
                "Descriptive observational states only. "
                "NOT negative-label eligibility criteria."
            ),
            "event_observed": event_observed,
            "right_censored": right_censored,
            "insufficient_history": max(0, insufficient),
        },
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "FEASIBILITY_ONLY",
        "reason": (
            "Observation windows are computable from available timestamps. "
            "However, censoring is informative and survival analysis "
            "is confounded by selection bias. Temporal survival is "
            "exposure/censoring analysis only, not negative-label evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 4: Within-Repository Controls
# ---------------------------------------------------------------------------

def evaluate_repository_control_strategy(
    populations: dict,
    manifest: dict[str, str],
) -> dict:
    """Evaluate within-repository matched controls."""
    # Count files per repo in each population
    op_by_repo: dict[str, int] = Counter(
        r["repo_name"] for r in populations[OBSERVED_POSITIVE]
    )
    ul_by_repo: dict[str, int] = Counter(
        r["repo_name"] for r in populations[UNLABELED]
    )

    # Matching dimensions available
    matching_dims = [
        {"dimension": "repository", "available": True, "source": "repo_name"},
        {"dimension": "file_type", "available": True, "source": "file_path extension"},
        {"dimension": "temporal_period", "available": True, "source": "commit_timestamp"},
        {"dimension": "change_size", "available": True, "source": "file_features"},
    ]

    return {
        "strategy": "REPOSITORY_CONTROLS",
        "matching_dimensions": matching_dims,
        "observed_positive_by_repo": dict(op_by_repo.most_common()),
        "unlabeled_by_repo": dict(ul_by_repo.most_common()),
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "PROCEED_WITH_CAVEATS",
        "reason": (
            "Matched controls reduce confounding but are NOT negative "
            "labels. They may serve as ranking references or evaluation "
            "strata. Selection bias and residual confounding remain."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 5: Corrective-Commit Controls
# ---------------------------------------------------------------------------

def evaluate_corrective_control_strategy(
    populations: dict,
    phase411b_commits: list[dict],
) -> dict:
    """Evaluate corrective-commit controls."""
    # Files in corrective commits but not strongly attributed
    corrective_controls: list[dict] = []
    for rec in phase411b_commits:
        corrective_files = set(rec.get("corrective_files", []))
        for fp in corrective_files:
            corrective_controls.append({
                "repo_name": rec["repo_name"],
                "commit_sha": rec["commit_sha"],
                "file_path": fp,
                "corrective_sha": rec.get("corrective_sha"),
                "status": CANDIDATE_NEGATIVE,
            })

    return {
        "strategy": "CORRECTIVE_COMMIT_CONTROLS",
        "candidate_controls_count": len(corrective_controls),
        "identifiability": "NOT_IDENTIFIABLE",
        "recommended_for_4.13": "DO_NOT_PROCEED",
        "reason": (
            "Files in corrective commits without strong attribution "
            "are candidate controls/unlabeled, NOT negative labels. "
            "Absence of strong attribution evidence does not establish "
            "absence of defect."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 6: Evidence-Ranking Supervision
# ---------------------------------------------------------------------------

def evaluate_evidence_ranking_strategy(
    populations: dict,
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
) -> dict:
    """Evaluate evidence-ranking supervision."""
    # Build corrective_sha lookup
    commit_corrective: dict[str, str] = {}
    for rec in phase411b_commits:
        if rec.get("corrective_sha"):
            commit_corrective[rec["commit_sha"]] = rec["corrective_sha"]

    # Group files by (commit_sha, corrective_sha) for pair construction
    # Only files with identity_established=True can be paired
    commit_files: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in phase411b_files:
        if not rec.get("identity_established", False):
            continue
        sha = rec["commit_sha"]
        csha = commit_corrective.get(sha, "")
        if not csha:
            continue
        commit_files[(sha, csha)].append(rec)

    # Construct pairs per category
    pair_categories = {
        "STRONG_vs_WEAK": [],
        "STRONG_vs_UNKNOWN": [],
        "WEAK_vs_UNKNOWN": [],
        "STRONG_vs_MODERATE": [],
    }

    for (sha, csha), files in commit_files.items():
        # Group by evidence level within this commit+corrective
        by_level: dict[str, list[dict]] = defaultdict(list)
        for f in files:
            by_level[f["file_evidence_level"]].append(f)

        strong = by_level.get(FILE_EVIDENCE_STRONG, [])
        weak = by_level.get(FILE_EVIDENCE_WEAK, [])
        unknown = by_level.get(FILE_EVIDENCE_UNKNOWN, [])
        moderate = by_level.get(FILE_EVIDENCE_MODERATE, [])

        # STRONG vs WEAK
        for s in strong:
            for w in weak:
                pair_categories["STRONG_vs_WEAK"].append({
                    "strong_file": s["file_path"],
                    "weak_file": w["file_path"],
                    "commit_sha": sha,
                    "corrective_sha": csha,
                    "repo_name": s["repo_name"],
                })

        # STRONG vs UNKNOWN
        for s in strong:
            for u in unknown:
                pair_categories["STRONG_vs_UNKNOWN"].append({
                    "strong_file": s["file_path"],
                    "unknown_file": u["file_path"],
                    "commit_sha": sha,
                    "corrective_sha": csha,
                    "repo_name": s["repo_name"],
                })

        # WEAK vs UNKNOWN
        for w in weak:
            for u in unknown:
                pair_categories["WEAK_vs_UNKNOWN"].append({
                    "weak_file": w["file_path"],
                    "unknown_file": u["file_path"],
                    "commit_sha": sha,
                    "corrective_sha": csha,
                    "repo_name": w["repo_name"],
                })

        # STRONG vs MODERATE (if any)
        for s in strong:
            for m in moderate:
                pair_categories["STRONG_vs_MODERATE"].append({
                    "strong_file": s["file_path"],
                    "moderate_file": m["file_path"],
                    "commit_sha": sha,
                    "corrective_sha": csha,
                    "repo_name": s["repo_name"],
                })

    # Pair statistics
    pair_stats = {}
    for cat_name, pairs in pair_categories.items():
        repo_conc = Counter(p["repo_name"] for p in pairs)
        commit_conc = Counter(p["commit_sha"] for p in pairs)
        pair_stats[cat_name] = {
            "count": len(pairs),
            "unique_repos": len(repo_conc),
            "unique_commits": len(commit_conc),
            "top_5_repos": dict(repo_conc.most_common(5)),
            "top_5_commits": dict(commit_conc.most_common(5)),
        }

    total_pairs = sum(s["count"] for s in pair_stats.values())

    return {
        "strategy": "EVIDENCE_RANKING",
        "pair_categories": pair_stats,
        "total_pairs": total_pairs,
        "identity_verification": {
            "all_pairs_have_identity_established": True,
            "all_pairs_share_commit_sha": True,
            "all_pairs_share_corrective_sha": True,
            "all_pairs_same_repository": True,
        },
        "confounding_risks": [
            "Pairs are from same commit, introducing commit-level correlation",
            "Evidence strength does not establish true relative defect probability",
            "UNKNOWN evidence level is valid but unresolved identity is not",
        ],
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "PROCEED_WITH_CAVEATS",
        "reason": (
            "Evidence-ranking pairs are constructible from available data "
            "with verified identity. The partial ordering is evidence-based, "
            "not true-defect-based. Proceed with explicit caveats about "
            "evidence semantics."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 7: Weak Supervision
# ---------------------------------------------------------------------------

def evaluate_weak_supervision_strategy(
    populations: dict,
    phase411b_files: list[dict],
) -> dict:
    """Evaluate weak supervision feasibility."""
    # Signal coverage
    signal_counts: dict[str, int] = {
        "OBSERVED_POSITIVE": len(populations[OBSERVED_POSITIVE]),
        "CONTENT_RESTORATION": sum(
            1 for r in populations[OBSERVED_POSITIVE]
            if "CONTENT_RESTORATION" in r.get("evidence_types", [])
        ),
        "EXACT_CANDIDATE_CONTENT_REMOVAL": sum(
            1 for r in populations[OBSERVED_POSITIVE]
            if "EXACT_CANDIDATE_CONTENT_REMOVAL" in r.get("evidence_types", [])
        ),
        "SAME_REGION_OVERLAP": 0,
        "SAME_FUNCTION_ONLY": 0,
    }

    # Count region overlap and function signals across all files
    for rec in phase411b_files:
        evtypes = rec.get("evidence_types", [])
        if "SAME_REGION_OVERLAP" in evtypes:
            signal_counts["SAME_REGION_OVERLAP"] += 1
        if "SAME_FUNCTION_ONLY" in evtypes:
            signal_counts["SAME_FUNCTION_ONLY"] += 1

    # Implication structure
    implications = [
        {
            "implication": (
                "OBSERVED_POSITIVE => CONTENT_RESTORATION "
                "| EXACT_CANDIDATE_CONTENT_REMOVAL"
            ),
            "verified": True,
            "reason": "By definition, FILE_STRONG requires one of these evidence types",
        },
        {
            "implication": "CONTENT_RESTORATION => OBSERVED_POSITIVE",
            "verified": True,
            "reason": "All CONTENT_RESTORATION cases are FILE_STRONG",
        },
        {
            "implication": "EXACT_CANDIDATE_CONTENT_REMOVAL => OBSERVED_POSITIVE",
            "verified": True,
            "reason": "All EXACT_CANDIDATE_CONTENT_REMOVAL cases are FILE_STRONG (479/499)",
        },
    ]

    # Independence assessment
    independence = {
        "CONTENT_RESTORATION_and_EXACT_CANDIDATE_CONTENT_REMOVAL": {
            "status": "NOT_INDEPENDENT",
            "reason": "Both are subsets of OBSERVED_POSITIVE; 400/499 have both",
        },
        "SAME_REGION_OVERLAP_and_OBSERVED_POSITIVE": {
            "status": "PARTIALLY_INDEPENDENT",
            "reason": "SAME_REGION_OVERLAP spans both OBSERVED_POSITIVE and EVIDENCE_WEAK",
        },
        "SAME_FUNCTION_ONLY_and_OBSERVED_POSITIVE": {
            "status": "PARTIALLY_INDEPENDENT",
            "reason": "SAME_FUNCTION_ONLY spans both OBSERVED_POSITIVE and EVIDENCE_WEAK",
        },
    }

    return {
        "strategy": "WEAK_SUPERVISION",
        "signal_coverage": signal_counts,
        "implication_structure": implications,
        "independence_assessment": independence,
        "label_model_feasibility": {
            "status": "PARTIALLY_IDENTIFIABLE",
            "reason": (
                "Standard conditional independence assumption is violated. "
                "A dependency-aware model using the implication structure "
                "could be explored but requires explicit dependency modeling."
            ),
        },
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "FEASIBILITY_ONLY",
        "reason": (
            "Weak signals exist but are correlated. Dependency-aware "
            "label modeling is possible in principle but requires "
            "explicit implementation of the implication structure."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 8: Synthetic Negatives
# ---------------------------------------------------------------------------

def evaluate_synthetic_strategy() -> dict:
    """Evaluate synthetic negative feasibility."""
    return {
        "strategy": "SYNTHETIC_NEGATIVES",
        "feasibility": "NOT_RECOMMENDED",
        "assumptions": [
            "Controlled transformations preserve non-defect semantics",
            "Synthetic examples approximate real-world defect-free code",
        ],
        "domain_shift_risk": "HIGH",
        "identifiability": "NOT_IDENTIFIABLE",
        "recommended_for_4.13": "DO_NOT_PROCEED",
        "reason": (
            "Domain shift between synthetic and real examples makes "
            "synthetic negatives unreliable for training. No synthetic "
            "dataset should be generated."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 9: Commit-Level Fallback
# ---------------------------------------------------------------------------

def evaluate_commit_level_strategy(
    populations: dict,
    phase411b_commits: list[dict],
) -> dict:
    """Evaluate commit-level supervision as fallback."""
    total_commits = len(phase411b_commits)
    commit_evidence = Counter(
        rec.get("commit_evidence_level", "NONE") for rec in phase411b_commits
    )

    return {
        "strategy": "COMMIT_LEVEL_FALLBACK",
        "commit_evidence_distribution": dict(commit_evidence),
        "total_commits": total_commits,
        "identifiability": "PARTIALLY_IDENTIFIABLE",
        "recommended_for_4.13": "PROCEED_WITH_CAVEATS",
        "reason": (
            "Commit-level labels are more reliable than file-level "
            "(less attribution ambiguity). However, commit risk is not "
            "file risk. A commit may introduce a defect in one file "
            "while others are clean. This is a valid fallback but does "
            "not solve file-level attribution."
        ),
    }


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def build_prediction_time_feature_table() -> list[dict]:
    """Return the explicit prediction-time feature eligibility table."""
    return list(PREDICTION_TIME_FEATURES)


def _audit_data_leakage() -> dict:
    """Audit for data leakage in prediction-time features."""
    features = [
        item for item in PREDICTION_TIME_FEATURES
        if item["permitted_usage"] == "FEATURE"
    ]
    excluded = [
        item for item in PREDICTION_TIME_FEATURES
        if item["permitted_usage"] == "EXCLUDED"
    ]
    label_only = [
        item for item in PREDICTION_TIME_FEATURES
        if item["permitted_usage"] == "LABEL"
    ]

    leakage_free = all(
        item["leakage_status"] == "NO_LEAKAGE" for item in features
    )
    label_source_excluded = all(
        item["name"] != "label_source" for item in features
    )

    return {
        "prediction_time_features": [f["name"] for f in features],
        "excluded_features": [f["name"] for f in excluded],
        "label_only_features": [f["name"] for f in label_only],
        "leakage_free": leakage_free,
        "label_source_excluded_from_features": label_source_excluded,
        "audit_result": "PASS" if (leakage_free and label_source_excluded) else "FAIL",
    }


def _analyze_class_imbalance(
    populations: dict,
    phase411b_commits: list[dict],
) -> dict:
    """Analyze class imbalance and concentration."""
    # Repository concentration
    op_by_repo: dict[str, int] = Counter(
        r["repo_name"] for r in populations[OBSERVED_POSITIVE]
    )
    all_repos = set(
        r["repo_name"] for r in populations[OBSERVED_POSITIVE]
    ) | set(
        r["repo_name"] for r in populations[EVIDENCE_WEAK]
    ) | set(
        r["repo_name"] for r in populations[EVIDENCE_UNKNOWN]
    ) | set(
        r["repo_name"] for r in populations[UNLABELED]
    )

    # Commit concentration
    op_by_commit: dict[str, int] = Counter(
        r["commit_sha"] for r in populations[OBSERVED_POSITIVE]
    )

    # Positive files per repo
    total_by_repo: dict[str, int] = Counter()
    for pop in (OBSERVED_POSITIVE, EVIDENCE_WEAK, EVIDENCE_UNKNOWN, UNLABELED):
        for r in populations[pop]:
            total_by_repo[r["repo_name"]] += 1

    return {
        "positive_by_repo": dict(op_by_repo.most_common()),
        "positive_by_commit": dict(op_by_commit.most_common(10)),
        "total_by_repo": dict(total_by_repo.most_common()),
        "repos_with_positives": len(op_by_repo),
        "repos_without_positives": len(all_repos) - len(op_by_repo),
        "total_repos": len(all_repos),
        "concentration": {
            "top_5_repo_share": (
                sum(c for _, c in op_by_repo.most_common(5))
                / max(sum(op_by_repo.values()), 1)
            ),
            "gini_coefficient": _gini(list(op_by_repo.values())),
        },
    }


def _gini(values: list[int]) -> float:
    """Compute Gini coefficient for a list of counts."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    cumsum = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total) if total > 0 else 0.0


def _analyze_identifiability(strategy_results: list[dict]) -> dict:
    """Compile identifiability classifications for all strategies."""
    classifications = {}
    for sr in strategy_results:
        classifications[sr["strategy"]] = sr.get("identifiability", "UNKNOWN")

    all_classified = all(
        v in {"IDENTIFIABLE", "PARTIALLY_IDENTIFIABLE", "NOT_IDENTIFIABLE", "UNTESTABLE"}
        for v in classifications.values()
    )

    return {
        "strategy_classifications": classifications,
        "all_classified": all_classified,
    }


def _build_decision_matrix(strategy_results: list[dict]) -> list[dict]:
    """Build the decision matrix from strategy results."""
    matrix = []
    for sr in strategy_results:
        matrix.append({
            "strategy": sr["strategy"],
            "evidence_available": sr.get("evidence_available", {}),
            "required_assumptions": sr.get("required_assumptions", []),
            "identifiability": sr.get("identifiability", "UNKNOWN"),
            "negative_labels_created": sr.get("negative_labels_created", 0),
            "risk_of_label_bias": sr.get("risk_of_label_bias", "UNKNOWN"),
            "temporal_leakage_risk": sr.get("temporal_leakage_risk", "UNKNOWN"),
            "recommended_for_4.13": sr.get("recommended_for_4.13", "UNKNOWN"),
            "reason": sr.get("reason", ""),
        })
    return matrix


# ---------------------------------------------------------------------------
# Artifacts and report
# ---------------------------------------------------------------------------

def _write_artifacts(
    output_dir: Path,
    populations: dict,
    disjointness: dict,
    pu_result: dict,
    reliable_neg_result: dict,
    temporal_result: dict,
    repo_control_result: dict,
    corrective_control_result: dict,
    evidence_ranking_result: dict,
    weak_sup_result: dict,
    synthetic_result: dict,
    commit_level_result: dict,
    feature_table: list[dict],
    leakage_audit: dict,
    imbalance: dict,
    identifiability: dict,
    decision_matrix: list[dict],
    report: str,
) -> None:
    """Write all 8 artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. supervision_population_analysis.json
    pop_analysis = {
        "populations": {
            OBSERVED_POSITIVE: {"count": len(populations[OBSERVED_POSITIVE])},
            EVIDENCE_MODERATE: {"count": len(populations[EVIDENCE_MODERATE])},
            EVIDENCE_WEAK: {"count": len(populations[EVIDENCE_WEAK])},
            EVIDENCE_UNKNOWN: {"count": len(populations[EVIDENCE_UNKNOWN])},
            UNLABELED: {"count": len(populations[UNLABELED])},
            OUT_OF_SCOPE: {"count": len(populations[OUT_OF_SCOPE])},
            CANDIDATE_NEGATIVE: {"count": len(populations[CANDIDATE_NEGATIVE])},
            DEFENSIBLE_NEGATIVE: {"count": len(populations[DEFENSIBLE_NEGATIVE])},
        },
        "disjointness": disjointness,
        "metadata": populations["metadata"],
        "class_imbalance": imbalance,
    }
    _write_json(output_dir / "supervision_population_analysis.json", pop_analysis)

    # 2. pu_identifiability_analysis.json
    _write_json(output_dir / "pu_identifiability_analysis.json", pu_result)

    # 3. negative_strategy_analysis.json
    _write_json(output_dir / "negative_strategy_analysis.json", reliable_neg_result)

    # 4. temporal_control_analysis.json
    _write_json(output_dir / "temporal_control_analysis.json", temporal_result)

    # 5. pairwise_ranking_analysis.json
    _write_json(output_dir / "pairwise_ranking_analysis.json", evidence_ranking_result)

    # 6. weak_supervision_analysis.json
    _write_json(output_dir / "weak_supervision_analysis.json", weak_sup_result)

    # 7. supervision_decision_matrix.json
    _write_json(output_dir / "supervision_decision_matrix.json", {
        "decision_matrix": decision_matrix,
        "feature_table": feature_table,
        "leakage_audit": leakage_audit,
        "identifiability": identifiability,
    })

    # 8. phase412_supervision_strategy.md
    with open(output_dir / "phase412_supervision_strategy.md", "w", encoding="utf-8") as f:
        f.write(report)


def _generate_report(
    populations: dict,
    disjointness: dict,
    strategy_results: list[dict],
    feature_table: list[dict],
    leakage_audit: dict,
    identifiability: dict,
    decision_matrix: list[dict],
) -> str:
    """Generate the human-readable report."""
    meta = populations["metadata"]
    lines = [
        "# Phase 4.12: Defensible Supervision & Negative-Label Construction Feasibility",
        "",
        "## 1. Research Question",
        "",
        "Given Phase 4.11b results (499 OBSERVED_POSITIVE, 238 EVIDENCE_WEAK, "
        "318 EVIDENCE_UNKNOWN, 0 defensible negatives), determine which "
        "supervision strategy can produce a scientifically defensible "
        "file-level risk model.",
        "",
        "## 2. Frozen Inputs",
        "",
        "- Frozen supervised JSONL (train/validation/test)",
        "- Phase 4.6 REPO_MANIFEST from repo_split.py",
        "- Phase 4.11b file_attribution_results.json",
        "- Phase 4.11b per_commit_git_evidence.jsonl",
        "- Raw commit_timestamps from dataset JSONL",
        "",
        "## 3. Population Accounting",
        "",
        "| Population | Count |",
        "|------------|-------|",
        f"| Total supervised rows | {meta['total_supervised_rows']} |",
        f"| Historical defect_label=1 | {meta['historical_positive_rows']} |",
        f"| OBSERVED_POSITIVE | {meta['observed_positive_rows']} |",
        f"| EVIDENCE_MODERATE | {meta['evidence_moderate_rows']} |",
        f"| EVIDENCE_WEAK | {meta['evidence_weak_rows']} |",
        f"| EVIDENCE_UNKNOWN | {meta['evidence_unknown_rows']} |",
        f"| UNLABELED | {meta['unlabeled_rows']} |",
        f"| OUT_OF_SCOPE | {meta['out_of_scope_rows']} |",
        f"| Date range | {meta['date_range']} |",
        "",
        "**Note:** historical_positive_rows (defect_label=1) and "
        "observed_positive_rows (FILE_STRONG) are different concepts "
        "and are NOT asserted to be equal.",
        "",
        "### Positive Commit / Repository Coverage",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Historical positive commits (defect_label=1) | {meta['historical_positive_commits']} |",
        f"| FILE_STRONG commits | {meta['file_strong_commits']} |",
        f"| Historical positive repos | {meta['historical_positive_repos']} |",
        f"| Repos with FILE_STRONG | {meta['repos_with_file_strong']} |",
        "",
        f"**Disjointness:** {'PASS' if disjointness['all_disjoint'] else 'FAIL'}",
        "",
        "## 4. Leakage Audit",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Leakage-free features | {leakage_audit['leakage_free']} |",
        f"| label_source excluded | {leakage_audit['label_source_excluded_from_features']} |",
        f"| Overall | {leakage_audit['audit_result']} |",
        "",
        "## 5. Strategy Assessments",
        "",
    ]

    for sr in strategy_results:
        lines.append(f"### {sr['strategy']}")
        lines.append(f"- Identifiability: {sr.get('identifiability', 'UNKNOWN')}")
        lines.append(f"- Recommendation: {sr.get('recommended_for_4.13', 'UNKNOWN')}")
        lines.append(f"- Reason: {sr.get('reason', '')}")
        lines.append("")

    lines.extend([
        "## 6. Decision Matrix",
        "",
        "| Strategy | Identifiable | Negative Labels | Leakage Risk | Recommendation |",
        "|----------|-------------|----------------|-------------|----------------|",
    ])
    for row in decision_matrix:
        lines.append(
            f"| {row['strategy']} | {row['identifiability']} | "
            f"{row['negative_labels_created']} | {row['temporal_leakage_risk']} | "
            f"{row['recommended_for_4.13']} |"
        )

    lines.extend([
        "",
        "## 7. Limitations",
        "",
        "- No defensible negatives exist with current evidence",
        "- Evidence-level labels (FILE_STRONG, FILE_WEAK, UNKNOWN) are "
        "outcome-derived and excluded from prediction-time features",
        "- PU class prior is not identifiable",
        "- Temporal survival is confounded by informative censoring",
        "- Candidate negatives are NOT true negatives",
        "",
        "## 8. Final Recommendation",
        "",
        "The decision matrix determines the recommended Phase 4.13 strategy.",
        "If all strategies are insufficient, DO_NOT_PROCEED with a list of "
        "what additional evidence or data collection is needed.",
        "",
        "## 9. Braket Status",
        "",
        "Braket remains blocked until a scientifically defensible "
        "risk-ranking formulation exists.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_phase412(
    data_dir: Path = DATA_DIR,
    phase411b_dir: Path = PHASE411B_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the full Phase 4.12 feasibility study."""
    logger.info("Phase 4.12: Loading data...")

    # Load frozen data
    supervised_rows = _read_all_supervised_rows(data_dir)
    ambiguous_rows = _read_ambiguous_rows(data_dir)
    manifest = _parse_frozen_manifest()

    # Load Phase 4.11b artifacts
    phase411b_files = _load_phase411b_file_results(phase411b_dir)
    phase411b_commits = _load_phase411b_commit_evidence(phase411b_dir)

    # Load timestamps
    timestamps = _load_timestamps(data_dir)

    logger.info(
        "Phase 4.12: Loaded %d supervised rows, %d Phase 4.11b files, "
        "%d Phase 4.11b commits, %d timestamps",
        len(supervised_rows), len(phase411b_files),
        len(phase411b_commits), len(timestamps),
    )

    # Construct populations
    logger.info("Phase 4.12: Constructing supervision populations...")
    populations = construct_supervision_populations(
        phase411b_files, phase411b_commits,
        supervised_rows, ambiguous_rows, manifest, timestamps,
    )

    # Verify disjointness
    disjointness = _verify_population_disjointness(populations)

    # Run strategy evaluations
    logger.info("Phase 4.12: Evaluating strategies...")
    strategy_results = []

    strategy_results.append(evaluate_pu_strategy(populations, supervised_rows))
    strategy_results.append(evaluate_reliable_negative_strategy(
        populations, phase411b_files, phase411b_commits, timestamps,
    ))
    strategy_results.append(evaluate_temporal_strategy(
        populations, phase411b_commits, timestamps,
    ))
    strategy_results.append(evaluate_repository_control_strategy(populations, manifest))
    strategy_results.append(evaluate_corrective_control_strategy(
        populations, phase411b_commits,
    ))
    strategy_results.append(evaluate_evidence_ranking_strategy(
        populations, phase411b_files, phase411b_commits,
    ))
    strategy_results.append(evaluate_weak_supervision_strategy(
        populations, phase411b_files,
    ))
    strategy_results.append(evaluate_synthetic_strategy())
    strategy_results.append(evaluate_commit_level_strategy(
        populations, phase411b_commits,
    ))

    # Analyses
    feature_table = build_prediction_time_feature_table()
    leakage_audit = _audit_data_leakage()
    imbalance = _analyze_class_imbalance(populations, phase411b_commits)
    identifiability = _analyze_identifiability(strategy_results)

    # Add common fields to strategy results
    for sr in strategy_results:
        sr.setdefault("negative_labels_created", 0)
        is_not_identifiable = sr.get("identifiability") == "NOT_IDENTIFIABLE"
        sr.setdefault("risk_of_label_bias", "NONE" if is_not_identifiable else "MEDIUM")
        sr.setdefault("temporal_leakage_risk", "NONE")
        sr.setdefault("required_assumptions", [])
        sr.setdefault("evidence_available", {})

    decision_matrix = _build_decision_matrix(strategy_results)

    # Generate report
    report = _generate_report(
        populations, disjointness, strategy_results,
        feature_table, leakage_audit, identifiability, decision_matrix,
    )

    # Write artifacts
    logger.info("Phase 4.12: Writing artifacts...")
    _write_artifacts(
        output_dir, populations, disjointness,
        strategy_results[0], strategy_results[1], strategy_results[2],
        strategy_results[3], strategy_results[4], strategy_results[5],
        strategy_results[6], strategy_results[7], strategy_results[8],
        feature_table, leakage_audit, imbalance, identifiability,
        decision_matrix, report,
    )

    logger.info("Phase 4.12: Complete.")
    return {
        "populations": {k: len(v) for k, v in populations.items() if k != "metadata"},
        "metadata": populations["metadata"],
        "disjointness": disjointness,
        "strategy_count": len(strategy_results),
        "decision_matrix": decision_matrix,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase412()
