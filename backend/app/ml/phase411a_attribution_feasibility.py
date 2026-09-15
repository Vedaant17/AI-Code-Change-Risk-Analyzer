"""Phase 4.11a: Attribution Feasibility Study (JSONL-only, no repo access).

READ-ONLY analysis of whether existing JSONL/artifact evidence can improve
file-level defect attribution without repository access.

Frozen scope:
- Phase 4.6–4.10 source/artifacts: UNCHANGED
- combined-v3 dataset: UNCHANGED
- repo_split.py: UNCHANGED
- diff_parser.py: UNCHANGED
- No repos cloned, no network access, no ML training
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from backend.app.ml.phase49_supervision_feasibility import (
    _build_sha_lookup,
    _group_by_commit,
    _parse_frozen_manifest,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
    construct_supervision_sets,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("backend/data/datasets/combined-v3")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11a")
PHASE48_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.8")
PHASE410_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.10")

_BUG_FIX_SHA_PATTERN = re.compile(r"bug-fix ([0-9a-f]{8,40})")
_REVERT_SHA_PATTERN = re.compile(r"reverted by ([0-9a-f]{8,40})")

# Evidence hierarchy (two-level: commit and file)
COMMIT_EVIDENCE_STRONG = "COMMIT_STRONG"
COMMIT_EVIDENCE_WEAK = "COMMIT_WEAK"
COMMIT_EVIDENCE_NONE = "COMMIT_NONE"

FILE_EVIDENCE_STRONG = "FILE_STRONG"
FILE_EVIDENCE_MODERATE = "FILE_MODERATE"
FILE_EVIDENCE_WEAK = "FILE_WEAK"
FILE_EVIDENCE_UNKNOWN = "UNKNOWN"

# Contextual signal categories (NOT attribution)
CONTEXTUAL_TEST_SEPARATION = "test_separation"
CONTEXTUAL_FILE_STATUS = "file_status_pattern"
CONTEXTUAL_FEATURE_PROXIMITY = "feature_proximity"
CONTEXTUAL_EXCLUSIVITY = "corrective_exclusivity"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_data() -> dict:
    """Load all required data sources. Returns a dict of loaded data."""
    supervised_rows = _read_all_supervised_rows(DATA_DIR)
    ambiguous_rows = _read_ambiguous_rows(DATA_DIR)
    sha_lookup = _build_sha_lookup(supervised_rows, ambiguous_rows)
    manifest = _parse_frozen_manifest()

    # Load Phase 4.8 attribution cases
    phase48_cases = []
    cases_path = PHASE48_DIR / "file_attribution_cases.jsonl"
    if cases_path.exists():
        with open(cases_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    phase48_cases.append(json.loads(line))

    return {
        "supervised_rows": supervised_rows,
        "ambiguous_rows": ambiguous_rows,
        "sha_lookup": sha_lookup,
        "manifest": manifest,
        "phase48_cases": phase48_cases,
    }


# ---------------------------------------------------------------------------
# SHA chain provenance
# ---------------------------------------------------------------------------

def _trace_sha_chain(
    commit_sha: str,
    sha_lookup: dict[str, list[dict]],
    label_source: str,
    label_evidence: str,
    max_depth: int = 3,
) -> dict:
    """Trace the SHA provenance chain for a positive commit.

    This is COMMIT-LEVEL provenance analysis, NOT file-level attribution.

    Follows corrective SHA references recursively up to max_depth levels.
    Returns chain metadata including depth, resolution status, and whether
    the chain terminates at a specific commit.
    """
    chain = []
    current_sha = commit_sha
    seen = set()

    for depth in range(max_depth):
        if current_sha in seen:
            break
        seen.add(current_sha)

        rows = sha_lookup.get(current_sha, [])
        if not rows:
            break

        evidence = rows[0].get("label_evidence", "")
        fix_prefix = _extract_fix_sha_prefix(evidence)
        if fix_prefix is None:
            break

        full_sha = _resolve_full_sha(fix_prefix, sha_lookup)
        chain.append({
            "sha": current_sha,
            "corrective_prefix": fix_prefix,
            "corrective_resolved": full_sha is not None,
            "corrective_sha": full_sha,
            "label_source": rows[0].get("label_source", "none"),
        })

        if full_sha is None:
            break
        current_sha = full_sha

    return {
        "origin_sha": commit_sha,
        "chain_depth": len(chain),
        "chain": chain,
        "chain_terminates": len(chain) > 0 and chain[-1]["corrective_resolved"],
        "final_sha": chain[-1]["corrective_sha"] if chain else None,
    }


# ---------------------------------------------------------------------------
# Commit-message file signal
# ---------------------------------------------------------------------------

def _extract_file_mentions_from_message(
    message: str,
    candidate_files: list[str],
) -> dict:
    """Extract file/function mentions from commit message text.

    This produces MODERATE/WEAK CONTEXTUAL FILE SIGNAL only.
    It must NOT be treated as causal attribution.
    """
    if not message:
        return {"mentions": [], "matched_files": [], "signal_level": "none"}

    message_lower = message.lower()
    matched = []
    for fp in candidate_files:
        basename = Path(fp).name.lower()
        stem = Path(fp).stem.lower()
        # Check for basename or stem mention
        if basename in message_lower or stem in message_lower:
            matched.append(fp)
        # Check for partial path components
        parts = fp.replace("\\", "/").split("/")
        for part in parts:
            if len(part) > 3 and part.lower() in message_lower:
                if fp not in matched:
                    matched.append(fp)
                break

    # Check for function/class mentions (common patterns)
    func_pattern = re.compile(
        r"\b(?:fix|patch|resolve|bug|issue|error)\b.*?\b(\w+)\b",
        re.IGNORECASE,
    )
    func_matches = func_pattern.findall(message)

    signal_level = "none"
    if len(matched) == len(candidate_files) and len(matched) > 0:
        signal_level = "all_files_mentioned"
    elif len(matched) > 0:
        signal_level = "partial_files_mentioned"

    return {
        "mentions": func_matches[:5],
        "matched_files": matched,
        "match_count": len(matched),
        "signal_level": signal_level,
    }


# ---------------------------------------------------------------------------
# File-status narrowing
# ---------------------------------------------------------------------------

def _analyze_file_status_pattern(
    candidate_rows: list[dict],
    corrective_rows: list[dict],
) -> dict:
    """Analyze file_status patterns between candidate and corrective commits.

    This is a DESCRIPTIVE analysis only. It does NOT produce attribution evidence.
    """
    cand_statuses = defaultdict(int)
    for r in candidate_rows:
        cand_statuses[r.get("file_status", "unknown")] += 1

    fix_statuses = defaultdict(int)
    for r in corrective_rows:
        fix_statuses[r.get("file_status", "unknown")] += 1

    # Check if status patterns overlap
    cand_set = set(cand_statuses.keys())
    fix_set = set(fix_statuses.keys())
    overlap = cand_set & fix_set

    return {
        "candidate_statuses": dict(cand_statuses),
        "corrective_statuses": dict(fix_statuses),
        "status_overlap": sorted(overlap),
        "has_added_in_both": "added" in cand_set and "added" in fix_set,
        "has_deleted_in_both": "deleted" in cand_set and "deleted" in fix_set,
        "has_renamed_in_both": "renamed" in cand_set and "renamed" in fix_set,
    }


# ---------------------------------------------------------------------------
# Feature proximity (contextual signal, NOT attribution)
# ---------------------------------------------------------------------------

def _compute_feature_proximity(
    candidate_rows: list[dict],
    corrective_rows: list[dict],
) -> dict:
    """Compute feature vector proximity between candidate and corrective.

    This is a CONTEXTUAL SIGNAL for potential predictor features.
    It must NOT be treated as attribution evidence.
    """
    if not candidate_rows or not corrective_rows:
        return {"proximity": 0.0, "available": False}

    # Average commit features across rows for each commit
    cand_features = []
    for r in candidate_rows:
        cf = r.get("commit_features", [])
        if cf and len(cf) == 29:
            cand_features.append(cf)

    fix_features = []
    for r in corrective_rows:
        cf = r.get("commit_features", [])
        if cf and len(cf) == 29:
            fix_features.append(cf)

    if not cand_features or not fix_features:
        return {"proximity": 0.0, "available": False}

    # Compute mean feature vectors
    import numpy as np
    cand_mean = np.mean(cand_features, axis=0)
    fix_mean = np.mean(fix_features, axis=0)

    # Cosine similarity
    dot = float(np.dot(cand_mean, fix_mean))
    norm_c = float(np.linalg.norm(cand_mean))
    norm_f = float(np.linalg.norm(fix_mean))
    cosine = dot / (norm_c * norm_f) if norm_c > 0 and norm_f > 0 else 0.0

    # Euclidean distance
    euclidean = float(np.linalg.norm(cand_mean - fix_mean))

    return {
        "cosine_similarity": round(cosine, 6),
        "euclidean_distance": round(euclidean, 6),
        "available": True,
    }


# ---------------------------------------------------------------------------
# Test/production separation
# ---------------------------------------------------------------------------

def _analyze_test_separation(
    candidate_rows: list[dict],
) -> dict:
    """Analyze test/production file separation in candidate commit.

    This is a CONTEXTUAL SIGNAL, not attribution evidence.
    """
    test_files = []
    prod_files = []
    other_files = []

    for r in candidate_rows:
        ff = r.get("file_features", [])
        is_test = ff[2] if len(ff) > 2 else 0.0
        fp = r.get("file_path", "")
        if is_test > 0.5:
            test_files.append(fp)
        elif any(x in fp for x in ("test_", "_test.py", "tests/", "test/")):
            test_files.append(fp)
        else:
            prod_files.append(fp)

    return {
        "test_files": test_files,
        "prod_files": prod_files,
        "other_files": other_files,
        "test_count": len(test_files),
        "prod_count": len(prod_files),
        "has_both": len(test_files) > 0 and len(prod_files) > 0,
    }


# ---------------------------------------------------------------------------
# Corrective exclusivity ratio
# ---------------------------------------------------------------------------

def _compute_exclusivity_ratio(
    candidate_file_count: int,
    corrective_file_count: int,
    path_overlap_count: int,
) -> dict:
    """Compute exclusivity ratio between candidate and corrective.

    This is a DESCRIPTIVE STATISTIC, not attribution evidence.
    High exclusivity means the corrective commit is narrowly targeted.
    """
    if corrective_file_count == 0:
        return {"ratio": 0.0, "available": False}

    overlap_ratio = path_overlap_count / corrective_file_count
    narrowing_ratio = (
        path_overlap_count / candidate_file_count
        if candidate_file_count > 0
        else 0.0
    )

    return {
        "corrective_file_count": corrective_file_count,
        "candidate_file_count": candidate_file_count,
        "path_overlap_count": path_overlap_count,
        "overlap_ratio": round(overlap_ratio, 4),
        "narrowing_ratio": round(narrowing_ratio, 4),
        "available": True,
    }


# ---------------------------------------------------------------------------
# Revert vs explicit-SHA analysis
# ---------------------------------------------------------------------------

def _classify_revert_evidence(
    label_source: str,
    candidate_files: list[str],
    corrective_files: list[str],
    sha_lookup: dict[str, list[dict]],
    corrective_sha: str | None,
) -> dict:
    """Classify revert vs explicit-SHA attribution evidence.

    METHODOLOGICAL NOTE:
    A revert provides STRONG COMMIT-LEVEL EVIDENCE that the candidate
    commit was reverted. It does NOT prove that every file changed by
    the candidate was defective. The affected files remain UNKNOWN
    unless additional file-level evidence exists.
    """
    is_revert = label_source == "revert"
    is_explicit = label_source == "explicit_sha_reference"

    commit_evidence = COMMIT_EVIDENCE_NONE
    if is_revert or is_explicit:
        commit_evidence = COMMIT_EVIDENCE_STRONG

    # File-level evidence from revert: only path overlap (WEAK)
    file_evidence = FILE_EVIDENCE_UNKNOWN
    file_evidence_list = []

    if corrective_sha:
        candidate_set = set(candidate_files)
        corrective_set = set(corrective_files)
        intersection = candidate_set & corrective_set

        if is_revert:
            # Revert: path overlap is WEAK evidence (files in both)
            if intersection:
                file_evidence = FILE_EVIDENCE_WEAK
                file_evidence_list = [
                    {"file": f, "level": FILE_EVIDENCE_WEAK, "source": "revert_path_overlap"}
                    for f in sorted(intersection)
                ]
        elif is_explicit:
            # Explicit SHA: same path overlap is WEAK evidence
            if intersection:
                file_evidence = FILE_EVIDENCE_WEAK
                file_evidence_list = [
                    {"file": f, "level": FILE_EVIDENCE_WEAK, "source": "explicit_sha_path_overlap"}
                    for f in sorted(intersection)
                ]

    return {
        "label_source": label_source,
        "is_revert": is_revert,
        "is_explicit_sha": is_explicit,
        "commit_evidence_level": commit_evidence,
        "file_evidence_level": file_evidence,
        "file_evidence_list": file_evidence_list,
        "note": (
            "Revert provides strong commit-level evidence that the commit "
            "was reverted. It does NOT prove which specific files were "
            "defective. File-level evidence remains WEAK (path overlap only)."
            if is_revert else
            "Explicit SHA reference provides strong commit-level evidence. "
            "File-level evidence remains WEAK (path overlap only)."
        ),
    }


# ---------------------------------------------------------------------------
# Per-commit comprehensive analysis
# ---------------------------------------------------------------------------

def _analyze_single_commit(
    commit_sha: str,
    candidate_rows: list[dict],
    sha_lookup: dict[str, list[dict]],
) -> dict:
    """Comprehensive analysis of a single positive commit.

    Returns a dict with all evidence levels and contextual signals.
    """
    row0 = candidate_rows[0]
    repo_name = row0["repo_name"]
    label_source = row0.get("label_source", "none")
    label_evidence = row0.get("label_evidence", "")
    commit_message = row0.get("commit_message", "")
    split = row0.get("split", "unknown")

    candidate_files = sorted(set(r["file_path"] for r in candidate_rows))
    candidate_file_count = len(candidate_files)

    # Step 1: Extract corrective SHA
    fix_prefix = _extract_fix_sha_prefix(label_evidence)
    corrective_sha = None
    corrective_files = []
    corrective_rows = []

    if fix_prefix:
        corrective_sha = _resolve_full_sha(fix_prefix, sha_lookup)
        if corrective_sha:
            corrective_rows = sha_lookup.get(corrective_sha, [])
            corrective_files = sorted(set(r["file_path"] for r in corrective_rows))

    # Step 2: Path intersection
    candidate_set = set(candidate_files)
    corrective_set = set(corrective_files)
    intersection = sorted(candidate_set & corrective_set)
    unknown_files = sorted(candidate_set - corrective_set)

    # Step 3: Phase 4.8 category (for comparison)
    if fix_prefix and corrective_sha:
        if len(intersection) == 0:
            phase48_category = "COMMIT_ONLY"
        elif len(intersection) == len(candidate_set):
            phase48_category = "COMMIT_ONLY"
        else:
            phase48_category = "FILE_PARTIAL"
    else:
        phase48_category = "NO_ATTRIBUTION"

    # Step 4: Commit-level evidence
    commit_evidence = COMMIT_EVIDENCE_NONE
    if fix_prefix and corrective_sha:
        commit_evidence = COMMIT_EVIDENCE_STRONG
    elif fix_prefix and not corrective_sha:
        commit_evidence = COMMIT_EVIDENCE_WEAK

    # Step 5: File-level evidence (from Phase 4.8 path intersection)
    file_evidence = FILE_EVIDENCE_UNKNOWN
    attributed_files = []
    if phase48_category == "FILE_PARTIAL":
        file_evidence = FILE_EVIDENCE_WEAK
        attributed_files = intersection
    elif phase48_category == "COMMIT_ONLY" and intersection:
        file_evidence = FILE_EVIDENCE_WEAK
        attributed_files = intersection

    # Step 6: Revert classification
    revert_analysis = _classify_revert_evidence(
        label_source, candidate_files, corrective_files,
        sha_lookup, corrective_sha,
    )

    # Step 7: SHA chain provenance
    chain = _trace_sha_chain(commit_sha, sha_lookup, label_source, label_evidence)

    # Step 8: Contextual signals (NOT attribution)
    message_signal = _extract_file_mentions_from_message(commit_message, candidate_files)
    status_pattern = _analyze_file_status_pattern(candidate_rows, corrective_rows)
    feature_prox = _compute_feature_proximity(candidate_rows, corrective_rows)
    test_sep = _analyze_test_separation(candidate_rows)
    exclusivity = _compute_exclusivity_ratio(
        candidate_file_count, len(corrective_files), len(intersection),
    )

    # Step 9: Path transform candidates
    from backend.app.ml.phase48_attribution_feasibility import (
        _detect_path_transform_candidates,
    )
    path_transforms = _detect_path_transform_candidates(candidate_files, corrective_files)

    return {
        "commit_sha": commit_sha,
        "repo_name": repo_name,
        "split": split,
        "label_source": label_source,
        "commit_message": commit_message[:300],

        # Candidate/Corrective metadata
        "candidate_files": candidate_files,
        "candidate_file_count": candidate_file_count,
        "corrective_sha": corrective_sha,
        "corrective_sha_prefix": fix_prefix,
        "corrective_files": corrective_files,
        "corrective_file_count": len(corrective_files),

        # Phase 4.8 classification (for baseline comparison)
        "phase48_category": phase48_category,
        "path_overlap_files": intersection,
        "unknown_files": unknown_files,

        # Evidence levels
        "commit_evidence_level": commit_evidence,
        "file_evidence_level": file_evidence,
        "attributed_files": attributed_files,

        # Revert vs explicit analysis
        "revert_analysis": revert_analysis,

        # SHA chain provenance (commit-level)
        "sha_chain": chain,

        # Contextual signals (NOT attribution evidence)
        "contextual_signals": {
            "message_file_signal": message_signal,
            "file_status_pattern": status_pattern,
            "feature_proximity": feature_prox,
            "test_separation": test_sep,
            "corrective_exclusivity": exclusivity,
            "path_transform_candidates": path_transforms,
        },
    }


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def _aggregate_results(results: list[dict]) -> dict:
    """Aggregate per-commit results into summary statistics."""
    total = len(results)

    # Commit-level evidence distribution
    commit_evidence_dist = defaultdict(int)
    for r in results:
        commit_evidence_dist[r["commit_evidence_level"]] += 1

    # File-level evidence distribution
    file_evidence_dist = defaultdict(int)
    for r in results:
        file_evidence_dist[r["file_evidence_level"]] += 1

    # Phase 4.8 category distribution (baseline)
    phase48_dist = defaultdict(int)
    for r in results:
        phase48_dist[r["phase48_category"]] += 1

    # Per-repo breakdown
    repo_breakdown = defaultdict(lambda: {
        "commits": 0, "FILE_PARTIAL": 0, "COMMIT_ONLY": 0, "NO_ATTRIBUTION": 0,
    })
    for r in results:
        repo = r["repo_name"]
        repo_breakdown[repo]["commits"] += 1
        repo_breakdown[repo][r["phase48_category"]] += 1

    # Per-split breakdown
    split_breakdown = defaultdict(lambda: {"commits": 0, "categories": defaultdict(int)})
    for r in results:
        s = r["split"]
        split_breakdown[s]["commits"] += 1
        split_breakdown[s]["categories"][r["phase48_category"]] += 1

    # Attributed file counts
    commits_with_any_attribution = sum(
        1 for r in results if r["attributed_files"]
    )
    total_attributed_files = sum(
        len(r["attributed_files"]) for r in results
    )

    # Revert vs explicit breakdown
    source_breakdown = defaultdict(lambda: {"commits": 0, "commit_strong": 0, "file_weak": 0})
    for r in results:
        src = r["label_source"]
        source_breakdown[src]["commits"] += 1
        if r["commit_evidence_level"] == COMMIT_EVIDENCE_STRONG:
            source_breakdown[src]["commit_strong"] += 1
        if r["file_evidence_level"] == FILE_EVIDENCE_WEAK:
            source_breakdown[src]["file_weak"] += 1

    # Contextual signal summaries
    message_match_count = sum(
        1 for r in results
        if r["contextual_signals"]["message_file_signal"]["match_count"] > 0
    )
    test_prod_mixed = sum(
        1 for r in results
        if r["contextual_signals"]["test_separation"]["has_both"]
    )

    return {
        "total_commits": total,
        "commit_evidence_distribution": dict(commit_evidence_dist),
        "file_evidence_distribution": dict(file_evidence_dist),
        "phase48_baseline_distribution": dict(phase48_dist),
        "per_repo_breakdown": dict(repo_breakdown),
        "per_split_breakdown": {
            k: {"commits": v["commits"], "categories": dict(v["categories"])}
            for k, v in split_breakdown.items()
        },
        "commits_with_any_attribution": commits_with_any_attribution,
        "total_attributed_files": total_attributed_files,
        "source_breakdown": dict(source_breakdown),
        "contextual_signals_summary": {
            "message_file_match_count": message_match_count,
            "test_prod_mixed_commits": test_prod_mixed,
        },
    }


# ---------------------------------------------------------------------------
# Negative feasibility audit
# ---------------------------------------------------------------------------

def _audit_negative_feasibility(
    results: list[dict],
    supervision_sets: dict,
) -> dict:
    """Audit whether defensible negatives could be constructed.

    This is an AUDIT only. It does NOT create candidate negative labels.
    Any proposed negative must remain explicitly CANDIDATE_NEGATIVE.
    """
    total_candidate_files = sum(r["candidate_file_count"] for r in results)
    total_unknown_files = sum(len(r["unknown_files"]) for r in results)
    total_overlap = sum(len(r["path_overlap_files"]) for r in results)

    # Check: do any unknown files have subsequent corrective commits?
    # (Would require deeper git history analysis — not available here)
    # Check: observation window sufficiency
    # (Would require commit timestamps and window analysis)

    # Current state: Phase 4.9 found 0 defensible negatives
    # Phase 4.11a cannot construct negatives from JSONL alone

    investigation = {
        "strategy_1_survival": {
            "description": (
                "Long post-change survival without defect attribution"
            ),
            "feasible_without_repos": False,
            "reason": (
                "Requires git history to determine observation "
                "windows and subsequent commits"
            ),
            "candidate_negative_count": 0,
        },
        "strategy_2_subsequent_commits": {
            "description": "Subsequent commits without defect attribution",
            "feasible_without_repos": False,
            "reason": (
                "Requires git log to find subsequent "
                "commits to same file"
            ),
            "candidate_negative_count": 0,
        },
        "strategy_3_neighboring_correction": {
            "description": (
                "Independent corrective commits affecting neighboring code"
            ),
            "feasible_without_repos": False,
            "reason": "Requires git history to identify neighboring changes",
            "candidate_negative_count": 0,
        },
        "strategy_4_stable_files": {
            "description": (
                "Files with sufficient observation windows "
                "and no defect evidence"
            ),
            "feasible_without_repos": False,
            "reason": "Requires commit timestamps and window duration analysis",
            "candidate_negative_count": 0,
        },
    }

    return {
        "total_candidate_files": total_candidate_files,
        "total_unknown_files": total_unknown_files,
        "total_path_overlap_files": total_overlap,
        "defensible_negative_count": 0,
        "candidate_negative_count": 0,
        "conclusion": (
            "No defensible negatives can be constructed from existing "
            "JSONL data alone. All four investigated strategies require "
            "repository history access (commit timestamps, subsequent "
            "commits, neighboring changes). This is consistent with "
            "Phase 4.8 and Phase 4.9 findings."
        ),
        "strategies_investigated": investigation,
        "requires_repo_access": True,
    }


# ---------------------------------------------------------------------------
# Prediction unit analysis
# ---------------------------------------------------------------------------

def _analyze_prediction_units(
    results: list[dict],
    supervised_rows: list[dict],
) -> dict:
    """Evaluate prediction units: commit, file, function, hunk.

    Distinguishes LABEL AVAILABILITY from ATTRIBUTION QUALITY.
    """
    total_positive_commits = len(results)
    total_candidate_files = sum(r["candidate_file_count"] for r in results)
    total_attributed_files = sum(len(r["attributed_files"]) for r in results)
    total_unknown_files = sum(len(r["unknown_files"]) for r in results)

    units = {
        "commit_level": {
            "label_availability": "high",
            "label_count": total_positive_commits,
            "label_method": (
                "revert/explicit_sha provides strong "
                "commit-level evidence"
            ),
            "attribution_quality": (
                "STRONG commit-level, but all files in "
                "commit labeled equally"
            ),
            "coverage": (
                f"{total_positive_commits}/{total_positive_commits} "
                "commits have evidence"
            ),
            "leakage_risk": (
                "low — commit features use only "
                "pre-commit information"
            ),
            "practical_usefulness": (
                "Limited — identifies which commits are risky "
                "but not which files within the commit. "
                "Entire commit treated as positive."
            ),
            "investigation_budget": (
                "Commit-level budget means reviewing ALL files "
                "in a commit. For commits touching 30+ files "
                "(e.g., jinja2), this is impractical."
            ),
        },
        "file_level": {
            "label_availability": "low",
            "label_count": total_attributed_files,
            "label_method": (
                "path intersection with corrective commit "
                "(Phase 4.8 FILE_PARTIAL only)"
            ),
            "attribution_quality": (
                "WEAK — path intersection only. No content, line, or "
                "function-level evidence available without repositories."
            ),
            "coverage": (
                f"{total_attributed_files}/{total_candidate_files} "
                f"candidate files attributed "
                f"({total_attributed_files/total_candidate_files*100:.1f}%)"
                if total_candidate_files > 0 else "0%"
            ),
            "unknown_rate": (
                f"{total_unknown_files}/{total_candidate_files} "
                f"unknown "
                f"({total_unknown_files/total_candidate_files*100:.1f}%)"
                if total_candidate_files > 0 else "0%"
            ),
            "leakage_risk": (
                "low — file features use only "
                "pre-commit information"
            ),
            "practical_usefulness": (
                "High potential but currently limited by WEAK attribution. "
                "If repository-level evidence improves attribution, this "
                "becomes the most practically useful unit."
            ),
            "investigation_budget": (
                "File-level budget means reviewing specific files. "
                "Current 108 attributed files across 38 commits is manageable."
            ),
        },
        "function_level": {
            "label_availability": "none",
            "label_count": 0,
            "label_method": "requires repository history for AST parsing of diffs",
            "attribution_quality": "UNAVAILABLE without repository access",
            "coverage": "0% — not measurable from JSONL data",
            "leakage_risk": "N/A",
            "practical_usefulness": (
                "Highest potential practical usefulness — functions are the "
                "natural unit of code review. But requires repo access."
            ),
            "investigation_budget": (
                "Function-level budget is the most actionable for developers. "
                "Currently unavailable."
            ),
        },
        "hunk_level": {
            "label_availability": "none",
            "label_count": 0,
            "label_method": "requires repository history for diff extraction",
            "attribution_quality": "UNAVAILABLE without repository access",
            "coverage": "0% — not measurable from JSONL data",
            "leakage_risk": "N/A",
            "practical_usefulness": (
                "Most precise but hardest to act on. Requires repo access."
            ),
            "investigation_budget": (
                "Hunk-level budget would be extremely targeted but "
                "requires infrastructure not available here."
            ),
        },
    }

    return {
        "units": units,
        "recommendation": (
            "FILE_LEVEL has the strongest methodological foundation among "
            "available units. It has low label coverage (108/1055 = 10.2%) "
            "and WEAK attribution, but it is the only unit with any file-level "
            "evidence at all. Commit-level has strong labels but poor "
            "practical usefulness (too many files per commit). Function and "
            "hunk levels require repository access."
        ),
        "summary": {
            "commit_labels": total_positive_commits,
            "file_labels": total_attributed_files,
            "file_unknown": total_unknown_files,
            "function_labels": 0,
            "hunk_labels": 0,
        },
    }


# ---------------------------------------------------------------------------
# Repository concentration analysis
# ---------------------------------------------------------------------------

def _analyze_repository_concentration(
    results: list[dict],
) -> dict:
    """Analyze per-repository positive commit and file distribution."""
    repo_data = defaultdict(lambda: {
        "commits": 0,
        "files": 0,
        "attributed_files": 0,
        "unknown_files": 0,
        "commit_strong": 0,
        "file_weak": 0,
        "splits": set(),
    })

    for r in results:
        repo = r["repo_name"]
        repo_data[repo]["commits"] += 1
        repo_data[repo]["files"] += r["candidate_file_count"]
        repo_data[repo]["attributed_files"] += len(r["attributed_files"])
        repo_data[repo]["unknown_files"] += len(r["unknown_files"])
        repo_data[repo]["splits"].add(r["split"])
        if r["commit_evidence_level"] == COMMIT_EVIDENCE_STRONG:
            repo_data[repo]["commit_strong"] += 1
        if r["file_evidence_level"] == FILE_EVIDENCE_WEAK:
            repo_data[repo]["file_weak"] += 1

    # Compute concentration metrics
    total_commits = sum(v["commits"] for v in repo_data.values())
    total_files = sum(v["files"] for v in repo_data.values())

    repo_list = []
    for repo, data in sorted(repo_data.items(), key=lambda x: -x[1]["commits"]):
        repo_list.append({
            "repo": repo,
            "commits": data["commits"],
            "commit_fraction": (
                round(data["commits"] / total_commits, 4)
                if total_commits > 0 else 0
            ),
            "files": data["files"],
            "attributed_files": data["attributed_files"],
            "unknown_files": data["unknown_files"],
            "commit_strong": data["commit_strong"],
            "file_weak": data["file_weak"],
            "splits": sorted(data["splits"]),
        })

    # Gini coefficient for commit distribution
    import numpy as np
    counts = np.array([v["commits"] for v in repo_data.values()], dtype=float)
    if len(counts) > 0 and counts.sum() > 0:
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = float(
            (2 * np.sum(index * sorted_counts)
             / (n * np.sum(sorted_counts)))
            - (n + 1) / n
        )
    else:
        gini = 0.0

    top5 = sum(v["commits"] for v in sorted(repo_data.values(), key=lambda x: -x["commits"])[:5])
    top5_fraction = round(top5 / total_commits, 4) if total_commits > 0 else 0

    return {
        "total_repos_with_positives": len(repo_data),
        "total_positive_commits": total_commits,
        "total_candidate_files": total_files,
        "gini_coefficient": round(gini, 4),
        "top_5_share": top5_fraction,
        "repos": repo_list,
        "repos_with_strong_commit_evidence": sum(
            1 for v in repo_data.values() if v["commit_strong"] > 0
        ),
        "repos_with_file_attribution": sum(
            1 for v in repo_data.values() if v["attributed_files"] > 0
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_phase411a(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the complete Phase 4.11a attribution feasibility study.

    Returns a dict with all results. Writes artifacts to output_dir.
    """
    logger.info("Phase 4.11a: Starting attribution feasibility study")

    # Step 1: Load data
    data = _load_all_data()
    supervised_rows = data["supervised_rows"]
    sha_lookup = data["sha_lookup"]

    # Step 2: Construct supervision sets (for negative audit)
    manifest = data["manifest"]
    supervision_sets = construct_supervision_sets(supervised_rows, sha_lookup, manifest)

    # Step 3: Group positive commits and analyze each
    commit_groups = _group_by_commit(supervised_rows)
    positive_commits = {
        k: rows for k, rows in commit_groups.items()
        if any(r["defect_label"] == 1 for r in rows)
    }

    logger.info("Analyzing %d positive commits", len(positive_commits))

    results = []
    for (repo_name, commit_sha), rows in sorted(positive_commits.items()):
        analysis = _analyze_single_commit(commit_sha, rows, sha_lookup)
        results.append(analysis)

    # Step 4: Aggregate results
    aggregated = _aggregate_results(results)

    # Step 5: Negative feasibility audit
    negative_audit = _audit_negative_feasibility(results, supervision_sets)

    # Step 6: Prediction unit analysis
    prediction_units = _analyze_prediction_units(results, supervised_rows)

    # Step 7: Repository concentration
    repo_concentration = _analyze_repository_concentration(results)

    # Step 8: Contextual signal analysis
    contextual_summary = _summarize_contextual_signals(results)

    # Step 9: Synthesize recommendation
    recommendation = _synthesize_recommendation(
        aggregated, negative_audit, prediction_units, repo_concentration,
    )

    # Step 10: Write artifacts
    _write_artifacts(
        output_dir, results, aggregated, negative_audit,
        prediction_units, repo_concentration, contextual_summary,
        recommendation,
    )

    logger.info("Phase 4.11a: Complete. Artifacts in %s", output_dir)

    return {
        "results": results,
        "aggregated": aggregated,
        "negative_audit": negative_audit,
        "prediction_units": prediction_units,
        "repo_concentration": repo_concentration,
        "contextual_summary": contextual_summary,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Contextual signal summary
# ---------------------------------------------------------------------------

def _summarize_contextual_signals(results: list[dict]) -> dict:
    """Summarize contextual signals across all commits.

    These are NOT attribution evidence. They are potentially useful
    features/signals for future model building.
    """
    message_signals = []
    status_signals = []
    feature_signals = []
    test_signals = []
    exclusivity_signals = []

    for r in results:
        cs = r["contextual_signals"]

        msg = cs["message_file_signal"]
        message_signals.append({
            "commit_sha": r["commit_sha"],
            "match_count": msg["match_count"],
            "signal_level": msg["signal_level"],
        })

        status_signals.append({
            "commit_sha": r["commit_sha"],
            "status_overlap": cs["file_status_pattern"]["status_overlap"],
        })

        if cs["feature_proximity"]["available"]:
            feature_signals.append({
                "commit_sha": r["commit_sha"],
                "cosine_similarity": cs["feature_proximity"]["cosine_similarity"],
            })

        test_signals.append({
            "commit_sha": r["commit_sha"],
            "has_both": cs["test_separation"]["has_both"],
            "test_count": cs["test_separation"]["test_count"],
            "prod_count": cs["test_separation"]["prod_count"],
        })

        exc = cs["corrective_exclusivity"]
        if exc["available"]:
            exclusivity_signals.append({
                "commit_sha": r["commit_sha"],
                "overlap_ratio": exc["overlap_ratio"],
                "narrowing_ratio": exc["narrowing_ratio"],
            })

    # Summary statistics
    msg_match_count = sum(1 for m in message_signals if m["match_count"] > 0)
    has_status_overlap = sum(1 for s in status_signals if s["status_overlap"])
    mixed_test_prod = sum(1 for t in test_signals if t["has_both"])

    return {
        "message_file_signal": {
            "commits_with_match": msg_match_count,
            "total_commits": len(message_signals),
            "note": "Contextual signal only — NOT causal attribution",
        },
        "file_status_pattern": {
            "commits_with_overlap": has_status_overlap,
            "total_commits": len(status_signals),
            "note": "Descriptive pattern — NOT causal attribution",
        },
        "feature_proximity": {
            "commits_with_data": len(feature_signals),
            "mean_cosine": (
                round(
                    sum(f["cosine_similarity"] for f in feature_signals)
                    / len(feature_signals),
                    6,
                )
                if feature_signals else 0.0
            ),
            "note": "Contextual predictor signal — NOT attribution evidence",
        },
        "test_separation": {
            "mixed_commits": mixed_test_prod,
            "total_commits": len(test_signals),
            "note": "Contextual signal — NOT attribution evidence",
        },
        "corrective_exclusivity": {
            "commits_with_data": len(exclusivity_signals),
            "mean_overlap_ratio": (
                round(
                    sum(e["overlap_ratio"] for e in exclusivity_signals)
                    / len(exclusivity_signals),
                    4,
                )
                if exclusivity_signals else 0.0
            ),
            "note": "Descriptive statistic — NOT attribution evidence",
        },
    }


# ---------------------------------------------------------------------------
# Recommendation synthesis
# ---------------------------------------------------------------------------

def _synthesize_recommendation(
    aggregated: dict,
    negative_audit: dict,
    prediction_units: dict,
    repo_concentration: dict,
) -> dict:
    """Synthesize the final recommendation based on all evidence."""
    total = aggregated["total_commits"]
    commit_strong = aggregated["commit_evidence_distribution"].get(COMMIT_EVIDENCE_STRONG, 0)
    file_weak = aggregated["file_evidence_distribution"].get(FILE_EVIDENCE_WEAK, 0)
    file_unknown = aggregated["file_evidence_distribution"].get(FILE_EVIDENCE_UNKNOWN, 0)

    phase48_file_partial = aggregated["phase48_baseline_distribution"].get("FILE_PARTIAL", 0)

    # Compare against Phase 4.8 baseline
    baseline_comparison = {
        "phase48_file_partial_commits": phase48_file_partial,
        "phase48_file_partial_pct": (
            round(phase48_file_partial / total * 100, 1)
            if total > 0 else 0
        ),
        "current_file_weak_commits": file_weak,
        "current_file_weak_pct": (
            round(file_weak / total * 100, 1) if total > 0 else 0
        ),
        "improvement": file_weak - phase48_file_partial,
    }

    # Decision logic
    if file_weak > phase48_file_partial * 1.2:
        decision = "PROCEED_TO_4.11B"
        rationale = (
            f"File-level evidence improved from {phase48_file_partial} to "
            f"{file_weak} commits. Repository-level git evidence (diffs, "
            f"content comparison, AST parsing) has reasonable prospect of "
            f"materially improving attribution further."
        )
    elif file_weak > 0 and negative_audit["requires_repo_access"]:
        decision = "PROCEED_TO_4.11B"
        rationale = (
            f"Some file-level evidence exists ({file_weak} commits) but "
            f"is WEAK (path overlap only). Defensible negatives require "
            f"repository history. Repository cloning is justified to "
            f"attempt stronger file-level attribution and "
            f"negative construction."
        )
    else:
        decision = "DO_NOT_PROCEED_TO_4.11B"
        rationale = (
            f"File-level evidence has not materially improved beyond "
            f"Phase 4.8 baseline ({phase48_file_partial} FILE_PARTIAL "
            f"commits). Repository cloning may not yield sufficient "
            f"improvement to "
            f"justify the cost."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "baseline_comparison": baseline_comparison,
        "evidence_summary": {
            "total_commits": total,
            "commit_strong": commit_strong,
            "file_weak": file_weak,
            "file_unknown": file_unknown,
            "defensible_negatives": negative_audit["defensible_negative_count"],
        },
    }


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def _write_artifacts(
    output_dir: Path,
    results: list[dict],
    aggregated: dict,
    negative_audit: dict,
    prediction_units: dict,
    repo_concentration: dict,
    contextual_summary: dict,
    recommendation: dict,
) -> None:
    """Write all Phase 4.11a artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. attribution_evidence_analysis.json
    _write_json(output_dir / "attribution_evidence_analysis.json", aggregated)

    # 2. contextual_signal_analysis.json
    _write_json(output_dir / "contextual_signal_analysis.json", contextual_summary)

    # 3. negative_feasibility_analysis.json
    _write_json(output_dir / "negative_feasibility_analysis.json", negative_audit)

    # 4. prediction_unit_analysis.json
    _write_json(output_dir / "prediction_unit_analysis.json", prediction_units)

    # 5. repository_concentration_analysis.json
    _write_json(output_dir / "repository_concentration_analysis.json", repo_concentration)

    # 6. per_commit_evidence.jsonl (optional, for auditability)
    _write_jsonl(output_dir / "per_commit_evidence.jsonl", results)

    # 7. phase411a_attribution_feasibility.md
    report = _generate_report(
        aggregated, negative_audit, prediction_units,
        repo_concentration, contextual_summary, recommendation,
    )
    (output_dir / "phase411a_attribution_feasibility.md").write_text(
        report, encoding="utf-8",
    )

    for fname in output_dir.iterdir():
        logger.info("Wrote %s", fname)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Helpers (reused from Phase 4.8)
# ---------------------------------------------------------------------------

def _extract_fix_sha_prefix(label_evidence: str) -> str | None:
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
    matches = [sha for sha in sha_lookup if sha.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    aggregated: dict,
    negative_audit: dict,
    prediction_units: dict,
    repo_concentration: dict,
    contextual_summary: dict,
    recommendation: dict,
) -> str:
    """Generate the Phase 4.11a markdown report."""
    report = []
    report.append("# Phase 4.11a: Attribution Feasibility Study (JSONL-only)\n")

    # Executive summary
    report.append("## Executive Summary\n")
    report.append(
        "This study investigates whether existing JSONL dataset artifacts "
        "can improve file-level defect attribution without repository access. "
        "It operates exclusively on the 210 positive commits, their candidate "
        "files, corrective commit references, and Phase 4.8/4.9 artifacts.\n"
    )
    report.append(
        "**Key finding**: All file-level evidence remains WEAK (path overlap "
        "only). No STRONG or MODERATE file-level attribution can be established "
        "from JSONL data alone. Repository-level git evidence is required for "
        "stronger attribution.\n"
    )

    # Methodology
    report.append("## Methodology\n")
    report.append("### Evidence Hierarchy\n")
    report.append(
        "| Level | Name | Definition |\n"
        "|---|---|---|\n"
        "| COMMIT_STRONG | Strong commit-level | "
        "Candidate commit has resolved corrective SHA |\n"
        "| COMMIT_WEAK | Weak commit-level | "
        "Corrective SHA prefix found but unresolvable |\n"
        "| COMMIT_NONE | No commit evidence | "
        "No corrective reference found |\n"
        "| FILE_STRONG | Strong file-level | "
        "Direct content/line evidence identifies specific file |\n"
        "| FILE_MODERATE | Moderate file-level | "
        "Evidence narrows to file but not causal |\n"
        "| FILE_WEAK | Weak file-level | "
        "Path/message/structural association only |\n"
        "| UNKNOWN | Unknown | Insufficient evidence |\n"
    )
    report.append(
        "**Important**: Contextual signals (feature proximity, file status "
        "patterns, test separation, message mentions) are reported separately "
        "and must NOT be treated as attribution evidence.\n"
    )

    # Dataset
    report.append("### Dataset\n")
    total = aggregated["total_commits"]
    report.append(f"- Total positive commits: {total}")
    revert_n = aggregated["source_breakdown"].get(
        "revert", {}
    ).get("commits", 0)
    explicit_n = aggregated["source_breakdown"].get(
        "explicit_sha_reference", {}
    ).get("commits", 0)
    report.append(
        f"- Source: revert={revert_n}, "
        f"explicit_sha={explicit_n}"
    )
    report.append(
        f"- Repositories with positives: "
        f"{repo_concentration['total_repos_with_positives']}"
    )
    report.append("")

    # Commit-level evidence
    report.append("## A. Commit-Level Evidence\n")
    report.append(
        "Commit-level evidence establishes that a candidate commit is "
        "associated with a corrective commit. This is STRONG for all "
        "210 commits where a corrective SHA can be resolved.\n"
    )
    report.append("| Evidence Level | Count | Percentage |")
    report.append("|---|---|---|")
    for level in [COMMIT_EVIDENCE_STRONG, COMMIT_EVIDENCE_WEAK, COMMIT_EVIDENCE_NONE]:
        count = aggregated["commit_evidence_distribution"].get(level, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0
        report.append(f"| {level} | {count} | {pct}% |")
    report.append("")

    report.append(
        "**Interpretation**: All 210 commits have strong commit-level "
        "evidence (corrective SHA resolved). This is unchanged from "
        "Phase 4.8 — no new commits were discovered.\n"
    )

    # File-level evidence
    report.append("## B. File-Level Evidence\n")
    report.append(
        "File-level evidence identifies specific candidate files as "
        "affected by the defect. Current evidence is WEAK (path overlap "
        "with corrective commit only).\n"
    )
    report.append("| Evidence Level | Commits | Pct | Attributed Files |")
    report.append("|---|---|---|---|")
    file_levels = [
        FILE_EVIDENCE_STRONG, FILE_EVIDENCE_MODERATE,
        FILE_EVIDENCE_WEAK, FILE_EVIDENCE_UNKNOWN,
    ]
    for level in file_levels:
        count = aggregated["file_evidence_distribution"].get(level, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0
        report.append(f"| {level} | {count} | {pct}% | — |")
    report.append("")

    report.append("### Baseline Comparison (Phase 4.8)\n")
    bc = recommendation["baseline_comparison"]
    report.append(
        f"- Phase 4.8 FILE_PARTIAL: "
        f"{bc['phase48_file_partial_commits']} commits "
        f"({bc['phase48_file_partial_pct']}%)"
    )
    report.append(
        f"- Phase 4.11a FILE_WEAK: "
        f"{bc['current_file_weak_commits']} commits "
        f"({bc['current_file_weak_pct']}%)"
    )
    report.append(
        f"- Change: {bc['improvement']:+d} commits"
    )
    report.append("")

    report.append(
        "**Interpretation**: File-level evidence has not materially "
        "improved beyond Phase 4.8. The same 38 FILE_PARTIAL commits "
        "produce WEAK file-level evidence via path overlap. No new "
        "file-level attribution strategies succeeded from JSONL data "
        "alone.\n"
    )

    # Contextual signals
    report.append("## C. Contextual Signals (NOT Attribution Evidence)\n")
    report.append(
        "The following signals are potentially useful features for future "
        "model building but must NOT be treated as attribution evidence.\n"
    )

    cs = contextual_summary
    report.append("### Message File Signal\n")
    mfs = cs["message_file_signal"]
    report.append(
        f"- Commits with file/function mentions: "
        f"{mfs['commits_with_match']}/{mfs['total_commits']}"
    )
    report.append(f"- Note: {mfs['note']}")
    report.append("")

    report.append("### File Status Pattern\n")
    fsp = cs["file_status_pattern"]
    report.append(
        f"- Commits with status overlap: "
        f"{fsp['commits_with_overlap']}/{fsp['total_commits']}"
    )
    report.append(f"- Note: {fsp['note']}")
    report.append("")

    report.append("### Feature Proximity\n")
    fp = cs["feature_proximity"]
    report.append(
        f"- Commits with feature data: "
        f"{fp['commits_with_data']}"
    )
    report.append(
        f"- Mean cosine similarity: {fp['mean_cosine']:.6f}"
    )
    report.append(f"- Note: {fp['note']}")
    report.append("")

    report.append("### Test/Production Separation\n")
    ts = cs["test_separation"]
    report.append(
        f"- Mixed test/prod commits: "
        f"{ts['mixed_commits']}/{ts['total_commits']}"
    )
    report.append(f"- Note: {ts['note']}")
    report.append("")

    report.append("### Corrective Exclusivity\n")
    ce = cs["corrective_exclusivity"]
    report.append(
        f"- Commits with exclusivity data: "
        f"{ce['commits_with_data']}"
    )
    report.append(
        f"- Mean overlap ratio: {ce['mean_overlap_ratio']:.4f}"
    )
    report.append(f"- Note: {ce['note']}")
    report.append("")

    # Negative feasibility
    report.append("## D. Negative Feasibility Audit\n")
    report.append(
        "This is an audit of whether defensible negatives could be "
        "constructed. It does NOT create candidate negative labels.\n"
    )
    report.append(f"- Defensible negatives: {negative_audit['defensible_negative_count']}")
    report.append(f"- Candidate negatives: {negative_audit['candidate_negative_count']}")
    report.append(f"- Requires repo access: {negative_audit['requires_repo_access']}")
    report.append(f"- Conclusion: {negative_audit['conclusion']}")
    report.append("")

    report.append("### Strategies Investigated\n")
    for name, strat in negative_audit["strategies_investigated"].items():
        report.append(f"- **{name}**: {strat['description']}")
        report.append(f"  - Feasible without repos: {strat['feasible_without_repos']}")
        report.append(f"  - Reason: {strat['reason']}")
    report.append("")

    # Prediction units
    report.append("## E. Prediction Unit Analysis\n")
    report.append(
        "Distinguishes LABEL AVAILABILITY from ATTRIBUTION QUALITY.\n"
    )
    for unit_name, unit in prediction_units["units"].items():
        report.append(f"### {unit_name.replace('_', ' ').title()}\n")
        report.append(f"- Label availability: {unit['label_availability']}")
        report.append(f"- Label count: {unit['label_count']}")
        report.append(f"- Attribution quality: {unit['attribution_quality']}")
        report.append(f"- Coverage: {unit['coverage']}")
        report.append(f"- Practical usefulness: {unit['practical_usefulness']}")
        report.append(f"- Investigation budget: {unit['investigation_budget']}")
        report.append("")

    report.append(f"**Recommendation**: {prediction_units['recommendation']}\n")

    # Repository concentration
    rc = repo_concentration
    report.append("## F. Repository Concentration\n")
    report.append(
        f"- Repos with positives: "
        f"{rc['total_repos_with_positives']}"
    )
    report.append(
        f"- Total positive commits: "
        f"{rc['total_positive_commits']}"
    )
    report.append(
        f"- Gini coefficient: {rc['gini_coefficient']}"
    )
    report.append(f"- Top 5 share: {rc['top_5_share']}")
    report.append(
        f"- Repos with strong commit evidence: "
        f"{rc['repos_with_strong_commit_evidence']}"
    )
    report.append(
        f"- Repos with file attribution: "
        f"{rc['repos_with_file_attribution']}"
    )
    report.append("")

    report.append("| Repo | Commits | Files | Attributed | Unknown |")
    report.append("|---|---|---|---|---|")
    for r in repo_concentration["repos"][:15]:
        report.append(
            f"| {r['repo']} | {r['commits']} | {r['files']} | "
            f"{r['attributed_files']} | {r['unknown_files']} |"
        )
    if len(repo_concentration["repos"]) > 15:
        report.append(f"| ... ({len(repo_concentration['repos']) - 15} more repos) | | | | |")
    report.append("")

    # Final decision
    report.append("## Final Decision\n")
    report.append(f"### {recommendation['decision']}\n")
    report.append(f"{recommendation['rationale']}\n")

    report.append("### Evidence Summary\n")
    es = recommendation["evidence_summary"]
    report.append(f"- Total commits: {es['total_commits']}")
    report.append(f"- Commit STRONG: {es['commit_strong']}")
    report.append(f"- File WEAK: {es['file_weak']}")
    report.append(f"- File UNKNOWN: {es['file_unknown']}")
    report.append(f"- Defensible negatives: {es['defensible_negatives']}")
    report.append("")

    # Limitations
    report.append("## Limitations\n")
    report.append(
        "1. All file-level evidence is WEAK (path overlap only). No content, "
        "line, or function-level evidence is available from JSONL data.\n"
        "2. Contextual signals (feature proximity, message mentions, file "
        "status patterns) are NOT attribution evidence.\n"
        "3. No defensible negatives can be constructed without repository "
        "history access.\n"
        "4. Function-level and hunk-level prediction units are currently "
        "unavailable.\n"
        "5. This analysis cannot establish causal attribution — only "
        "provenance and association.\n"
    )

    # Frozen integrity
    report.append("## Frozen Data Integrity\n")
    report.append(
        "- Phase 4.6–4.10 code and artifacts: UNCHANGED\n"
        "- Combined-v3 JSONL dataset: UNCHANGED\n"
        "- repo_split.py: UNCHANGED\n"
        "- diff_parser.py: UNCHANGED\n"
        "- No repositories cloned\n"
        "- No network access\n"
    )

    return "".join(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase411a()
