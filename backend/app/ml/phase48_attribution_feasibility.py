"""Phase 4.8: File-Level Attribution Feasibility Study.

READ-ONLY analysis of whether the frozen combined-v3 dataset supports
defensible file-level defect attribution. Inspects path-level evidence
between candidate commits and their corrective (fix/revert) commits
using only data available in the JSONL files.

Key methodological constraints:
- Path overlap is NOT equivalent to file-level defect attribution.
- COMMIT_ONLY cases have empty attributed_files; intersection stored
  as path_overlap_files only.
- FILE_PARTIAL cases store intersected files as path_associated_files
  (not proven defective files). Remaining files are UNKNOWN.
- No negative labels are manufactured.
- All counts are derived programmatically from the frozen data.
- Corrective/revert commit data is retrospective label evidence only
  and MUST NOT enter candidate feature vectors.
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
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.8")

EVIDENCE_CATEGORIES = frozenset({
    "FILE_PARTIAL",
    "COMMIT_ONLY",
    "NO_ATTRIBUTION",
    "UNUSABLE",
})

_BUG_FIX_SHA_PATTERN = re.compile(r"bug-fix ([0-9a-f]{8,40})")
_REVERT_SHA_PATTERN = re.compile(r"reverted by ([0-9a-f]{8,40})")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_all_supervised_rows(data_dir: Path) -> list[dict]:
    """Read all rows from train/validation/test JSONL files."""
    all_rows: list[dict] = []
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
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_no} in {path}: {e}"
                    ) from e
                all_rows.append(obj)
    logger.info("Loaded %d supervised rows from %s", len(all_rows), data_dir)
    return all_rows


def _read_ambiguous_rows(data_dir: Path) -> list[dict]:
    """Read all rows from ambiguous.jsonl."""
    path = data_dir / "ambiguous.jsonl"
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
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {path}: {e}"
                ) from e
            rows.append(obj)
    logger.info("Loaded %d ambiguous rows from %s", len(rows), path)
    return rows


def _build_sha_lookup(
    supervised_rows: list[dict],
    ambiguous_rows: list[dict],
) -> dict[str, list[dict]]:
    """Build SHA -> list[file_rows] lookup from all available data.

    Includes both supervised and ambiguous rows so that fix/revert
    commits referenced by label_evidence can be located.
    """
    lookup: dict[str, list[dict]] = defaultdict(list)
    for r in supervised_rows:
        lookup[r["commit_sha"]].append(r)
    for r in ambiguous_rows:
        lookup[r["commit_sha"]].append(r)
    logger.info("Built SHA lookup with %d unique SHAs", len(lookup))
    return dict(lookup)


def _group_by_commit(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (repo_name, commit_sha)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["repo_name"], row["commit_sha"])
        groups[key].append(row)
    return dict(groups)


# ---------------------------------------------------------------------------
# Attribution extraction
# ---------------------------------------------------------------------------

def _extract_fix_sha_prefix(label_evidence: str) -> str | None:
    """Extract the corrective commit SHA prefix from label_evidence.

    Returns the 12-character hex prefix of the bug-fix or revert commit,
    or None if not found.
    """
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
    """Resolve a 12-char SHA prefix to the full SHA in the lookup.

    Returns the full SHA if exactly one match is found, None otherwise.
    """
    matches = [sha for sha in sha_lookup if sha.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------

def classify_commit_evidence(
    candidate_rows: list[dict],
    sha_lookup: dict[str, list[dict]],
) -> dict:
    """Classify the evidence category for a single positive commit.

    Returns a dict with:
        repo_name: str
        commit_sha: str
        label_source: str
        label_evidence: str
        commit_message: str
        split: str
        candidate_files: list[str]  (sorted, deduplicated)
        candidate_file_count: int
        corrective_sha: str | None  (full SHA if resolved)
        corrective_sha_prefix: str | None
        corrective_files: list[str]  (sorted, deduplicated)
        corrective_file_count: int
        evidence_category: str
        path_overlap_files: list[str]  (intersection, for COMMIT_ONLY)
        path_associated_files: list[str]  (intersection, for FILE_PARTIAL)
        unknown_files: list[str]  (candidate files without intersection)
        path_transform_candidates: list[dict]  (potential renames)
    """
    row0 = candidate_rows[0]
    repo_name = row0["repo_name"]
    commit_sha = row0["commit_sha"]
    label_source = row0.get("label_source", "none")
    label_evidence = row0.get("label_evidence", "")
    commit_message = row0.get("commit_message", "")
    split = row0.get("split", "unknown")

    candidate_files = sorted(set(r["file_path"] for r in candidate_rows))

    result = {
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "label_source": label_source,
        "label_evidence": label_evidence,
        "commit_message": commit_message[:200],
        "split": split,
        "candidate_files": candidate_files,
        "candidate_file_count": len(candidate_files),
        "corrective_sha": None,
        "corrective_sha_prefix": None,
        "corrective_files": [],
        "corrective_file_count": 0,
        "evidence_category": "NO_ATTRIBUTION",
        "attributed_files": [],
        "path_overlap_files": [],
        "path_associated_files": [],
        "unknown_files": list(candidate_files),
        "path_transform_candidates": [],
    }

    # Extract corrective SHA prefix
    fix_prefix = _extract_fix_sha_prefix(label_evidence)
    if fix_prefix is None:
        result["evidence_category"] = "NO_ATTRIBUTION"
        return result

    result["corrective_sha_prefix"] = fix_prefix

    # Resolve full SHA
    full_sha = _resolve_full_sha(fix_prefix, sha_lookup)
    if full_sha is None:
        result["evidence_category"] = "NO_ATTRIBUTION"
        return result

    result["corrective_sha"] = full_sha

    # Get corrective commit files
    fix_rows = sha_lookup.get(full_sha, [])
    if not fix_rows:
        result["evidence_category"] = "NO_ATTRIBUTION"
        return result

    corrective_files = sorted(set(r["file_path"] for r in fix_rows))
    result["corrective_files"] = corrective_files
    result["corrective_file_count"] = len(corrective_files)

    # Compute path intersection
    candidate_set = set(candidate_files)
    corrective_set = set(corrective_files)
    intersection = sorted(candidate_set & corrective_set)
    unknown = sorted(candidate_set - corrective_set)

    # Detect path transform candidates (same basename, different prefix)
    path_transforms = _detect_path_transform_candidates(
        candidate_files, corrective_files
    )
    result["path_transform_candidates"] = path_transforms

    # Classify evidence category
    if len(intersection) == 0:
        # No path overlap
        result["evidence_category"] = "COMMIT_ONLY"
        result["attributed_files"] = []
        result["path_overlap_files"] = []
        result["unknown_files"] = unknown
    elif len(intersection) == len(candidate_set):
        # Full overlap: all candidate files appear in corrective commit
        result["evidence_category"] = "COMMIT_ONLY"
        result["attributed_files"] = []
        result["path_overlap_files"] = intersection
        result["unknown_files"] = []
    else:
        # Partial overlap: corrective commit touches subset of candidate files
        result["evidence_category"] = "FILE_PARTIAL"
        result["attributed_files"] = list(intersection)
        result["path_associated_files"] = intersection
        result["unknown_files"] = unknown

    return result


def _detect_path_transform_candidates(
    candidate_files: list[str],
    corrective_files: list[str],
) -> list[dict]:
    """Detect potential path transforms (same basename, different prefix).

    These are NOT automatically matched. They are reported for manual
    investigation as potential renames or layout changes.
    """
    candidates: list[dict] = []
    cand_basenames: dict[str, str] = {}
    for f in candidate_files:
        basename = Path(f).name
        cand_basenames[basename] = f

    for f in corrective_files:
        basename = Path(f).name
        if basename in cand_basenames:
            cand_path = cand_basenames[basename]
            if cand_path != f:
                candidates.append({
                    "candidate_path": cand_path,
                    "corrective_path": f,
                    "basename": basename,
                })

    return candidates


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def analyze_all_positive_commits(
    supervised_rows: list[dict],
    sha_lookup: dict[str, list[dict]],
) -> list[dict]:
    """Analyze all positive commits and classify their evidence.

    Returns a list of per-commit classification dicts.
    """
    commit_groups = _group_by_commit(supervised_rows)

    positive_commits = {
        k: rows for k, rows in commit_groups.items()
        if any(r["defect_label"] == 1 for r in rows)
    }

    logger.info(
        "Analyzing %d positive commits from %d total commits",
        len(positive_commits), len(commit_groups),
    )

    results: list[dict] = []
    for _key, rows in positive_commits.items():
        classification = classify_commit_evidence(rows, sha_lookup)
        results.append(classification)

    results.sort(key=lambda c: (c["repo_name"], c["commit_sha"]))
    return results


def compute_evidence_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics from evidence classification results."""
    total = len(results)

    by_category: dict[str, int] = defaultdict(int)
    for r in results:
        by_category[r["evidence_category"]] += 1

    by_source: dict[str, dict] = defaultdict(lambda: {"count": 0})
    for r in results:
        src = r["label_source"]
        by_source[src]["count"] += 1
        by_category_key = r["evidence_category"]
        by_source[src][by_category_key] = (
            by_source[src].get(by_category_key, 0) + 1
        )

    # File-level yield
    total_candidate_files = sum(r["candidate_file_count"] for r in results)
    total_path_overlap = sum(len(r["path_overlap_files"]) for r in results)
    total_path_associated = sum(
        len(r["path_associated_files"]) for r in results
    )
    total_unknown = sum(len(r["unknown_files"]) for r in results)
    total_corrective_files = sum(r["corrective_file_count"] for r in results)

    # Path transform candidates
    total_path_transforms = sum(
        len(r["path_transform_candidates"]) for r in results
    )

    # Per-category file counts
    file_partial_results = [
        r for r in results if r["evidence_category"] == "FILE_PARTIAL"
    ]
    commit_only_results = [
        r for r in results if r["evidence_category"] == "COMMIT_ONLY"
    ]
    no_attr_results = [
        r for r in results if r["evidence_category"] == "NO_ATTRIBUTION"
    ]
    unusable_results = [
        r for r in results if r["evidence_category"] == "UNUSABLE"
    ]

    fp_positive_files = sum(
        len(r["path_associated_files"]) for r in file_partial_results
    )
    fp_unknown_files = sum(
        len(r["unknown_files"]) for r in file_partial_results
    )
    co_overlap_files = sum(
        len(r["path_overlap_files"]) for r in commit_only_results
    )
    co_unknown_files = sum(
        len(r["unknown_files"]) for r in commit_only_results
    )

    return {
        "total_positive_commits": total,
        "by_category": dict(by_category),
        "by_source": dict(by_source),
        "file_yield": {
            "total_candidate_files": total_candidate_files,
            "total_path_overlap_files": total_path_overlap,
            "total_path_associated_files": total_path_associated,
            "total_unknown_files": total_unknown,
            "total_corrective_files": total_corrective_files,
            "total_path_transform_candidates": total_path_transforms,
            "path_overlap_ratio": (
                total_path_overlap / total_candidate_files
                if total_candidate_files else 0.0
            ),
        },
        "file_partial_details": {
            "commits": len(file_partial_results),
            "positive_file_rows": fp_positive_files,
            "unknown_file_rows": fp_unknown_files,
            "path_associated_per_commit": _distribution(
                [len(r["path_associated_files"]) for r in file_partial_results]
            ),
        },
        "commit_only_details": {
            "commits": len(commit_only_results),
            "path_overlap_file_rows": co_overlap_files,
            "unknown_file_rows": co_unknown_files,
            "full_overlap_commits": sum(
                1 for r in commit_only_results
                if len(r["path_overlap_files"]) == r["candidate_file_count"]
                and r["candidate_file_count"] > 0
            ),
            "zero_overlap_commits": sum(
                1 for r in commit_only_results
                if len(r["path_overlap_files"]) == 0
            ),
        },
        "no_attribution_details": {
            "commits": len(no_attr_results),
        },
        "unusable_details": {
            "commits": len(unusable_results),
        },
    }


def compute_split_analysis(results: list[dict]) -> dict:
    """Compute per-split evidence statistics."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r)

    split_stats: dict[str, dict] = {}
    for split_name in ("train", "validation", "test"):
        split_results = by_split.get(split_name, [])
        stats = compute_evidence_statistics(split_results)
        split_stats[split_name] = stats

    return split_stats


def compute_repository_analysis(results: list[dict]) -> dict:
    """Compute per-repository evidence statistics."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_repo[r["repo_name"]].append(r)

    repos: list[dict] = []
    for repo_name in sorted(by_repo):
        repo_results = by_repo[repo_name]
        stats = compute_evidence_statistics(repo_results)
        repos.append({
            "repo_name": repo_name,
            **stats,
        })

    repos.sort(key=lambda r: -r["total_positive_commits"])
    return {"repos": repos}


def _distribution(vals: list[int]) -> dict:
    """Compute distribution statistics for a list of integers."""
    if not vals:
        return {
            "count": 0, "mean": 0.0, "median": 0.0,
            "min": 0, "max": 0, "std": 0.0,
        }
    a = np.array(vals, dtype=np.float64)
    return {
        "count": len(vals),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": int(a.min()),
        "max": int(a.max()),
        "std": float(a.std()),
    }


# ---------------------------------------------------------------------------
# Negative label investigation
# ---------------------------------------------------------------------------

def investigate_negative_labels(results: list[dict]) -> dict:
    """Investigate whether defensible negative labels can be constructed.

    Analyzes whether files absent from the corrective commit can be
    defensibly labeled negative, or whether they must remain UNKNOWN.
    """
    total_candidate_files = sum(r["candidate_file_count"] for r in results)
    total_unknown = sum(len(r["unknown_files"]) for r in results)
    total_path_associated = sum(
        len(r["path_associated_files"]) for r in results
    )

    # For FILE_PARTIAL cases, unknown files are candidate files not in
    # the corrective commit. Can these be negative?
    fp_results = [
        r for r in results if r["evidence_category"] == "FILE_PARTIAL"
    ]
    fp_unknown_count = sum(len(r["unknown_files"]) for r in fp_results)

    # Reasons unknown files cannot be negative:
    # 1. Corrective commit may be partial (fixes only one symptom)
    # 2. File may have been renamed between candidate and corrective
    # 3. Defect may involve interaction between files
    # 4. Corrective commit may address different manifestation

    # For COMMIT_ONLY with zero overlap: all candidate files are unknown
    co_zero = [
        r for r in results
        if r["evidence_category"] == "COMMIT_ONLY"
        and len(r["path_overlap_files"]) == 0
    ]
    co_zero_files = sum(r["candidate_file_count"] for r in co_zero)

    return {
        "total_candidate_files": total_candidate_files,
        "total_path_associated_files": total_path_associated,
        "total_unknown_files": total_unknown,
        "defensible_negative_files": 0,
        "negative_investigation": {
            "file_partial_unknown_files": fp_unknown_count,
            "commit_only_zero_overlap_files": co_zero_files,
            "reasons_not_negative": [
                "Corrective commit may be partial",
                "File may have been renamed between candidate and corrective",
                "Defect may involve interaction between files",
                "Corrective commit may address different manifestation",
                "No content-level analysis possible without repository checkout",
            ],
        },
        "conclusion": (
            "No defensible negative labels can be constructed from "
            "path-level evidence alone. All non-intersected candidate "
            "files remain UNKNOWN."
        ),
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def derive_recommendation(
    evidence_stats: dict,
    negative_investigation: dict,
) -> dict:
    """Derive the recommendation based on measured evidence."""
    cats = evidence_stats["by_category"]
    total = evidence_stats["total_positive_commits"]
    fp_count = cats.get("FILE_PARTIAL", 0)
    co_count = cats.get("COMMIT_ONLY", 0)
    na_count = cats.get("NO_ATTRIBUTION", 0)
    uu_count = cats.get("UNUSABLE", 0)

    fp_positive_files = evidence_stats["file_partial_details"]["positive_file_rows"]
    fp_unknown_files = evidence_stats["file_partial_details"]["unknown_file_rows"]
    total_candidate = evidence_stats["file_yield"]["total_candidate_files"]
    total_unknown = evidence_stats["file_yield"]["total_unknown_files"]

    fp_pct = fp_count / total * 100 if total else 0.0
    unknown_rate = (
        total_unknown / total_candidate if total_candidate else 0.0
    )

    # Decision logic
    if fp_count == 0:
        recommendation = "FILE_LEVEL_ATTRIBUTION_NOT_FEASIBLE"
        reasoning = (
            f"Zero FILE_PARTIAL cases found among {total} positive commits. "
            f"No file-level narrowing is possible from path-level evidence."
        )
    elif unknown_rate > 0.8:
        recommendation = "PATH_LEVEL_EVIDENCE_ONLY"
        reasoning = (
            f"{fp_count} FILE_PARTIAL cases exist ({fp_pct:.1f}%), but "
            f"the unknown rate is {unknown_rate:.1%}, meaning most "
            f"candidate files cannot be attributed. Path-level evidence "
            f"exists but cannot be upgraded to file-level attribution "
            f"for the majority of files."
        )
    elif negative_investigation["defensible_negative_files"] == 0:
        recommendation = "PATH_LEVEL_EVIDENCE_ONLY"
        reasoning = (
            f"{fp_count} FILE_PARTIAL cases exist ({fp_pct:.1f}%) with "
            f"{fp_positive_files} path-associated file rows and "
            f"{fp_unknown_files} unknown file rows. However, no defensible "
            f"negative labels can be constructed. The supervision is "
            f"positive/unknown (positive-unlabeled), not binary "
            f"positive/negative. This requires a PU learning approach "
            f"rather than standard binary classification."
        )
    else:
        recommendation = "FILE_LEVEL_ATTRIBUTION_FEASIBLE_WITH_RESTRICTIONS"
        reasoning = (
            f"{fp_count} FILE_PARTIAL cases provide file-level narrowing "
            f"with {fp_positive_files} attributed files."
        )

    return {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "factors": {
            "file_partial_commits": fp_count,
            "file_partial_pct": fp_pct,
            "commit_only_commits": co_count,
            "no_attribution_commits": na_count,
            "unusable_commits": uu_count,
            "total_positive_commits": total,
            "path_associated_file_rows": fp_positive_files,
            "unknown_file_rows": fp_unknown_files,
            "total_candidate_file_rows": total_candidate,
            "unknown_rate": unknown_rate,
            "defensible_negatives": (
                negative_investigation["defensible_negative_files"]
            ),
            "supervision_type": "positive/unknown (PU)",
        },
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

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
            line = json.dumps(rec, default=_json_default, sort_keys=True)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_report(
    evidence_stats: dict,
    split_analysis: dict,
    repo_analysis: dict,
    negative_investigation: dict,
    recommendation: dict,
    output_dir: Path,
) -> None:
    """Write the Phase 4.8 scientific report."""
    lines: list[str] = [
        "# Phase 4.8: File-Level Attribution Feasibility Study",
        "",
        "## A. OBSERVED FACTS",
        "",
        "### Dataset Overview",
        "",
        f"- Total positive commits analyzed: "
        f"{evidence_stats['total_positive_commits']}",
        f"- Attribution sources: "
        f"{json.dumps(evidence_stats['by_source'], indent=2)}",
        "",
        "### Evidence Category Distribution",
        "",
        "| Category | Count | % |",
        "|----------|-------|---|",
    ]

    total = evidence_stats["total_positive_commits"]
    for cat in ("FILE_PARTIAL", "COMMIT_ONLY", "NO_ATTRIBUTION", "UNUSABLE"):
        count = evidence_stats["by_category"].get(cat, 0)
        pct = count / total * 100 if total else 0.0
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "### File-Level Yield",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        (f"| Total candidate files | "
         f"{evidence_stats['file_yield']['total_candidate_files']} |"),
        (f"| Path-overlap files (COMMIT_ONLY) | "
         f"{evidence_stats['file_yield']['total_path_overlap_files']} |"),
        (f"| Path-associated files (FILE_PARTIAL) | "
         f"{evidence_stats['file_yield']['total_path_associated_files']} |"),
        (f"| Unknown files | "
         f"{evidence_stats['file_yield']['total_unknown_files']} |"),
        (f"| Path transform candidates | "
         f"{evidence_stats['file_yield']['total_path_transform_candidates']} |"),
        "",
        "---",
        "",
        "## B. IMPLEMENTATION FACTS",
        "",
        "### What the Analysis Does",
        "",
        "1. Reads all rows from frozen train/validation/test JSONL files.",
        "2. Reads all rows from ambiguous.jsonl for SHA resolution.",
        "3. For each positive commit, extracts the corrective SHA from "
        "`label_evidence` text.",
        "4. Resolves the full SHA across all JSONL data.",
        "5. Extracts file paths from the corrective commit's rows.",
        "6. Computes exact path intersection with candidate files.",
        "7. Classifies evidence category.",
        "",
        "### What the Analysis Does NOT Do",
        "",
        "- Does NOT access repository checkouts or raw diffs.",
        "- Does NOT perform line-level or content-level analysis.",
        "- Does NOT normalize or transform paths.",
        "- Does NOT modify any frozen files.",
        "- Does NOT train ML models.",
        "- Does NOT use corrective commit data as features.",
        "",
        "---",
        "",
        "## C. INFERENCES",
        "",
        "### Evidence Category: FILE_PARTIAL",
        "",
        (f"- {evidence_stats['file_partial_details']['commits']} commits "
         f"where the corrective change touches a strict subset of "
         f"candidate files."),
        (f"- {evidence_stats['file_partial_details']['positive_file_rows']} "
         f"path-associated file rows."),
        (f"- {evidence_stats['file_partial_details']['unknown_file_rows']} "
         f"unknown file rows."),
        "",
        "These represent the primary potentially useful file-level signal. "
        "The intersected files have path-level association with the "
        "corrective change, but causality is not uniquely established.",
        "",
        "### Evidence Category: COMMIT_ONLY",
        "",
        (f"- {evidence_stats['commit_only_details']['commits']} commits "
         f"where path intersection does not narrow the candidate set."),
        (f"  - {evidence_stats['commit_only_details']['full_overlap_commits']} "
         f"with full overlap (all candidate files in corrective)."),
        (f"  - {evidence_stats['commit_only_details']['zero_overlap_commits']} "
         f"with zero overlap (no candidate files in corrective)."),
        (f"- {evidence_stats['commit_only_details']['path_overlap_file_rows']} "
         f"path-overlap files (stored separately, NOT labeled positive)."),
        (f"- {evidence_stats['commit_only_details']['unknown_file_rows']} "
         f"unknown files."),
        "",
        "Path overlap in COMMIT_ONLY cases establishes commit-level "
        "association but does NOT provide file-level narrowing.",
        "",
        "### Evidence Category: NO_ATTRIBUTION",
        "",
        (f"- {evidence_stats['no_attribution_details']['commits']} commits "
         f"where no corrective commit was found or resolved."),
        "",
        "### Evidence Category: UNUSABLE",
        "",
        (f"- {evidence_stats['unusable_details']['commits']} commits "
         f"with inconsistent or unavailable data."),
        "",
        "---",
        "",
        "## D. LIMITATIONS",
        "",
        "1. **No content-level analysis**: The frozen JSONL contains only "
        "file paths, not diffs or patch contents. Line-level attribution "
        "is impossible without repository checkouts.",
        "2. **Path overlap ≠ defect attribution**: Sharing file paths "
        "between candidate and corrective commits establishes association, "
        "not causation.",
        "3. **No defensible negatives**: Files absent from the corrective "
        "commit cannot be labeled negative (partial fixes, renames, "
        "interaction effects).",
        "4. **Path transform ambiguity**: Some 'no overlap' cases may be "
        "renames or layout changes, but these are not automatically "
        "resolved.",
        "5. **Positive-unlabeled supervision**: The resulting dataset "
        "has POSITIVE and UNKNOWN labels, not binary POSITIVE/NEGATIVE.",
        "",
        "---",
        "",
        "## E. RECOMMENDATION",
        "",
        f"**{recommendation['recommendation']}**",
        "",
        f"{recommendation['reasoning']}",
        "",
        "### Decision Factors",
        "",
    ])

    for key, val in recommendation["factors"].items():
        if isinstance(val, float):
            lines.append(f"- **{key}**: {val:.4f}")
        else:
            lines.append(f"- **{key}**: {val}")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Split Breakdown",
        "",
        "| Split | Commits | FILE_PARTIAL | COMMIT_ONLY "
        "| NO_ATTRIBUTION | UNUSABLE | Pos Files | Unknown |",
        "|-------|---------|-------------|-----------"
        "|---------------|----------|-----------|---------|",
    ])

    for sn in ("train", "validation", "test"):
        s = split_analysis.get(sn, {})
        cats = s.get("by_category", {})
        fp = s.get("file_partial_details", {})
        lines.append(
            f"| {sn} | {s.get('total_positive_commits', 0)} | "
            f"{cats.get('FILE_PARTIAL', 0)} | "
            f"{cats.get('COMMIT_ONLY', 0)} | "
            f"{cats.get('NO_ATTRIBUTION', 0)} | "
            f"{cats.get('UNUSABLE', 0)} | "
            f"{fp.get('positive_file_rows', 0)} | "
            f"{fp.get('unknown_file_rows', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Per-Repository Summary",
        "",
        "| Repo | Commits | FILE_PARTIAL | COMMIT_ONLY | Pos Files |",
        "|------|---------|-------------|-------------|-----------|",
    ])

    for r in repo_analysis.get("repos", []):
        cats = r.get("by_category", {})
        fp = r.get("file_partial_details", {})
        lines.append(
            f"| {r['repo_name']} | {r['total_positive_commits']} | "
            f"{cats.get('FILE_PARTIAL', 0)} | "
            f"{cats.get('COMMIT_ONLY', 0)} | "
            f"{fp.get('positive_file_rows', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Frozen Data Integrity",
        "",
        "- No frozen dataset files (JSONL) were modified.",
        "- No frozen labeling logic was modified.",
        "- No frozen builder logic was modified.",
        "- No Phase 4/4.5/4.6/4.7 artifacts were modified.",
        "- No dataset regeneration was performed.",
        "- All analyses read frozen JSONL files and ambiguous.jsonl.",
        "",
    ])

    report_path = output_dir / "phase48_attribution_feasibility.md"
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
    """Run the complete Phase 4.8 attribution feasibility study.

    Returns a dict with all analysis results for programmatic use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 4.8: File-Level Attribution Feasibility Study")
    logger.info("Reading frozen dataset from %s", data_dir)

    supervised_rows = _read_all_supervised_rows(data_dir)
    ambiguous_rows = _read_ambiguous_rows(data_dir)
    sha_lookup = _build_sha_lookup(supervised_rows, ambiguous_rows)

    logger.info("Analyzing positive commits...")
    results = analyze_all_positive_commits(supervised_rows, sha_lookup)

    logger.info("Computing evidence statistics...")
    evidence_stats = compute_evidence_statistics(results)

    logger.info("Computing split analysis...")
    split_analysis = compute_split_analysis(results)

    logger.info("Computing repository analysis...")
    repo_analysis = compute_repository_analysis(results)

    logger.info("Investigating negative labels...")
    negative_inv = investigate_negative_labels(results)

    logger.info("Deriving recommendation...")
    recommendation = derive_recommendation(evidence_stats, negative_inv)

    # Write artifacts
    logger.info("Writing artifacts...")
    _write_json(output_dir / "attribution_feasibility.json", {
        "evidence_statistics": evidence_stats,
        "negative_label_investigation": negative_inv,
        "recommendation": recommendation,
    })

    _write_jsonl(
        output_dir / "file_attribution_cases.jsonl",
        results,
    )

    _write_json(
        output_dir / "repository_analysis.json",
        repo_analysis,
    )

    _write_json(
        output_dir / "split_analysis.json",
        split_analysis,
    )

    logger.info("Generating report...")
    _write_report(
        evidence_stats,
        split_analysis,
        repo_analysis,
        negative_inv,
        recommendation,
        output_dir,
    )

    logger.info("Phase 4.8 complete. Artifacts: %s", output_dir)

    return {
        "evidence_stats": evidence_stats,
        "split_analysis": split_analysis,
        "repo_analysis": repo_analysis,
        "negative_investigation": negative_inv,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_feasibility_study()
