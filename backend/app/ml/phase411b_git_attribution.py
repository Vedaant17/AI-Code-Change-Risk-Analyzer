"""Phase 4.11b: Repository-Backed Git-History Attribution Feasibility.

READ-ONLY experiment investigating whether repository-level Git content
evidence materially improves file-level defect attribution beyond the
Phase 4.11a JSONL-only baseline.

No ML training, no dataset modification, no network access beyond
cloning public GitHub repositories.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import git

from backend.app.features.ast_diff import (
    _FUNC_NODE_TYPES,
    _collect_nodes,
    _parse_ast,
)
from backend.app.ml.phase48_attribution_feasibility import (
    _detect_path_transform_candidates,
    _extract_fix_sha_prefix,
    _resolve_full_sha,
)
from backend.app.ml.phase49_supervision_feasibility import (
    _build_sha_lookup,
    _group_by_commit,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
)
from backend.app.schemas.diff import FileDiff, Hunk
from backend.app.services.diff_parser import parse_unified_diff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("backend/data/datasets/combined-v3")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")
PHASE48_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.8")
PHASE411A_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11a")

_BUG_FIX_SHA_PATTERN = re.compile(
    r"bug-fix ([0-9a-f]{8,40})"
)
_REVERT_SHA_PATTERN = re.compile(
    r"reverted by ([0-9a-f]{8,40})"
)

COMMIT_EVIDENCE_STRONG = "COMMIT_STRONG"
COMMIT_EVIDENCE_WEAK = "COMMIT_WEAK"
COMMIT_EVIDENCE_NONE = "COMMIT_NONE"

FILE_EVIDENCE_STRONG = "FILE_STRONG"
FILE_EVIDENCE_MODERATE = "FILE_MODERATE"
FILE_EVIDENCE_WEAK = "FILE_WEAK"
FILE_EVIDENCE_UNKNOWN = "UNKNOWN"

EVIDENCE_PRECEDENCE = [
    FILE_EVIDENCE_STRONG,
    FILE_EVIDENCE_MODERATE,
    FILE_EVIDENCE_WEAK,
    FILE_EVIDENCE_UNKNOWN,
]


# ---------------------------------------------------------------------------
# Git Object Access (immutable, no checkout)
# ---------------------------------------------------------------------------


def _get_file_content_at_commit(
    repo: git.Repo,
    commit_sha: str,
    file_path: str,
) -> str | None:
    """Get file content at a specific commit using git show.

    No checkout required. Returns file content as string or None.
    """
    try:
        blob = repo.commit(commit_sha)
        content = blob.tree / file_path
        return content.data_stream.read().decode("utf-8", errors="replace")
    except (git.GitCommandError, KeyError, OSError):
        return None


def _get_commit_diff_raw(
    repo: git.Repo,
    commit_sha: str,
    file_path: str | None = None,
) -> str:
    """Get raw unified diff for a commit.

    Uses git diff {commit_sha}^ {commit_sha} [-- path].
    No checkout required.
    """
    try:
        parent = repo.commit(commit_sha + "^")
    except (git.GitCommandError, IndexError):
        try:
            return repo.git.diff("--root", commit_sha)
        except git.GitCommandError:
            return ""

    try:
        if file_path:
            return repo.git.diff(parent.hexsha, commit_sha, "--", file_path)
        return repo.git.diff(parent.hexsha, commit_sha)
    except git.GitCommandError:
        return ""


def _get_parent_sha(
    repo: git.Repo,
    commit_sha: str,
) -> str | None:
    """Get parent commit SHA using git rev-parse."""
    try:
        commit = repo.commit(commit_sha)
        if commit.parents:
            return commit.parents[0].hexsha
        return None
    except (git.GitCommandError, IndexError):
        return None


def _verify_git_objects_available(
    repo: git.Repo,
    candidate_sha: str,
    corrective_sha: str | None,
    candidate_files: list[str],
    corrective_files: list[str],
) -> dict:
    """Verify all required Git objects are available before analysis.

    Returns availability status for each required object.
    """
    availability = {
        "candidate_commit": False,
        "candidate_parent": False,
        "corrective_commit": False,
        "corrective_parent": False,
        "candidate_blobs": {},
        "corrective_blobs": {},
        "failures": [],
    }

    # Candidate commit
    try:
        repo.commit(candidate_sha)
        availability["candidate_commit"] = True
    except (git.GitCommandError, ValueError):
        availability["failures"].append(
            f"OBJECT_UNAVAILABLE: candidate {candidate_sha}"
        )

    # Candidate parent
    parent_sha = _get_parent_sha(repo, candidate_sha)
    if parent_sha:
        availability["candidate_parent"] = True
    else:
        availability["failures"].append(
            f"PARENT_UNAVAILABLE: candidate {candidate_sha}"
        )

    # Candidate file blobs
    for fp in candidate_files:
        try:
            content = _get_file_content_at_commit(repo, candidate_sha, fp)
            availability["candidate_blobs"][fp] = content is not None
            if content is None:
                availability["failures"].append(
                    f"BLOB_UNAVAILABLE: candidate {candidate_sha} {fp}"
                )
        except Exception:
            availability["candidate_blobs"][fp] = False
            availability["failures"].append(
                f"BLOB_UNAVAILABLE: candidate {candidate_sha} {fp}"
            )

    # Corrective commit (if exists)
    if corrective_sha:
        try:
            repo.commit(corrective_sha)
            availability["corrective_commit"] = True
        except (git.GitCommandError, ValueError):
            availability["failures"].append(
                f"OBJECT_UNAVAILABLE: corrective {corrective_sha}"
            )

        r_parent = _get_parent_sha(repo, corrective_sha)
        if r_parent:
            availability["corrective_parent"] = True
        else:
            availability["failures"].append(
                f"PARENT_UNAVAILABLE: corrective {corrective_sha}"
            )

        for fp in corrective_files:
            try:
                content = _get_file_content_at_commit(
                    repo, corrective_sha, fp
                )
                availability["corrective_blobs"][fp] = content is not None
                if content is None:
                    availability["failures"].append(
                        f"BLOB_UNAVAILABLE: corrective {corrective_sha} {fp}"
                    )
            except Exception:
                availability["corrective_blobs"][fp] = False
                availability["failures"].append(
                    f"BLOB_UNAVAILABLE: corrective {corrective_sha} {fp}"
                )

    return availability


# ---------------------------------------------------------------------------
# File Identity Resolution
# ---------------------------------------------------------------------------


def _resolve_file_identity_across_commits(
    file_path: str,
    candidate_diff_files: list[FileDiff],
    corrective_diff_files: list[FileDiff],
) -> dict:
    """Resolve whether candidate file F is the same file as corrective file G.

    Evidence sources (in order):
    1. Exact path match
    2. FileDiff.old_path match (Git rename metadata)
    3. Phase 4.8 path_transform_candidates
    4. None → identity not established
    """
    # 1. Exact path match
    for cfd in corrective_diff_files:
        if cfd.path == file_path:
            return {
                "identity_established": True,
                "identity_method": "exact_path",
                "candidate_file": file_path,
                "corrective_file": cfd.path,
            }

    # 2. Git rename metadata
    for cfd in corrective_diff_files:
        if cfd.old_path and cfd.old_path == file_path:
            return {
                "identity_established": True,
                "identity_method": "git_rename",
                "candidate_file": file_path,
                "corrective_file": cfd.path,
            }
        if cfd.old_path and file_path.endswith(cfd.old_path):
            return {
                "identity_established": True,
                "identity_method": "git_rename",
                "candidate_file": file_path,
                "corrective_file": cfd.path,
            }

    # 3. Path transform candidates from Phase 4.8
    candidate_paths = [f.path for f in candidate_diff_files]
    corrective_paths = [f.path for f in corrective_diff_files]
    transforms = _detect_path_transform_candidates(
        candidate_paths, corrective_paths
    )
    for t in transforms:
        if t["candidate_path"] == file_path:
            return {
                "identity_established": True,
                "identity_method": "path_transform",
                "candidate_file": file_path,
                "corrective_file": t["corrective_path"],
            }

    return {
        "identity_established": False,
        "identity_method": "none",
        "candidate_file": file_path,
        "corrective_file": None,
    }


# ---------------------------------------------------------------------------
# Content Extraction and Matching
# ---------------------------------------------------------------------------


def _normalize_line(line: str) -> str:
    """Deterministic line normalization.

    Strip trailing whitespace, normalize line endings.
    Do NOT strip leading whitespace.
    """
    return line.rstrip().replace("\r\n", "\n")


def _extract_added_lines_from_diff(
    raw_diff: str,
    file_path: str,
) -> list[str]:
    """Extract added lines from a unified diff for a specific file."""
    file_diffs = parse_unified_diff(raw_diff)
    for fd in file_diffs:
        if fd.path == file_path or (
            fd.old_path and file_path.endswith(fd.old_path)
        ):
            added = []
            for hunk in fd.hunks:
                for line in hunk.content.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        added.append(_normalize_line(line[1:]))
            return added
    return []


def _extract_deleted_lines_from_diff(
    raw_diff: str,
    file_path: str,
) -> list[str]:
    """Extract deleted lines from a unified diff for a specific file."""
    file_diffs = parse_unified_diff(raw_diff)
    for fd in file_diffs:
        if fd.path == file_path or (
            fd.old_path and file_path.endswith(fd.old_path)
        ):
            deleted = []
            for hunk in fd.hunks:
                for line in hunk.content.splitlines():
                    if line.startswith("-") and not line.startswith("---"):
                        deleted.append(_normalize_line(line[1:]))
            return deleted
    return []


def _extract_candidate_introduced_content(
    content_before: str | None,
    content_after: str | None,
    candidate_hunks: list[dict],
) -> dict:
    """Extract lines introduced by candidate commit C."""
    if content_before is None or content_after is None:
        return {
            "added_lines": [],
            "removed_lines": [],
            "added_count": 0,
            "removed_count": 0,
        }

    before_lines = set(
        _normalize_line(ln) for ln in content_before.splitlines()
    )
    after_lines = set(
        _normalize_line(ln) for ln in content_after.splitlines()
    )

    added = sorted(after_lines - before_lines)
    removed = sorted(before_lines - after_lines)

    return {
        "added_lines": added,
        "removed_lines": removed,
        "added_count": len(added),
        "removed_count": len(removed),
    }


def _extract_corrective_removed_content(
    content_before: str | None,
    content_after: str | None,
) -> dict:
    """Extract lines removed by corrective commit R."""
    if content_before is None or content_after is None:
        return {
            "removed_lines": [],
            "added_lines": [],
            "removed_count": 0,
            "added_count": 0,
        }

    before_lines = set(
        _normalize_line(ln) for ln in content_before.splitlines()
    )
    after_lines = set(
        _normalize_line(ln) for ln in content_after.splitlines()
    )

    removed = sorted(before_lines - after_lines)
    added = sorted(after_lines - before_lines)

    return {
        "removed_lines": removed,
        "added_lines": added,
        "removed_count": len(removed),
        "added_count": len(added),
    }


def _analyze_content_correspondence(
    candidate_introduced: dict,
    corrective_removed: dict,
) -> dict:
    """Deterministic content-matching between candidate and corrective.

    Reports exact and partial correspondence separately.
    Similarity ratios are descriptive evidence only.
    """
    c_added = set(candidate_introduced["added_lines"])
    r_removed = set(corrective_removed["removed_lines"])

    exact_matches = sorted(c_added & r_removed)
    unmatched_candidate = sorted(c_added - r_removed)
    unmatched_corrective = sorted(r_removed - c_added)

    # Partial matching: for each unmatched candidate line,
    # check if any corrective line has >50% token overlap
    partial_matches = []
    c_tokens = {}
    for line in unmatched_candidate:
        c_tokens[line] = set(line.split())
    r_tokens = {}
    for line in unmatched_corrective:
        r_tokens[line] = set(line.split())

    used_corrective = set()
    for c_line, c_toks in c_tokens.items():
        best_overlap = 0
        best_r_line = None
        for r_line, r_toks in r_tokens.items():
            if r_line in used_corrective:
                continue
            if not c_toks or not r_toks:
                continue
            overlap = len(c_toks & r_toks) / len(c_toks | r_toks)
            if overlap > best_overlap and overlap > 0.3:
                best_overlap = overlap
                best_r_line = r_line
        if best_r_line:
            partial_matches.append({
                "candidate_line": c_line,
                "corrective_line": best_r_line,
                "token_similarity": round(best_overlap, 4),
            })
            used_corrective.add(best_r_line)

    candidate_count = max(candidate_introduced["added_count"], 1)
    corrective_count = max(corrective_removed["removed_count"], 1)
    denom = max(candidate_count, corrective_count)

    return {
        "correspondence_exists": len(exact_matches) > 0,
        "correspondence_type": (
            "exact_removal" if exact_matches
            else "partial_overlap" if partial_matches
            else "none"
        ),
        "exact_matches": [
            {"line": m} for m in exact_matches
        ],
        "partial_matches": partial_matches,
        "unmatched_candidate_lines": unmatched_candidate,
        "unmatched_corrective_lines": unmatched_corrective,
        "candidate_lines_considered": candidate_introduced["added_count"],
        "corrective_lines_considered": corrective_removed["removed_count"],
        "exact_match_count": len(exact_matches),
        "partial_match_count": len(partial_matches),
        "exact_match_ratio": round(len(exact_matches) / denom, 4),
        "partial_match_ratio": round(
            (len(exact_matches) + len(partial_matches)) / denom, 4
        ),
        "matching_method": "content_equality",
    }


# ---------------------------------------------------------------------------
# Evidence Collectors
# ---------------------------------------------------------------------------


def _check_content_restoration(
    content_C_parent: str | None,
    content_R_current: str | None,
) -> dict:
    """Check if corrective commit restores parent-of-C content."""
    if content_C_parent is None or content_R_current is None:
        return {
            "restored": False,
            "line_diff_count": -1,
            "similarity_ratio": 0.0,
        }

    c_lines = content_C_parent.splitlines()
    r_lines = content_R_current.splitlines()
    c_norm = [_normalize_line(ln) for ln in c_lines]
    r_norm = [_normalize_line(ln) for ln in r_lines]

    restored = c_norm == r_norm
    diff_count = sum(
        1 for a, b in zip(c_norm, r_norm) if a != b
    ) + abs(len(c_norm) - len(r_norm))

    if max(len(c_norm), len(r_norm)) > 0:
        matches = sum(
            1 for a, b in zip(c_norm, r_norm) if a == b
        )
        ratio = matches / max(len(c_norm), len(r_norm))
    else:
        ratio = 1.0 if restored else 0.0

    return {
        "restored": restored,
        "line_diff_count": diff_count,
        "similarity_ratio": round(ratio, 4),
    }


def _compute_region_overlap(
    candidate_hunks: list[dict],
    corrective_hunks: list[dict],
) -> dict:
    """Compute line-range overlap between candidate and corrective changes."""
    if not candidate_hunks or not corrective_hunks:
        return {
            "overlap_exists": False,
            "candidate_line_range": (0, 0),
            "corrective_line_range": (0, 0),
            "overlap_start": None,
            "overlap_end": None,
            "overlap_line_count": 0,
        }

    c_lines = set()
    for h in candidate_hunks:
        for ln in range(h["new_start"], h["new_start"] + h["new_count"]):
            c_lines.add(ln)

    r_lines = set()
    for h in corrective_hunks:
        for ln in range(h["new_start"], h["new_start"] + h["new_count"]):
            r_lines.add(ln)

    overlap = c_lines & r_lines
    c_range = (
        min(c_lines) if c_lines else 0,
        max(c_lines) if c_lines else 0,
    )
    r_range = (
        min(r_lines) if r_lines else 0,
        max(r_lines) if r_lines else 0,
    )

    return {
        "overlap_exists": len(overlap) > 0,
        "candidate_line_range": c_range,
        "corrective_line_range": r_range,
        "overlap_start": min(overlap) if overlap else None,
        "overlap_end": max(overlap) if overlap else None,
        "overlap_line_count": len(overlap),
        "candidate_lines_in_overlap": len(overlap & c_lines),
        "corrective_lines_in_overlap": len(overlap & r_lines),
    }


def _extract_functions_by_name(
    content: str,
) -> dict[str, tuple[int, int]]:
    """Extract function name -> (start_line, end_line) using AST."""
    tree = _parse_ast(content)
    if tree is None:
        return {}

    functions = {}
    for node in _collect_nodes(tree, _FUNC_NODE_TYPES):
        name = getattr(node, "name", None)
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if name and start and end:
            functions[name] = (start, end)
    return functions


def _check_same_function_with_content(
    candidate_content_before: str | None,
    candidate_content_after: str | None,
    corrective_content_before: str | None,
    corrective_content_after: str | None,
    candidate_diff_files: list[FileDiff],
    corrective_diff_files: list[FileDiff],
    file_path: str,
) -> dict:
    """Check if candidate and corrective modify the same function,
    AND candidate-introduced content exists within that function.
    """
    result = {
        "same_function": False,
        "function_name": None,
        "candidate_content_in_function": False,
        "structural_relationship": False,
        "ast_parse_success": False,
    }

    if candidate_content_after is None or corrective_content_after is None:
        return result

    c_funcs = _extract_functions_by_name(candidate_content_after)
    r_funcs = _extract_functions_by_name(corrective_content_after)

    if not c_funcs or not r_funcs:
        return result

    result["ast_parse_success"] = True

    # Find functions modified by candidate
    c_added = _extract_candidate_introduced_content(
        candidate_content_before, candidate_content_after, []
    )
    c_added_set = set(c_added["added_lines"])

    # Find which functions contain candidate-introduced content
    c_lines_after = candidate_content_after.splitlines()
    candidate_functions_with_content = set()
    for fname, (start, end) in c_funcs.items():
        for ln_idx in range(start - 1, min(end, len(c_lines_after))):
            if _normalize_line(c_lines_after[ln_idx]) in c_added_set:
                candidate_functions_with_content.add(fname)
                break

    # Find functions modified by corrective
    r_removed = _extract_corrective_removed_content(
        corrective_content_before, corrective_content_after
    )
    r_removed_set = set(r_removed["removed_lines"])

    r_lines_after = corrective_content_after.splitlines()
    corrective_functions_modified = set()
    for fname, (start, end) in r_funcs.items():
        for ln_idx in range(start - 1, min(end, len(r_lines_after))):
            if _normalize_line(r_lines_after[ln_idx]) in r_removed_set:
                corrective_functions_modified.add(fname)
                break

    # Check overlap
    common = candidate_functions_with_content & corrective_functions_modified
    if common:
        best = min(common)
        result["same_function"] = True
        result["function_name"] = best
        result["candidate_content_in_function"] = True
        result["structural_relationship"] = True
    elif c_funcs.keys() & r_funcs.keys():
        common_names = c_funcs.keys() & r_funcs.keys()
        best = min(common_names)
        result["same_function"] = True
        result["function_name"] = best
        result["candidate_content_in_function"] = (
            best in candidate_functions_with_content
        )
        result["structural_relationship"] = (
            result["same_function"]
            and result["candidate_content_in_function"]
        )

    return result


# ---------------------------------------------------------------------------
# Evidence Classification
# ---------------------------------------------------------------------------


def _classify_file_evidence(
    identity_established: bool,
    content_restoration: dict,
    content_correspondence: dict,
    region_overlap: dict,
    function_analysis: dict,
) -> str:
    """Classify file evidence per rules from the approved plan.

    Returns exactly one of: FILE_STRONG, FILE_MODERATE, FILE_WEAK, UNKNOWN
    """
    if not identity_established:
        return FILE_EVIDENCE_UNKNOWN

    # FILE_STRONG: exact content removal or restoration
    if content_restoration["restored"]:
        return FILE_EVIDENCE_STRONG

    if content_correspondence["exact_match_count"] > 0:
        return FILE_EVIDENCE_STRONG

    # FILE_MODERATE: content correspondence + structural relationship
    has_content = content_correspondence["correspondence_exists"]
    has_structural = (
        region_overlap["overlap_exists"]
        or function_analysis["structural_relationship"]
    )

    if has_content and has_structural:
        return FILE_EVIDENCE_MODERATE

    # FILE_WEAK: path overlap or other association
    return FILE_EVIDENCE_WEAK


# ---------------------------------------------------------------------------
# Per-Commit Analysis
# ---------------------------------------------------------------------------


def _hunks_as_dicts(hunks: list[Hunk]) -> list[dict]:
    """Convert Hunk models to dicts for compatibility."""
    return [
        {
            "old_start": h.old_start,
            "old_count": h.old_count,
            "new_start": h.new_start,
            "new_count": h.new_count,
            "content": h.content,
        }
        for h in hunks
    ]


def _analyze_single_commit_git(
    commit_sha: str,
    candidate_rows: list[dict],
    repo_path: Path,
    sha_lookup: dict[str, list[dict]],
) -> dict:
    """Per-commit git-level attribution analysis."""
    repo_name = candidate_rows[0]["repo_name"]
    split = candidate_rows[0].get("split", "unknown")
    label_source = candidate_rows[0].get("label_source", "none")
    label_evidence = candidate_rows[0].get("label_evidence", "")

    candidate_files = sorted(set(r["file_path"] for r in candidate_rows))

    # Extract corrective SHA
    fix_prefix = _extract_fix_sha_prefix(label_evidence)
    corrective_sha = None
    if fix_prefix:
        corrective_sha = _resolve_full_sha(fix_prefix, sha_lookup)

    # Get corrective files from sha_lookup
    corrective_files = []
    if corrective_sha and corrective_sha in sha_lookup:
        corrective_files = sorted(set(
            r["file_path"] for r in sha_lookup[corrective_sha]
        ))

    # Determine commit-level evidence
    if corrective_sha:
        commit_evidence = COMMIT_EVIDENCE_STRONG
    elif fix_prefix:
        commit_evidence = COMMIT_EVIDENCE_WEAK
    else:
        commit_evidence = COMMIT_EVIDENCE_NONE

    # Verify Git objects
    try:
        repo = git.Repo(str(repo_path))
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return {
            "commit_sha": commit_sha,
            "repo_name": repo_name,
            "split": split,
            "label_source": label_source,
            "commit_evidence_level": commit_evidence,
            "corrective_sha": corrective_sha,
            "candidate_files": candidate_files,
            "corrective_files": corrective_files,
            "git_object_availability": {
                "candidate_commit": False,
                "candidate_parent": False,
                "corrective_commit": False,
                "corrective_parent": False,
                "candidate_blobs": {},
                "corrective_blobs": {},
                "failures": ["REPO_UNAVAILABLE"],
            },
            "file_results": [],
            "unknown_files": list(candidate_files),
            "failure_modes": ["REPO_UNAVAILABLE"],
        }

    availability = _verify_git_objects_available(
        repo, commit_sha, corrective_sha,
        candidate_files, corrective_files,
    )

    # Get diffs
    candidate_diff_raw = _get_commit_diff_raw(repo, commit_sha)
    candidate_diff_files = parse_unified_diff(candidate_diff_raw)

    corrective_diff_files = []
    corrective_diff_raw = ""
    if corrective_sha:
        corrective_diff_raw = _get_commit_diff_raw(repo, corrective_sha)
        corrective_diff_files = parse_unified_diff(corrective_diff_raw)

    # Get parent SHAs
    c_parent_sha = _get_parent_sha(repo, commit_sha)
    r_parent_sha = (
        _get_parent_sha(repo, corrective_sha) if corrective_sha else None
    )

    # Analyze each candidate file
    file_results = []
    unknown_files = []

    for fp in candidate_files:
        # Resolve file identity
        identity = _resolve_file_identity_across_commits(
            fp, candidate_diff_files, corrective_diff_files
        )

        if not identity["identity_established"]:
            unknown_files.append(fp)
            file_results.append({
                "file_path": fp,
                "file_evidence_level": FILE_EVIDENCE_UNKNOWN,
                "identity_established": False,
                "identity_method": "none",
                "evidence_types": [],
                "content_correspondence": None,
                "content_restoration": None,
                "region_overlap": None,
                "function_analysis": None,
                "git_objects_used": {},
                "rationale": "File identity not established",
            })
            continue

        corrective_file = identity["corrective_file"]

        # Check if corrective file is in corrective's file list
        if corrective_file not in corrective_files:
            unknown_files.append(fp)
            file_results.append({
                "file_path": fp,
                "file_evidence_level": FILE_EVIDENCE_UNKNOWN,
                "identity_established": True,
                "identity_method": identity["identity_method"],
                "evidence_types": [],
                "content_correspondence": None,
                "content_restoration": None,
                "region_overlap": None,
                "function_analysis": None,
                "git_objects_used": {},
                "rationale": (
                    f"File {fp} not in corrective commit "
                    f"{corrective_sha} file list"
                ),
            })
            continue

        # Get file content at relevant commits
        c_before = (
            _get_file_content_at_commit(repo, c_parent_sha, fp)
            if c_parent_sha else None
        )
        c_after = _get_file_content_at_commit(repo, commit_sha, fp)

        r_before = None
        r_after = None
        if corrective_sha and r_parent_sha:
            r_before = _get_file_content_at_commit(
                repo, r_parent_sha, corrective_file
            )
            r_after = _get_file_content_at_commit(
                repo, corrective_sha, corrective_file
            )

        # Content restoration check
        content_restoration = _check_content_restoration(c_before, r_after)

        # Content correspondence
        candidate_introduced = _extract_candidate_introduced_content(
            c_before, c_after, []
        )
        corrective_removed = _extract_corrective_removed_content(
            r_before, r_after
        )
        content_correspondence = _analyze_content_correspondence(
            candidate_introduced, corrective_removed
        )

        # Region overlap
        c_file_diffs = [
            f for f in candidate_diff_files if f.path == fp
        ]
        r_file_diffs = [
            f for f in corrective_diff_files if f.path == corrective_file
        ]

        c_hunks_dict = []
        for fd in c_file_diffs:
            c_hunks_dict.extend(_hunks_as_dicts(fd.hunks))

        r_hunks_dict = []
        for fd in r_file_diffs:
            r_hunks_dict.extend(_hunks_as_dicts(fd.hunks))

        region_overlap = _compute_region_overlap(c_hunks_dict, r_hunks_dict)

        # Function analysis
        function_analysis = _check_same_function_with_content(
            c_before, c_after, r_before, r_after,
            c_file_diffs, r_file_diffs, fp,
        )

        # Classify
        evidence_level = _classify_file_evidence(
            identity["identity_established"],
            content_restoration,
            content_correspondence,
            region_overlap,
            function_analysis,
        )

        # Determine evidence types
        evidence_types = []
        if content_restoration["restored"]:
            evidence_types.append("CONTENT_RESTORATION")
        if content_correspondence["exact_match_count"] > 0:
            evidence_types.append("EXACT_CANDIDATE_CONTENT_REMOVAL")
        if (
            content_correspondence["correspondence_exists"]
            and content_correspondence["exact_match_count"] == 0
        ):
            evidence_types.append("CANDIDATE_CONTENT_OVERLAP")
        if region_overlap["overlap_exists"]:
            evidence_types.append("SAME_REGION_OVERLAP")
        if function_analysis["structural_relationship"]:
            evidence_types.append("SAME_FUNCTION_CANDIDATE_CONTENT")
        elif function_analysis["same_function"]:
            evidence_types.append("SAME_FUNCTION_ONLY")

        file_results.append({
            "file_path": fp,
            "file_evidence_level": evidence_level,
            "identity_established": True,
            "identity_method": identity["identity_method"],
            "corrective_file": corrective_file,
            "evidence_types": evidence_types,
            "content_correspondence": content_correspondence,
            "content_restoration": content_restoration,
            "region_overlap": region_overlap,
            "function_analysis": function_analysis,
            "git_objects_used": {
                "candidate_commit": commit_sha,
                "candidate_parent": c_parent_sha,
                "corrective_commit": corrective_sha,
                "corrective_parent": r_parent_sha,
            },
            "rationale": _build_rationale(
                evidence_level, evidence_types,
                content_correspondence, region_overlap,
                function_analysis,
            ),
        })

    # Build result
    result = {
        "commit_sha": commit_sha,
        "repo_name": repo_name,
        "split": split,
        "label_source": label_source,
        "commit_evidence_level": commit_evidence,
        "corrective_sha": corrective_sha,
        "candidate_files": candidate_files,
        "corrective_files": corrective_files,
        "git_object_availability": availability,
        "file_results": file_results,
        "unknown_files": unknown_files,
        "failure_modes": availability["failures"],
    }

    return result


def _build_rationale(
    evidence_level: str,
    evidence_types: list[str],
    content_correspondence: dict,
    region_overlap: dict,
    function_analysis: dict,
) -> str:
    """Build human-readable rationale for the classification."""
    if evidence_level == FILE_EVIDENCE_STRONG:
        if "CONTENT_RESTORATION" in evidence_types:
            return "Corrective restored parent-of-C content (exact)"
        return (
            f"Exact candidate-content removal: "
            f"{content_correspondence['exact_match_count']} lines matched"
        )

    if evidence_level == FILE_EVIDENCE_MODERATE:
        parts = []
        if content_correspondence["correspondence_exists"]:
            parts.append(
                f"content correspondence "
                f"({content_correspondence['exact_match_count']} exact, "
                f"{content_correspondence['partial_match_count']} partial)"
            )
        if region_overlap["overlap_exists"]:
            parts.append(
                f"region overlap ({region_overlap['overlap_line_count']} lines)"
            )
        if function_analysis["structural_relationship"]:
            parts.append(
                f"same function ({function_analysis['function_name']})"
            )
        return " + ".join(parts)

    if evidence_level == FILE_EVIDENCE_WEAK:
        return "Path overlap or file association only"

    return "Insufficient evidence"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(results: list[dict]) -> dict:
    """Aggregate per-commit results into summary statistics."""
    total = len(results)

    commit_dist = defaultdict(int)
    file_dist = defaultdict(int)
    evidence_type_counts = defaultdict(int)
    per_repo = defaultdict(lambda: {"commits": 0, "files": 0})
    per_split = defaultdict(lambda: {"commits": 0, "files": 0})
    total_attributed_files = 0
    total_candidate_files = 0
    total_unknown_files = 0
    object_failures = 0

    for r in results:
        commit_dist[r["commit_evidence_level"]] += 1
        per_repo[r["repo_name"]]["commits"] += 1
        per_split[r["split"]]["commits"] += 1

        for fr in r["file_results"]:
            file_dist[fr["file_evidence_level"]] += 1
            total_candidate_files += 1
            if fr["file_evidence_level"] != FILE_EVIDENCE_UNKNOWN:
                total_attributed_files += 1
            for et in fr["evidence_types"]:
                evidence_type_counts[et] += 1

        total_unknown_files += len(r["unknown_files"])
        if r["failure_modes"]:
            object_failures += 1

    # Build baseline comparison
    phase48_cases = _load_phase48_cases()
    phase48_file_partial = sum(
        1 for c in phase48_cases
        if c.get("evidence_category") == "FILE_PARTIAL"
    )

    return {
        "total_commits": total,
        "commit_evidence_distribution": dict(commit_dist),
        "file_evidence_distribution": dict(file_dist),
        "evidence_type_distribution": dict(evidence_type_counts),
        "total_attributed_files": total_attributed_files,
        "total_candidate_files": total_candidate_files,
        "total_unknown_files": total_unknown_files,
        "repos_with_positives": len(per_repo),
        "per_repo_breakdown": dict(per_repo),
        "per_split_breakdown": dict(per_split),
        "object_failure_count": object_failures,
        "baseline_comparison": {
            "phase48_file_partial": phase48_file_partial,
            "phase48_file_exact": 0,
            "phase411a_file_weak": 185,
            "phase411a_file_moderate": 0,
            "phase411a_file_strong": 0,
        },
    }


# ---------------------------------------------------------------------------
# Prediction Unit Analysis
# ---------------------------------------------------------------------------


def _analyze_prediction_units(
    results: list[dict],
    supervised_rows: list[dict],
) -> dict:
    """Evaluate prediction units with label availability vs attribution quality."""
    total_commits = len(results)
    total_files = sum(
        len(r["file_results"]) for r in results
    )

    # Count by evidence level
    file_strong = sum(
        1 for r in results
        for fr in r["file_results"]
        if fr["file_evidence_level"] == FILE_EVIDENCE_STRONG
    )
    file_moderate = sum(
        1 for r in results
        for fr in r["file_results"]
        if fr["file_evidence_level"] == FILE_EVIDENCE_MODERATE
    )
    file_weak = sum(
        1 for r in results
        for fr in r["file_results"]
        if fr["file_evidence_level"] == FILE_EVIDENCE_WEAK
    )

    return {
        "commit_level": {
            "label_availability": "high",
            "label_count": total_commits,
            "attribution_quality": "STRONG commit-level",
            "coverage": f"{total_commits}/{total_commits}",
            "practical_usefulness": (
                "Limited — identifies risky commits but not "
                "which files within the commit"
            ),
        },
        "file_level": {
            "label_availability": "medium",
            "label_count": total_files,
            "attribution_quality": (
                f"STRONG={file_strong}, MODERATE={file_moderate}, "
                f"WEAK={file_weak}"
            ),
            "coverage": f"{total_files} files analyzed",
            "practical_usefulness": (
                "High potential if MODERATE/STRONG evidence exists"
            ),
        },
        "function_level": {
            "label_availability": "limited",
            "label_count": sum(
                1 for r in results
                for fr in r["file_results"]
                if (fr.get("function_analysis") or {}).get("same_function")
            ),
            "attribution_quality": (
                "Requires AST parse + content correspondence"
            ),
            "coverage": "Python files only",
            "practical_usefulness": (
                "Most actionable for developers if available"
            ),
        },
        "hunk_level": {
            "label_availability": "unavailable",
            "label_count": 0,
            "attribution_quality": "UNAVAILABLE without diff analysis",
            "coverage": "Requires patch reconstruction",
            "practical_usefulness": "Most precise but hardest to act on",
        },
    }


# ---------------------------------------------------------------------------
# Negative Feasibility
# ---------------------------------------------------------------------------


def _investigate_negative_feasibility(
    results: list[dict],
) -> dict:
    """Investigate whether defensible negatives can be constructed.

    All proposed negatives remain CANDIDATE_NEGATIVE.
    No labels without independent justification.
    """
    candidate_negatives = []

    for r in results:
        if not r.get("corrective_sha"):
            continue

        corrective_set = set(r.get("corrective_files", []))
        candidate_set = set(r.get("candidate_files", []))

        # Files in corrective diff but not in candidate list
        for cf in corrective_set:
            if cf not in candidate_set:
                candidate_negatives.append({
                    "commit_sha": r["commit_sha"],
                    "repo_name": r.get("repo_name", "unknown"),
                    "file_path": cf,
                    "strategy": "unaffected_in_corrective",
                    "status": "CANDIDATE_NEGATIVE",
                    "justification_required": True,
                })

    return {
        "candidate_negative_count": len(candidate_negatives),
        "defensible_negative_count": 0,
        "candidate_negatives": candidate_negatives[:50],
        "conclusion": (
            "No defensible negatives constructed. All proposed "
            "negatives remain CANDIDATE_NEGATIVE requiring "
            "independent justification."
        ),
        "strategies_investigated": [
            "unaffected_files_in_corrective_commit",
        ],
    }


# ---------------------------------------------------------------------------
# Recommendation Synthesis
# ---------------------------------------------------------------------------


def _synthesize_recommendation(
    aggregated: dict,
    negative_investigation: dict,
) -> dict:
    """Synthesize the final recommendation.

    Qualitative comparison against Phase 4.8 and 4.11a baselines.
    No arbitrary thresholds.
    """
    fd = aggregated["file_evidence_distribution"]
    file_strong = fd.get(FILE_EVIDENCE_STRONG, 0)
    file_moderate = fd.get(FILE_EVIDENCE_MODERATE, 0)
    file_weak = fd.get(FILE_EVIDENCE_WEAK, 0)
    file_unknown = fd.get(FILE_EVIDENCE_UNKNOWN, 0)
    total = aggregated["total_candidate_files"]

    # Evidence summary
    evidence_summary = {
        "file_strong": file_strong,
        "file_moderate": file_moderate,
        "file_weak": file_weak,
        "file_unknown": file_unknown,
        "total_files": total,
    }

    # Baselines
    bc = aggregated["baseline_comparison"]

    # Decision: qualitative comparison
    has_material_improvement = (
        file_strong > 0 or file_moderate > 0
    )

    if has_material_improvement:
        decision = "MATERIAL_IMPROVEMENT_OBSERVED"
        rationale = (
            f"Git content evidence produced {file_strong} FILE_STRONG "
            f"and {file_moderate} FILE_MODERATE cases, materially "
            f"improving beyond Phase 4.11a's 0/0 baseline."
        )
    else:
        decision = "NO_MATERIAL_IMPROVEMENT"
        rationale = (
            f"Git content evidence produced {file_strong} FILE_STRONG "
            f"and {file_moderate} FILE_MODERATE cases. All attributed "
            f"files remain FILE_WEAK ({file_weak}). No material "
            f"improvement beyond Phase 4.11a baseline."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "evidence_summary": evidence_summary,
        "baseline_comparison": bc,
        "negative_feasibility": {
            "candidate_negatives": (
                negative_investigation["candidate_negative_count"]
            ),
            "defensible_negatives": (
                negative_investigation["defensible_negative_count"]
            ),
        },
    }


# ---------------------------------------------------------------------------
# Artifact Writing
# ---------------------------------------------------------------------------


def _write_artifacts(
    output_dir: Path,
    aggregated: dict,
    results: list[dict],
    prediction_units: dict,
    negative_investigation: dict,
    recommendation: dict,
    execution_log: dict,
    required_repos: dict,
) -> None:
    """Write all Phase 4.11b artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. git_evidence_analysis.json
    (output_dir / "git_evidence_analysis.json").write_text(
        json.dumps(aggregated, indent=2, default=str),
        encoding="utf-8",
    )

    # 2. file_attribution_results.json
    all_file_results = []
    for r in results:
        for fr in r["file_results"]:
            all_file_results.append({
                "commit_sha": r["commit_sha"],
                "repo_name": r["repo_name"],
                **fr,
            })
    (output_dir / "file_attribution_results.json").write_text(
        json.dumps(all_file_results, indent=2, default=str),
        encoding="utf-8",
    )

    # 3. per_commit_git_evidence.jsonl
    with open(
        output_dir / "per_commit_git_evidence.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # 4. function_hunk_feasibility.json
    func_results = {
        "python_files_analyzed": 0,
        "function_attributable": 0,
        "same_function_count": 0,
        "function_with_content_count": 0,
        "details": [],
    }
    for r in results:
        for fr in r["file_results"]:
            fa = fr.get("function_analysis")
            if fa and fa.get("ast_parse_success"):
                func_results["python_files_analyzed"] += 1
                if fa.get("same_function"):
                    func_results["same_function_count"] += 1
                if fa.get("structural_relationship"):
                    func_results["function_with_content_count"] += 1
                    func_results["function_attributable"] += 1
                func_results["details"].append({
                    "commit_sha": r["commit_sha"],
                    "file_path": fr["file_path"],
                    **fa,
                })
    (output_dir / "function_hunk_feasibility.json").write_text(
        json.dumps(func_results, indent=2, default=str),
        encoding="utf-8",
    )

    # 5. negative_feasibility_analysis.json
    (output_dir / "negative_feasibility_analysis.json").write_text(
        json.dumps(negative_investigation, indent=2, default=str),
        encoding="utf-8",
    )

    # 6. prediction_unit_analysis.json
    (output_dir / "prediction_unit_analysis.json").write_text(
        json.dumps(prediction_units, indent=2, default=str),
        encoding="utf-8",
    )

    # 7. repository_execution_log.json
    (output_dir / "repository_execution_log.json").write_text(
        json.dumps(execution_log, indent=2, default=str),
        encoding="utf-8",
    )

    # 8. phase411b_git_attribution.md
    report = _generate_report(
        aggregated, prediction_units, negative_investigation,
        recommendation, execution_log, required_repos,
    )
    (output_dir / "phase411b_git_attribution.md").write_text(
        report, encoding="utf-8",
    )


def _generate_report(
    aggregated: dict,
    prediction_units: dict,
    negative_investigation: dict,
    recommendation: dict,
    execution_log: dict,
    required_repos: dict,
) -> str:
    """Generate the Phase 4.11b markdown report."""
    report = []
    report.append(
        "# Phase 4.11b: Repository-Backed Git-History "
        "Attribution Feasibility\n"
    )

    # Executive summary
    report.append("## Executive Summary\n")
    report.append(
        "This study investigates whether repository-level Git content "
        "evidence materially improves file-level defect attribution "
        "beyond the Phase 4.11a JSONL-only baseline.\n"
    )

    fd = aggregated["file_evidence_distribution"]
    report.append(
        f"**Result**: {fd.get(FILE_EVIDENCE_STRONG, 0)} FILE_STRONG, "
        f"{fd.get(FILE_EVIDENCE_MODERATE, 0)} FILE_MODERATE, "
        f"{fd.get(FILE_EVIDENCE_WEAK, 0)} FILE_WEAK, "
        f"{fd.get(FILE_EVIDENCE_UNKNOWN, 0)} UNKNOWN "
        f"across {aggregated['total_candidate_files']} candidate files.\n"
    )

    # Methodology
    report.append("## Methodology\n")
    report.append(
        "Git objects accessed via immutable inspection "
        "(git show, git diff). No checkout used for core analysis.\n"
    )

    # Evidence hierarchy
    report.append("### Evidence Hierarchy\n")
    report.append("| Level | Name | Requirement |")
    report.append("|---|---|---|")
    report.append(
        "| COMMIT_STRONG | Strong commit-level | "
        "Corrective SHA resolved |"
    )
    report.append(
        "| FILE_STRONG | Strong file-level | "
        "Exact content removal or restoration |"
    )
    report.append(
        "| FILE_MODERATE | Moderate file-level | "
        "Content correspondence + structural relationship |"
    )
    report.append(
        "| FILE_WEAK | Weak file-level | "
        "Path overlap or file association only |"
    )
    report.append(
        "| UNKNOWN | Unknown | "
        "File identity not established |"
    )
    report.append("")

    # Commit-level evidence
    report.append("## A. Commit-Level Evidence\n")
    cd = aggregated["commit_evidence_distribution"]
    total = aggregated["total_commits"]
    for level in [
        COMMIT_EVIDENCE_STRONG, COMMIT_EVIDENCE_WEAK, COMMIT_EVIDENCE_NONE
    ]:
        count = cd.get(level, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0
        report.append(f"- {level}: {count} ({pct}%)")
    report.append("")

    # File-level evidence
    report.append("## B. File-Level Evidence\n")
    report.append(
        "| Evidence Level | Files | Percentage |")
    report.append("|---|---|---|")
    for level in EVIDENCE_PRECEDENCE:
        count = fd.get(level, 0)
        pct = (
            round(count / aggregated["total_candidate_files"] * 100, 1)
            if aggregated["total_candidate_files"] > 0 else 0
        )
        report.append(f"| {level} | {count} | {pct}% |")
    report.append("")

    # Baseline comparison
    report.append("### Baseline Comparison\n")
    bc = aggregated["baseline_comparison"]
    report.append(
        f"- Phase 4.8: {bc['phase48_file_partial']} FILE_PARTIAL, "
        f"0 FILE_EXACT"
    )
    report.append(
        f"- Phase 4.11a: {bc['phase411a_file_weak']} FILE_WEAK, "
        f"0 FILE_MODERATE, 0 FILE_STRONG"
    )
    report.append(
        f"- Phase 4.11b: {fd.get(FILE_EVIDENCE_STRONG, 0)} FILE_STRONG, "
        f"{fd.get(FILE_EVIDENCE_MODERATE, 0)} FILE_MODERATE, "
        f"{fd.get(FILE_EVIDENCE_WEAK, 0)} FILE_WEAK"
    )
    report.append("")

    # Evidence type distribution
    report.append("### Evidence Type Distribution\n")
    etd = aggregated.get("evidence_type_distribution", {})
    for etype, count in sorted(etd.items()):
        report.append(f"- {etype}: {count}")
    report.append("")

    # Function/hunk feasibility
    report.append("## C. Function/Hunk Feasibility\n")
    fu = prediction_units.get("function_level", {})
    report.append(
        f"- Python files analyzed: {fu.get('label_count', 0)}"
    )
    report.append(
        f"- Function-attributable: "
        f"{fu.get('attribution_quality', 'N/A')}"
    )
    report.append("")

    # Negative feasibility
    report.append("## D. Negative Feasibility\n")
    report.append(
        f"- Candidate negatives: "
        f"{negative_investigation['candidate_negative_count']}"
    )
    report.append(
        f"- Defensible negatives: "
        f"{negative_investigation['defensible_negative_count']}"
    )
    report.append(
        f"- Conclusion: {negative_investigation['conclusion']}"
    )
    report.append("")

    # Prediction units
    report.append("## E. Prediction Unit Analysis\n")
    for unit_name in [
        "commit_level", "file_level", "function_level", "hunk_level"
    ]:
        unit = prediction_units.get(unit_name, {})
        report.append(f"### {unit_name.replace('_', ' ').title()}\n")
        report.append(
            f"- Label availability: {unit.get('label_availability', 'N/A')}"
        )
        report.append(
            f"- Attribution quality: "
            f"{unit.get('attribution_quality', 'N/A')}"
        )
        report.append(
            f"- Practical usefulness: "
            f"{unit.get('practical_usefulness', 'N/A')}"
        )
        report.append("")

    # Repository execution
    report.append("## F. Repository Execution\n")
    report.append(
        f"- Repos required: {len(required_repos)}"
    )
    report.append(
        f"- Repos cloned: "
        f"{execution_log.get('repos_cloned', 0)}"
    )
    report.append(
        f"- Object failures: "
        f"{execution_log.get('object_failures', 0)}"
    )
    report.append("")

    # Final decision
    report.append("## Final Decision\n")
    report.append(f"### {recommendation['decision']}\n")
    report.append(f"{recommendation['rationale']}\n")

    # Key distinction
    report.append("### Key Distinction\n")
    report.append(
        "**Git evidence exists** does not mean "
        "**Git evidence is strong enough to justify "
        "file-level supervision.**\n"
    )
    report.append(
        "Content similarity between candidate and corrective changes "
        "is reported as evidence. It is not equated with causal "
        "attribution.\n"
    )

    # Limitations
    report.append("## Limitations\n")
    report.append(
        "1. All file-level evidence is based on Git content analysis. "
        "No independent validation.\n"
    )
    report.append(
        "2. Function attribution is limited to Python files with "
        "successful AST parsing.\n"
    )
    report.append(
        "3. No defensible negatives were constructed.\n"
    )
    report.append(
        "4. Similarity ratios are descriptive, not causal.\n"
    )

    # Frozen integrity
    report.append("## Frozen Data Integrity\n")
    report.append("- Phase 4.6–4.11a code and artifacts: UNCHANGED\n")
    report.append("- Combined-v3 JSONL dataset: UNCHANGED\n")
    report.append("- No repositories were modified\n")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Repo URL Loading
# ---------------------------------------------------------------------------


def _load_repo_urls(data_dir: Path) -> dict[str, str]:
    """Load repo_name → repo_url mapping from JSONL data."""
    urls = {}
    for split_name in ["train", "validation", "test"]:
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("repo_name", "")
                url = row.get("repo_url", "")
                if name and url and name not in urls:
                    urls[name] = url
    return urls


def _load_phase48_cases() -> list[dict]:
    """Load Phase 4.8 attribution cases."""
    path = PHASE48_DIR / "file_attribution_cases.jsonl"
    if not path.exists():
        return []
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Repository Cloning
# ---------------------------------------------------------------------------


def _clone_repo(
    repo_url: str,
    dest_dir: Path,
    repo_name: str,
) -> Path | None:
    """Clone a single repository. Returns local path or None."""
    local_path = dest_dir / repo_name
    if local_path.exists():
        return local_path

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(repo_url, str(local_path))
        return local_path
    except (git.GitCommandError, OSError) as exc:
        logger.warning("Failed to clone %s: %s", repo_url, exc)
        return None


def _clone_repos_staged(
    required_repos: dict[str, str],
    repo_base_dir: Path,
    supervised_rows: list[dict],
) -> dict:
    """Clone repositories in stages.

    Stage A: repos with >= 5 positive commits
    Stage B: repos with 2-4 positive commits
    Stage C: repos with exactly 1 positive commit
    """
    # Count positive commits per repo
    pos_counts = defaultdict(int)
    for row in supervised_rows:
        if row.get("defect_label") == 1:
            pos_counts[row["repo_name"]] += 1

    stages = {"A": [], "B": [], "C": []}
    for repo_name in required_repos:
        count = pos_counts.get(repo_name, 0)
        if count >= 5:
            stages["A"].append(repo_name)
        elif count >= 2:
            stages["B"].append(repo_name)
        else:
            stages["C"].append(repo_name)

    execution_log = {
        "stages": {},
        "repos_cloned": 0,
        "repos_failed": 0,
        "object_failures": 0,
        "clone_times": {},
    }

    cloned = {}
    for stage_name in ["A", "B", "C"]:
        repos_in_stage = stages[stage_name]
        stage_log = {
            "repos": repos_in_stage,
            "repo_count": len(repos_in_stage),
            "commit_counts": {
                r: pos_counts.get(r, 0) for r in repos_in_stage
            },
            "cloned": [],
            "failed": [],
        }

        for repo_name in repos_in_stage:
            url = required_repos[repo_name]
            start = time.monotonic()
            path = _clone_repo(url, repo_base_dir, repo_name)
            elapsed = time.monotonic() - start

            if path:
                cloned[repo_name] = path
                stage_log["cloned"].append(repo_name)
                execution_log["repos_cloned"] += 1
                execution_log["clone_times"][repo_name] = round(
                    elapsed, 2
                )
            else:
                stage_log["failed"].append(repo_name)
                execution_log["repos_failed"] += 1

        execution_log["stages"][stage_name] = stage_log

    return {
        "cloned": cloned,
        "execution_log": execution_log,
        "stages": stages,
        "pos_counts": dict(pos_counts),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def run_phase411b(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    repo_base_dir: Path | None = None,
) -> dict:
    """Run the full Phase 4.11b experiment."""
    if repo_base_dir is None:
        repo_base_dir = Path("backend/data/repos")

    logger.info("Phase 4.11b: Loading data...")
    supervised_rows = _read_all_supervised_rows(data_dir)
    ambiguous_rows = _read_ambiguous_rows(data_dir)
    sha_lookup = _build_sha_lookup(supervised_rows, ambiguous_rows)

    # Determine required repos
    required_repos = _load_repo_urls(data_dir)
    pos_repos = set()
    for row in supervised_rows:
        if row.get("defect_label") == 1:
            pos_repos.add(row["repo_name"])
    required_repos = {
        k: v for k, v in required_repos.items() if k in pos_repos
    }

    logger.info("Phase 4.11b: Required repos: %d", len(required_repos))

    # Clone repos
    logger.info("Phase 4.11b: Cloning repositories...")
    clone_result = _clone_repos_staged(
        required_repos, repo_base_dir, supervised_rows
    )
    cloned = clone_result["cloned"]
    execution_log = clone_result["execution_log"]

    # Group positive commits
    groups = _group_by_commit(supervised_rows)
    pos_commits = {
        k: v for k, v in groups.items()
        if any(r["defect_label"] == 1 for r in v)
    }

    logger.info(
        "Phase 4.11b: Analyzing %d positive commits...",
        len(pos_commits),
    )

    # Analyze each positive commit
    results = []
    for (repo_name, commit_sha), rows in pos_commits.items():
        repo_path = cloned.get(repo_name)
        if repo_path is None:
            results.append({
                "commit_sha": commit_sha,
                "repo_name": repo_name,
                "split": rows[0].get("split", "unknown"),
                "label_source": rows[0].get("label_source", "none"),
                "commit_evidence_level": COMMIT_EVIDENCE_NONE,
                "corrective_sha": None,
                "candidate_files": sorted(
                    set(r["file_path"] for r in rows)
                ),
                "corrective_files": [],
                "git_object_availability": {
                    "candidate_commit": False,
                    "candidate_parent": False,
                    "corrective_commit": False,
                    "corrective_parent": False,
                    "candidate_blobs": {},
                    "corrective_blobs": {},
                    "failures": ["REPO_NOT_CLONED"],
                },
                "file_results": [],
                "unknown_files": sorted(
                    set(r["file_path"] for r in rows)
                ),
                "failure_modes": ["REPO_NOT_CLONED"],
            })
            continue

        result = _analyze_single_commit_git(
            commit_sha, rows, repo_path, sha_lookup,
        )
        results.append(result)

    # Aggregate
    logger.info("Phase 4.11b: Aggregating results...")
    aggregated = _aggregate_results(results)

    # Prediction units
    prediction_units = _analyze_prediction_units(results, supervised_rows)

    # Negative feasibility
    negative_investigation = _investigate_negative_feasibility(results)

    # Recommendation
    recommendation = _synthesize_recommendation(
        aggregated, negative_investigation,
    )

    # Write artifacts
    logger.info("Phase 4.11b: Writing artifacts...")
    _write_artifacts(
        output_dir, aggregated, results, prediction_units,
        negative_investigation, recommendation, execution_log,
        required_repos,
    )

    logger.info(
        "Phase 4.11b: Complete. Decision: %s",
        recommendation["decision"],
    )

    return {
        "aggregated": aggregated,
        "prediction_units": prediction_units,
        "negative_investigation": negative_investigation,
        "recommendation": recommendation,
        "execution_log": execution_log,
    }
