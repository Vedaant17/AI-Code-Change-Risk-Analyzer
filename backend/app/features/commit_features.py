"""Commit-level feature extraction (Phase 2).

Aggregates per-file features and diff-level metadata into a single
deterministic ``CommitFeatures`` vector.
"""

from __future__ import annotations

import math
from collections import Counter

from backend.app.features.file_features import extract_file_features
from backend.app.features.language import LANGUAGE_LIST, detect_language, is_test_file
from backend.app.features.schemas import CommitFeatures, FileFeatures
from backend.app.schemas.diff import CommitInfo


def _change_entropy(total_lines_changed_per_file: list[int]) -> float:
    """Shannon entropy of the per-file line-change distribution.

    Measures whether changes are concentrated in one file or spread
    across many.  Returns 0.0 for a single file or zero total.
    """
    total = sum(total_lines_changed_per_file)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in total_lines_changed_per_file:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def extract_commit_features(
    commit_info: CommitInfo,
    file_features: list[FileFeatures] | None = None,
) -> CommitFeatures:
    """Extract deterministic commit-level features from a ``CommitInfo``.

    Parameters
    ----------
    commit_info:
        Phase 1 structured commit output.
    file_features:
        Pre-computed file-level features.  If ``None``, they are computed
        internally (the extractor always computes them for consistency).

    Returns
    -------
    CommitFeatures
    """
    if file_features is None:
        file_features = [extract_file_features(f) for f in commit_info.files]

    stats = commit_info.stats

    # ── Change volume ──
    lines_added = stats.total_lines_added
    lines_deleted = stats.total_lines_deleted
    total_lines_changed = lines_added + lines_deleted

    # ── File counts ──
    files_changed = stats.total_files
    files_added = stats.files_added
    files_deleted = stats.files_deleted
    files_modified = stats.files_modified
    files_renamed = stats.files_renamed
    files_binary = stats.files_binary

    # ── Hunk density ──
    total_hunks = sum(len(f.hunks) for f in commit_info.files)
    avg_hunks_per_file = total_hunks / max(files_changed, 1)

    # ── Ratios ──
    additions_ratio = lines_added / max(total_lines_changed, 1)
    deletions_ratio = lines_deleted / max(total_lines_changed, 1)

    # ── Test / production coupling ──
    test_count = sum(1 for ff in file_features if ff.is_test_file)
    prod_count = files_changed - test_count
    test_ratio = test_count / max(files_changed, 1)
    has_test = test_count > 0
    has_prod = prod_count > 0
    test_prod_coupling = has_test and has_prod

    # ── Language diversity ──
    lang_counter: Counter[str] = Counter()
    line_weight: Counter[str] = Counter()
    for f, ff in zip(commit_info.files, file_features):
        lang = detect_language(f.path)
        lang_counter[lang] += 1
        line_weight[lang] = f.total_lines_changed
    languages_touched = len(lang_counter)

    # Primary language: language with most lines changed
    if line_weight:
        primary_lang_name = line_weight.most_common(1)[0][0]
        primary_lang_idx = LANGUAGE_LIST.index(primary_lang_name) if primary_lang_name in LANGUAGE_LIST else len(LANGUAGE_LIST) - 1
        primary_lang_float = primary_lang_idx / max(len(LANGUAGE_LIST) - 1, 1)
    else:
        primary_lang_float = 0.0

    # ── Change distribution ──
    per_file_lines = [f.total_lines_changed for f in commit_info.files]
    avg_file_changes = sum(per_file_lines) / max(files_changed, 1)
    max_file_changes = max(per_file_lines) if per_file_lines else 0

    # ── Aggregate heuristic code-level signals ──
    total_func_decl = sum(
        ff.function_declarations_added + ff.function_declarations_deleted
        for ff in file_features
    )
    total_cls_decl = sum(
        ff.class_declarations_added + ff.class_declarations_deleted
        for ff in file_features
    )
    total_imports = sum(ff.imports_added + ff.imports_deleted for ff in file_features)

    # ── Indentation statistics (structural proxy) ──
    # Collect all changed lines across all files
    all_indents: list[int] = []
    for f in commit_info.files:
        for hunk in f.hunks:
            for line in hunk.content.splitlines():
                if (line.startswith("+") and not line.startswith("+++")) or (
                    line.startswith("-") and not line.startswith("---")
                ):
                    code = line[1:]
                    all_indents.append(len(code) - len(code.lstrip()))
    avg_indent = (sum(all_indents) / len(all_indents)) if all_indents else 0.0
    max_indent = max(all_indents) if all_indents else 0

    # ── Change entropy ──
    entropy = _change_entropy(per_file_lines)

    return CommitFeatures(
        commit_sha=commit_info.sha,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_lines_changed=total_lines_changed,
        files_changed=files_changed,
        files_added=files_added,
        files_deleted=files_deleted,
        files_modified=files_modified,
        files_renamed=files_renamed,
        files_binary=files_binary,
        total_hunks=total_hunks,
        avg_hunks_per_file=avg_hunks_per_file,
        additions_ratio=additions_ratio,
        deletions_ratio=deletions_ratio,
        test_files_changed=test_count,
        prod_files_changed=prod_count,
        test_ratio=test_ratio,
        has_test_changes=has_test,
        has_prod_changes=has_prod,
        test_prod_coupling=test_prod_coupling,
        languages_touched=languages_touched,
        primary_language=primary_lang_float,
        avg_file_changes=avg_file_changes,
        max_file_changes=max_file_changes,
        total_function_declarations_changed=total_func_decl,
        total_class_declarations_changed=total_cls_decl,
        total_imports_changed=total_imports,
        avg_changed_line_indent=avg_indent,
        max_changed_line_indent=max_indent,
        change_entropy=entropy,
    )
