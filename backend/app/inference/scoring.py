"""B1 scoring implementation (Phase 5.0).

Implements the B1_CHANGE_SIZE investigation-priority heuristic.

Scoring rule
------------
1. For each file, ``score = file_total_lines_changed``.
2. Binary files receive ``score = 0.0`` (never ranked).
3. Non-binary files are rank-normalized to [0, 1] via
   ``1.0 - rank / max_rank`` where ``rank`` is 0-indexed descending.
4. Equal scores are broken by stable file order (diff order).
5. No random component.  Fully deterministic.

Output is ordered by descending score, then by original diff order for ties.
Binary files appear last with score 0.0.

This is an **investigation-priority heuristic**, NOT:
  - a defect probability
  - a calibrated confidence
  - a validated ML model
"""
from __future__ import annotations

import logging

from backend.app.inference.feature_pipeline import FileFeatures, PipelineResult

logger = logging.getLogger(__name__)


def score_files(pipeline_result: PipelineResult) -> list[tuple[FileFeatures, float]]:
    """Compute B1 investigation-priority scores for all files.

    Parameters
    ----------
    pipeline_result:
        Output of the feature pipeline for a single commit.

    Returns
    -------
    list[tuple[FileFeatures, float]]
        Pairs of (file_features, score) where score is in [0, 1].
        Ordered by descending score, then by original diff order for ties.
        Binary files always have score 0.0 and appear last.
    """
    files = pipeline_result.file_features
    if not files:
        return []

    # Separate binary and non-binary
    binary_indices: list[int] = []
    non_binary_indices: list[int] = []
    for i, f in enumerate(files):
        if f.is_binary:
            binary_indices.append(i)
        else:
            non_binary_indices.append(i)

    # Score non-binary files by rank normalization
    n_non_binary = len(non_binary_indices)
    max_rank = max(n_non_binary - 1, 1)

    # Sort non-binary by (-total_lines_changed, original_index)
    ranked = sorted(non_binary_indices, key=lambda i: (-files[i].total_lines_changed, i))

    normalized: dict[int, float] = {}
    for rank, orig_idx in enumerate(ranked):
        normalized[orig_idx] = 1.0 - rank / max_rank

    # Build result: non-binary first (descending score), then binary (0.0)
    result: list[tuple[FileFeatures, float]] = []
    for orig_idx in ranked:
        result.append((files[orig_idx], normalized[orig_idx]))
    for orig_idx in binary_indices:
        result.append((files[orig_idx], 0.0))

    logger.debug(
        "B1 scoring: commit=%s, files=%d, non_binary=%d, max_raw=%.0f",
        pipeline_result.commit_info.sha[:8],
        len(files),
        n_non_binary,
        max((files[i].total_lines_changed for i in non_binary_indices), default=0.0),
    )

    return result
