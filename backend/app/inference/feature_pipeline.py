"""Production feature pipeline (Phase 5.0).

Wraps existing Phase 1 (Git ingestion) and Phase 2 (feature extraction)
into a single callable for the inference service.

Pipeline
--------
``commit_info`` (CommitInfo)
    → ``FeatureExtractor.extract()``
    → ``FeatureExtractionResult`` (commit_features + file_features)
    → per-file ``total_lines_changed`` for B1 scoring

This module does NOT:
- load training data
- build PU labels
- compute experimental features (E1/E2/E4)
- depend on any Phase 4.x artifacts
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.app.features.extractor import FeatureExtractor
from backend.app.features.schemas import FeatureExtractionResult
from backend.app.schemas.diff import CommitInfo, FileDiff

logger = logging.getLogger(__name__)


@dataclass
class FileFeatures:
    """Lightweight per-file feature container for B1 scoring."""

    path: str
    status: str
    total_lines_changed: int
    lines_added: int
    lines_deleted: int
    is_binary: bool
    language: float
    is_test_file: bool


@dataclass
class PipelineResult:
    """Output of the feature pipeline for a single commit."""

    commit_info: CommitInfo
    extraction_result: FeatureExtractionResult
    file_features: list[FileFeatures]


def _map_file_status(file_diff: FileDiff) -> str:
    """Map ``FileStatus`` enum to a plain string for the response."""
    return file_diff.status.value


def _build_file_features(
    commit_info: CommitInfo,
    extraction_result: FeatureExtractionResult,
) -> list[FileFeatures]:
    """Combine diff information with extracted features.

    For each file in the commit, produces a ``FileFeatures`` that carries
    the ``total_lines_changed`` value needed by the B1 scoring heuristic.
    """
    results: list[FileFeatures] = []
    for file_diff, file_feat in zip(
        commit_info.files, extraction_result.file_features
    ):
        results.append(
            FileFeatures(
                path=file_diff.path,
                status=_map_file_status(file_diff),
                total_lines_changed=file_feat.total_lines_changed,
                lines_added=file_diff.lines_added,
                lines_deleted=file_diff.lines_deleted,
                is_binary=file_diff.is_binary,
                language=file_feat.language,
                is_test_file=file_feat.is_test_file,
            )
        )
    return results


def extract_features(commit_info: CommitInfo) -> PipelineResult:
    """Run the full canonical feature pipeline on a ``CommitInfo``.

    Parameters
    ----------
    commit_info:
        Structured commit output from the Phase 1 Git/diff layer.

    Returns
    -------
    PipelineResult
        Contains the original ``CommitInfo``, the full
        ``FeatureExtractionResult``, and per-file ``FileFeatures``
        ready for scoring.
    """
    extractor = FeatureExtractor()
    extraction_result = extractor.extract(commit_info)
    file_features = _build_file_features(commit_info, extraction_result)

    logger.info(
        "Feature pipeline: commit=%s, files=%d, feature_version=%s",
        commit_info.sha[:8],
        len(file_features),
        extraction_result.feature_version,
    )

    return PipelineResult(
        commit_info=commit_info,
        extraction_result=extraction_result,
        file_features=file_features,
    )
