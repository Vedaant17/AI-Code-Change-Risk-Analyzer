"""Evaluation population loader (Phase 7).

Loads the frozen combined-v3 test split and constructs production-compatible
FileFeatures objects for scoring by the frozen B1 scorer.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.dataset.schemas import DatasetRow
from backend.app.features.schemas import (
    FILE_FEATURE_NAMES,
    CommitFeatures,
    FeatureExtractionResult,
)
from backend.app.inference.feature_pipeline import FileFeatures, PipelineResult
from backend.app.inference.scoring import score_files
from backend.app.schemas.diff import CommitInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")

# Feature indices within the 16-dim file feature vector
_IDX_LANGUAGE = FILE_FEATURE_NAMES.index("language")           # 0
_IDX_IS_BINARY = FILE_FEATURE_NAMES.index("is_binary")       # 1
_IDX_IS_TEST = FILE_FEATURE_NAMES.index("is_test_file")      # 2
_IDX_LINES_ADDED = FILE_FEATURE_NAMES.index("lines_added")   # 3
_IDX_LINES_DELETED = FILE_FEATURE_NAMES.index("lines_deleted")  # 4
_IDX_TOTAL_LINES = FILE_FEATURE_NAMES.index("total_lines_changed")  # 5


@dataclass
class EvaluationPopulation:
    """Loaded and grouped evaluation data from a dataset split."""

    rows: list[DatasetRow]
    metadata: list[dict]
    positive_keys: set[tuple[str, str]]
    commit_groups: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_commits(self) -> int:
        return len(self.commit_groups)

    @property
    def n_positives(self) -> int:
        return sum(1 for r in self.rows if r.defect_label == 1)

    @property
    def positive_rate(self) -> float:
        return self.n_positives / self.n_rows if self.n_rows > 0 else 0.0

    @property
    def repositories(self) -> list[str]:
        return sorted({m["repo_name"] for m in self.metadata})


def load_population(data_dir: Path = DATA_DIR) -> EvaluationPopulation:
    """Load the frozen test split and build evaluation structures.

    Parameters
    ----------
    data_dir:
        Path to the combined-v3 dataset directory.

    Returns
    -------
    EvaluationPopulation
        Loaded rows, metadata, positive keys, and commit groups.
    """
    rows: list[DatasetRow] = []
    metadata: list[dict] = []

    with open(data_dir / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = DatasetRow.model_validate_json(line)
            rows.append(row)

    # Rebuild metadata using the same schema as dataset_loader
    for row in rows:
        metadata.append({
            "commit_sha": row.commit_sha,
            "file_path": row.file_path,
            "split": row.split,
            "label_status": row.label_status,
            "label_source": row.label_source,
            "label_confidence": row.label_confidence,
            "label_evidence": row.label_evidence,
            "commit_message": row.commit_message,
            "author": row.author,
            "file_status": row.file_status,
            "repo_name": row.repo_name,
        })

    # Build positive keys
    positive_keys: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        if row.defect_label == 1:
            positive_keys.add((row.commit_sha, row.file_path))

    # Group by commit
    commit_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        commit_groups[(m["repo_name"], m["commit_sha"])].append(i)

    population = EvaluationPopulation(
        rows=rows,
        metadata=metadata,
        positive_keys=positive_keys,
        commit_groups=dict(commit_groups),
    )

    logger.info(
        "Loaded evaluation population: %d rows, %d commits, %d positives (%.4f%%)",
        population.n_rows,
        population.n_commits,
        population.n_positives,
        population.positive_rate * 100,
    )

    return population


def construct_file_features(row: DatasetRow) -> FileFeatures:
    """Construct a production-compatible FileFeatures from a dataset row.

    Parameters
    ----------
    row:
        A single DatasetRow from the frozen dataset.

    Returns
    -------
    FileFeatures
        Production-compatible FileFeatures for B1 scoring.
    """
    ff = row.file_features
    return FileFeatures(
        path=row.file_path,
        status=row.file_status,
        total_lines_changed=int(round(ff[_IDX_TOTAL_LINES])),
        lines_added=int(round(ff[_IDX_LINES_ADDED])),
        lines_deleted=int(round(ff[_IDX_LINES_DELETED])),
        is_binary=ff[_IDX_IS_BINARY] > 0.5,
        language=ff[_IDX_LANGUAGE],
        is_test_file=ff[_IDX_IS_TEST] > 0.5,
    )


def score_commit(
    rows: list[DatasetRow],
    metadata: list[dict],
    commit_sha: str,
) -> list[float]:
    """Score all files in a commit using the frozen B1 scorer.

    Parameters
    ----------
    rows:
        DatasetRows for this commit.
    metadata:
        Metadata dicts aligned with rows.
    commit_sha:
        The commit SHA (for PipelineResult construction).

    Returns
    -------
    list[float]
        Scores aligned with the input metadata order.
    """
    if not rows:
        return []

    file_features_list = [construct_file_features(r) for r in rows]

    pipeline_result = PipelineResult(
        commit_info=CommitInfo(sha=commit_sha),
        extraction_result=FeatureExtractionResult(
            commit_features=CommitFeatures(commit_sha=commit_sha),
            file_features=[],
            feature_version="v1",
        ),
        file_features=file_features_list,
    )

    scored = score_files(pipeline_result)

    id_to_score = {id(ff): score for ff, score in scored}
    return [id_to_score[id(ff)] for ff in file_features_list]
