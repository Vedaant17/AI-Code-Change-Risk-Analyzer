"""Dataset schemas for Phase 3 historical dataset construction.

Defines the row-level and manifest-level Pydantic models for the
supervised dataset.  All fields are explicitly separated into
metadata, features, labels, and split.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.features.schemas import FEATURE_VERSION


class DatasetRow(BaseModel):
    """One row = one file change in one commit, with label.

    Splits are assigned at the commit level and inherited by all
    file rows from that commit.
    """

    # ── Metadata ──
    dataset_version: str = Field(default="v1")
    repo_url: str = Field(default="")
    repo_name: str = Field(default="")
    commit_sha: str = Field(default="")
    commit_timestamp: datetime | None = None
    commit_message: str = Field(default="")
    author: str = Field(default="")
    file_path: str = Field(default="")
    file_status: str = Field(
        default="",
        description="added/modified/deleted/renamed/binary",
    )

    # ── Features (Phase 2 contracts) ──
    feature_version: str = Field(default=FEATURE_VERSION)
    commit_features: list[float] = Field(default_factory=list)
    file_features: list[float] = Field(default_factory=list)

    # ── Label ──
    label_status: str = Field(
        default="negative",
        description="positive | negative | ambiguous",
    )
    defect_label: int = Field(
        default=0,
        description="1 if positive, 0 if negative, -1 if ambiguous",
    )
    label_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0.0 for negative, 1.0 for high-confidence positive",
    )
    label_source: str = Field(
        default="none",
        description="explicit_sha_reference | revert | line_overlap | none | ambiguous",
    )
    label_evidence: str = Field(
        default="",
        description="Human-readable description of the attribution evidence",
    )

    # ── Split (assigned at commit level, inherited by file rows) ──
    split: str = Field(
        default="train",
        description="train | validation | test",
    )


class DatasetManifest(BaseModel):
    """Metadata about the assembled dataset.

    Contains no volatile values (no datetime.now, no random IDs).
    All fields are deterministic given the same repository and config.
    """

    dataset_version: str
    feature_version: str
    labeling_version: str
    config: dict

    repo_url: str
    repo_name: str
    repo_revision: str  # HEAD SHA at processing time

    # ── Counts ──
    total_commits_processed: int
    total_commits_included: int
    total_commits_excluded: int
    total_file_examples: int

    positive_examples: int
    negative_examples: int
    ambiguous_examples: int
    positive_rate: float

    # ── Splits ──
    train_commits: int
    validation_commits: int
    test_commits: int
    train_examples: int
    validation_examples: int
    test_examples: int

    # ── Distributions ──
    commits_per_language: dict[str, int]
    examples_per_language: dict[str, int]
    time_range_start: str | None  # ISO format
    time_range_end: str | None

    # ── Attribution statistics ──
    bug_fix_commits_found: int
    revert_commits_found: int
    explicit_sha_attributions: int
    revert_attributions: int
    line_overlap_attributions: int
    ambiguous_attributions: int

    # ── Exclusions ──
    exclusion_counts: dict[str, int]
