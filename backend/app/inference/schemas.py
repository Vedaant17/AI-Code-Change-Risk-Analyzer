"""Production inference request/response schemas (Phase 5.0).

Stable domain models for the inference service contract.  These schemas
define the public API boundary between the inference layer and its callers.

Score semantics
---------------
``investigation_priority_score`` is a **ranking score** in [0, 1].

It is NOT:
  - a defect probability
  - a calibrated confidence
  - a defect prevalence estimate

Higher scores indicate files that should be investigated first, based on
change magnitude.  Normalization to [0, 1] is a presentation convenience,
not a probability transformation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AnalyzeRiskRequest(BaseModel):
    """Request for commit-level investigation-priority analysis."""

    repo_url: str = Field(
        ...,
        min_length=1,
        description="HTTPS URL or local filesystem path to the repository",
    )
    commit_sha: str = Field(
        ...,
        min_length=1,
        description="Full or abbreviated commit SHA to analyse",
    )


# ---------------------------------------------------------------------------
# Per-file result
# ---------------------------------------------------------------------------


class FileRiskResult(BaseModel):
    """Investigation-priority result for a single file within a commit."""

    path: str = Field(..., description="File path relative to repository root")
    status: str = Field(
        ...,
        description="File status: added, modified, deleted, renamed, or binary",
    )
    investigation_priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Ranking score in [0, 1]. NOT a probability. "
            "Higher = investigate first."
        ),
    )
    rank: int = Field(
        ...,
        ge=0,
        description="1-indexed rank within the commit (0 = not scored)",
    )
    total_files_in_commit: int = Field(
        ...,
        ge=0,
        description="Total number of files in this commit",
    )
    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)
    is_binary: bool = Field(default=False)
    language: str = Field(default="", description="Detected language name")
    is_test_file: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

_LIMITATIONS: list[str] = [
    "Scores are investigation-priority rankings, not defect probabilities.",
    "No defensible negatives exist in the training data.",
    "FILE_STRONG represents labeling pipeline coverage, not confirmed defectives.",
    "The ranking heuristic is based on file-level change magnitude only.",
    "Repository selection bias: training used 50 non-random repositories.",
    "Score is not calibrated to any external frequency.",
]

StrategyLiteral = Literal["B1_CHANGE_SIZE"]


class AnalyzeRiskResponse(BaseModel):
    """Structured response from commit-level risk analysis."""

    # -- Identity --
    repo_url: str
    commit_sha: str
    short_sha: str

    # -- Strategy metadata --
    strategy: str = Field(
        default="B1_CHANGE_SIZE",
        description="Production ranking strategy identifier",
    )
    strategy_version: str = Field(
        default="1.0.0",
        description="Version of the ranking strategy",
    )
    feature_version: str = Field(
        default="v1",
        description="Canonical feature schema version",
    )

    # -- Analysis context --
    analyzed_at: str = Field(
        default="",
        description="ISO 8601 UTC timestamp of the analysis",
    )
    elapsed_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock analysis time in milliseconds",
    )

    # -- Commit summary --
    total_files: int = Field(default=0, ge=0)
    files_analyzed: int = Field(
        default=0,
        ge=0,
        description="Files that received a ranking score",
    )
    files_skipped: int = Field(
        default=0,
        ge=0,
        description="Files excluded from ranking (binary, etc.)",
    )

    # -- Results --
    files: list[FileRiskResult] = Field(default_factory=list)

    # -- Warnings --
    warnings: list[str] = Field(default_factory=list)

    # -- Semantics (immutable) --
    score_semantics: str = Field(
        default="investigation_priority_ranking",
        description=(
            "Score type identifier. " "NOT a probability or calibrated risk."
        ),
    )
    limitations: list[str] = Field(default_factory=lambda: list(_LIMITATIONS))
