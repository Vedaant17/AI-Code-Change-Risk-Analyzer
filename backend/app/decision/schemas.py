"""Investigation-priority presentation schemas (Phase 8.0).

Defines the deterministic presentation layer for B1 investigation priority
and Phase 6 evidence. PriorityBand is a rank-position grouping — not a risk
category, not a severity level, not a defect classification.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PriorityBand(str, Enum):
    """Deterministic rank-position grouping derived from B1 output order.

    Semantics: investigation-priority rank-position grouping only.
    NOT risk. NOT severity. NOT probability. NOT defect classification.
    """

    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceSummary(BaseModel):
    """Strict field-by-field projection of a Phase 6 EvidenceItem.

    Same fields, same values, no transformation.
    """

    category: str = Field(
        ...,
        description="Evidence category (verbatim from EvidenceItem)",
    )
    evidence_type: str = Field(
        ...,
        description="Specific evidence type (verbatim from EvidenceItem)",
    )
    description: str = Field(
        ...,
        description="Observable, source-backed fact (verbatim from EvidenceItem)",
    )
    source: str = Field(
        ...,
        description="Data source (verbatim from EvidenceItem)",
    )
    provenance: str = Field(
        ...,
        description="Whether directly observed or derived (verbatim from EvidenceItem)",
    )


class FileDecision(BaseModel):
    """Per-file investigation-priority presentation.

    B1 fields are copied verbatim from FileEvidence.
    Priority bands are derived from B1 output position + evidence types.
    """

    path: str = Field(..., description="File path relative to repository root")
    priority_band: PriorityBand = Field(
        ...,
        description="Deterministic rank-position band",
    )
    b1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="B1 investigation-priority score (verbatim from FileEvidence)",
    )
    b1_position: int = Field(
        ...,
        ge=0,
        description="0-indexed position in B1 output (verbatim from FileEvidence)",
    )
    total_lines_changed: int = Field(
        ...,
        ge=0,
        description="Raw B1 signal (verbatim from FileEvidence)",
    )
    is_binary: bool = Field(
        ...,
        description="Derived from evidence_type == 'binary_file' presence",
    )
    is_test_file: bool = Field(
        ...,
        description="Derived from evidence_type == 'test_file' presence",
    )
    evidence_count: int = Field(
        ...,
        ge=0,
        description="Number of evidence items for this file",
    )
    evidence_summaries: list[EvidenceSummary] = Field(
        default_factory=list,
        description="Projected evidence items for this file",
    )
    explanation: str = Field(
        default="",
        description="Deterministic text composed from evidence + B1 metadata",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="What evidence is absent or limited for this file",
    )


class PrioritySummary(BaseModel):
    """Commit-level summary counts derived from B1 output + evidence."""

    total_files: int = Field(
        ...,
        ge=0,
        description="Total files analyzed (verbatim from InvestigationResult)",
    )
    files_ranked: int = Field(
        ...,
        ge=0,
        description="Non-binary files (total_files - files_binary)",
    )
    files_binary: int = Field(
        ...,
        ge=0,
        description="Binary files (evidence contains 'binary_file' type)",
    )
    highest_count: int = Field(
        ..., ge=0, description="Files in HIGHEST band"
    )
    high_count: int = Field(..., ge=0, description="Files in HIGH band")
    medium_count: int = Field(
        ..., ge=0, description="Files in MEDIUM band"
    )
    low_count: int = Field(..., ge=0, description="Files in LOW band")
    evidence_available: int = Field(
        ..., ge=0, description="Files with at least one evidence item"
    )
    evidence_unavailable: int = Field(
        ..., ge=0, description="Files with zero evidence items"
    )


class DecisionResult(BaseModel):
    """Deterministic investigation-priority presentation.

    All semantic fields are deterministic given identical InvestigationResult
    input. Runtime metadata (analyzed_at, elapsed_ms) is excluded.
    """

    repo_url: str = Field(..., description="Repository URL")
    commit_sha: str = Field(..., description="Full commit SHA")
    short_sha: str = Field(..., description="Short commit SHA")
    strategy: str = Field(
        default="B1_CHANGE_SIZE",
        description="Investigation strategy used",
    )
    strategy_version: str = Field(
        default="1.0.0",
        description="Strategy version",
    )
    feature_version: str = Field(
        default="v1",
        description="Feature extraction version",
    )
    evidence_version: str = Field(
        default="1.0.0",
        description="Evidence collection version",
    )
    decision_version: str = Field(
        default="1.0.0",
        description="Decision layer version",
    )
    summary: PrioritySummary = Field(
        ..., description="Commit-level summary counts"
    )
    files: list[FileDecision] = Field(
        default_factory=list,
        description="Per-file investigation-priority decisions",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this analysis",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from the analysis pipeline",
    )


class DecisionResponse(BaseModel):
    """API envelope separating deterministic semantic content from runtime metadata.

    The `decision` field contains all deterministic semantic content.
    `analyzed_at` and `elapsed_ms` are runtime-only and do not affect
    deterministic output.
    """

    decision: DecisionResult = Field(
        ..., description="Deterministic investigation-priority presentation"
    )
    analyzed_at: str = Field(
        ..., description="ISO 8601 UTC timestamp of analysis start (runtime only)"
    )
    elapsed_ms: float = Field(
        ..., description="Wall-clock time in milliseconds (runtime only)"
    )
