"""Evidence schemas (Phase 6).

Defines the data contracts for investigation evidence items and
per-file evidence bundles.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A single observable fact for investigation context.

    Every description must contain an observable, source-backed fact
    and must not make unsupported causal, predictive, or subjective
    risk claims.
    """

    category: str = Field(
        ...,
        description="Evidence category: change, structural, test, historical, context",
    )
    evidence_type: str = Field(
        ...,
        description="Specific evidence type, e.g., lines_changed, recent_commits",
    )
    description: str = Field(
        ...,
        description="Observable, source-backed fact",
    )
    source: str = Field(
        ...,
        description="Data source: diff, feature_extraction, git_log, path_convention, b1_ranking",
    )
    provenance: str = Field(
        ...,
        description="Whether the evidence is directly observed or derived: direct | derived",
    )
    ref_commit: str = Field(
        default="",
        description="Related commit SHA (for historical evidence only)",
    )
    ref_file: str = Field(
        default="",
        description="Related file path (for cross-file evidence)",
    )


class FileEvidence(BaseModel):
    """Investigation evidence for a single file in B1 order.

    Scores and positions are copied verbatim from the frozen B1 output.
    """

    path: str = Field(..., description="File path relative to repository root")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="B1 score, copied verbatim — not modified",
    )
    position: int = Field(
        ...,
        ge=0,
        description="0-indexed position in frozen B1 output",
    )
    total_lines_changed: int = Field(
        ...,
        ge=0,
        description="Raw B1 signal, included for transparency",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence items for this file",
    )
