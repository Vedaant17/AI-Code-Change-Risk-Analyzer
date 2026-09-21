"""Investigation service schemas (Phase 6).

Defines request and response models for the evidence-backed investigation
endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.evidence.schemas import FileEvidence


class InvestigateRequest(BaseModel):
    """Request for evidence-backed investigation of a commit."""

    repo_url: str = Field(
        ...,
        min_length=1,
        description="HTTPS URL or local filesystem path to the repository",
    )
    commit_sha: str = Field(
        ...,
        min_length=1,
        description="Full or abbreviated commit SHA to investigate",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Maximum number of eligible files for historical evidence "
            "analysis. Binary and deleted files do not consume slots."
        ),
    )


class InvestigationResult(BaseModel):
    """Deterministic evidence-backed investigation result for a commit.

    For the same repository state, commit SHA, top_k, and available Git
    history, repeated runs produce byte-for-byte identical content.
    """

    repo_url: str = Field(..., description="Source repository URL or path")
    commit_sha: str = Field(..., description="Full commit SHA")
    short_sha: str = Field(..., description="Abbreviated commit SHA")
    strategy: str = Field(
        default="B1_CHANGE_SIZE",
        description="Production ranking strategy identifier",
    )
    total_files: int = Field(
        default=0,
        ge=0,
        description="Total number of files in this commit",
    )
    files: list[FileEvidence] = Field(
        default_factory=list,
        description="Files in frozen B1 output order with evidence",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Analysis warnings",
    )
    evidence_version: str = Field(
        default="1.0.0",
        description="Evidence engine version",
    )
    top_k_configured: int = Field(
        default=10,
        ge=0,
        description="Requested maximum number of eligible files for historical analysis",
    )
    top_k_analyzed: int = Field(
        default=0,
        ge=0,
        description="Actual number of eligible files selected for historical analysis",
    )
    git_subprocess_count: int = Field(
        default=0,
        ge=0,
        description="Actual number of historical Git invocations attempted",
    )
