"""Request / response schemas for the FastAPI layer (Phase 1)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalyzeCommitRequest(BaseModel):
    """Payload for ``POST /analysis/commit``."""

    repo_url: str = Field(
        ..., min_length=1, description="HTTPS URL or local path to the repository"
    )
    commit_sha: str = Field(
        ..., min_length=1, description="Full or abbreviated commit SHA"
    )


class AnalyzePRRequest(BaseModel):
    """Payload for ``POST /analysis/pull-request``."""

    repo_url: str = Field(
        ..., min_length=1, description="HTTPS URL or local path to the repository"
    )
    base_branch: str = Field(
        ..., min_length=1, description="Target / base branch"
    )
    head_branch: str = Field(
        ..., min_length=1, description="Source / feature branch"
    )


# ---------------------------------------------------------------------------
# Response schemas (compact subset of CommitInfo for the API)
# ---------------------------------------------------------------------------

class FileDiffResponse(BaseModel):
    path: str
    status: str
    lines_added: int
    lines_deleted: int
    is_binary: bool
    old_path: str | None = None


class DiffStatsResponse(BaseModel):
    total_files: int
    total_lines_added: int
    total_lines_deleted: int
    files_added: int
    files_deleted: int
    files_modified: int
    files_renamed: int
    files_binary: int


class CommitAnalysisResponse(BaseModel):
    """Returned by ``POST /analysis/commit`` and ``POST /analysis/pull-request``."""

    sha: str
    short_sha: str
    author: str
    author_date: datetime | None = None
    message: str
    files: list[FileDiffResponse]
    stats: DiffStatsResponse


class HealthResponse(BaseModel):
    status: str = "ok"
