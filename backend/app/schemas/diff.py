"""Domain schemas for Git diff ingestion (Phase 1).

These models represent structured output from diff parsing and serve as
the contract between the ingestion layer and downstream feature extraction.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class FileStatus(str, enum.Enum):
    """Status of a file within a commit or diff."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    BINARY = "binary"


class Hunk(BaseModel):
    """A single diff hunk within a file."""

    old_start: int = Field(..., description="Starting line in the old file")
    old_count: int = Field(..., description="Number of lines from old file in this hunk")
    new_start: int = Field(..., description="Starting line in the new file")
    new_count: int = Field(..., description="Number of lines from new file in this hunk")
    content: str = Field(default="", description="Raw hunk content including context lines")


class FileDiff(BaseModel):
    """Structured diff information for a single file."""

    path: str = Field(..., description="File path relative to repo root")
    old_path: str | None = Field(
        default=None, description="Previous path if renamed"
    )
    status: FileStatus
    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)
    is_binary: bool = Field(default=False)
    hunks: list[Hunk] = Field(default_factory=list)

    @property
    def total_lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted


class DiffStats(BaseModel):
    """Aggregated statistics across all files in a diff."""

    total_files: int = Field(default=0, ge=0)
    total_lines_added: int = Field(default=0, ge=0)
    total_lines_deleted: int = Field(default=0, ge=0)
    files_added: int = Field(default=0, ge=0)
    files_deleted: int = Field(default=0, ge=0)
    files_modified: int = Field(default=0, ge=0)
    files_renamed: int = Field(default=0, ge=0)
    files_binary: int = Field(default=0, ge=0)

    @property
    def total_lines_changed(self) -> int:
        return self.total_lines_added + self.total_lines_deleted


class CommitInfo(BaseModel):
    """Complete structured output for a single commit diff."""

    sha: str = Field(..., description="Full commit SHA")
    short_sha: str = Field(default="", description="Abbreviated commit SHA")
    author: str = Field(default="")
    author_date: datetime | None = None
    message: str = Field(default="")
    files: list[FileDiff] = Field(default_factory=list)
    stats: DiffStats = Field(default_factory=DiffStats)
    repo_url: str = Field(default="", description="Source repository URL")
    base_ref: str = Field(
        default="", description="Parent commit SHA or base branch"
    )

    def compute_stats(self) -> DiffStats:
        """Compute and cache aggregate stats from the file list."""
        stats = DiffStats(total_files=len(self.files))
        for f in self.files:
            stats.total_lines_added += f.lines_added
            stats.total_lines_deleted += f.lines_deleted
            match f.status:
                case FileStatus.ADDED:
                    stats.files_added += 1
                case FileStatus.DELETED:
                    stats.files_deleted += 1
                case FileStatus.MODIFIED:
                    stats.files_modified += 1
                case FileStatus.RENAMED:
                    stats.files_renamed += 1
                case FileStatus.BINARY:
                    stats.files_binary += 1
        self.stats = stats
        return stats
