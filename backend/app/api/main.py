"""FastAPI application — Phase 1 skeleton.

Endpoints:
    GET  /health
    POST /analysis/commit
    POST /analysis/pull-request
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from backend.app.analyzers.diff_analyzer import DiffAnalyzer, DiffAnalyzerError
from backend.app.api.schemas import (
    AnalyzeCommitRequest,
    AnalyzePRRequest,
    CommitAnalysisResponse,
    DiffStatsResponse,
    FileDiffResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Code Change-Risk Analyzer",
    version="0.1.0",
    description="MVP — Git diff ingestion and risk analysis.",
)

analyzer = DiffAnalyzer()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()


@app.post("/analysis/commit", response_model=CommitAnalysisResponse)
async def analyze_commit(req: AnalyzeCommitRequest) -> CommitAnalysisResponse:
    """Analyse a single commit in a repository."""
    try:
        info = analyzer.analyze_commit(req.repo_url, req.commit_sha)
    except DiffAnalyzerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CommitAnalysisResponse(
        sha=info.sha,
        short_sha=info.short_sha,
        author=info.author,
        author_date=info.author_date,
        message=info.message,
        stats=DiffStatsResponse(**info.stats.model_dump()),
        files=[
            FileDiffResponse(
                path=f.path,
                status=f.status.value,
                lines_added=f.lines_added,
                lines_deleted=f.lines_deleted,
                is_binary=f.is_binary,
                old_path=f.old_path,
            )
            for f in info.files
        ],
    )


@app.post("/analysis/pull-request", response_model=CommitAnalysisResponse)
async def analyze_pull_request(req: AnalyzePRRequest) -> CommitAnalysisResponse:
    """Analyse the diff between two branches (simulates PR analysis)."""
    try:
        info = analyzer.analyze_pull_request(
            req.repo_url, req.base_branch, req.head_branch
        )
    except DiffAnalyzerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CommitAnalysisResponse(
        sha=info.sha,
        short_sha=info.short_sha,
        author=info.author,
        author_date=info.author_date,
        message=info.message,
        stats=DiffStatsResponse(**info.stats.model_dump()),
        files=[
            FileDiffResponse(
                path=f.path,
                status=f.status.value,
                lines_added=f.lines_added,
                lines_deleted=f.lines_deleted,
                is_binary=f.is_binary,
                old_path=f.old_path,
            )
            for f in info.files
        ],
    )
