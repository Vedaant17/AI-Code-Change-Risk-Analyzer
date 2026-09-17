"""FastAPI application — Phase 1 skeleton + Phase 5.1 inference endpoint.

Endpoints:
    GET  /health
    POST /analysis/commit
    POST /analysis/pull-request
    POST /analysis/risk          (Phase 5.1)
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
from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
)
from backend.app.inference.schemas import AnalyzeRiskRequest, AnalyzeRiskResponse
from backend.app.inference.service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Code Change-Risk Analyzer",
    version="0.1.0",
    description="MVP — Git diff ingestion and risk analysis.",
)

analyzer = DiffAnalyzer()
inference_service = InferenceService()


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


@app.post(
    "/analysis/risk",
    response_model=AnalyzeRiskResponse,
    summary="Analyse commit investigation-priority ranking",
    description=(
        "Rank files within a commit by investigation priority using the "
        "B1_CHANGE_SIZE heuristic. Scores are ranking scores in [0, 1] "
        "-- NOT defect probabilities, calibrated confidence, or likelihood. "
        "Higher scores indicate files that should be investigated first "
        "based on change magnitude."
    ),
    responses={
        404: {"description": "Commit not found in repository"},
        422: {"description": "Repository inaccessible or invalid request"},
        500: {"description": "Feature extraction or inference failure"},
    },
)
async def analyze_risk(req: AnalyzeRiskRequest) -> AnalyzeRiskResponse:
    """Analyse a commit and return investigation-priority file rankings.

    Uses the B1_CHANGE_SIZE production heuristic. Scores represent
    investigation priority, NOT defect probability.
    """
    try:
        return inference_service.analyze_commit(
            repo_url=req.repo_url,
            commit_sha=req.commit_sha,
        )
    except CommitNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FeatureExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
