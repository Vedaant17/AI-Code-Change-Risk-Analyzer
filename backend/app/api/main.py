"""FastAPI application — Phase 1/5.1/6/8 endpoints.

Endpoints:
    GET  /health
    POST /analysis/commit
    POST /analysis/pull-request
    POST /analysis/risk          (Phase 5.1)
    POST /analysis/investigate   (Phase 6)
    POST /analysis/decision      (Phase 8.0)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles

from backend.app.analyzers.diff_analyzer import DiffAnalyzer, DiffAnalyzerError
from backend.app.api.schemas import (
    AnalyzeCommitRequest,
    AnalyzePRRequest,
    CommitAnalysisResponse,
    DiffStatsResponse,
    FileDiffResponse,
    HealthResponse,
)
from backend.app.decision.engine import DecisionEngine
from backend.app.decision.schemas import DecisionResponse
from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
)
from backend.app.inference.schemas import AnalyzeRiskRequest, AnalyzeRiskResponse
from backend.app.inference.service import InferenceService
from backend.app.investigation.schemas import InvestigateRequest, InvestigationResult
from backend.app.investigation.service import InvestigationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Code Change-Risk Analyzer",
    version="0.1.0",
    description="MVP — Git diff ingestion and risk analysis.",
)

analyzer = DiffAnalyzer()
inference_service = InferenceService()
investigation_service = InvestigationService()
decision_engine = DecisionEngine()


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


@app.post(
    "/analysis/investigate",
    response_model=InvestigationResult,
    summary="Evidence-backed investigation of commit",
    description=(
        "Rank files by B1 investigation priority and attach observable "
        "evidence for each file. Scores are the same B1 ranking scores "
        "returned by /analysis/risk. Evidence is factual, provenance-tracked "
        "context — not risk prediction."
    ),
    responses={
        404: {"description": "Commit not found in repository"},
        422: {"description": "Repository inaccessible or invalid request"},
        500: {"description": "Feature extraction or inference failure"},
    },
)
async def investigate(req: InvestigateRequest) -> InvestigationResult:
    """Investigate a commit and return evidence-backed file rankings.

    Uses the B1_CHANGE_SIZE production heuristic for ranking. Evidence
    is factual context, not risk prediction.
    """
    try:
        return investigation_service.investigate(
            repo_url=req.repo_url,
            commit_sha=req.commit_sha,
            top_k=req.top_k,
        )
    except CommitNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FeatureExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/analysis/decision",
    response_model=DecisionResponse,
    summary="Deterministic investigation-priority presentation",
    description=(
        "Present B1 investigation priority and Phase 6 evidence "
        "in a structured, deterministic format with priority bands "
        "and evidence-grounded explanations."
    ),
    responses={
        404: {"description": "Commit not found in repository"},
        422: {"description": "Repository inaccessible or invalid request"},
        500: {"description": "Feature extraction or inference failure"},
    },
)
async def decide(req: InvestigateRequest) -> DecisionResponse:
    """Present deterministic investigation-priority ranking with evidence.

    Uses the frozen B1_CHANGE_SIZE heuristic for ranking and Phase 6
    evidence for context. Priority bands are rank-position groupings,
    not risk categories.
    """
    try:
        start = datetime.now(UTC)
        result = investigation_service.investigate(
            repo_url=req.repo_url,
            commit_sha=req.commit_sha,
            top_k=req.top_k,
        )
        decision = decision_engine.decide(result)
        elapsed_ms = (datetime.now(UTC) - start).total_seconds() * 1000
        return decision_engine.wrap_response(
            decision=decision,
            analyzed_at=start.isoformat(),
            elapsed_ms=elapsed_ms,
        )
    except CommitNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FeatureExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Static frontend (Phase 5.2) ────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parents[3] / "frontend"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="frontend")
