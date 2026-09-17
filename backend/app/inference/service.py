"""Production inference service (Phase 5.0).

Single entry point for commit-level investigation-priority analysis.

Usage::

    from backend.app.inference.service import InferenceService

    service = InferenceService()
    result = service.analyze_commit(
        repo_url="https://github.com/user/repo",
        commit_sha="abc123",
    )
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
)
from backend.app.inference.feature_pipeline import PipelineResult, extract_features
from backend.app.inference.registry import (
    DEFAULT_STRATEGY,
    StrategyMeta,
    get_production_strategy,
)
from backend.app.inference.schemas import (
    AnalyzeRiskResponse,
    FileRiskResult,
)
from backend.app.inference.scoring import score_files
from backend.app.schemas.diff import FileStatus
from backend.app.services.git_service import GitService, GitServiceError

logger = logging.getLogger(__name__)


def _is_analyzable(file_status: FileStatus) -> bool:
    """Return True if the file status warrants feature extraction."""
    return file_status not in (FileStatus.BINARY,)


def _make_short_sha(sha: str) -> str:
    """Return the first 8 characters of a SHA."""
    return sha[:8]


class InferenceService:
    """Production inference service for investigation-priority analysis.

    Wraps existing Git ingestion (Phase 1), feature extraction (Phase 2),
    and B1 scoring into a single callable.

    Parameters
    ----------
    git_service:
        Optional pre-configured ``GitService`` instance.
    """

    def __init__(self, git_service: GitService | None = None) -> None:
        self._git_service = git_service or GitService()

    def analyze_commit(
        self,
        repo_url: str,
        commit_sha: str,
        strategy: str = DEFAULT_STRATEGY,
    ) -> AnalyzeRiskResponse:
        """Analyse a commit and return investigation-priority rankings.

        Parameters
        ----------
        repo_url:
            HTTPS URL or local filesystem path to the repository.
        commit_sha:
            Full or abbreviated commit SHA.
        strategy:
            Ranking strategy.  Must be a production-safe strategy.
            Default: ``B1_CHANGE_SIZE``.

        Returns
        -------
        AnalyzeRiskResponse
            Structured response with ranked files.

        Raises
        ------
        UnsupportedStrategyError
            If the requested strategy is not production-safe.
        RepositoryAccessError
            If the repository cannot be cloned or opened.
        CommitNotFoundError
            If the commit SHA does not exist.
        FeatureExtractionError
            If feature extraction fails.
        """
        start = time.monotonic()

        # Validate strategy
        strategy_meta = get_production_strategy(strategy)

        # Step 1: Git ingestion
        local_path = self._clone_or_open(repo_url)
        try:
            commit_info = self._get_commit(local_path, commit_sha, repo_url)
        except InferenceError:
            raise
        except GitServiceError as exc:
            raise RepositoryAccessError(str(exc)) from exc

        # Step 2: Feature extraction
        try:
            pipeline_result = extract_features(commit_info)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Feature extraction failed for {commit_sha[:8]}: {exc}"
            ) from exc

        # Step 3: B1 scoring
        scored = score_files(pipeline_result)

        # Step 4: Build response
        response = self._build_response(
            repo_url=repo_url,
            commit_info=commit_info,
            pipeline_result=pipeline_result,
            scored=scored,
            strategy_meta=strategy_meta,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

        logger.info(
            "Analyzed %s/%s: %d files scored in %.1fms",
            repo_url,
            commit_sha[:8],
            response.files_analyzed,
            response.elapsed_ms,
        )

        return response

    def analyze_local_commit(
        self,
        repo_path: str,
        commit_sha: str,
        strategy: str = DEFAULT_STRATEGY,
    ) -> AnalyzeRiskResponse:
        """Analyse a commit in an already-cloned local repository.

        Parameters
        ----------
        repo_path:
            Local filesystem path to the repository.
        commit_sha:
            Full or abbreviated commit SHA.
        strategy:
            Ranking strategy.  Default: ``B1_CHANGE_SIZE``.

        Returns
        -------
        AnalyzeRiskResponse
        """
        start = time.monotonic()

        strategy_meta = get_production_strategy(strategy)

        try:
            commit_info = self._git_service.get_commit_diff(
                repo_path, commit_sha
            )
        except GitServiceError as exc:
            raise CommitNotFoundError(str(exc)) from exc

        try:
            pipeline_result = extract_features(commit_info)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Feature extraction failed for {commit_sha[:8]}: {exc}"
            ) from exc

        scored = score_files(pipeline_result)

        return self._build_response(
            repo_url=repo_path,
            commit_info=commit_info,
            pipeline_result=pipeline_result,
            scored=scored,
            strategy_meta=strategy_meta,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    # -- Internal helpers ---------------------------------------------------

    def _clone_or_open(self, repo_url: str) -> str:
        """Clone or open the repository.  Returns local path."""
        try:
            return self._git_service.clone_repo(repo_url)
        except GitServiceError as exc:
            raise RepositoryAccessError(str(exc)) from exc

    def _get_commit(
        self, local_path: str, commit_sha: str, repo_url: str
    ) -> object:
        """Get structured commit info from a local repo."""
        try:
            return self._git_service.get_commit_diff(local_path, commit_sha)
        except GitServiceError as exc:
            msg = str(exc).lower()
            if "unknown commit" in msg or "badname" in msg:
                raise CommitNotFoundError(
                    f"Commit {commit_sha!r} not found in {repo_url}"
                ) from exc
            raise RepositoryAccessError(str(exc)) from exc

    def _build_response(
        self,
        repo_url: str,
        commit_info: object,
        pipeline_result: PipelineResult,
        scored: list[tuple[object, float]],
        strategy_meta: StrategyMeta,
        elapsed_ms: float,
    ) -> AnalyzeRiskResponse:
        """Build the structured response from scored files."""
        ci = pipeline_result.commit_info
        files_result: list[FileRiskResult] = []
        warnings: list[str] = []
        files_analyzed = 0
        files_skipped = 0

        total_in_commit = len(ci.files)

        for file_feat, score in scored:
            if file_feat.is_binary:
                files_skipped += 1
                warnings.append(
                    f"Binary file {file_feat.path} not scored"
                )
                rank = 0
            else:
                files_analyzed += 1
                rank = files_analyzed

            files_result.append(
                FileRiskResult(
                    path=file_feat.path,
                    status=file_feat.status,
                    investigation_priority_score=score,
                    rank=rank,
                    total_files_in_commit=total_in_commit,
                    lines_added=file_feat.lines_added,
                    lines_deleted=file_feat.lines_deleted,
                    is_binary=file_feat.is_binary,
                    language=_language_name(file_feat.language),
                    is_test_file=file_feat.is_test_file,
                )
            )

        return AnalyzeRiskResponse(
            repo_url=repo_url,
            commit_sha=ci.sha,
            short_sha=_make_short_sha(ci.sha),
            strategy=strategy_meta.name,
            strategy_version=strategy_meta.strategy_version,
            feature_version=strategy_meta.feature_version,
            analyzed_at=datetime.now(tz=UTC).isoformat(),
            elapsed_ms=elapsed_ms,
            total_files=total_in_commit,
            files_analyzed=files_analyzed,
            files_skipped=files_skipped,
            files=files_result,
            warnings=warnings,
        )


def _language_name(lang_index: float) -> str:
    """Convert a language index back to a human-readable name."""
    from backend.app.features.language import LANGUAGE_LIST

    idx = int(lang_index * (len(LANGUAGE_LIST) - 1) + 0.5)
    idx = max(0, min(idx, len(LANGUAGE_LIST) - 1))
    return LANGUAGE_LIST[idx]
