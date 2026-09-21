"""Investigation service (Phase 6).

Composes frozen B1 scoring with independent evidence collection into
an evidence-backed investigation result.
"""
from __future__ import annotations

import logging

from backend.app.evidence.collector import EvidenceEngine
from backend.app.evidence.schemas import FileEvidence
from backend.app.features.language import LANGUAGE_LIST
from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
)
from backend.app.inference.feature_pipeline import extract_features
from backend.app.inference.registry import DEFAULT_STRATEGY, get_production_strategy
from backend.app.inference.scoring import score_files
from backend.app.investigation.schemas import InvestigationResult
from backend.app.services.git_service import GitService, GitServiceError

logger = logging.getLogger(__name__)


def _language_name(lang_index: float) -> str:
    """Convert the language index back to a human-readable name."""
    idx = int(lang_index * (len(LANGUAGE_LIST) - 1) + 0.5)
    idx = max(0, min(idx, len(LANGUAGE_LIST) - 1))
    return LANGUAGE_LIST[idx]


class InvestigationService:
    """Evidence-backed investigation service for commit analysis.

    Composes frozen B1 scoring with independent evidence collection.
    B1 scoring is untouched; evidence is collected separately and
    merged into the investigation result.

    Parameters
    ----------
    git_service:
        Optional pre-configured ``GitService`` instance.
    """

    def __init__(self, git_service: GitService | None = None) -> None:
        self._git_service = git_service or GitService()
        self._evidence_engine = EvidenceEngine()

    def investigate(
        self,
        repo_url: str,
        commit_sha: str,
        top_k: int = 10,
    ) -> InvestigationResult:
        """Investigate a commit and return evidence-backed results.

        Parameters
        ----------
        repo_url:
            HTTPS URL or local filesystem path to the repository.
        commit_sha:
            Full or abbreviated commit SHA.
        top_k:
            Maximum number of eligible files for historical evidence.

        Returns
        -------
        InvestigationResult
        """
        strategy_meta = get_production_strategy(DEFAULT_STRATEGY)

        # [1-3] Same frozen pipeline as InferenceService
        local_path = self._clone_or_open(repo_url)
        try:
            commit_info = self._get_commit(local_path, commit_sha, repo_url)
        except InferenceError:
            raise
        except GitServiceError as exc:
            raise RepositoryAccessError(str(exc)) from exc

        try:
            pipeline_result = extract_features(commit_info)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Feature extraction failed for {commit_sha[:8]}: {exc}"
            ) from exc

        # [4] Frozen B1 scoring — all files including binary
        b1_scored = score_files(pipeline_result)

        # [5] Construct B1 context — preserves exact frozen B1 output order
        # Position is 0-indexed in the frozen B1 output list
        b1_context = [
            (file_feat.path, score, position)
            for position, (file_feat, score) in enumerate(b1_scored)
        ]

        # [6-7] Collect evidence (cheap + historical)
        evidence_by_path, git_count, top_k_analyzed = (
            self._evidence_engine.collect(
                pipeline_result=pipeline_result,
                b1_context=b1_context,
                local_path=local_path,
                top_k=top_k,
            )
        )

        # [8] Compose — files in exact frozen B1 output order, scores verbatim
        warnings: list[str] = []
        files: list[FileEvidence] = []

        for position, (file_feat, score) in enumerate(b1_scored):
            file_evidence = evidence_by_path.get(file_feat.path, [])
            files.append(FileEvidence(
                path=file_feat.path,
                score=score,
                position=position,
                total_lines_changed=file_feat.total_lines_changed,
                evidence=file_evidence,
            ))

        logger.info(
            "Investigated %s/%s: %d files, %d with historical evidence",
            repo_url,
            commit_sha[:8],
            len(files),
            top_k_analyzed,
        )

        return InvestigationResult(
            repo_url=repo_url,
            commit_sha=commit_info.sha,
            short_sha=commit_info.sha[:8],
            strategy=strategy_meta.name,
            total_files=len(commit_info.files),
            files=files,
            warnings=warnings,
            top_k_configured=top_k,
            top_k_analyzed=top_k_analyzed,
            git_subprocess_count=git_count,
        )

    def investigate_local(
        self,
        repo_path: str,
        commit_sha: str,
        top_k: int = 10,
    ) -> InvestigationResult:
        """Investigate a commit in an already-cloned local repository.

        Parameters
        ----------
        repo_path:
            Local filesystem path to the repository.
        commit_sha:
            Full or abbreviated commit SHA.
        top_k:
            Maximum number of eligible files for historical evidence.

        Returns
        -------
        InvestigationResult
        """
        strategy_meta = get_production_strategy(DEFAULT_STRATEGY)

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

        b1_scored = score_files(pipeline_result)

        b1_context = [
            (file_feat.path, score, position)
            for position, (file_feat, score) in enumerate(b1_scored)
        ]

        evidence_by_path, git_count, top_k_analyzed = (
            self._evidence_engine.collect(
                pipeline_result=pipeline_result,
                b1_context=b1_context,
                local_path=repo_path,
                top_k=top_k,
            )
        )

        files: list[FileEvidence] = []
        for position, (file_feat, score) in enumerate(b1_scored):
            file_evidence = evidence_by_path.get(file_feat.path, [])
            files.append(FileEvidence(
                path=file_feat.path,
                score=score,
                position=position,
                total_lines_changed=file_feat.total_lines_changed,
                evidence=file_evidence,
            ))

        return InvestigationResult(
            repo_url=repo_path,
            commit_sha=commit_info.sha,
            short_sha=commit_info.sha[:8],
            strategy=strategy_meta.name,
            total_files=len(commit_info.files),
            files=files,
            warnings=[],
            top_k_configured=top_k,
            top_k_analyzed=top_k_analyzed,
            git_subprocess_count=git_count,
        )

    # -- Internal helpers ---------------------------------------------------

    def _clone_or_open(self, repo_url: str) -> str:
        """Clone or open the repository. Returns local path."""
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
