"""Diff analyzer orchestration layer (Phase 1).

Thin wrapper that connects repository access with diff parsing to provide
a single entry point for commit / PR analysis.
"""

from __future__ import annotations

import logging

from backend.app.schemas.diff import CommitInfo
from backend.app.services.git_service import GitService, GitServiceError

logger = logging.getLogger(__name__)


class DiffAnalyzerError(Exception):
    """Raised when analysis fails."""


class DiffAnalyzer:
    """Orchestrates Git diff extraction and parsing."""

    def __init__(self, git_service: GitService | None = None) -> None:
        self.git_service = git_service or GitService()

    def analyze_commit(self, repo_url: str, commit_sha: str) -> CommitInfo:
        """Clone a repo (or use a local path) and analyse a single commit.

        Parameters
        ----------
        repo_url:
            HTTPS URL **or** local filesystem path to the repository.
        commit_sha:
            The commit to analyse.

        Returns
        -------
        CommitInfo
        """
        local_path = self.git_service.clone_repo(repo_url)
        try:
            return self.git_service.get_commit_diff(local_path, commit_sha)
        except GitServiceError as exc:
            raise DiffAnalyzerError(str(exc)) from exc
        finally:
            self.git_service.cleanup(local_path)

    def analyze_pull_request(
        self, repo_url: str, base_branch: str, head_branch: str
    ) -> CommitInfo:
        """Clone a repo and analyse the diff between two branches.

        Parameters
        ----------
        repo_url:
            HTTPS URL **or** local path to the repository.
        base_branch:
            Target / base branch.
        head_branch:
            Source / feature branch.

        Returns
        -------
        CommitInfo
        """
        local_path = self.git_service.clone_repo(repo_url)
        try:
            return self.git_service.get_branch_diff(local_path, base_branch, head_branch)
        except GitServiceError as exc:
            raise DiffAnalyzerError(str(exc)) from exc
        finally:
            self.git_service.cleanup(local_path)

    def analyze_local_commit(self, repo_path: str, commit_sha: str) -> CommitInfo:
        """Analyse a commit in an already-cloned local repository.

        Parameters
        ----------
        repo_path:
            Local path to the repository.
        commit_sha:
            The commit to analyse.

        Returns
        -------
        CommitInfo
        """
        try:
            return self.git_service.get_commit_diff(repo_path, commit_sha)
        except GitServiceError as exc:
            raise DiffAnalyzerError(str(exc)) from exc

    def analyze_local_branch_diff(
        self, repo_path: str, base_branch: str, head_branch: str
    ) -> CommitInfo:
        """Analyse a branch diff in an already-cloned local repository.

        Parameters
        ----------
        repo_path:
            Local path to the repository.
        base_branch:
            Target / base branch.
        head_branch:
            Source / feature branch.

        Returns
        -------
        CommitInfo
        """
        try:
            return self.git_service.get_branch_diff(repo_path, base_branch, head_branch)
        except GitServiceError as exc:
            raise DiffAnalyzerError(str(exc)) from exc
