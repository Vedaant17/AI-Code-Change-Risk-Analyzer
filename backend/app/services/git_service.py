"""Git repository operations service (Phase 1).

Handles cloning, diff extraction, and commit/PR inspection using GitPython.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone

import git

from backend.app.schemas.diff import CommitInfo, FileDiff
from backend.app.services.diff_parser import parse_unified_diff

logger = logging.getLogger(__name__)


class GitServiceError(Exception):
    """Raised when a Git operation fails."""


class GitService:
    """Provides Git repository access and diff extraction."""

    def __init__(self, github_token: str | None = None) -> None:
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    def _make_clone_url(self, repo_url: str) -> str:
        """Inject GitHub token into HTTPS URL for private repo access."""
        if self.github_token and "github.com" in repo_url:
            return repo_url.replace("https://", f"https://x-access-token:{self.github_token}@")
        return repo_url

    def clone_repo(self, repo_url: str, dest_dir: str | None = None) -> str:
        """Clone a remote repository and return the local path.

        Parameters
        ----------
        repo_url:
            HTTPS URL of the repository.
        dest_dir:
            Optional parent directory.  A subdirectory is created inside it.

        Returns
        -------
        str
            Absolute path to the cloned repository.
        """
        if dest_dir is None:
            dest_dir = tempfile.mkdtemp(prefix="aira_")

        repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        local_path = os.path.join(dest_dir, repo_name)

        clone_url = self._make_clone_url(repo_url)

        try:
            logger.info("Cloning %s -> %s", repo_url, local_path)
            git.Repo.clone_from(clone_url, local_path)
        except git.GitCommandError as exc:
            raise GitServiceError(f"Failed to clone {repo_url}: {exc}") from exc

        return local_path

    def open_repo(self, local_path: str) -> git.Repo:
        """Open an existing local repository."""
        try:
            return git.Repo(local_path)
        except (git.InvalidGitRepositoryError, git.NoSuchPathError) as exc:
            raise GitServiceError(f"Cannot open repo at {local_path}: {exc}") from exc

    def get_commit_diff(self, repo_path: str, commit_sha: str) -> CommitInfo:
        """Extract structured diff for a single commit.

        Parameters
        ----------
        repo_path:
            Local path to a cloned repository.
        commit_sha:
            Full or abbreviated SHA of the commit.

        Returns
        -------
        CommitInfo
            Structured commit information including per-file diffs.
        """
        repo = self.open_repo(repo_path)

        try:
            commit = repo.commit(commit_sha)
        except git.BadName as exc:
            raise GitServiceError(f"Unknown commit: {commit_sha}") from exc

        parent = commit.parents[0] if commit.parents else None

        # Use repo.git.diff to get raw unified diff text.
        if parent:
            raw_diff_text: str = repo.git.diff(parent.hexsha, commit.hexsha)
        else:
            # Root commit — diff against empty tree
            raw_diff_text = repo.git.diff("--root", commit.hexsha)

        file_diffs = parse_unified_diff(raw_diff_text)

        # Parse author date
        author_date: datetime | None = None
        try:
            author_date = datetime.fromtimestamp(
                commit.committed_date, tz=timezone.utc
            )
        except (OSError, ValueError):
            pass

        commit_info = CommitInfo(
            sha=commit.hexsha,
            short_sha=commit.hexsha[:8],
            author=str(commit.author),
            author_date=author_date,
            message=commit.message.strip(),
            files=file_diffs,
            repo_url=repo_path,
            base_ref=parent.hexsha if parent else "",
        )
        commit_info.compute_stats()
        return commit_info

    def get_branch_diff(
        self, repo_path: str, base_branch: str, head_branch: str
    ) -> CommitInfo:
        """Extract the diff between two branches (simulates a PR diff).

        Parameters
        ----------
        repo_path:
            Local path to a cloned repository.
        base_branch:
            The target / base branch name.
        head_branch:
            The source / feature branch name.

        Returns
        -------
        CommitInfo
            Structured diff between the two branches.
        """
        repo = self.open_repo(repo_path)

        try:
            base = repo.commit(base_branch)
            head = repo.commit(head_branch)
        except git.BadName as exc:
            raise GitServiceError(f"Invalid branch ref: {exc}") from exc

        raw_diff_text: str = repo.git.diff(base.hexsha, head.hexsha)
        file_diffs = parse_unified_diff(raw_diff_text)

        commit_info = CommitInfo(
            sha=head.hexsha,
            short_sha=head.hexsha[:8],
            author=str(head.author),
            author_date=datetime.fromtimestamp(
                head.committed_date, tz=timezone.utc
            )
            if head.committed_date
            else None,
            message=f"Diff {base_branch}..{head_branch}",
            files=file_diffs,
            repo_url=repo_path,
            base_ref=base.hexsha,
        )
        commit_info.compute_stats()
        return commit_info

    def cleanup(self, repo_path: str) -> None:
        """Remove a cloned repository from disk."""
        if os.path.isdir(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
            logger.info("Cleaned up %s", repo_path)
