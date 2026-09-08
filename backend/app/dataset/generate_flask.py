"""Generate the real Phase 3 dataset from the pinned Flask revision.

This script:
1. Clones Flask at the pinned revision d318b683471101618febed18996405ad26462110
2. Collects the last 500 ancestors
3. Builds CommitInfo for each
4. Runs DatasetBuilder
5. Persists output to backend/data/datasets/flask/
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import UTC
from pathlib import Path

import git

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

from backend.app.dataset.builder import DatasetBuilder  # noqa: E402
from backend.app.dataset.config import DatasetConfig  # noqa: E402
from backend.app.schemas.diff import CommitInfo  # noqa: E402
from backend.app.services.diff_parser import parse_unified_diff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PINNED_SHA = "d318b683471101618febed18996405ad26462110"
REPO_URL = "https://github.com/pallets/flask.git"
NUM_COMMITS = 500
OUTPUT_DIR = project_root / "backend" / "data" / "datasets" / "flask"


def build_commit_info(commit: git.Commit) -> CommitInfo:
    """Build a CommitInfo from a git.Commit object."""
    parent = commit.parents[0] if commit.parents else None

    if parent:
        raw_diff_text: str = commit.repo.git.diff(parent.hexsha, commit.hexsha)
    else:
        raw_diff_text = commit.repo.git.diff("--root", commit.hexsha)

    file_diffs = parse_unified_diff(raw_diff_text)

    from datetime import datetime

    author_date: datetime | None = None
    try:
        author_date = datetime.fromtimestamp(
            commit.committed_date, tz=UTC
        )
    except (OSError, ValueError):
        pass

    info = CommitInfo(
        sha=commit.hexsha,
        short_sha=commit.hexsha[:8],
        author=str(commit.author),
        author_date=author_date,
        message=commit.message.strip(),
        files=file_diffs,
    )
    info.compute_stats()
    return info


def main() -> None:
    """Clone Flask and generate the dataset."""
    logger.info("Cloning Flask at %s ...", PINNED_SHA[:12])

    with tempfile.TemporaryDirectory(prefix="flask_") as tmp_dir:
        repo_path = Path(tmp_dir) / "flask"
        git.Repo.clone_from(REPO_URL, str(repo_path))
        repo = git.Repo(str(repo_path))

        # Checkout pinned revision
        repo.git.checkout(PINNED_SHA)
        logger.info("Checked out %s", repo.head.commit.hexsha[:12])

        # Collect last 500 ancestor commits (chronological order)
        logger.info("Collecting last %d commits ...", NUM_COMMITS)
        commits_iter = repo.iter_commits(rev=PINNED_SHA, max_count=NUM_COMMITS)
        git_commits = list(commits_iter)
        # Reverse to chronological order (oldest first)
        git_commits.reverse()
        logger.info("Collected %d commits", len(git_commits))

        # Build CommitInfo objects
        logger.info("Building CommitInfo objects ...")
        commit_infos: list[CommitInfo] = []
        for i, gc in enumerate(git_commits):
            try:
                info = build_commit_info(gc)
                commit_infos.append(info)
                if (i + 1) % 100 == 0:
                    logger.info("  %d/%d commits processed", i + 1, len(git_commits))
            except Exception as e:
                logger.warning("Failed to process commit %s: %s", gc.hexsha[:12], e)
                continue

        logger.info("Built %d CommitInfo objects", len(commit_infos))

        # Build dataset
        config = DatasetConfig()
        builder = DatasetBuilder(config=config)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        manifest = builder.build(
            commits=commit_infos,
            output_dir=OUTPUT_DIR,
            repo_url=REPO_URL,
            repo_name="flask",
            repo_revision=PINNED_SHA,
        )

        logger.info("Dataset built successfully!")
        logger.info("  Total commits: %d", manifest.total_commits_processed)
        logger.info("  Total file examples: %d", manifest.total_file_examples)
        logger.info("  Positive: %d", manifest.positive_examples)
        logger.info("  Negative: %d", manifest.negative_examples)
        logger.info("  Ambiguous: %d", manifest.ambiguous_examples)
        logger.info("  Train: %d examples", manifest.train_examples)
        logger.info("  Validation: %d examples", manifest.validation_examples)
        logger.info("  Test: %d examples", manifest.test_examples)
        logger.info("  Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
