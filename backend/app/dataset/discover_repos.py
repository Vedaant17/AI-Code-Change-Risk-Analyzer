"""Discover valid pinned revisions for candidate repositories.

Clones each repo, finds the HEAD SHA, then cleans up.
This runs once to establish the pinned revisions.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import git

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES = [
    ("https://github.com/pallets/flask.git", "flask"),
    ("https://github.com/pallets/jinja2.git", "jinja2"),
    ("https://github.com/pallets/werkzeug.git", "werkzeug"),
    ("https://github.com/pallets/click.git", "click"),
    ("https://github.com/psf/requests.git", "requests"),
    ("https://github.com/pypa/setuptools.git", "setuptools"),
]

NUM_COMMITS = 500


def main() -> None:
    for url, name in CANDIDATES:
        logger.info("Cloning %s ...", name)
        try:
            with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp_dir:
                repo_path = Path(tmp_dir) / name
                repo = git.Repo.clone_from(url, str(repo_path), depth=1)
                head_sha = repo.head.commit.hexsha

                # Now fetch full history for commit count
                repo.git.fetch("--unshallow")
                commits = list(repo.iter_commits(max_count=NUM_COMMITS))

                print(f"{name}:")
                print(f"  URL: {url}")
                print(f"  HEAD SHA: {head_sha}")
                print(f"  Available commits: {len(commits)}")
                print(f"  Oldest: {commits[-1].hexsha[:12]} ({commits[-1].authored_datetime})")
                print(f"  Newest: {commits[0].hexsha[:12]} ({commits[0].authored_datetime})")
                print()
        except Exception as e:
            logger.error("Failed to clone %s: %s", name, e)
            print(f"{name}: FAILED - {e}")
            print()


if __name__ == "__main__":
    main()
