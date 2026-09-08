"""Discover candidate repositories for Phase 3.6 expansion.

Clones each candidate repo, runs the existing attribution system,
and reports positive counts to select the best repositories.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

import git  # noqa: E402

from backend.app.dataset.config import DatasetConfig  # noqa: E402
from backend.app.dataset.labeling import DefectLabeler  # noqa: E402
from backend.app.schemas.diff import CommitInfo  # noqa: E402
from backend.app.services.diff_parser import parse_unified_diff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES = [
    # Web frameworks
    ("https://github.com/encode/django-rest-framework.git", "drf", "Django REST framework"),
    ("https://github.com/encode/httpx.git", "httpx", "Async HTTP client"),
    ("https://github.com/encode/starlette.git", "starlette", "ASGI framework"),
    ("https://github.com/tiangolo/fastapi.git", "fastapi", "Modern web framework"),
    # Testing
    ("https://github.com/pytest-dev/pytest.git", "pytest", "Testing framework"),
    ("https://github.com/pytest-dev/pytest-xdist.git", "pytest-xdist", "Pytest parallel testing"),
    # Database/ORM
    ("https://github.com/sqlalchemy/sqlalchemy.git", "sqlalchemy", "Database toolkit"),
    ("https://github.com/coleifer/peewee.git", "peewee", "Lightweight ORM"),
    # CLI/Dev tools
    ("https://github.com/psf/black.git", "black", "Code formatter"),
    ("https://github.com/pydantic/pydantic.git", "pydantic", "Data validation"),
    ("https://github.com/astral-sh/ruff.git", "ruff", "Python linter"),
    # Task queues / Async
    ("https://github.com/celery/celery.git", "celery", "Task queue"),
    ("https://github.com/encode/uvicorn.git", "uvicorn", "ASGI server"),
    # HTTP/Networking
    ("https://github.com/urllib3/urllib3.git", "urllib3", "HTTP client library"),
    # Data / Serialization
    ("https://github.com/marshmallow-code/marshmallow.git", "marshmallow", "Serialization library"),
    ("https://github.com/pallets/itsdangerous.git", "itsdangerous", "Data signing"),
    # Configuration
    ("https://github.com/toml-lang/toml.git", "toml", "TOML parser"),
    ("https://github.com/twisted/treq.git", "treq", "HTTP client for Twisted"),
    # Misc
    ("https://github.com/pallets-eco/flask-login.git", "flask-login", "Flask login management"),
    ("https://github.com/mitsuhiko/flask-sqlalchemy.git", "flask-sqlalchemy", "Flask SQLAlchemy"),
    ("https://github.com/pallets-eco/werkzeug.git", "werkzeug-eco", "WSGI utility (eco fork)"),
    ("https://github.com/PyCQA/isort.git", "isort", "Import sorter"),
    ("https://github.com/PyCQA/pycodestyle.git", "pycodestyle", "Python style checker"),
    ("https://github.com/PyCQA/flake8.git", "flake8", "Python linter"),
    ("https://github.com/python/cpython.git", "cpython", "Python reference implementation"),
]

NUM_COMMITS = 500


def build_commit_infos(git_commits: list, repo: git.Repo) -> list[CommitInfo]:
    """Build CommitInfo objects from git commits."""
    from datetime import UTC, datetime

    infos: list[CommitInfo] = []
    for gc in git_commits:
        try:
            parent = gc.parents[0] if gc.parents else None
            if parent:
                raw_diff = repo.git.diff(parent.hexsha, gc.hexsha)
            else:
                raw_diff = repo.git.diff("--root", gc.hexsha)

            file_diffs = parse_unified_diff(raw_diff)

            author_date = None
            try:
                author_date = datetime.fromtimestamp(gc.committed_date, tz=UTC)
            except (OSError, ValueError):
                pass

            info = CommitInfo(
                sha=gc.hexsha,
                short_sha=gc.hexsha[:8],
                author=str(gc.author),
                author_date=author_date,
                message=gc.message.strip(),
                files=file_diffs,
            )
            info.compute_stats()
            infos.append(info)
        except Exception as e:
            logger.warning("  Failed commit %s: %s", gc.hexsha[:12], e)
            continue
    return infos


def analyze_repo(url: str, name: str, description: str) -> dict | None:
    """Clone a repo, run attribution, return stats."""
    logger.info("Analyzing %s ...", name)
    try:
        with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp_dir:
            repo_path = Path(tmp_dir) / name
            repo = git.Repo.clone_from(url, str(repo_path))

            # Ensure full history
            try:
                repo.git.fetch("--unshallow")
            except git.GitCommandError:
                pass

            # Collect commits
            commits_iter = repo.iter_commits(max_count=NUM_COMMITS)
            git_commits = list(commits_iter)
            git_commits.reverse()

            if len(git_commits) < 50:
                logger.warning("  %s: Only %d commits, skipping", name, len(git_commits))
                return None

            head_sha = git_commits[-1].hexsha if git_commits else "unknown"

            # Build CommitInfo objects
            commit_infos = build_commit_infos(git_commits, repo)
            if not commit_infos:
                return None

            # Run attribution
            config = DatasetConfig()
            labeler = DefectLabeler(config)
            bug_fix_shas = labeler.identify_bug_fixes(commit_infos)
            revert_map = labeler.identify_reverts(commit_infos)
            results = labeler.attribute_defects(
                commit_infos, bug_fix_shas, revert_map, repo
            )

            # Count stats
            pos_count = sum(1 for v in results.values() if v[0] == "positive")
            neg_count = sum(1 for v in results.values() if v[0] == "negative")
            amb_count = sum(1 for v in results.values() if v[0] == "ambiguous")

            # Count attribution sources
            explicit_sha = sum(1 for v in results.values() if v[3] == "explicit_sha_reference")
            revert = sum(1 for v in results.values() if v[3] == "revert")
            line_overlap = sum(1 for v in results.values() if v[3] == "line_overlap")

            # Count positive files (commit-level positive * avg files per commit)
            pos_files = 0
            for sha, status in results.items():
                if status[0] == "positive":
                    for ci in commit_infos:
                        if ci.sha == sha:
                            pos_files += len(ci.files)
                            break

            return {
                "name": name,
                "url": url,
                "description": description,
                "head_sha": head_sha,
                "commits_analyzed": len(commit_infos),
                "bug_fixes": len(bug_fix_shas),
                "reverts": len(revert_map),
                "positive_commits": pos_count,
                "negative_commits": neg_count,
                "ambiguous_commits": amb_count,
                "positive_files": pos_files,
                "explicit_sha": explicit_sha,
                "revert_attributions": revert,
                "line_overlap_attributions": line_overlap,
            }
    except Exception as e:
        logger.error("  Failed to analyze %s: %s", name, e)
        return None


def main() -> None:
    """Analyze all candidate repositories."""
    print("=" * 80)
    print("PHASE 3.6: Repository Discovery")
    print("=" * 80)

    results = []
    for url, name, desc in CANDIDATES:
        stats = analyze_repo(url, name, desc)
        if stats:
            results.append(stats)
            print(f"\n  {name}:")
            print(f"    HEAD SHA:     {stats['head_sha'][:12]}...")
            print(f"    Commits:      {stats['commits_analyzed']}")
            print(f"    Bug-fixes:    {stats['bug_fixes']}")
            print(f"    Reverts:      {stats['reverts']}")
            print(f"    Pos commits:  {stats['positive_commits']}")
            print(f"    Pos files:    {stats['positive_files']}")
            print(f"    Explicit SHA: {stats['explicit_sha']}")
            print(f"    Revert attr:  {stats['revert_attributions']}")
            print(f"    Line overlap: {stats['line_overlap_attributions']}")
        else:
            print(f"\n  {name}: SKIPPED (insufficient commits or error)")

    # Sort by positive commits (descending), then positive files
    results.sort(key=lambda r: (r["positive_commits"], r["positive_files"]), reverse=True)

    print("\n" + "=" * 80)
    print("RANKED BY POSITIVE COMMITS")
    print("=" * 80)
    header = (f"\n{'Name':<25} {'Pos Comms':>10} {'Pos Files':>10} "
              f"{'SHA':>6} {'Rev':>5} {'LO':>5} {'Commits':>8}")
    print(header)
    print("-" * 80)
    for r in results:
        print(f"  {r['name']:<25} {r['positive_commits']:>8} {r['positive_files']:>10} "
              f"{r['explicit_sha']:>6} {r['revert_attributions']:>5} "
              f"{r['line_overlap_attributions']:>5} {r['commits_analyzed']:>8}")

    # Summary
    total_pos = sum(r["positive_commits"] for r in results)
    total_files = sum(r["positive_files"] for r in results)
    print(f"\n  Total positive commits across all repos: {total_pos}")
    print(f"  Total positive files across all repos:   {total_files}")
    print("=" * 80)


if __name__ == "__main__":
    main()
