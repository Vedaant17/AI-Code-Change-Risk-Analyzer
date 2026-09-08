"""Discover HEAD SHAs for candidate repositories (no cleanup needed)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import git

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES = [
    ("https://github.com/pytest-dev/pytest.git", "pytest"),
    ("https://github.com/encode/httpx.git", "httpx"),
    ("https://github.com/encode/starlette.git", "starlette"),
    ("https://github.com/tiangolo/fastapi.git", "fastapi"),
    ("https://github.com/sqlalchemy/sqlalchemy.git", "sqlalchemy"),
    ("https://github.com/psf/black.git", "black"),
    ("https://github.com/celery/celery.git", "celery"),
    ("https://github.com/urllib3/urllib3.git", "urllib3"),
    ("https://github.com/marshmallow-code/marshmallow.git", "marshmallow"),
    ("https://github.com/pydantic/pydantic.git", "pydantic"),
    ("https://github.com/encode/uvicorn.git", "uvicorn"),
    ("https://github.com/pallets/itsdangerous.git", "itsdangerous"),
    ("https://github.com/mitsuhiko/flask-sqlalchemy.git", "flask-sqlalchemy"),
    ("https://github.com/PyCQA/isort.git", "isort"),
    ("https://github.com/PyCQA/pycodestyle.git", "pycodestyle"),
    ("https://github.com/PyCQA/flake8.git", "flake8"),
    ("https://github.com/twisted/treq.git", "treq"),
    ("https://github.com/coleifer/peewee.git", "peewee"),
    ("https://github.com/pallets-eco/flask-wtf.git", "flask-wtf"),
    ("https://github.com/miguelgrinberg/flask-migrate.git", "flask-migrate"),
    ("https://github.com/jarus/flask-testing.git", "flask-testing"),
]

# Use a fixed output file instead of context manager
OUTPUT = Path("backend/data/discovered_shas.txt")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for url, name in CANDIDATES:
        clone_dir = tempfile.mkdtemp(prefix=f"{name}_sha_")
        try:
            repo_path = Path(clone_dir) / name
            repo = git.Repo.clone_from(url, str(repo_path), depth=1)
            sha = repo.head.commit.hexsha
            lines.append(f"{name}|{url}|{sha}")
            print(f"OK  {name}: {sha}")
        except Exception as e:
            print(f"ERR {name}: {e}")
        finally:
            try:
                import shutil
                shutil.rmtree(clone_dir, ignore_errors=True)
            except Exception:
                pass

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to {OUTPUT}")


if __name__ == "__main__":
    main()
