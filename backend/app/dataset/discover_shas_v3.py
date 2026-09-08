"""Discover and pin HEAD SHAs for Phase 3.7 candidate repositories.

This script clones each candidate repository with --depth=1 to obtain
and verify the HEAD SHA. It does NOT discover positive-producing
repositories or infer positive yield. Actual positive yield is determined
solely by the DatasetBuilder and frozen attribution pipeline.

Output: backend/data/discovered_shas_v3.txt (name|url|sha format)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

OUTPUT_FILE = Path("backend/data/discovered_shas_v3.txt")

# V3 candidate repositories.
# Each entry is (name, https_url, description).
# These have been checked against V1/V2 names to avoid duplicates.
CANDIDATES: list[tuple[str, str, str]] = [
    # HTTP / API testing
    ("httpie", "https://github.com/httpie/cli.git", "CLI for HTTP APIs"),
    # Async HTTP
    ("aiohttp", "https://github.com/aio-libs/aiohttp.git", "Async HTTP client/server"),
    # Database drivers
    ("psycopg2", "https://github.com/psycopg/psycopg2.git", "PostgreSQL adapter"),
    ("pymongo", "https://github.com/mongodb/pymongo.git", "MongoDB driver"),
    # Cache clients
    ("redis", "https://github.com/redis/redis-py.git", "Redis client"),
    ("pymemcache", "https://github.com/pinterest/pymemcache.git", "Memcached client"),
    # Async frameworks
    ("twisted", "https://github.com/twisted/twisted.git", "Event-driven networking engine"),
    ("tornado", "https://github.com/tornadoweb/tornado.git", "Async web framework"),
    # Web frameworks
    ("bottle", "https://github.com/bottlepy/bottle.git", "Minimalist web framework"),
    ("falcon", "https://github.com/falconry/falcon.git", "REST API framework"),
    ("cherrypy", "https://github.com/cherrypy/cherrypy.git", "Object-oriented web framework"),
    # Serialization / Config
    ("pyyaml", "https://github.com/yaml/pyyaml.git", "YAML parser and emitter"),
    # Image processing
    ("Pillow", "https://github.com/python-pillow/Pillow.git", "Image processing library"),
    # Security / Networking
    ("paramiko", "https://github.com/paramiko/paramiko.git", "SSHv2 protocol library"),
    # Dev tools
    ("invoke", "https://github.com/pyinvoke/invoke.git", "Task execution tool"),
    ("fabric", "https://github.com/fabric/fabric.git", "SSH/deployment tool"),
    ("mypy", "https://github.com/python/mypy.git", "Static type checker"),
    ("tox", "https://github.com/tox-dev/tox.git", "Test automation tool"),
    ("nox", "https://github.com/theacodes/nox.git", "Test automation alternative"),
    ("pre-commit", "https://github.com/pre-commit/pre-commit.git", "Git hooks framework"),
    ("bandit", "https://github.com/PyCQA/bandit.git", "Security linter"),
    # Data / Scientific
    ("networkx", "https://github.com/networkx/networkx.git", "Graph algorithms"),
    ("sympy", "https://github.com/sympy/sympy.git", "Symbolic mathematics"),
    # Terminal UI
    ("rich", "https://github.com/Textualize/rich.git", "Rich terminal output"),
]

HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def clone_and_get_sha(name: str, url: str) -> str | None:
    """Clone a repository with --depth=1 and return its HEAD SHA.

    Returns None if cloning or SHA retrieval fails.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"discover_v3_{name}_")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, tmpdir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  FAIL clone {name}: {result.stderr.strip()[:200]}")
            return None

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  FAIL rev-parse {name}: {result.stderr.strip()[:200]}")
            return None

        sha = result.stdout.strip()
        if not HEX_SHA_RE.match(sha):
            print(f"  FAIL invalid SHA for {name}: {sha}")
            return None

        return sha

    except subprocess.TimeoutExpired:
        print(f"  FAIL timeout {name}")
        return None
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    """Discover and verify HEAD SHAs for all V3 candidates."""
    print("=" * 60)
    print("Phase 3.7 SHA Discovery (V3 Candidates)")
    print("=" * 60)
    print(f"\nCandidates: {len(CANDIDATES)}")

    results: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str, str]] = []

    for name, url, desc in CANDIDATES:
        print(f"\n  {name} ({url})")
        sha = clone_and_get_sha(name, url)
        if sha:
            print(f"    SHA: {sha}")
            results.append((name, url, sha))
        else:
            failures.append((name, url, desc))

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name, url, sha in results:
            f.write(f"{name}|{url}|{sha}\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successfully pinned: {len(results)}")
    print(f"  Failed:              {len(failures)}")

    if failures:
        print("\n  Failed repositories:")
        for name, url, desc in failures:
            print(f"    {name}: {desc} ({url})")

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
