"""Multi-repository configuration for Phase 3.7 dataset expansion.

Extends the Phase 3.6 repository set with additional repositories
selected for diverse project types and independent commit histories.

All SHAs pinned at discovery time for reproducibility.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RepoConfig(BaseModel):
    """Configuration for a single repository in the combined dataset."""

    repo_url: str = Field(description="HTTPS clone URL")
    repo_name: str = Field(description="Short name for directory/manifest")
    repo_revision: str = Field(description="Pinned HEAD SHA")
    max_commits: int = Field(
        default=1000,
        description="Max ancestor commits to collect from the pinned revision",
    )
    description: str = Field(default="", description="Human-readable description")


# ── Phase 3.5 original repositories ─────────────────────────────────────
REPOSITORIES_V1: list[RepoConfig] = [
    RepoConfig(
        repo_url="https://github.com/pallets/flask.git",
        repo_name="flask",
        repo_revision="d318b683471101618febed18996405ad26462110",
        max_commits=500,
        description="Flask micro web framework",
    ),
    RepoConfig(
        repo_url="https://github.com/pallets/jinja2.git",
        repo_name="jinja2",
        repo_revision="5ef70112a1ff19c05324ff889dd30405b1002044",
        max_commits=500,
        description="Template engine for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/pallets/werkzeug.git",
        repo_name="werkzeug",
        repo_revision="bb2c10a1db220815ddea9fdec05dd561ae009731",
        max_commits=500,
        description="WSGI utility library for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/psf/requests.git",
        repo_name="requests",
        repo_revision="dae7ef63b4df6eded86637f251fc4e3a06c3b479",
        max_commits=500,
        description="HTTP library for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/pallets/click.git",
        repo_name="click",
        repo_revision="6aabf099bfdd4c1e75fe8d0e0d4241372b988ab1",
        max_commits=500,
        description="Command-line interface toolkit",
    ),
]

# ── Phase 3.6 additional repositories ───────────────────────────────────
REPOSITORIES_V2: list[RepoConfig] = [
    RepoConfig(
        repo_url="https://github.com/pytest-dev/pytest.git",
        repo_name="pytest",
        repo_revision="431f3e1f5fd70b9b0f8afa2d20a10421542e5c6a",
        max_commits=500,
        description="Testing framework for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/encode/httpx.git",
        repo_name="httpx",
        repo_revision="b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
        max_commits=500,
        description="Async HTTP client for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/urllib3/urllib3.git",
        repo_name="urllib3",
        repo_revision="278d98d7bf6cbafbccc924e7aa73dbbe06474dc2",
        max_commits=500,
        description="HTTP client library for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/encode/starlette.git",
        repo_name="starlette",
        repo_revision="41db6a707f2636526847dbda0e5610e732e6b4fc",
        max_commits=500,
        description="Lightweight ASGI framework",
    ),
    RepoConfig(
        repo_url="https://github.com/tiangolo/fastapi.git",
        repo_name="fastapi",
        repo_revision="50113da16fec53b66b80d75e80a89296de4fa5a5",
        max_commits=500,
        description="Modern web framework for building APIs",
    ),
    RepoConfig(
        repo_url="https://github.com/encode/uvicorn.git",
        repo_name="uvicorn",
        repo_revision="eed8e7212a60681a9c3c865305cdd227a2b16f90",
        max_commits=500,
        description="Lightning-fast ASGI server",
    ),
    RepoConfig(
        repo_url="https://github.com/sqlalchemy/sqlalchemy.git",
        repo_name="sqlalchemy",
        repo_revision="de83fa72d787136785624fd8981e1d71e9f427ab",
        max_commits=500,
        description="Database toolkit and ORM for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/coleifer/peewee.git",
        repo_name="peewee",
        repo_revision="c6328d6c91120b3d04878921a0af64a9ee78c866",
        max_commits=500,
        description="Small, expressive ORM for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/psf/black.git",
        repo_name="black",
        repo_revision="20622e1259c29bda81831962ace1348ba1921c84",
        max_commits=500,
        description="Uncompromising Python code formatter",
    ),
    RepoConfig(
        repo_url="https://github.com/PyCQA/isort.git",
        repo_name="isort",
        repo_revision="131f4adcd5582bfc53928ab0d740eceb8b506b6c",
        max_commits=500,
        description="Python import sorter",
    ),
    RepoConfig(
        repo_url="https://github.com/PyCQA/pycodestyle.git",
        repo_name="pycodestyle",
        repo_revision="d6c38543a95c3adec7958fc659f4081972a3f1d1",
        max_commits=500,
        description="Python style checker (PEP 8)",
    ),
    RepoConfig(
        repo_url="https://github.com/PyCQA/flake8.git",
        repo_name="flake8",
        repo_revision="efe6750405be77b661f64e60002c4ef8e922daf5",
        max_commits=500,
        description="Python linter",
    ),
    RepoConfig(
        repo_url="https://github.com/pydantic/pydantic.git",
        repo_name="pydantic",
        repo_revision="c23cb86ef197693fc016437614f174252a3d189a",
        max_commits=500,
        description="Data validation using Python type annotations",
    ),
    RepoConfig(
        repo_url="https://github.com/marshmallow-code/marshmallow.git",
        repo_name="marshmallow",
        repo_revision="74e0043254d2390afe7f8ac83162bb99fd0aafe1",
        max_commits=500,
        description="Object serialization/deserialization library",
    ),
    RepoConfig(
        repo_url="https://github.com/celery/celery.git",
        repo_name="celery",
        repo_revision="07ee451b5313b520ed4f57cfe93d35ac81d0548c",
        max_commits=500,
        description="Distributed task queue",
    ),
    RepoConfig(
        repo_url="https://github.com/pallets/itsdangerous.git",
        repo_name="itsdangerous",
        repo_revision="672971d66a2ef9f85151e53283113f33d642dabd",
        max_commits=500,
        description="Data signing for Python",
    ),
    RepoConfig(
        repo_url="https://github.com/twisted/treq.git",
        repo_name="treq",
        repo_revision="99f17121ab7e42fa4f4d12578e45cf3a26e118d9",
        max_commits=500,
        description="HTTP client for Twisted",
    ),
    RepoConfig(
        repo_url="https://github.com/mitsuhiko/flask-sqlalchemy.git",
        repo_name="flask-sqlalchemy",
        repo_revision="80ac6ef26bd12a6b5472a494d541a7d2442a38a4",
        max_commits=500,
        description="Flask SQLAlchemy integration",
    ),
    RepoConfig(
        repo_url="https://github.com/pallets-eco/flask-wtf.git",
        repo_name="flask-wtf",
        repo_revision="63e2d71269d7568307799cf43d1ee11f2d859b98",
        max_commits=500,
        description="Flask WTForms integration",
    ),
    RepoConfig(
        repo_url="https://github.com/miguelgrinberg/flask-migrate.git",
        repo_name="flask-migrate",
        repo_revision="bce4d35a0c61d2516c76744a1f168bc6af26d55e",
        max_commits=500,
        description="Flask database migration support",
    ),
    RepoConfig(
        repo_url="https://github.com/jarus/flask-testing.git",
        repo_name="flask-testing",
        repo_revision="5107691011fa891835c01547e73e991c484fa07f",
        max_commits=500,
        description="Flask testing utilities",
    ),
]

# ── Phase 3.7 additional repositories ───────────────────────────────────
# Selected for: diverse project types, independent commit histories,
# mature codebases, Python-primary. No duplicates of V1/V2.
REPOSITORIES_V3: list[RepoConfig] = [
    # HTTP / API testing
    RepoConfig(
        repo_url="https://github.com/httpie/cli.git",
        repo_name="httpie",
        repo_revision="5b604c37c6c67e18e7c3e9aee6c88a8c22b98345",
        description="CLI for HTTP APIs",
    ),
    # Async HTTP
    RepoConfig(
        repo_url="https://github.com/aio-libs/aiohttp.git",
        repo_name="aiohttp",
        repo_revision="a5fba5b4d3ce962db5f4c40865fdb270170d7840",
        description="Async HTTP client/server",
    ),
    # Database drivers
    RepoConfig(
        repo_url="https://github.com/psycopg/psycopg2.git",
        repo_name="psycopg2",
        repo_revision="3a6d9d6ddc6b53eaa80b712f5fa6b23abbdc38db",
        description="PostgreSQL adapter",
    ),
    RepoConfig(
        repo_url="https://github.com/mongodb/mongo-python-driver.git",
        repo_name="pymongo",
        repo_revision="4dd73032d98146fd97bedfb8434bff2f458bc26f",
        description="MongoDB driver",
    ),
    # Cache clients
    RepoConfig(
        repo_url="https://github.com/redis/redis-py.git",
        repo_name="redis",
        repo_revision="c73fdd92c955b9d19fece2def1f8f2bd3de80c27",
        description="Redis client",
    ),
    RepoConfig(
        repo_url="https://github.com/pinterest/pymemcache.git",
        repo_name="pymemcache",
        repo_revision="3ef28efac1a6cca5fb47c88981b7df4b33f1aac5",
        description="Memcached client",
    ),
    # Async frameworks
    RepoConfig(
        repo_url="https://github.com/twisted/twisted.git",
        repo_name="twisted",
        repo_revision="ac919a6d1b2f91b7dea806bacfae99abab5291d4",
        description="Event-driven networking engine",
    ),
    RepoConfig(
        repo_url="https://github.com/tornadoweb/tornado.git",
        repo_name="tornado",
        repo_revision="0096f2897c98facdcd9716009ee934a7381af5ef",
        description="Async web framework",
    ),
    # Web frameworks
    RepoConfig(
        repo_url="https://github.com/bottlepy/bottle.git",
        repo_name="bottle",
        repo_revision="457a8fa82f2c96d18c6f5934288387ad83ac12fa",
        description="Minimalist web framework",
    ),
    RepoConfig(
        repo_url="https://github.com/falconry/falcon.git",
        repo_name="falcon",
        repo_revision="eb27592b159fea6678b3c70b16f195b0f990faa9",
        description="REST API framework",
    ),
    RepoConfig(
        repo_url="https://github.com/cherrypy/cherrypy.git",
        repo_name="cherrypy",
        repo_revision="1f75bc9eed8e0e385f64f368bd69f58d96fb8c2b",
        description="Object-oriented web framework",
    ),
    # Serialization / Config
    RepoConfig(
        repo_url="https://github.com/yaml/pyyaml.git",
        repo_name="pyyaml",
        repo_revision="34a9bf82357f4952d8f194a5a31f1c39743652d0",
        description="YAML parser and emitter",
    ),
    # Image processing
    RepoConfig(
        repo_url="https://github.com/python-pillow/Pillow.git",
        repo_name="Pillow",
        repo_revision="5b262f5d42509158ef520a7126cc450e4fb57ce2",
        description="Image processing library",
    ),
    # Security / Networking
    RepoConfig(
        repo_url="https://github.com/paramiko/paramiko.git",
        repo_name="paramiko",
        repo_revision="142f593e40ad767c5e3556cbace66dc84589620c",
        description="SSHv2 protocol library",
    ),
    # Dev tools
    RepoConfig(
        repo_url="https://github.com/pyinvoke/invoke.git",
        repo_name="invoke",
        repo_revision="6a71e680c535ba6520e935c497099fbca011d03c",
        description="Task execution tool",
    ),
    RepoConfig(
        repo_url="https://github.com/fabric/fabric.git",
        repo_name="fabric",
        repo_revision="ded51893f02c33d2bc7c157624c44a039a952037",
        description="SSH/deployment tool",
    ),
    RepoConfig(
        repo_url="https://github.com/python/mypy.git",
        repo_name="mypy",
        repo_revision="0ff707d475eb967d1709409472dfe45644a8e6d3",
        description="Static type checker",
    ),
    RepoConfig(
        repo_url="https://github.com/tox-dev/tox.git",
        repo_name="tox",
        repo_revision="6485a013e46b3cfe8f43002f6aa80e6e390efbfe",
        description="Test automation tool",
    ),
    RepoConfig(
        repo_url="https://github.com/theacodes/nox.git",
        repo_name="nox",
        repo_revision="a525fb69765e172bfecfe50aaacfeec7f224bbb6",
        description="Test automation alternative",
    ),
    RepoConfig(
        repo_url="https://github.com/pre-commit/pre-commit.git",
        repo_name="pre-commit",
        repo_revision="a9bba55a3f74068b53f4bd4d831d7e05e34eae6c",
        description="Git hooks framework",
    ),
    RepoConfig(
        repo_url="https://github.com/PyCQA/bandit.git",
        repo_name="bandit",
        repo_revision="1d3053df070c91fe0fde002a21536c277d67e5d9",
        description="Security linter",
    ),
    # Data / Scientific
    RepoConfig(
        repo_url="https://github.com/networkx/networkx.git",
        repo_name="networkx",
        repo_revision="0db8227000872d7a9f6ce84c54ba1e5e99429122",
        description="Graph algorithms",
    ),
    RepoConfig(
        repo_url="https://github.com/sympy/sympy.git",
        repo_name="sympy",
        repo_revision="ad538b41ffe5316fc13464920f3f8c823573d12a",
        description="Symbolic mathematics",
    ),
    # Terminal UI
    RepoConfig(
        repo_url="https://github.com/Textualize/rich.git",
        repo_name="rich",
        repo_revision="9d8f9a372cc5916fd4781fec207ced7ddac2f08f",
        description="Rich terminal output",
    ),
]

# ── Combined repository set for Phase 3.7 ───────────────────────────────
REPOSITORIES: list[RepoConfig] = REPOSITORIES_V1 + REPOSITORIES_V2 + REPOSITORIES_V3


class MultiRepoConfig(BaseModel):
    """Configuration for the combined multi-repository dataset."""

    repositories: list[RepoConfig] = Field(default_factory=lambda: REPOSITORIES)
    dataset_version: str = Field(default="v2-multi-phase3.6")
    combined_dir_name: str = Field(default="combined-v2")
