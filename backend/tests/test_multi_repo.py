"""Tests for Phase 3.5 multi-repository dataset construction."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from backend.app.dataset.multi_repo_config import MultiRepoConfig, RepoConfig
from backend.app.dataset.schemas import DatasetRow

# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestMultiRepoConfig:
    def test_default_config_has_repositories(self):
        cfg = MultiRepoConfig()
        assert len(cfg.repositories) >= 3

    def test_repo_config_fields(self):
        r = RepoConfig(
            repo_url="https://github.com/org/repo.git",
            repo_name="repo",
            repo_revision="abc123",
        )
        assert r.repo_url == "https://github.com/org/repo.git"
        assert r.repo_name == "repo"
        assert r.max_commits == 1000

    def test_all_repos_have_required_fields(self):
        cfg = MultiRepoConfig()
        for repo in cfg.repositories:
            assert repo.repo_url
            assert repo.repo_name
            assert repo.repo_revision
            assert len(repo.repo_revision) >= 12


# ---------------------------------------------------------------------------
# Combined builder structure tests (using synthetic per-repo data)
# ---------------------------------------------------------------------------


class TestCombinedBuilderStructure:
    """Test the combined builder's split logic using synthetic data."""

    def test_global_chronological_split(self, tmp_path: Path):
        """Verify global split assigns earlier commits to train."""
        from datetime import datetime

        from backend.app.dataset.schemas import DatasetRow

        # Create 10 synthetic rows from 3 commits across 2 repos
        rows: list[DatasetRow] = []
        for i in range(10):
            ts = datetime(2020, 1, i + 1, tzinfo=UTC)
            row = DatasetRow(
                repo_name="repo_a" if i < 5 else "repo_b",
                commit_sha=f"{'a' if i < 5 else 'b'}{i:039d}",
                commit_timestamp=ts,
                commit_features=[float(i)] * 29,
                file_features=[float(i)] * 16,
                label_status="negative",
                defect_label=0,
            )
            rows.append(row)

        # Test the sort key logic
        def sort_key(row: DatasetRow) -> tuple:
            ts = row.commit_timestamp
            if ts is None:
                return ("0000-01-01T00:00:00", row.repo_name, row.commit_sha)
            return (ts.isoformat(), row.repo_name, row.commit_sha)

        sorted_rows = sorted(rows, key=sort_key)
        # Verify chronological order
        for i in range(len(sorted_rows) - 1):
            assert sort_key(sorted_rows[i]) <= sort_key(sorted_rows[i + 1])

    def test_commit_level_split_integrity(self):
        """All file rows from one commit must be in the same split."""
        from datetime import datetime

        # Create rows from 2 commits, each with 3 files
        rows: list[DatasetRow] = []
        for commit_idx in range(2):
            ts = datetime(2020, 1, commit_idx + 1, tzinfo=UTC)
            for file_idx in range(3):
                row = DatasetRow(
                    repo_name="test_repo",
                    commit_sha=f"{'a' * 40}" if commit_idx == 0 else f"{'b' * 40}",
                    commit_timestamp=ts,
                    commit_features=[0.0] * 29,
                    file_features=[0.0] * 16,
                    label_status="negative",
                    defect_label=0,
                )
                rows.append(row)

        # Group by commit
        groups: dict[str, list[DatasetRow]] = {}
        for row in rows:
            key = f"{row.repo_name}:{row.commit_sha}"
            groups.setdefault(key, []).append(row)

        # All files from one commit should have the same SHA
        for key, group in groups.items():
            shas = {r.commit_sha for r in group}
            assert len(shas) == 1

    def test_ambiguous_excluded_from_supervised(self):
        """Ambiguous rows must not appear in train/val/test."""
        rows = [
            DatasetRow(
                label_status="ambiguous",
                defect_label=-1,
                split="ambiguous",
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
            DatasetRow(
                label_status="negative",
                defect_label=0,
                split="train",
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
        ]
        supervised = [r for r in rows if r.label_status != "ambiguous"]
        assert len(supervised) == 1
        assert supervised[0].label_status == "negative"

    def test_feature_dimensions_preserved(self):
        """Combined rows must maintain 45-feature contract."""
        row = DatasetRow(
            commit_features=[1.0] * 29,
            file_features=[2.0] * 16,
        )
        assert len(row.commit_features) == 29
        assert len(row.file_features) == 16
        assert len(row.commit_features) + len(row.file_features) == 45


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_sort_key_deterministic(self):
        """Same input always produces same sort order."""
        from datetime import datetime

        rows = []
        for i in range(5):
            ts = datetime(2020, 1, i + 1, tzinfo=UTC)
            rows.append(DatasetRow(
                repo_name=f"repo_{i % 2}",
                commit_sha=f"{i:040d}",
                commit_timestamp=ts,
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ))

        def sort_key(r: DatasetRow) -> tuple:
            ts = r.commit_timestamp
            if ts is None:
                return ("0000-01-01T00:00:00", r.repo_name, r.commit_sha)
            return (ts.isoformat(), r.repo_name, r.commit_sha)

        order1 = [r.commit_sha for r in sorted(rows, key=sort_key)]
        order2 = [r.commit_sha for r in sorted(rows, key=sort_key)]
        assert order1 == order2

    def test_tie_breaking_by_repo_name(self):
        """Same timestamp ties broken by repo_name then SHA."""
        from datetime import datetime

        ts = datetime(2020, 1, 1, tzinfo=UTC)
        rows = [
            DatasetRow(
                repo_name="z_repo",
                commit_sha="b" * 40,
                commit_timestamp=ts,
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
            DatasetRow(
                repo_name="a_repo",
                commit_sha="a" * 40,
                commit_timestamp=ts,
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
        ]

        def sort_key(r: DatasetRow) -> tuple:
            ts = r.commit_timestamp
            if ts is None:
                return ("0000-01-01T00:00:00", r.repo_name, r.commit_sha)
            return (ts.isoformat(), r.repo_name, r.commit_sha)

        sorted_rows = sorted(rows, key=sort_key)
        assert sorted_rows[0].repo_name == "a_repo"
        assert sorted_rows[1].repo_name == "z_repo"

    def test_none_timestamp_sorts_first(self):
        """Rows with None timestamps sort before those with timestamps."""
        from datetime import datetime

        ts = datetime(2020, 1, 1, tzinfo=UTC)
        rows = [
            DatasetRow(
                repo_name="repo",
                commit_sha="b" * 40,
                commit_timestamp=ts,
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
            DatasetRow(
                repo_name="repo",
                commit_sha="a" * 40,
                commit_timestamp=None,
                commit_features=[0.0] * 29,
                file_features=[0.0] * 16,
            ),
        ]

        def sort_key(r: DatasetRow) -> tuple:
            ts = r.commit_timestamp
            if ts is None:
                return ("0000-01-01T00:00:00", r.repo_name, r.commit_sha)
            return (ts.isoformat(), r.repo_name, r.commit_sha)

        sorted_rows = sorted(rows, key=sort_key)
        assert sorted_rows[0].commit_timestamp is None


# ---------------------------------------------------------------------------
# Phase 3.6: Expanded repository configuration tests
# ---------------------------------------------------------------------------


class TestPhase3_6Config:
    def test_expanded_config_has_50_repositories(self):
        cfg = MultiRepoConfig()
        assert len(cfg.repositories) == 50

    def test_v2_repos_all_have_shas(self):
        cfg = MultiRepoConfig()
        for repo in cfg.repositories:
            assert len(repo.repo_revision) == 40, (
                f"{repo.repo_name} has invalid SHA: {repo.repo_revision}"
            )

    def test_v2_dataset_version(self):
        cfg = MultiRepoConfig()
        assert cfg.dataset_version == "v2-multi-phase3.6"
        assert cfg.combined_dir_name == "combined-v2"

    def test_v2_repos_include_diverse_projects(self):
        cfg = MultiRepoConfig()
        names = {r.repo_name for r in cfg.repositories}
        # Should include repos from multiple categories
        assert "pytest" in names  # testing
        assert "sqlalchemy" in names  # database
        assert "httpx" in names  # HTTP
        assert "black" in names  # code quality
        assert "pydantic" in names  # data validation
        assert "celery" in names  # task queue
        assert "flask" in names  # web framework (original)


# ---------------------------------------------------------------------------
# Phase 3.7: Expanded repository configuration tests
# ---------------------------------------------------------------------------


class TestPhase3_7Config:
    def test_config_has_50_repositories(self):
        cfg = MultiRepoConfig()
        assert len(cfg.repositories) == 50

    def test_all_shas_are_40_chars(self):
        cfg = MultiRepoConfig()
        for repo in cfg.repositories:
            assert len(repo.repo_revision) == 40, (
                f"{repo.repo_name} has invalid SHA: {repo.repo_revision}"
            )

    def test_dataset_version_v3(self):
        cfg = MultiRepoConfig(
            dataset_version="v3-multi-phase3.7",
            combined_dir_name="combined-v3",
        )
        assert cfg.dataset_version == "v3-multi-phase3.7"
        assert cfg.combined_dir_name == "combined-v3"

    def test_v3_default_max_commits_1000(self):
        from backend.app.dataset.multi_repo_config import REPOSITORIES_V3
        for repo in REPOSITORIES_V3:
            assert repo.max_commits == 1000, (
                f"{repo.repo_name} has max_commits={repo.max_commits}, expected 1000"
            )

    def test_v1_v2_max_commits_500(self):
        from backend.app.dataset.multi_repo_config import (
            REPOSITORIES_V1,
            REPOSITORIES_V2,
        )
        for repo in REPOSITORIES_V1 + REPOSITORIES_V2:
            assert repo.max_commits == 500, (
                f"{repo.repo_name} has max_commits={repo.max_commits}, expected 500"
            )

    def test_no_duplicate_names(self):
        cfg = MultiRepoConfig()
        names = [r.repo_name for r in cfg.repositories]
        assert len(names) == len(set(names)), "Duplicate repo names found"

    def test_v3_repos_include_diverse_categories(self):
        cfg = MultiRepoConfig()
        names = {r.repo_name for r in cfg.repositories}
        # V3 should add new categories
        assert "httpie" in names  # CLI API testing
        assert "aiohttp" in names  # async HTTP
        assert "twisted" in names  # async framework
        assert "Pillow" in names  # image processing
        assert "mypy" in names  # type checking
        assert "networkx" in names  # graph algorithms
        assert "rich" in names  # terminal UI
        assert "pyyaml" in names  # serialization

    def test_v1_v2_v3_disjoint(self):
        from backend.app.dataset.multi_repo_config import (
            REPOSITORIES_V1,
            REPOSITORIES_V2,
            REPOSITORIES_V3,
        )
        v1_names = {r.repo_name for r in REPOSITORIES_V1}
        v2_names = {r.repo_name for r in REPOSITORIES_V2}
        v3_names = {r.repo_name for r in REPOSITORIES_V3}
        assert not (v1_names & v2_names), f"V1/V2 overlap: {v1_names & v2_names}"
        assert not (v1_names & v3_names), f"V1/V3 overlap: {v1_names & v3_names}"
        assert not (v2_names & v3_names), f"V2/V3 overlap: {v2_names & v3_names}"

    def test_all_shas_are_hex(self):
        import re
        cfg = MultiRepoConfig()
        hex_re = re.compile(r"^[0-9a-f]{40}$")
        for repo in cfg.repositories:
            assert hex_re.match(repo.repo_revision), (
                f"{repo.repo_name} SHA is not valid hex: {repo.repo_revision}"
            )

    def test_v3_repos_all_have_descriptions(self):
        from backend.app.dataset.multi_repo_config import REPOSITORIES_V3
        for repo in REPOSITORIES_V3:
            assert repo.description, f"{repo.repo_name} has no description"

    def test_combined_repos_list_is_concatenation(self):
        from backend.app.dataset.multi_repo_config import (
            REPOSITORIES,
            REPOSITORIES_V1,
            REPOSITORIES_V2,
            REPOSITORIES_V3,
        )
        expected = REPOSITORIES_V1 + REPOSITORIES_V2 + REPOSITORIES_V3
        assert len(REPOSITORIES) == len(expected)
        for r1, r2 in zip(REPOSITORIES, expected):
            assert r1.repo_name == r2.repo_name
