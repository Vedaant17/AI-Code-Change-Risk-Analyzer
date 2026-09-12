"""Tests for Phase 4.6 unseen-repository generalization and ranking evaluation.

Verifies:
- Repo manifest completeness and non-overlap
- Split reassignment correctness
- Feature dimensions for variants A/B/C
- Per-commit ranking correctness
- Determinism
- No commit/repo leakage
- Global metrics match sklearn
- Output contract
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.ml.evaluation import (
    dominance_check,
    global_classification_metrics,
    macro_average_per_repo,
    per_commit_ranking,
    per_repo_metrics,
)
from backend.app.ml.repo_split import (
    REPO_MANIFEST,
    TEST_REPOS,
    TRAIN_REPOS,
    VALIDATION_REPOS,
    build_repo_split_dataset,
    load_all_rows,
    reassign_splits,
)

V3_DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DATA_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")


class TestRepoManifest:
    """Tests for the repository manifest."""

    def test_manifest_completeness(self):
        """All 50 repos present exactly once."""
        assert len(REPO_MANIFEST) == 50

    def test_no_repo_overlap(self):
        """No repo appears in multiple splits."""
        assert len(set(TRAIN_REPOS) & set(VALIDATION_REPOS)) == 0
        assert len(set(TRAIN_REPOS) & set(TEST_REPOS)) == 0
        assert len(set(VALIDATION_REPOS) & set(TEST_REPOS)) == 0

    def test_split_counts(self):
        """Correct number of repos per split."""
        assert len(TRAIN_REPOS) == 17
        assert len(VALIDATION_REPOS) == 17
        assert len(TEST_REPOS) == 16

    def test_all_expected_repos_present(self):
        """Every expected repo is in the manifest."""
        expected_train = {
            "aiohttp", "bandit", "falcon", "flask-migrate", "flask-sqlalchemy",
            "flask-testing", "invoke", "jinja2", "marshmallow", "networkx",
            "paramiko", "pycodestyle", "pymongo", "rich", "sqlalchemy",
            "tornado", "tox",
        }
        expected_test = {
            "Pillow", "black", "celery", "cherrypy", "click", "flake8",
            "flask", "httpx", "itsdangerous", "mypy", "nox", "psycopg2",
            "pydantic", "redis", "treq", "werkzeug",
        }
        assert set(TRAIN_REPOS) == expected_train
        assert set(TEST_REPOS) == expected_test


class TestSplitReassignment:
    """Tests for the split reassignment logic."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_all_rows_assigned(self):
        """Every row from JSONL files maps to a split."""
        all_rows = load_all_rows(V3_DATA_DIR)
        split_rows = reassign_splits(all_rows)
        total = sum(len(v) for v in split_rows.values())
        assert total == len(all_rows)

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_no_repo_leakage(self):
        """No repo_name in multiple splits after reassignment."""
        all_rows = load_all_rows(V3_DATA_DIR)
        split_rows = reassign_splits(all_rows)
        train_repos = set(r.repo_name for r in split_rows["train"])
        val_repos = set(r.repo_name for r in split_rows["validation"])
        test_repos = set(r.repo_name for r in split_rows["test"])
        assert len(train_repos & val_repos) == 0
        assert len(train_repos & test_repos) == 0
        assert len(val_repos & test_repos) == 0

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_no_commit_leakage(self):
        """No commit_sha appears in multiple splits."""
        all_rows = load_all_rows(V3_DATA_DIR)
        split_rows = reassign_splits(all_rows)
        train_commits = set(r.commit_sha for r in split_rows["train"])
        val_commits = set(r.commit_sha for r in split_rows["validation"])
        test_commits = set(r.commit_sha for r in split_rows["test"])
        assert len(train_commits & val_commits) == 0
        assert len(train_commits & test_commits) == 0
        assert len(val_commits & test_commits) == 0


class TestFeatureDimensions:
    """Tests for feature dimensions across variants."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists() or not EXP_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_variant_a_45_features(self):
        """Variant A (E0) has 45 features."""
        train, val, test = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E0")
        assert train.X.shape[1] == 45
        assert val.X.shape[1] == 45
        assert test.X.shape[1] == 45

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists() or not EXP_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_variant_b_48_features(self):
        """Variant B (E2) has 48 features."""
        train, val, test = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E2")
        assert train.X.shape[1] == 48
        assert val.X.shape[1] == 48
        assert test.X.shape[1] == 48

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists() or not EXP_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_variant_c_51_features(self):
        """Variant C (E4) has 51 features."""
        train, val, test = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E4")
        assert train.X.shape[1] == 51
        assert val.X.shape[1] == 51
        assert test.X.shape[1] == 51

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists() or not EXP_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_baseline_feature_names(self):
        """Variant A feature names match existing FEATURE_NAMES."""
        from backend.app.ml.dataset_loader import FEATURE_NAMES
        train, _, _ = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E0")
        assert train.feature_names == FEATURE_NAMES


class TestPerCommitRanking:
    """Tests for per-commit ranking metrics."""

    def test_basic_ranking(self):
        """Manual: 5-file commit, 2 positives at ranks 1 and 3."""
        metadata = [
            {"repo_name": "test_repo", "commit_sha": "abc123", "file_path": f"file_{i}.py"}
            for i in range(5)
        ]
        y_true = np.array([1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.1])

        result = per_commit_ranking(metadata, y_true, y_prob)

        assert result.num_commits_total == 1
        assert result.num_commits_with_positives == 1
        assert len(result.per_commit_details) == 1

        detail = result.per_commit_details[0]
        assert detail.total_files == 5
        assert detail.positive_files == 2
        assert detail.precision_at_1 == 1.0
        assert detail.recall_at_1 == 0.5
        assert detail.rank_of_first_positive == 1
        assert detail.top_1_capture == 1
        assert detail.top_3_capture == 1

    def test_no_positives(self):
        """Commit with 0 positives: positive-dependent metrics are None."""
        metadata = [
            {"repo_name": "test_repo", "commit_sha": "abc123", "file_path": f"file_{i}.py"}
            for i in range(3)
        ]
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.9, 0.5, 0.1])

        result = per_commit_ranking(metadata, y_true, y_prob)

        assert result.num_commits_with_positives == 0
        assert result.num_commits_zero_positives == 1
        detail = result.per_commit_details[0]
        assert detail.precision_at_1 is None
        assert detail.recall_at_1 is None
        assert detail.rank_of_first_positive is None
        assert detail.reciprocal_rank is None
        assert detail.top_1_capture is None

    def test_single_file_commit_positive(self):
        """1-file commit that is positive."""
        metadata = [{"repo_name": "r", "commit_sha": "c1", "file_path": "a.py"}]
        y_true = np.array([1])
        y_prob = np.array([0.8])

        result = per_commit_ranking(metadata, y_true, y_prob)
        detail = result.per_commit_details[0]

        assert detail.total_files == 1
        assert detail.positive_files == 1
        assert detail.precision_at_1 == 1.0
        assert detail.recall_at_1 == 1.0
        assert detail.rank_of_first_positive == 1
        assert detail.reciprocal_rank == 1.0

    def test_single_file_commit_negative(self):
        """1-file commit that is negative."""
        metadata = [{"repo_name": "r", "commit_sha": "c1", "file_path": "a.py"}]
        y_true = np.array([0])
        y_prob = np.array([0.8])

        result = per_commit_ranking(metadata, y_true, y_prob)
        detail = result.per_commit_details[0]

        assert detail.positive_files == 0
        assert detail.precision_at_1 is None

    def test_tie_breaking_deterministic(self):
        """Equal scores are broken by file_path lexicographic."""
        metadata = [
            {"repo_name": "r", "commit_sha": "c1", "file_path": "c.py"},
            {"repo_name": "r", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "c1", "file_path": "b.py"},
        ]
        y_true = np.array([1, 0, 0])
        y_prob = np.array([0.5, 0.5, 0.5])

        result = per_commit_ranking(metadata, y_true, y_prob)
        detail = result.per_commit_details[0]

        # a.py comes first lexicographically, so it's ranked first
        # a.py is negative, so rank_of_first_positive = 3 (c.py at rank 3)
        assert detail.rank_of_first_positive == 3

    def test_multiple_commits(self):
        """Multiple commits across repos."""
        metadata = [
            {"repo_name": "r1", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "r1", "commit_sha": "c1", "file_path": "b.py"},
            {"repo_name": "r2", "commit_sha": "c2", "file_path": "x.py"},
            {"repo_name": "r2", "commit_sha": "c2", "file_path": "y.py"},
        ]
        y_true = np.array([1, 0, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7])

        result = per_commit_ranking(metadata, y_true, y_prob)
        assert result.num_commits_total == 2
        assert result.num_commits_with_positives == 2


class TestGlobalMetrics:
    """Tests for global classification metrics."""

    def test_matches_sklearn(self):
        """ROC-AUC and PR-AUC match sklearn directly."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.4, 0.5, 0.6, 0.15])

        result = global_classification_metrics(y_true, y_prob)

        assert abs(result.roc_auc - roc_auc_score(y_true, y_prob)) < 1e-10
        assert abs(result.pr_auc - average_precision_score(y_true, y_prob)) < 1e-10

    def test_all_positives(self):
        """Edge case: all labels are positive."""
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7])

        result = global_classification_metrics(y_true, y_prob)
        assert result.roc_auc == 0.0
        assert result.pr_auc == 0.0
        assert result.positive_prevalence == 1.0

    def test_all_negatives(self):
        """Edge case: all labels are negative."""
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])

        result = global_classification_metrics(y_true, y_prob)
        assert result.roc_auc == 0.0
        assert result.positive_prevalence == 0.0


class TestPerRepoMetrics:
    """Tests for per-repository metrics."""

    def test_per_repo_grouping(self):
        """Metrics computed per repo correctly."""
        metadata = [
            {"repo_name": "repo_a", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "repo_a", "commit_sha": "c1", "file_path": "b.py"},
            {"repo_name": "repo_b", "commit_sha": "c2", "file_path": "x.py"},
            {"repo_name": "repo_b", "commit_sha": "c2", "file_path": "y.py"},
        ]
        y_true = np.array([1, 0, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7])

        results = per_repo_metrics(metadata, y_true, y_prob)
        assert len(results) == 2
        repos = {r.repo_name: r for r in results}
        assert repos["repo_a"].total_rows == 2
        assert repos["repo_a"].positive_rows == 1
        assert repos["repo_b"].total_rows == 2
        assert repos["repo_b"].positive_rows == 1

    def test_expected_repos_filter(self):
        """Only compute for expected repos."""
        metadata = [
            {"repo_name": "repo_a", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "repo_b", "commit_sha": "c2", "file_path": "x.py"},
        ]
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.1])

        results = per_repo_metrics(metadata, y_true, y_prob, expected_repos=["repo_a"])
        assert len(results) == 1
        assert results[0].repo_name == "repo_a"

    def test_macro_average(self):
        """Macro average is unweighted mean across repos."""
        results = per_repo_metrics(
            [
                {"repo_name": "a", "commit_sha": "c1", "file_path": "a.py"},
                {"repo_name": "b", "commit_sha": "c2", "file_path": "x.py"},
            ],
            np.array([1, 0]),
            np.array([0.9, 0.1]),
        )
        macro = macro_average_per_repo(results)
        assert macro["num_repos"] == 2


class TestDominanceCheck:
    """Tests for dominance check."""

    def test_no_dominance(self):
        """No repo > 50% when positives are spread."""
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "b", "commit_sha": "c2", "file_path": "x.py"},
            {"repo_name": "c", "commit_sha": "c3", "file_path": "y.py"},
        ]
        y_true = np.array([1, 1, 1])

        result = dominance_check(metadata, y_true, threshold=0.5)
        assert not result["any_dominates"]

    def test_dominance(self):
        """One repo > 50% of positives."""
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "a.py"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "b.py"},
            {"repo_name": "b", "commit_sha": "c3", "file_path": "x.py"},
        ]
        y_true = np.array([1, 1, 1])

        result = dominance_check(metadata, y_true, threshold=0.5)
        assert result["any_dominates"]
        assert result["repos"]["a"]["dominates"]


class TestDeterminism:
    """Tests for deterministic output."""

    @pytest.mark.skipif(
        not V3_DATA_DIR.exists() or not EXP_DATA_DIR.exists(),
        reason="Dataset not available",
    )
    def test_deterministic_predictions(self):
        """Same input produces same predictions."""
        train1, val1, test1 = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E0")
        train2, val2, test2 = build_repo_split_dataset(V3_DATA_DIR, EXP_DATA_DIR, mode="E0")

        assert np.array_equal(train1.X, train2.X)
        assert np.array_equal(train1.y, train2.y)
        assert len(train1.metadata) == len(train2.metadata)

        for m1, m2 in zip(train1.metadata, train2.metadata):
            assert m1["commit_sha"] == m2["commit_sha"]
            assert m1["file_path"] == m2["file_path"]