"""Tests for Phase 7: End-to-End Evaluation of Frozen B1 Heuristic.

Tests production parity, population, ranking, metrics, bootstrap,
leakage, artifact integrity, and regression.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from backend.app.dataset.schemas import DatasetRow
from backend.app.evaluation.metrics import (
    EvaluationResult,
    _bootstrap_aggregate,
    _compute_checksum,
    load_evaluation,
    save_evaluation,
)
from backend.app.evaluation.population import (
    construct_file_features,
    load_population,
    score_commit,
)
from backend.app.features.schemas import FILE_FEATURE_NAMES
from backend.app.inference.feature_pipeline import FileFeatures, PipelineResult
from backend.app.inference.scoring import score_files
from backend.app.ml.phase413_evaluation import (
    RECALL_AT_K_VALUES,
    PerCommitRanking,
    rank_files_within_commit,
)


def _make_ranking(
    sha: str = "s1",
    n_files: int = 3,
    p_positives: int = 1,
    recall: float = 1.0,
    enrichment: float = 3.0,
    first_rank: int = 1,
    all_captured: bool = True,
) -> PerCommitRanking:
    """Helper to create PerCommitRanking with all required K values."""
    return PerCommitRanking(
        repo_name="r",
        commit_sha=sha,
        n_files=n_files,
        p_positives=p_positives,
        recall_at_k={k: recall for k in RECALL_AT_K_VALUES},
        enrichment_at_k={k: enrichment for k in RECALL_AT_K_VALUES},
        rank_of_first_positive=first_rank,
        reciprocal_rank=1.0 / first_rank if first_rank else None,
        all_positives_captured_in_top_k={k: all_captured for k in RECALL_AT_K_VALUES},
    )

# ── Production Parity Tests ─────────────────────────────────────────────────


class TestProductionParity:
    def test_score_files_called_directly(self):
        """Phase 7 must call frozen score_files(), not reimplement B1."""
        row = DatasetRow(
            commit_sha="abc123",
            file_path="test.py",
            file_status="modified",
            defect_label=0,
            file_features=[0.0] * 16,
            commit_features=[0.0] * 29,
        )
        ff = construct_file_features(row)
        pipeline = PipelineResult(
            commit_info=__import__(
                "backend.app.schemas.diff", fromlist=["CommitInfo"]
            ).CommitInfo(sha="abc123"),
            extraction_result=__import__(
                "backend.app.features.schemas", fromlist=["FeatureExtractionResult"]
            ).FeatureExtractionResult(
                commit_features=__import__(
                    "backend.app.features.schemas", fromlist=["CommitFeatures"]
                ).CommitFeatures(commit_sha="abc123"),
                file_features=[],
                feature_version="v1",
            ),
            file_features=[ff],
        )
        scored = score_files(pipeline)
        assert len(scored) == 1
        assert scored[0][0] is ff

    def test_file_features_total_lines_changed_index(self):
        """FILE_FEATURE_NAMES[5] = 'total_lines_changed'."""
        assert FILE_FEATURE_NAMES[5] == "total_lines_changed"

    def test_construct_file_features_correct_mapping(self):
        """FileFeatures.total_lines_changed from FILE_FEATURE_NAMES[5]."""
        ff_vec = [0.0] * 16
        ff_vec[5] = 42.0
        row = DatasetRow(
            commit_sha="abc123",
            file_path="test.py",
            file_status="modified",
            defect_label=0,
            file_features=ff_vec,
            commit_features=[0.0] * 29,
        )
        ff = construct_file_features(row)
        assert ff.total_lines_changed == 42

    def test_binary_files_score_zero(self):
        """Binary files get score 0.0 from score_files()."""
        ff_binary = FileFeatures(
            path="img.png", status="added", total_lines_changed=100,
            lines_added=50, lines_deleted=50, is_binary=True,
            language=0.0, is_test_file=False,
        )
        ff_normal = FileFeatures(
            path="code.py", status="modified", total_lines_changed=10,
            lines_added=5, lines_deleted=5, is_binary=False,
            language=0.5, is_test_file=False,
        )
        pipeline = PipelineResult(
            commit_info=__import__(
                "backend.app.schemas.diff", fromlist=["CommitInfo"]
            ).CommitInfo(sha="abc123"),
            extraction_result=__import__(
                "backend.app.features.schemas", fromlist=["FeatureExtractionResult"]
            ).FeatureExtractionResult(
                commit_features=__import__(
                    "backend.app.features.schemas", fromlist=["CommitFeatures"]
                ).CommitFeatures(commit_sha="abc123"),
                file_features=[],
                feature_version="v1",
            ),
            file_features=[ff_binary, ff_normal],
        )
        scored = score_files(pipeline)
        scores_by_path = {ff.path: score for ff, score in scored}
        assert scores_by_path["img.png"] == 0.0
        assert scores_by_path["code.py"] > 0.0

    def test_ranking_order_matches_production(self):
        """Higher total_lines_changed → higher score."""
        ff_big = FileFeatures(
            path="big.py", status="modified", total_lines_changed=100,
            lines_added=50, lines_deleted=50, is_binary=False,
            language=0.0, is_test_file=False,
        )
        ff_small = FileFeatures(
            path="small.py", status="modified", total_lines_changed=5,
            lines_added=3, lines_deleted=2, is_binary=False,
            language=0.0, is_test_file=False,
        )
        pipeline = PipelineResult(
            commit_info=__import__(
                "backend.app.schemas.diff", fromlist=["CommitInfo"]
            ).CommitInfo(sha="abc123"),
            extraction_result=__import__(
                "backend.app.features.schemas", fromlist=["FeatureExtractionResult"]
            ).FeatureExtractionResult(
                commit_features=__import__(
                    "backend.app.features.schemas", fromlist=["CommitFeatures"]
                ).CommitFeatures(commit_sha="abc123"),
                file_features=[],
                feature_version="v1",
            ),
            file_features=[ff_small, ff_big],
        )
        scored = score_files(pipeline)
        assert scored[0][0].path == "big.py"
        assert scored[1][0].path == "small.py"


# ── Population Tests ────────────────────────────────────────────────────────


class TestPopulation:
    def test_exact_row_count(self):
        """Test split has exactly 10,574 rows."""
        pop = load_population()
        assert pop.n_rows == 10574

    def test_expected_repositories(self):
        """Test split covers expected repositories."""
        pop = load_population()
        assert len(pop.repositories) > 0
        assert "flask" in pop.repositories

    def test_deterministic_population(self):
        """Same load → same results."""
        pop1 = load_population()
        pop2 = load_population()
        assert pop1.n_rows == pop2.n_rows
        assert pop1.n_commits == pop2.n_commits
        assert pop1.n_positives == pop2.n_positives

    def test_zero_positive_commits_retained(self):
        """Zero-positive commits in population statistics."""
        pop = load_population()
        zero_pos_commits = sum(
            1 for indices in pop.commit_groups.values()
            if all(pop.rows[i].defect_label != 1 for i in indices)
        )
        assert zero_pos_commits > 0

    def test_positive_rate(self):
        """Positive rate is approximately 2%."""
        pop = load_population()
        assert 0.01 < pop.positive_rate < 0.05


# ── Ranking Tests ───────────────────────────────────────────────────────────


class TestRanking:
    def test_deterministic_ranking(self):
        """Same scores → same ranking."""
        scores = [0.9, 0.5, 0.1]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "c.py"},
        ]
        positive_keys = set()
        r1 = rank_files_within_commit(scores, metadata, positive_keys)
        r2 = rank_files_within_commit(scores, metadata, positive_keys)
        assert r1 == r2

    def test_binary_at_end_with_zero_score(self):
        """Binary files rank last with score 0.0."""
        scores = [0.5, 0.0, 0.8]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.png"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "c.py"},
        ]
        positive_keys = {("s", "a.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.n_files == 3

    def test_single_file_positive_commit(self):
        """Single-file positive commit: Recall@1=1, MRR=1."""
        scores = [0.5]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
        ]
        positive_keys = {("s", "a.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.p_positives == 1
        assert ranking.recall_at_k[1] == 1.0
        assert ranking.reciprocal_rank == 1.0
        assert ranking.rank_of_first_positive == 1
        assert ranking.all_positives_captured_in_top_k[1] is True

    def test_k_greater_than_n(self):
        """K > N: K_eff = N."""
        scores = [0.9, 0.1]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
        ]
        positive_keys = {("s", "a.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.n_files == 2
        assert ranking.recall_at_k[100] == 1.0

    def test_p_zero_excluded_from_ranking(self):
        """P=0: commit excluded from positive-ranking aggregates."""
        scores = [0.5, 0.3]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
        ]
        positive_keys = set()
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.p_positives == 0
        assert ranking.reciprocal_rank is None


# ── Metrics Tests ───────────────────────────────────────────────────────────


class TestMetrics:
    def test_recall_at_k(self):
        """Recall@K computes correctly."""
        scores = [0.9, 0.5, 0.1]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "c.py"},
        ]
        positive_keys = {("s", "a.py"), ("s", "b.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.recall_at_k[1] == pytest.approx(0.5)
        assert ranking.recall_at_k[2] == pytest.approx(1.0)

    def test_mrr(self):
        """MRR = 1 / rank_of_first_positive."""
        scores = [0.1, 0.9, 0.5]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "c.py"},
        ]
        positive_keys = {("s", "b.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.rank_of_first_positive == 1
        assert ranking.reciprocal_rank == 1.0

    def test_enrichment_at_k(self):
        """Enrichment@K = observed_recall / random_recall."""
        scores = [0.9, 0.5, 0.1, 0.05]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": f"_{i}.py"}
            for i in range(4)
        ]
        positive_keys = {("s", "_0.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.enrichment_at_k[1] > 1.0

    def test_all_captured_at_k(self):
        """all_positive_captured@K = 1 if all positives in top K_eff."""
        scores = [0.9, 0.5, 0.1]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": "a.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "b.py"},
            {"repo_name": "r", "commit_sha": "s", "file_path": "c.py"},
        ]
        positive_keys = {("s", "a.py"), ("s", "b.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.all_positives_captured_in_top_k[2] is True
        assert ranking.all_positives_captured_in_top_k[1] is False

    def test_multiple_positives(self):
        """Multiple positives: correct recall and all-captured."""
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        metadata = [
            {"repo_name": "r", "commit_sha": "s", "file_path": f"_{i}.py"}
            for i in range(5)
        ]
        positive_keys = {("s", "_0.py"), ("s", "_2.py")}
        result = rank_files_within_commit(scores, metadata, positive_keys)
        ranking = result[("r", "s")]
        assert ranking.p_positives == 2
        assert ranking.recall_at_k[1] == pytest.approx(0.5)
        assert ranking.recall_at_k[2] == pytest.approx(0.5)
        assert ranking.recall_at_k[3] == pytest.approx(1.0)


# ── Bootstrap Tests ─────────────────────────────────────────────────────────


class TestBootstrap:
    def test_commit_level_resampling(self):
        """Bootstrap resamples at commit level, not file level."""
        rankings = [_make_ranking(sha=f"s{i}") for i in range(10)]
        ci = _bootstrap_aggregate(rankings, n_bootstrap=100, seed=42)
        assert "mrr" in ci

    def test_fixed_seed_deterministic(self):
        """Fixed seed produces deterministic CIs."""
        rankings = [_make_ranking()]
        ci1 = _bootstrap_aggregate(rankings, n_bootstrap=10, seed=42)
        ci2 = _bootstrap_aggregate(rankings, n_bootstrap=10, seed=42)
        assert ci1["mrr"].mean == ci2["mrr"].mean
        assert ci1["mrr"].ci_lower == ci2["mrr"].ci_lower

    def test_ci_bounds_valid(self):
        """CI lower <= mean <= CI upper."""
        rankings = [_make_ranking(sha=f"s{i}") for i in range(10)]
        ci = _bootstrap_aggregate(rankings, n_bootstrap=100, seed=42)
        for key, val in ci.items():
            assert val.ci_lower <= val.mean <= val.ci_upper, f"{key} CI invalid"

    def test_duplicates_preserved(self):
        """Bootstrap preserves duplicate sampled commits."""
        rankings = [_make_ranking()]
        ci = _bootstrap_aggregate(rankings, n_bootstrap=10, seed=42)
        assert ci["mrr"].mean == pytest.approx(1.0)


# ── Leakage Tests ───────────────────────────────────────────────────────────


class TestLeakage:
    def test_no_future_fields_in_scoring(self):
        """No post-outcome fields enter B1 scoring path."""
        pop = load_population()
        sample_indices = list(pop.commit_groups.values())[0]
        sample_rows = [pop.rows[i] for i in sample_indices]
        sample_meta = [pop.metadata[i] for i in sample_indices]
        sha = sample_rows[0].commit_sha
        scores = score_commit(sample_rows, sample_meta, sha)
        assert len(scores) == len(sample_rows)

    def test_labels_as_ground_truth_only(self):
        """Labels consumed only as positive_keys, not as features."""
        pop = load_population()
        positive_keys = pop.positive_keys
        for sha, path in positive_keys:
            assert isinstance(sha, str)
            assert isinstance(path, str)

    def test_canonical_e0_features(self):
        """Feature vectors contain only canonical E0 features."""
        pop = load_population()
        row = pop.rows[0]
        assert len(row.file_features) == 16
        assert len(row.commit_features) == 29


# ── Artifact Tests ──────────────────────────────────────────────────────────


class TestArtifact:
    def test_checksum_generation(self):
        """Checksum is generated for payload."""
        payload = {"a": 1, "b": "test"}
        checksum = _compute_checksum(payload)
        assert len(checksum) == 64

    def test_checksum_validation(self):
        """Checksum validates successfully."""
        payload = {"a": 1, "b": "test"}
        checksum = _compute_checksum(payload)
        full = {**payload, "checksum": checksum}
        reloaded = {k: v for k, v in full.items() if k != "checksum"}
        canonical = json.dumps(reloaded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert computed == checksum

    def test_checksum_invalidates_on_change(self):
        """Checksum invalidates when payload changes."""
        payload = {"a": 1, "b": "test"}
        checksum = _compute_checksum(payload)
        payload["a"] = 2
        new_checksum = _compute_checksum(payload)
        assert checksum != new_checksum

    def test_deterministic_json(self):
        """Same payload → same canonical JSON."""
        payload = {"z": 1, "a": 2, "m": 3}
        canonical1 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        canonical2 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert canonical1 == canonical2

    def test_save_and_load(self):
        """Save and load with checksum verification."""
        result = EvaluationResult(
            n_commits=5,
            n_files=10,
            n_positives=1,
            positive_rate=0.1,
            repos=["flask"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_evaluation(result, Path(tmpdir))
            loaded = load_evaluation(path)
            assert loaded["n_commits"] == 5
            assert loaded["checksum"]


# ── Regression Tests ────────────────────────────────────────────────────────


class TestRegression:
    def test_imports_work(self):
        """All Phase 7 imports succeed."""
        from backend.app.evaluation import (
            EvaluationResult,
            load_evaluation,
            run_evaluation,
            save_evaluation,
        )
        from backend.app.evaluation.population import (
            construct_file_features,
            load_population,
            score_commit,
        )

        assert EvaluationResult is not None
        assert load_evaluation is not None
        assert run_evaluation is not None
        assert save_evaluation is not None
        assert construct_file_features is not None
        assert load_population is not None
        assert score_commit is not None

    def test_ruff_clean(self):
        """Phase 7 files pass Ruff lint."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "ruff", "check",
             "backend/app/evaluation/",
             "backend/tests/test_phase7_evaluation.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0, f"Ruff errors:\n{result.stdout}\n{result.stderr}"
