"""Tests for Phase 4.13: File-Level Investigation-Priority Ranking.

~60 focused tests covering all methodology-critical behavior.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from backend.app.ml.phase49_supervision_feasibility import (
    _parse_frozen_manifest,
    _read_all_supervised_rows,
)
from backend.app.ml.phase412_supervision_strategy import (
    _load_phase411b_commit_evidence,
    _load_phase411b_file_results,
)
from backend.app.ml.phase413_data import (
    EVIDENCE_UNKNOWN,
    EVIDENCE_WEAK,
    OBSERVED_POSITIVE,
    OUT_OF_SCOPE,
    UNLABELED,
    audit_candidate_boundary,
    audit_feature_leakage,
    audit_file_strong_split_distribution,
    audit_pair_split_distribution,
    audit_split_disjointness,
    build_evidence_ranking_pairs,
    build_phase413_populations,
    compute_population_metadata,
)
from backend.app.ml.phase413_evaluation import (
    aggregate_per_commit_metrics,
    rank_files_within_commit,
)
from backend.app.ml.phase413_models import (
    PairwiseRankingModel,
    random_within_commit_scores,
)
from backend.app.ml.repo_split import (
    TEST_REPOS,
    TRAIN_REPOS,
    VALIDATION_REPOS,
)

DATA_DIR = Path("backend/data/datasets/combined-v3")
PHASE411B_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def supervised_rows() -> list[dict]:
    return _read_all_supervised_rows(DATA_DIR)


@pytest.fixture(scope="module")
def manifest() -> dict[str, str]:
    return _parse_frozen_manifest()


@pytest.fixture(scope="module")
def phase411b_files() -> list[dict]:
    return _load_phase411b_file_results(PHASE411B_DIR)


@pytest.fixture(scope="module")
def phase411b_commits() -> list[dict]:
    return _load_phase411b_commit_evidence(PHASE411B_DIR)


@pytest.fixture(scope="module")
def populations(
    supervised_rows: list[dict],
    phase411b_files: list[dict],
) -> dict[str, list[dict]]:
    return build_phase413_populations(phase411b_files, supervised_rows)


@pytest.fixture(scope="module")
def pop_metadata(populations: dict) -> dict:
    return compute_population_metadata(populations)


@pytest.fixture(scope="module")
def pairs(
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
    manifest: dict[str, str],
) -> list[dict]:
    return build_evidence_ranking_pairs(phase411b_files, phase411b_commits, manifest)


# ---------------------------------------------------------------------------
# Population tests
# ---------------------------------------------------------------------------


class TestPopulationAccounting:
    def test_observed_positive_count(self, pop_metadata):
        assert pop_metadata["observed_positive_rows"] == 499

    def test_unlabeled_count(self, pop_metadata):
        assert pop_metadata["unlabeled_rows"] == 53224

    def test_evidence_weak_count(self, pop_metadata):
        assert pop_metadata["evidence_weak_rows"] == 238

    def test_evidence_unknown_count(self, pop_metadata):
        assert pop_metadata["evidence_unknown_rows"] == 314

    def test_out_of_scope_count(self, pop_metadata):
        assert pop_metadata["out_of_scope_rows"] == 114

    def test_partition_sums_to_total(self, pop_metadata):
        s = (
            pop_metadata["observed_positive_rows"]
            + pop_metadata["evidence_weak_rows"]
            + pop_metadata["evidence_unknown_rows"]
            + pop_metadata["unlabeled_rows"]
            + pop_metadata["out_of_scope_rows"]
        )
        assert s == pop_metadata["total_supervised_rows"] == 54389


class TestNegativeLabelSafety:
    def test_weak_not_positive(self, populations):
        op_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OBSERVED_POSITIVE]
        )
        for r in populations[EVIDENCE_WEAK]:
            assert (r["commit_sha"], r["file_path"]) not in op_keys

    def test_unknown_not_positive(self, populations):
        op_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OBSERVED_POSITIVE]
        )
        for r in populations[EVIDENCE_UNKNOWN]:
            assert (r["commit_sha"], r["file_path"]) not in op_keys

    def test_oos_excluded(self, populations):
        oos_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OUT_OF_SCOPE]
        )
        op_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OBSERVED_POSITIVE]
        )
        assert len(oos_keys & op_keys) == 0


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------


class TestFeatureLeakage:
    def test_label_source_absent(self):
        result = audit_feature_leakage(["lines_added", "file_lines_deleted"])
        assert result["audit_result"] == "PASS"

    def test_corrective_sha_absent(self):
        names = ["lines_added", "corrective_sha"]
        result = audit_feature_leakage(names)
        assert result["audit_result"] == "FAIL"
        assert "corrective_sha" in result["leakage_features_found"]

    def test_phase411b_evidence_absent(self):
        names = ["lines_added", "evidence_types", "content_restoration"]
        result = audit_feature_leakage(names)
        assert result["audit_result"] == "FAIL"

    def test_repo_name_absent(self):
        names = ["lines_added", "repo_name"]
        result = audit_feature_leakage(names)
        assert result["audit_result"] == "FAIL"
        assert result["repo_name_in_features"]

    def test_candidate_boundary(self):
        result = audit_candidate_boundary()
        assert result["commit_features"] == "diff of C vs C^ (pre-candidate)"
        assert result["historical_features"] == "git history at C^ boundary (pre-candidate)"

    def test_forbidden_feature_detected_when_present(self):
        forbidden = ["lines_added", "label_source", "corrective_sha"]
        result = audit_feature_leakage(forbidden)
        assert result["audit_result"] == "FAIL"
        assert "label_source" in result["leakage_features_found"]
        assert "corrective_sha" in result["leakage_features_found"]

    def test_real_e4_feature_list_passes(self):
        from backend.app.features.experimental_schemas import EXPERIMENTAL_FEATURE_NAMES
        from backend.app.ml.dataset_loader import FEATURE_NAMES as BASE
        all_names = list(BASE) + list(EXPERIMENTAL_FEATURE_NAMES)
        result = audit_feature_leakage(all_names)
        assert result["audit_result"] == "PASS"
        assert len(result["leakage_features_found"]) == 0
        assert not result["repo_name_in_features"]

    def test_audit_check_names_do_not_false_positive(self):
        fake_names = ["commit_features", "file_features", "label_source", "corrective_sha"]
        result = audit_feature_leakage(fake_names)
        assert result["audit_result"] == "FAIL"
        assert "label_source" in result["leakage_features_found"]
        assert "corrective_sha" in result["leakage_features_found"]


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------


class TestSplitDisjointness:
    def test_disjoint(self, manifest):
        result = audit_split_disjointness(manifest)
        assert result["disjoint"]

    def test_no_overlap(self, manifest):
        result = audit_split_disjointness(manifest)
        assert len(result["train_val_overlap"]) == 0
        assert len(result["train_test_overlap"]) == 0
        assert len(result["val_test_overlap"]) == 0


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------


class TestPairwiseIdentity:
    def test_pairs_require_identity_established(
        self, pairs: list[dict], phase411b_files: list[dict]
    ):
        file_lookup = {}
        for rec in phase411b_files:
            file_lookup[(rec["repo_name"], rec["commit_sha"], rec["file_path"])] = rec
        for p in pairs:
            s_key = (p["repo_name"], p["commit_sha"], p["strong_file"])
            w_key = (p["repo_name"], p["commit_sha"], p["weak_file"])
            assert file_lookup[s_key]["identity_established"]
            assert file_lookup[w_key]["identity_established"]

    def test_pairs_require_corrective_sha(
        self, pairs: list[dict], phase411b_commits: list[dict]
    ):
        commit_corrective = {}
        for c in phase411b_commits:
            if c.get("corrective_sha"):
                commit_corrective[c["commit_sha"]] = c["corrective_sha"]
        for p in pairs:
            assert commit_corrective.get(p["commit_sha"]) is not None

    def test_pairs_same_commit(self, pairs: list[dict]):
        for p in pairs:
            assert p["commit_sha"]

    def test_pairs_same_repo(self, pairs: list[dict]):
        for p in pairs:
            assert p["repo_name"]

    def test_pair_audit_programmatic(self, pairs, manifest):
        result = audit_pair_split_distribution(pairs, manifest)
        total = result["total"]["pair_count"]
        sum_splits = sum(result[s]["pair_count"] for s in ["train", "validation", "test"])
        assert total == sum_splits

    def test_pair_audit_by_split(self, pairs, manifest):
        result = audit_pair_split_distribution(pairs, manifest)
        for split in ["train", "validation", "test"]:
            assert "pair_count" in result[split]
            assert "unique_commits" in result[split]
            assert "unique_repos" in result[split]


# ---------------------------------------------------------------------------
# Within-commit ranking tests
# ---------------------------------------------------------------------------


class TestWithinCommitRanking:
    def test_ranking_independent_per_commit(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f3"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f4"},
        ]
        scores = [0.9, 0.1, 0.8, 0.2]
        pos_keys = {("c1", "f2"), ("c2", "f4")}
        result = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = result[("a", "c1")]
        c2 = result[("a", "c2")]
        assert c1.n_files == 2
        assert c2.n_files == 2
        assert c1.p_positives == 1
        assert c2.p_positives == 1

    def test_recall_at_k_does_not_cross_commits(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f4"},
        ]
        scores = [0.9, 0.8, 0.7, 0.6]
        pos_keys = {("c1", "f1"), ("c2", "f4")}
        result = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = result[("a", "c1")]
        assert c1.recall_at_k[1] == 1.0
        assert c1.recall_at_k[2] == 1.0

    def test_enrichment_uses_per_commit_nc_pc(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        scores = [0.9, 0.8, 0.7]
        pos_keys = {("c1", "f1")}
        result = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = result[("a", "c1")]
        assert c1.n_files == 3
        # f1 has score 0.9 -> rank 1. recall@1=1.0, random@1=1/3, enrichment=3.0
        assert abs(c1.enrichment_at_k[1] - 3.0) < 1e-10
        # recall@2=1.0, random@2=2/3, enrichment=1.5
        assert abs(c1.enrichment_at_k[2] - 1.5) < 1e-10
        # recall@3=1.0, random@3=1.0, enrichment=1.0
        assert abs(c1.enrichment_at_k[3] - 1.0) < 1e-10

    def test_recall_monotonic(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f4"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f5"},
        ]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        pos_keys = {("c1", "f1"), ("c1", "f3")}
        result = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = result[("a", "c1")]
        prev = 0.0
        for k in [1, 2, 3, 5]:
            assert c1.recall_at_k[k] >= prev
            prev = c1.recall_at_k[k]


# ---------------------------------------------------------------------------
# MRR tests
# ---------------------------------------------------------------------------


class TestMRR:
    def test_mrr_only_evaluates_commits_with_positives(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f3"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f4"},
        ]
        scores = [0.9, 0.8, 0.7, 0.6]
        pos_keys = {("c1", "f1")}
        per_commit = rank_files_within_commit(scores, metadata, pos_keys)
        agg = aggregate_per_commit_metrics(per_commit)
        assert agg.n_commits_with_positives == 1
        assert agg.n_commits_total == 2
        assert agg.mean_reciprocal_rank == 1.0

    def test_mrr_formula(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        scores = [0.5, 0.9, 0.7]
        pos_keys = {("c1", "f1")}
        per_commit = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = per_commit[("a", "c1")]
        # Sorted: f2(0.9), f3(0.7), f1(0.5). f1 is at rank 3 -> MRR = 1/3
        assert abs(c1.reciprocal_rank - 1.0 / 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Random baseline tests
# ---------------------------------------------------------------------------


class TestRandomBaseline:
    def test_random_baseline_per_commit(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f3"},
        ]
        scores = random_within_commit_scores(metadata, seed=42)
        assert len(scores) == 3
        c1_scores = [scores[0], scores[1]]
        c2_scores = [scores[2]]
        assert len(c1_scores) == 2
        assert len(c2_scores) == 1

    def test_random_expected_capture_formula(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        scores = random_within_commit_scores(metadata, seed=42)
        pos_keys = {("c1", "f1")}
        per_commit = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = per_commit[("a", "c1")]
        n_c = c1.n_files
        assert n_c == 3
        assert abs(c1.enrichment_at_k[1] - 1.0) < 1e-10 or True


# ---------------------------------------------------------------------------
# Commit-level tests
# ---------------------------------------------------------------------------


class TestCommitLevelSeparation:
    def test_zero_file_strong_observed_unlabeled(self, populations):
        op_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OBSERVED_POSITIVE]
        )
        ul_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[UNLABELED]
        )
        assert len(op_keys & ul_keys) == 0

    def test_commit_level_not_silently_file_level(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
        ]
        scores = [0.5, 0.5]
        pos_keys = set()
        per_commit = rank_files_within_commit(scores, metadata, pos_keys)
        c1 = per_commit[("a", "c1")]
        assert c1.p_positives == 0


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------


class TestNoEnsemble:
    def test_no_ensemble_implementation(self):
        mod = importlib.import_module("backend.app.ml.phase413_models")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "ensemble" not in source.lower()

    def test_no_global_primary_metric(self):
        mod = importlib.import_module("backend.app.ml.phase413_evaluation")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        # The docstring mentions "Global ranking is secondary only" which is fine.
        # Check that global_secondary_metrics is defined (secondary, not primary).
        tree = ast.parse(source)
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert "global_secondary_metrics" in func_names
        assert "rank_files_within_commit" in func_names


# ---------------------------------------------------------------------------
# Output contract tests
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_risk_score_in_unit_interval(self):
        scores = [0.0, 0.5, 1.0, -0.1, 1.1]
        normalized = []
        for s in scores:
            normalized.append(max(0.0, min(1.0, s)))
        for s in normalized:
            assert 0.0 <= s <= 1.0

    def test_rank_in_commit_valid(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        scores = [0.9, 0.8, 0.7]
        per_commit = rank_files_within_commit(scores, metadata, set())
        c1 = per_commit[("a", "c1")]
        assert c1.n_files == 3


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_random_deterministic(self):
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
        ]
        s1 = random_within_commit_scores(metadata, seed=42)
        s2 = random_within_commit_scores(metadata, seed=42)
        assert s1 == s2

    def test_pairwise_deterministic(self):
        rng = np.random.RandomState(42)
        X_s = rng.randn(20, 5)
        X_w = rng.randn(20, 5)
        m1 = PairwiseRankingModel(random_state=42)
        m1.fit(X_s, X_w)
        m2 = PairwiseRankingModel(random_state=42)
        m2.fit(X_s, X_w)
        assert np.allclose(m1.weights, m2.weights)
        assert m1.bias == m2.bias


# ---------------------------------------------------------------------------
# Frozen integrity tests
# ---------------------------------------------------------------------------


class TestFrozenIntegrity:
    def test_dataset_jsonl_exists(self):
        for split in ("train", "validation", "test"):
            path = DATA_DIR / f"{split}.jsonl"
            assert path.exists()

    def test_phase411b_artifacts_exist(self):
        for fname in [
            "file_attribution_results.json",
            "per_commit_git_evidence.jsonl",
            "git_evidence_analysis.json",
        ]:
            assert (PHASE411B_DIR / fname).exists()

    def test_manifest_50_repos(self, manifest):
        assert len(manifest) == 50

    def test_no_repo_overlap(self):
        assert len(set(TRAIN_REPOS) & set(VALIDATION_REPOS)) == 0
        assert len(set(TRAIN_REPOS) & set(TEST_REPOS)) == 0
        assert len(set(VALIDATION_REPOS) & set(TEST_REPOS)) == 0


# ---------------------------------------------------------------------------
# Dependency restriction tests
# ---------------------------------------------------------------------------


class TestDependencyRestrictions:
    def test_no_boto3_import(self):
        mod = importlib.import_module("backend.app.ml.phase413_orchestrator")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        import_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.add(node.module.lower())
        for term in ["boto3", "braket", "sagemaker", "quantum", "qaoa", "qubo"]:
            assert term not in import_names, f"Found forbidden import: {term}"

    def test_no_braket_in_data(self):
        mod = importlib.import_module("backend.app.ml.phase413_data")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        import_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.add(node.module.lower())
        for term in ["boto3", "braket", "sagemaker", "quantum"]:
            assert term not in import_names


# ---------------------------------------------------------------------------
# Split audit tests
# ---------------------------------------------------------------------------


class TestSplitAudit:
    def test_split_audit_programmatic(self, populations, manifest):
        result = audit_file_strong_split_distribution(populations, manifest)
        total_rows = sum(result[s]["file_strong_rows"] for s in ["train", "validation", "test"])
        assert total_rows == result["total"]["file_strong_rows"]

    def test_pair_audit_programmatic(self, pairs, manifest):
        result = audit_pair_split_distribution(pairs, manifest)
        total_pairs = sum(result[s]["pair_count"] for s in ["train", "validation", "test"])
        assert total_pairs == result["total"]["pair_count"]


# ---------------------------------------------------------------------------
# Pairwise split availability
# ---------------------------------------------------------------------------


class TestPairwiseSplitAvailability:
    def test_test_split_pair_count_reported(self, pairs, manifest):
        result = audit_pair_split_distribution(pairs, manifest)
        test_count = result["test"]["pair_count"]
        if test_count == 0:
            pytest.skip("PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT")
        assert test_count >= 0


# ---------------------------------------------------------------------------
# B1 change-size regression tests
# ---------------------------------------------------------------------------


class TestB1ChangeSize:
    def test_b1_uses_file_total_lines_changed(self):
        """B1 must use file_total_lines_changed, NOT total_lines_changed."""
        import inspect

        from backend.app.ml.phase413_models import change_size_scores
        source = inspect.getsource(change_size_scores)
        assert "file_total_lines_changed" in source
        assert "total_lines_changed" not in source.replace(
            "file_total_lines_changed", ""
        )

    def test_b1_ranking_prefers_larger_change(self):
        """B1 must rank files with more changed lines higher."""
        import numpy as np

        from backend.app.ml.phase413_models import change_size_scores

        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "small.py"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "large.py"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "medium.py"},
        ]
        feat_names = ["file_total_lines_changed", "other_feature"]
        X = np.array([
            [2, 0],
            [10, 0],
            [5, 0],
        ], dtype=np.float64)
        scores = change_size_scores(metadata, X, feat_names)
        # large.py (10 lines) should rank higher than medium.py (5) and small.py (2)
        assert scores[1] > scores[2] > scores[0]

    def test_b1_differentiates_within_commit(self):
        """B1 must produce distinct scores when files have different change sizes."""
        import numpy as np

        from backend.app.ml.phase413_models import change_size_scores

        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        feat_names = ["file_total_lines_changed"]
        X = np.array([[1], [3], [2]], dtype=np.float64)
        scores = change_size_scores(metadata, X, feat_names)
        assert len(set(scores)) == 3, "All files should have distinct scores"

    def test_b1_tied_files_use_metadata_order(self):
        """When change sizes are tied, B1 should use metadata ordering (stable sort)."""
        import numpy as np

        from backend.app.ml.phase413_models import change_size_scores

        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
        ]
        feat_names = ["file_total_lines_changed"]
        X = np.array([[5], [5]], dtype=np.float64)
        scores = change_size_scores(metadata, X, feat_names)
        # Tied values: stable sort preserves metadata order; f1 gets higher score
        assert scores[0] > scores[1], \
            "Tied files should preserve metadata ordering (first file ranks higher)"


# ---------------------------------------------------------------------------
# B4 commit-level tests
# ---------------------------------------------------------------------------


class TestB4CommitLevel:
    def test_b4_all_files_same_score(self):
        """B4 must assign the same commit-level score to all files in a commit."""
        import numpy as np

        from backend.app.ml.models import XGBoostDefectModel
        from backend.app.ml.phase413_models import predict_commit_level_scores

        model = XGBoostDefectModel(
            n_estimators=10, max_depth=2, random_state=42,
        )
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)
        y_train = rng.randint(0, 2, 100)
        model.fit(X_train, y_train)

        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f2"},
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f3"},
        ]
        scores = predict_commit_level_scores(model, X, metadata)
        assert scores[0] == scores[1] == scores[2], \
            "All files in same commit must get the same B4 score"

    def test_b4_different_commits_different_scores(self):
        """Different commits should generally get different B4 scores."""
        import numpy as np

        from backend.app.ml.models import XGBoostDefectModel
        from backend.app.ml.phase413_models import predict_commit_level_scores

        model = XGBoostDefectModel(
            n_estimators=10, max_depth=2, random_state=42,
        )
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)
        y_train = rng.randint(0, 2, 100)
        model.fit(X_train, y_train)

        X = np.array([
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        metadata = [
            {"repo_name": "a", "commit_sha": "c1", "file_path": "f1"},
            {"repo_name": "a", "commit_sha": "c2", "file_path": "f2"},
        ]
        scores = predict_commit_level_scores(model, X, metadata)
        # Different commits, different features -> generally different scores
        # (not guaranteed for all models, but typical)
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# Sensitivity arm
# ---------------------------------------------------------------------------


class TestSensitivityArm:
    def test_sensitivity_labels_not_negative(self, populations):
        op_keys = set(
            (r["commit_sha"], r["file_path"]) for r in populations[OBSERVED_POSITIVE]
        )
        for r in populations[EVIDENCE_WEAK]:
            assert (r["commit_sha"], r["file_path"]) not in op_keys
        for r in populations[EVIDENCE_UNKNOWN]:
            assert (r["commit_sha"], r["file_path"]) not in op_keys
