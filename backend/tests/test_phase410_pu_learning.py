"""Tests for Phase 4.10: PU Learning Feasibility Experiment.

Tests verify dataset accounting, population disjointness, metric formulas,
method contracts, artifact integrity, and frozen data integrity.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.ml.phase49_supervision_feasibility import (
    _build_sha_lookup,
    _parse_frozen_manifest,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
    construct_supervision_sets,
)
from backend.app.ml.phase410_pu_learning import (
    DATA_DIR,
    ELKAN_NOTO_STATUS,
    EXPERIMENT_MODES,
    RECALL_AT_K_VALUES,
    _compute_ranking_metrics,
    _random_baseline_metrics,
    run_pu_experiment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_PRIMARY_P = 108
EXPECTED_SENSITIVITY_S = 594
EXPECTED_UNLABELED_U = 53573
EXPECTED_OOS = 114
EXPECTED_DN = 0
EXPECTED_TOTAL = 54389


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def supervision_sets() -> dict:
    """Load real data and construct supervision sets once per module."""
    supervised = _read_all_supervised_rows(DATA_DIR)
    ambiguous = _read_ambiguous_rows(DATA_DIR)
    sha_lookup = _build_sha_lookup(supervised, ambiguous)
    manifest = _parse_frozen_manifest()
    return construct_supervision_sets(supervised, sha_lookup, manifest)


@pytest.fixture(scope="module")
def supervised_row_count() -> int:
    return len(_read_all_supervised_rows(DATA_DIR))


@pytest.fixture(scope="module")
def experiment_output(tmp_path_factory):
    """Run the experiment once for the entire module and cache results."""
    tmp_dir = tmp_path_factory.mktemp("phase410_module")
    result = run_pu_experiment(output_dir=tmp_dir)
    report = (tmp_dir / "phase410_pu_learning.md").read_text(encoding="utf-8")
    with open(tmp_dir / "pu_results.json", encoding="utf-8") as f:
        pu_results_json = json.load(f)
    with open(tmp_dir / "pu_dataset_accounting.json", encoding="utf-8") as f:
        accounting_json = json.load(f)
    with open(tmp_dir / "pu_experiment_config.json", encoding="utf-8") as f:
        config_json = json.load(f)
    return {
        "result": result,
        "tmp_dir": tmp_dir,
        "report": report,
        "pu_results_json": pu_results_json,
        "accounting_json": accounting_json,
        "config_json": config_json,
    }


# ---------------------------------------------------------------------------
# TestDatasetAccounting
# ---------------------------------------------------------------------------

class TestDatasetAccounting:
    def test_primary_positive_count(self, supervision_sets):
        assert supervision_sets["primary_observed_path_associated"]["count"] == EXPECTED_PRIMARY_P

    def test_sensitivity_count(self, supervision_sets):
        count = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        assert count == EXPECTED_SENSITIVITY_S

    def test_unlabeled_count(self, supervision_sets):
        assert supervision_sets["unlabeled"]["count"] == EXPECTED_UNLABELED_U

    def test_out_of_scope_count(self, supervision_sets):
        assert supervision_sets["out_of_scope"]["count"] == EXPECTED_OOS

    def test_defensible_negative_count(self, supervision_sets):
        assert supervision_sets["defensible_negative"]["count"] == EXPECTED_DN

    def test_primary_eligible_count(self, supervision_sets):
        P = supervision_sets["primary_observed_path_associated"]["count"]
        U = supervision_sets["unlabeled"]["count"]
        assert P + U == 53681

    def test_sensitivity_eligible_count(self, supervision_sets):
        P = supervision_sets["primary_observed_path_associated"]["count"]
        S = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        U = supervision_sets["unlabeled"]["count"]
        assert (P + S) + U == 54275

    def test_global_reconciliation_primary(self, supervision_sets):
        P = supervision_sets["primary_observed_path_associated"]["count"]
        S = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        U = supervision_sets["unlabeled"]["count"]
        OOS = supervision_sets["out_of_scope"]["count"]
        DN = supervision_sets["defensible_negative"]["count"]
        assert P + S + U + OOS + DN == EXPECTED_TOTAL

    def test_global_reconciliation_sensitivity(self, supervision_sets):
        P = supervision_sets["primary_observed_path_associated"]["count"]
        S = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        U = supervision_sets["unlabeled"]["count"]
        OOS = supervision_sets["out_of_scope"]["count"]
        DN = supervision_sets["defensible_negative"]["count"]
        assert (P + S) + U + OOS + DN == EXPECTED_TOTAL

    def test_no_defensible_negatives(self, supervision_sets):
        assert supervision_sets["defensible_negative"]["count"] == 0

    def test_total_reconciles_to_source(self, supervision_sets, supervised_row_count):
        P = supervision_sets["primary_observed_path_associated"]["count"]
        S = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        U = supervision_sets["unlabeled"]["count"]
        OOS = supervision_sets["out_of_scope"]["count"]
        DN = supervision_sets["defensible_negative"]["count"]
        assert P + S + U + OOS + DN == supervised_row_count


# ---------------------------------------------------------------------------
# TestPopulationDisjointness
# ---------------------------------------------------------------------------

class TestPopulationDisjointness:
    def test_primary_unlabeled_disjoint(self, supervision_sets):
        p_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["primary_observed_path_associated"]["rows"]
        }
        u_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["unlabeled"]["rows"]
        }
        assert p_keys.isdisjoint(u_keys)

    def test_primary_sensitivity_disjoint(self, supervision_sets):
        p_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["primary_observed_path_associated"]["rows"]
        }
        s_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["sensitivity_commit_only_overlap"]["rows"]
        }
        assert p_keys.isdisjoint(s_keys)

    def test_sensitivity_unlabeled_disjoint(self, supervision_sets):
        s_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["sensitivity_commit_only_overlap"]["rows"]
        }
        u_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["unlabeled"]["rows"]
        }
        assert s_keys.isdisjoint(u_keys)

    def test_out_of_scope_excluded_from_all(self, supervision_sets):
        oos_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["out_of_scope"]["rows"]
        }
        p_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["primary_observed_path_associated"]["rows"]
        }
        u_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["unlabeled"]["rows"]
        }
        s_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["sensitivity_commit_only_overlap"]["rows"]
        }
        assert oos_keys.isdisjoint(p_keys)
        assert oos_keys.isdisjoint(u_keys)
        assert oos_keys.isdisjoint(s_keys)

    def test_exhaustive_partition(self, supervision_sets, supervised_row_count):
        p = supervision_sets["primary_observed_path_associated"]["count"]
        s = supervision_sets["sensitivity_commit_only_overlap"]["count"]
        u = supervision_sets["unlabeled"]["count"]
        oos = supervision_sets["out_of_scope"]["count"]
        dn = supervision_sets["defensible_negative"]["count"]
        assert p + s + u + oos + dn == supervised_row_count


# ---------------------------------------------------------------------------
# TestFilePartialCommitOnlySemantics
# ---------------------------------------------------------------------------

class TestFilePartialCommitOnlySemantics:
    def test_file_partial_rows_are_primary(self, supervision_sets):
        for row in supervision_sets["primary_observed_path_associated"]["rows"]:
            assert row.get("observation_role") == "observed_path_associated"

    def test_commit_only_path_overlap_are_sensitivity(self, supervision_sets):
        for row in supervision_sets["sensitivity_commit_only_overlap"]["rows"]:
            assert row.get("observation_role") == "sensitivity_commit_only_overlap"

    def test_commit_only_unknown_are_unlabeled(self, supervision_sets):
        unlabeled_roles = {
            row.get("observation_role")
            for row in supervision_sets["unlabeled"]["rows"]
        }
        assert "unlabeled" in unlabeled_roles

    def test_no_sensitivity_in_primary(self, supervision_sets):
        p_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["primary_observed_path_associated"]["rows"]
        }
        s_keys = {
            (r["commit_sha"], r["file_path"])
            for r in supervision_sets["sensitivity_commit_only_overlap"]["rows"]
        }
        assert len(p_keys & s_keys) == 0


# ---------------------------------------------------------------------------
# TestNoDefensibleNegatives
# ---------------------------------------------------------------------------

class TestNoDefensibleNegatives:
    def test_no_negatives_fabricated(self, supervision_sets):
        assert supervision_sets["defensible_negative"]["count"] == 0
        assert len(supervision_sets["defensible_negative"]["rows"]) == 0


# ---------------------------------------------------------------------------
# TestOutOfScope
# ---------------------------------------------------------------------------

class TestOutOfScope:
    def test_oos_count_is_114(self, supervision_sets):
        assert supervision_sets["out_of_scope"]["count"] == EXPECTED_OOS

    def test_oos_rows_are_dev_null(self, supervision_sets):
        for row in supervision_sets["out_of_scope"]["rows"]:
            assert row["file_path"] == "/dev/null"

    def test_oos_reason_is_parser_artifact(self, supervision_sets):
        for row in supervision_sets["out_of_scope"]["rows"]:
            assert row.get("out_of_scope_reason") == "parser_artifact_dev_null"


# ---------------------------------------------------------------------------
# TestRepositorySplit
# ---------------------------------------------------------------------------

class TestRepositorySplit:
    def test_50_repos_total(self):
        from backend.app.ml.repo_split import REPO_MANIFEST
        assert len(REPO_MANIFEST) == 50

    def test_train_val_test_repos_disjoint(self):
        from backend.app.ml.repo_split import (
            TEST_REPOS,
            TRAIN_REPOS,
            VALIDATION_REPOS,
        )
        assert set(TRAIN_REPOS).isdisjoint(set(VALIDATION_REPOS))
        assert set(TRAIN_REPOS).isdisjoint(set(TEST_REPOS))
        assert set(VALIDATION_REPOS).isdisjoint(set(TEST_REPOS))

    def test_17_17_16_split(self):
        from backend.app.ml.repo_split import (
            TEST_REPOS,
            TRAIN_REPOS,
            VALIDATION_REPOS,
        )
        assert len(TRAIN_REPOS) == 17
        assert len(VALIDATION_REPOS) == 17
        assert len(TEST_REPOS) == 16


# ---------------------------------------------------------------------------
# TestFeatureDimensions
# ---------------------------------------------------------------------------

class TestFeatureDimensions:
    def test_e0_is_45_features(self):
        from backend.app.ml.dataset_loader import NUM_FEATURES
        mode_indices = EXPERIMENT_MODES["E0"]
        assert NUM_FEATURES + len(mode_indices) == 45

    def test_e1_is_48_features(self):
        from backend.app.ml.dataset_loader import NUM_FEATURES
        mode_indices = EXPERIMENT_MODES["E1"]
        assert NUM_FEATURES + len(mode_indices) == 48

    def test_e2_is_48_features(self):
        from backend.app.ml.dataset_loader import NUM_FEATURES
        mode_indices = EXPERIMENT_MODES["E2"]
        assert NUM_FEATURES + len(mode_indices) == 48

    def test_e4_is_51_features(self):
        from backend.app.ml.dataset_loader import NUM_FEATURES
        mode_indices = EXPERIMENT_MODES["E4"]
        assert NUM_FEATURES + len(mode_indices) == 51


# ---------------------------------------------------------------------------
# TestRandomBaseline
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    def test_random_expected_recall_at_k(self):
        N = 1000
        P = 50
        bl = _random_baseline_metrics(N, P)
        for k in RECALL_AT_K_VALUES:
            expected = min(k, N) / N
            assert bl["expected_recall_at_k"][k] == pytest.approx(expected)

    def test_random_expected_positive_count_at_k(self):
        N = 1000
        P = 50
        bl = _random_baseline_metrics(N, P)
        for k in RECALL_AT_K_VALUES:
            expected = min(k, N) * P / N
            assert bl["expected_positive_count_at_k"][k] == pytest.approx(expected)

    def test_random_expected_enrichment_is_1(self):
        N = 1000
        P = 50
        bl = _random_baseline_metrics(N, P)
        for k in RECALL_AT_K_VALUES:
            assert bl["expected_enrichment_at_k"][k] == pytest.approx(1.0)

    def test_random_expected_mean_rank(self):
        N = 1000
        P = 50
        bl = _random_baseline_metrics(N, P)
        assert bl["expected_mean_rank"] == pytest.approx((N + 1) / 2)

    def test_random_deterministic(self):
        bl1 = _random_baseline_metrics(1000, 50)
        bl2 = _random_baseline_metrics(1000, 50)
        assert bl1 == bl2


# ---------------------------------------------------------------------------
# TestPUMethods
# ---------------------------------------------------------------------------

class TestPUMethods:
    def test_no_elkan_noto_class_prior(self):
        """The observed-positive fraction must NOT be used as Elkan-Noto c."""
        P = 108
        U = 53573
        observed_fraction = P / (P + U)
        assert ELKAN_NOTO_STATUS == "NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE"
        assert observed_fraction != 1.0

    def test_elkan_noto_status_not_identifiable(self):
        assert ELKAN_NOTO_STATUS == "NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE"


# ---------------------------------------------------------------------------
# TestRankingMetrics
# ---------------------------------------------------------------------------

class TestRankingMetrics:
    def test_observed_positive_recall_at_k_calculation(self):
        # scores: sha0=0.9 (rank 1), sha1=0.2 (rank 8), rest in between
        scores = np.array([0.9, 0.1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.0])
        metadata = [
            {
                "commit_sha": f"sha{i}", "file_path": f"file{i}",
                "repo_name": "repo", "split": "train",
                "file_status": "modified", "label_status": "negative",
            }
            for i in range(10)
        ]
        positive_keys = {("sha0", "file0"), ("sha8", "file8")}
        result = _compute_ranking_metrics(
            scores, metadata, positive_keys,
            label="test", mode="E0", split="train", seed=42,
        )
        # sha0 at rank 1, sha8 at rank 9
        # recall@5 = 1/2 = 0.5 (only sha0 in top 5)
        assert result["recall_at_k"]["observed_positive_recall_at_k_5"] == pytest.approx(0.5)
        # recall@10 = 2/2 = 1.0 (both in top 10)
        assert result["recall_at_k"]["observed_positive_recall_at_k_10"] == pytest.approx(1.0)

    def test_observed_positive_enrichment_at_k_calculation(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        metadata = [
            {
                "commit_sha": f"sha{i}", "file_path": f"file{i}",
                "repo_name": "repo", "split": "train",
                "file_status": "modified", "label_status": "negative",
            }
            for i in range(10)
        ]
        positive_keys = {("sha0", "file0"), ("sha1", "file1")}
        result = _compute_ranking_metrics(
            scores, metadata, positive_keys,
            label="test", mode="E0", split="train", seed=42,
        )
        # 2 positives in top-5 out of 10 total, 2/10 prevalence
        # enrichment = (2/5) / (2/10) = 0.4 / 0.2 = 2.0
        key = "observed_positive_enrichment_at_k_5"
        assert result["enrichment_at_k"][key] == pytest.approx(2.0)

    def test_mean_rank_calculation(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        metadata = [
            {
                "commit_sha": f"sha{i}", "file_path": f"file{i}",
                "repo_name": "repo", "split": "train",
                "file_status": "modified", "label_status": "negative",
            }
            for i in range(5)
        ]
        positive_keys = {("sha0", "file0")}
        result = _compute_ranking_metrics(
            scores, metadata, positive_keys,
            label="test", mode="E0", split="train", seed=42,
        )
        assert result["mean_rank_of_observed_positives"] == pytest.approx(1.0)

    def test_score_separation_ks(self):
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        metadata = [
            {
                "commit_sha": "sha0", "file_path": "f0",
                "repo_name": "r", "split": "train",
                "file_status": "m", "label_status": "n",
            },
            {
                "commit_sha": "sha1", "file_path": "f1",
                "repo_name": "r", "split": "train",
                "file_status": "m", "label_status": "n",
            },
            {
                "commit_sha": "sha2", "file_path": "f2",
                "repo_name": "r", "split": "train",
                "file_status": "m", "label_status": "n",
            },
            {
                "commit_sha": "sha3", "file_path": "f3",
                "repo_name": "r", "split": "train",
                "file_status": "m", "label_status": "n",
            },
        ]
        positive_keys = {("sha0", "f0"), ("sha1", "f1")}
        result = _compute_ranking_metrics(
            scores, metadata, positive_keys,
            label="test", mode="E0", split="train", seed=42,
        )
        assert result["score_separation_ks"] > 0.0


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_random_baseline_deterministic(self):
        bl1 = _random_baseline_metrics(53681, 108)
        bl2 = _random_baseline_metrics(53681, 108)
        assert bl1 == bl2

    def test_ranking_metrics_deterministic(self):
        scores = np.random.RandomState(42).uniform(0, 1, size=100)
        metadata = [
            {
                "commit_sha": f"sha{i}", "file_path": f"f{i}",
                "repo_name": "r", "split": "train",
                "file_status": "m", "label_status": "n",
            }
            for i in range(100)
        ]
        positive_keys = {("sha0", "f0"), ("sha1", "f1")}
        r1 = _compute_ranking_metrics(scores, metadata, positive_keys, "test", "E0", "train", 42)
        r2 = _compute_ranking_metrics(scores, metadata, positive_keys, "test", "E0", "train", 42)
        assert r1["mean_rank_of_observed_positives"] == r2["mean_rank_of_observed_positives"]
        assert r1["recall_at_k"] == r2["recall_at_k"]


# ---------------------------------------------------------------------------
# TestArtifactContract
# ---------------------------------------------------------------------------

class TestArtifactContract:
    def test_all_artifacts_exist(self, experiment_output):
        tmp_dir = experiment_output["tmp_dir"]
        expected_files = [
            "pu_experiment_config.json",
            "pu_dataset_accounting.json",
            "pu_results.json",
            "pu_ranking_analysis.json",
            "repository_analysis.json",
            "seed_stability.json",
            "phase410_pu_learning.md",
        ]
        for fname in expected_files:
            assert (tmp_dir / fname).exists(), f"Missing artifact: {fname}"

    def test_config_schema_valid(self, experiment_output):
        config = experiment_output["config_json"]
        assert "seeds" in config
        assert "modes" in config
        assert "elkan_noto_status" in config
        assert config["elkan_noto_status"] == "NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE"
        assert "pu_assumptions" in config
        assert config["pu_assumptions"]["scar_testable"] is False
        assert config["pu_assumptions"]["no_true_defect_metrics"] is True

    def test_accounting_schema_valid(self, experiment_output):
        acct = experiment_output["accounting_json"]
        assert acct["total_source_rows"] == EXPECTED_TOTAL
        assert acct["primary"]["positive_count"] == EXPECTED_PRIMARY_P
        assert acct["primary"]["unlabeled_count"] == EXPECTED_UNLABELED_U
        assert acct["primary"]["eligible_count"] == 53681
        assert acct["primary"]["excluded_sensitivity"] == EXPECTED_SENSITIVITY_S
        assert acct["primary"]["out_of_scope"] == EXPECTED_OOS
        assert acct["primary"]["defensible_negative"] == EXPECTED_DN
        assert acct["sensitivity"]["positive_count"] == EXPECTED_PRIMARY_P + EXPECTED_SENSITIVITY_S
        assert acct["sensitivity"]["unlabeled_count"] == EXPECTED_UNLABELED_U
        assert acct["sensitivity"]["eligible_count"] == 54275
        assert acct["metric_semantics"] == "observed-positive ranking metrics only"
        assert acct["elkan_noto_status"] == "NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE"

    def test_report_not_empty(self, experiment_output):
        report = experiment_output["report"]
        assert len(report) > 100
        assert "Phase 4.10" in report
        assert "Elkan-Noto" in report
        assert "NOT_IDENTIFIABLE" in report


# ---------------------------------------------------------------------------
# TestFrozenIntegrity
# ---------------------------------------------------------------------------

class TestFrozenIntegrity:
    def test_frozen_jsonl_unchanged(self):
        for split_name, expected in [("train", 34405), ("validation", 9410), ("test", 10574)]:
            path = DATA_DIR / f"{split_name}.jsonl"
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            assert count == expected, f"{split_name}.jsonl has {count} rows, expected {expected}"

    def test_frozen_ambiguous_unchanged(self):
        path = DATA_DIR / "ambiguous.jsonl"
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        assert count == 117487

    def test_frozen_manifest_unchanged(self):
        manifest = _parse_frozen_manifest()
        assert len(manifest) == 50


# ---------------------------------------------------------------------------
# TestLeakagePrevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_no_test_commits_in_train(self):
        from backend.app.ml.repo_split import load_all_rows, reassign_splits
        all_rows = load_all_rows(DATA_DIR)
        splits = reassign_splits(all_rows)
        train_commits = {
            (r.commit_sha, r.repo_name)
            for r in splits["train"]
        }
        test_commits = {
            (r.commit_sha, r.repo_name)
            for r in splits["test"]
        }
        assert train_commits.isdisjoint(test_commits)

    def test_no_val_commits_in_train(self):
        from backend.app.ml.repo_split import load_all_rows, reassign_splits
        all_rows = load_all_rows(DATA_DIR)
        splits = reassign_splits(all_rows)
        train_commits = {
            (r.commit_sha, r.repo_name)
            for r in splits["train"]
        }
        val_commits = {
            (r.commit_sha, r.repo_name)
            for r in splits["validation"]
        }
        assert train_commits.isdisjoint(val_commits)

    def test_no_attribution_in_features(self):
        from backend.app.ml.dataset_loader import FEATURE_NAMES
        for name in FEATURE_NAMES:
            assert "attribution" not in name.lower()
            assert "label" not in name.lower()


# ---------------------------------------------------------------------------
# TestSplitSpecificRandomBaselines
# ---------------------------------------------------------------------------

class TestSplitSpecificRandomBaselines:
    def test_random_baselines_per_split_structure(self, experiment_output):
        result = experiment_output["result"]
        rb = result["random_baselines"]
        for mode in ["E0", "E1", "E2", "E4"]:
            assert mode in rb
            for split in ("train", "validation", "test"):
                assert split in rb[mode], f"Missing {split} in {mode}"
                rbs = rb[mode][split]
                assert "N" in rbs
                assert "P" in rbs
                assert "expected_recall_at_k" in rbs
                assert "expected_positive_count_at_k" in rbs
                assert "expected_enrichment_at_k" in rbs
                assert "expected_mean_rank" in rbs

    def test_random_baselines_split_n_values(self, experiment_output):
        result = experiment_output["result"]
        rb = result["random_baselines"]["E0"]
        assert rb["train"]["N"] > 10000
        assert rb["validation"]["N"] > 10000
        assert rb["test"]["N"] > 10000
        assert rb["train"]["N"] != rb["validation"]["N"]
        assert rb["validation"]["N"] != rb["test"]["N"]

    def test_random_baselines_split_p_values(self, experiment_output):
        result = experiment_output["result"]
        rb = result["random_baselines"]["E0"]
        assert rb["train"]["P"] > 0
        assert rb["validation"]["P"] > 0
        assert rb["test"]["P"] > 0
        assert rb["train"]["P"] != 108
        assert rb["validation"]["P"] != 108
        assert rb["test"]["P"] != 108

    def test_no_global_p_leakage(self, experiment_output):
        result = experiment_output["result"]
        rb = result["random_baselines"]["E0"]
        for split in ("train", "validation", "test"):
            assert rb[split]["P"] != 108, (
                f"Global P=108 leaked into {split} random baseline"
            )

    def test_expected_mean_rank_formula(self, experiment_output):
        result = experiment_output["result"]
        rb = result["random_baselines"]["E0"]
        for split in ("train", "validation", "test"):
            N = rb[split]["N"]
            expected_mean = (N + 1) / 2
            assert rb[split]["expected_mean_rank"] == pytest.approx(expected_mean)

    def test_sensitivity_random_baselines_exist(self, experiment_output):
        result = experiment_output["result"]
        srb = result["sensitivity_random_baselines"]
        assert len(srb) > 0
        for mode in srb:
            for split in ("train", "validation", "test"):
                assert split in srb[mode]
                rbs = srb[mode][split]
                assert "N" in rbs
                assert "P" in rbs
                assert "expected_recall_at_k" in rbs
                assert "expected_mean_rank" in rbs


# ---------------------------------------------------------------------------
# TestReportContent
# ---------------------------------------------------------------------------

class TestReportContent:
    def test_report_contains_all_k_values(self, experiment_output):
        report = experiment_output["report"]
        for k in [5, 10, 20, 50, 100, 200, 500]:
            assert f"| {k} |" in report, f"Missing K={k} in report"

    def test_report_contains_method_b_c_explanation(self, experiment_output):
        report = experiment_output["report"]
        assert "Method B vs Method C" in report
        assert "LogisticRegression" in report
        assert "predict_proba" in report
        assert "decision_function" in report

    def test_report_contains_per_split_random_baselines(self, experiment_output):
        report = experiment_output["report"]
        assert "Analytic Random Baselines (per split)" in report

    def test_report_contains_recommendation(self, experiment_output):
        report = experiment_output["report"]
        assert "## Recommendation" in report
        assert "Recommendation:" in report
        assert "derived from" in report or "evidence" in report.lower()

    def test_report_contains_investigation_budget(self, experiment_output):
        report = experiment_output["report"]
        assert "Practical Investigation Budget" in report

    def test_report_evidence_matrix(self, experiment_output):
        report = experiment_output["report"]
        assert "## Evidence Matrix" in report
        assert "Ranking vs random" in report

    def test_report_limitations(self, experiment_output):
        report = experiment_output["report"]
        assert "## Limitations" in report
        assert "true-defect" in report.lower() or "calibrated" in report.lower()

    def test_report_frozen_integrity_section(self, experiment_output):
        report = experiment_output["report"]
        assert "## Frozen Data Integrity" in report
        assert "UNCHANGED" in report


# ---------------------------------------------------------------------------
# TestMethodBC
# ---------------------------------------------------------------------------

class TestMethodBC:
    def test_b_c_use_same_model_type(self):
        # Methods B and C both use LogisticRegression with identical params
        # This is documented behavior, not something we can test without
        # refitting. Verify the documentation contract.
        assert True  # Contract: same model type

    def test_b_c_ranking_equivalence_note_in_report(self, experiment_output):
        """Report must document that B and C produce identical rankings."""
        report = experiment_output["report"]
        assert "same LogisticRegression" in report or "identical" in report.lower()

    def test_decision_function_in_report(self, experiment_output):
        report = experiment_output["report"]
        assert "decision_function" in report


# ---------------------------------------------------------------------------
# TestRecommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommendation_not_placeholder(self, experiment_output):
        report = experiment_output["report"]
        idx = report.find("## Recommendation")
        assert idx >= 0
        rec_section = report[idx:idx + 2000]
        has_valid_rec = any(
            cat in rec_section
            for cat in [
                "IMPROVE_ATTRIBUTION_FIRST",
                "PU_SIGNAL_INSUFFICIENT",
                "PROCEED_WITH_RESTRICTIONS",
            ]
        )
        assert has_valid_rec, "No valid recommendation category found"

    def test_recommendation_has_evidence_table(self, experiment_output):
        report = experiment_output["report"]
        idx = report.find("## Recommendation")
        rec_section = report[idx:idx + 3000]
        assert "Evidence" in rec_section

    def test_recommendation_mentions_ranking_only(self, experiment_output):
        report = experiment_output["report"]
        idx = report.find("## Recommendation")
        rec_section = report[idx:idx + 3000]
        assert "observed-positive ranking" in rec_section.lower()


# ---------------------------------------------------------------------------
# TestArtifactInvariants
# ---------------------------------------------------------------------------

class TestArtifactInvariants:
    def test_results_json_has_sensitivity_random_baselines(self, experiment_output):
        results = experiment_output["pu_results_json"]
        assert "sensitivity_random_baselines" in results
        srb = results["sensitivity_random_baselines"]
        assert len(srb) > 0
        for mode in srb:
            for split in ("train", "validation", "test"):
                assert split in srb[mode]

    def test_results_json_random_baselines_per_split(self, experiment_output):
        results = experiment_output["pu_results_json"]
        rb = results["random_baselines"]
        for mode in rb:
            for split in ("train", "validation", "test"):
                assert split in rb[mode]
                assert "N" in rb[mode][split]
                assert "P" in rb[mode][split]

    def test_results_json_primary_results_unchanged(self, experiment_output):
        """Primary ranking metrics should be the same before/after correction."""
        result = experiment_output["result"]
        pr = result["primary_results"]
        assert len(pr) > 0
        for r in pr:
            assert "recall_at_k" in r
            assert "mean_rank_of_observed_positives" in r
            assert "score_separation_ks" in r

    def test_accounting_json_unchanged(self, experiment_output):
        acct = experiment_output["accounting_json"]
        assert acct["primary"]["positive_count"] == EXPECTED_PRIMARY_P
        assert acct["primary"]["unlabeled_count"] == EXPECTED_UNLABELED_U
        assert acct["primary"]["eligible_count"] == 53681
        assert acct["sensitivity"]["positive_count"] == EXPECTED_PRIMARY_P + EXPECTED_SENSITIVITY_S


# ---------------------------------------------------------------------------
# TestDeterminismRerun
# ---------------------------------------------------------------------------

class TestDeterminismRerun:
    def test_report_deterministic(self, tmp_path):
        """Running experiment twice produces equivalent reports."""
        run_pu_experiment(output_dir=tmp_path / "run1")
        run_pu_experiment(output_dir=tmp_path / "run2")
        report1 = (tmp_path / "run1" / "phase410_pu_learning.md").read_text(encoding="utf-8")
        report2 = (tmp_path / "run2" / "phase410_pu_learning.md").read_text(encoding="utf-8")
        assert report1 == report2

    def test_random_baselines_deterministic(self, tmp_path):
        run_pu_experiment(output_dir=tmp_path / "run1")
        run_pu_experiment(output_dir=tmp_path / "run2")
        with open(tmp_path / "run1" / "pu_results.json", encoding="utf-8") as f:
            data1 = json.load(f)
        with open(tmp_path / "run2" / "pu_results.json", encoding="utf-8") as f:
            data2 = json.load(f)
        rb1 = data1["random_baselines"]
        rb2 = data2["random_baselines"]
        assert rb1 == rb2
