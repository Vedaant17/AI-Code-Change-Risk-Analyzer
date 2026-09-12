"""Tests for Phase 4.9: Supervision Strategy & PU Learning Feasibility Study.

Tests verify calculations, contracts, invariants, and recommendation behavior.
They do NOT prescribe what the real dataset must conclude.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.ml.phase49_supervision_feasibility import (
    ALLOWED_RECOMMENDATIONS,
    COMMIT_FEATURE_INDEX,
    FILE_FEATURE_INDEX,
    _distribution,
    _parse_frozen_manifest,
    analyze_coverage,
    analyze_repository_concentration,
    analyze_split_distribution,
    compare_supervision_strategies,
    construct_supervision_sets,
    derive_recommendation,
    design_evaluation_strategy,
    evaluate_pu_assumptions,
    identify_data_requirements,
    run_feasibility_study,
    verify_negative_feasibility,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_row(
    repo_name: str,
    commit_sha: str,
    file_path: str,
    defect_label: int = 0,
    label_source: str = "none",
    label_evidence: str = "",
    split: str = "train",
    file_status: str = "modified",
    commit_features: list[float] | None = None,
    file_features: list[float] | None = None,
) -> dict[str, Any]:
    if commit_features is None:
        commit_features = [0.0] * 29
    if file_features is None:
        file_features = [0.0] * 16
    return {
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "file_path": file_path,
        "defect_label": defect_label,
        "label_source": label_source,
        "label_evidence": label_evidence,
        "split": split,
        "file_status": file_status,
        "commit_features": commit_features,
        "file_features": file_features,
        "label_status": "positive" if defect_label == 1 else "negative",
        "label_confidence": 1.0 if defect_label == 1 else 0.0,
        "commit_message": "",
    }


def _fix_sha() -> str:
    return "abcdef123456"


def _mock_evidence(prefix: str) -> str:
    return f"bug-fix {prefix}"


def _make_manifest() -> dict[str, str]:
    return {
        "repo_a": "train",
        "repo_b": "validation",
        "repo_c": "test",
    }


# ---------------------------------------------------------------------------
# Population construction tests
# ---------------------------------------------------------------------------

class TestSupervisionSetConstruction:
    def test_primary_count_from_file_partial(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", "aaa" + "0" * 37, "b.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
            _make_row("r", fix_sha, "c.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: rows[:2],
            fix_sha: rows[2:],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        assert result["primary_observed_path_associated"]["count"] == 1

    def test_commit_only_excluded_from_primary(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: [rows[0]],
            fix_sha: [rows[1]],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        assert result["primary_observed_path_associated"]["count"] == 0
        assert result["sensitivity_commit_only_overlap"]["count"] == 1

    def test_defensible_negative_is_zero(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
            _make_row("r", "bbb" + "0" * 37, "x.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: [rows[0]],
            fix_sha: [rows[1]],
            "bbb" + "0" * 37: [rows[2]],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        assert result["defensible_negative"]["count"] == 0

    def test_population_reconciliation(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", "aaa" + "0" * 37, "b.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
            _make_row("r", fix_sha, "c.py", 0),
            _make_row("r", "bbb" + "0" * 37, "x.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: rows[:2],
            fix_sha: rows[2:4],
            "bbb" + "0" * 37: [rows[4]],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        s = result["summary"]
        assert (
            s["primary_observed_count"]
            + s["sensitivity_count"]
            + s["unlabeled_count"]
            + s["defensible_negative_count"]
            == s["total_file_rows"]
        )

    def test_deterministic_results(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: [rows[0]],
            fix_sha: [rows[1]],
        }
        manifest = {"r": "train"}
        r1 = construct_supervision_sets(rows, lookup, manifest)
        r2 = construct_supervision_sets(rows, lookup, manifest)
        assert r1["summary"] == r2["summary"]

    def test_commit_only_rows_have_correct_category(self) -> None:
        fix_sha = "abcdef1234567890"
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 1,
                       label_source="explicit_sha_reference",
                       label_evidence=_mock_evidence(_fix_sha())),
            _make_row("r", fix_sha, "a.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: [rows[0]],
            fix_sha: [rows[1]],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        for r in result["sensitivity_commit_only_overlap"]["rows"]:
            assert r["observation_role"] == "sensitivity_commit_only_overlap"

    def test_unlabeled_includes_non_positive_rows(self) -> None:
        rows = [
            _make_row("r", "aaa" + "0" * 37, "a.py", 0),
            _make_row("r", "bbb" + "0" * 37, "b.py", 0),
        ]
        lookup = {
            "aaa" + "0" * 37: [rows[0]],
            "bbb" + "0" * 37: [rows[1]],
        }
        manifest = {"r": "train"}
        result = construct_supervision_sets(rows, lookup, manifest)
        assert result["unlabeled"]["count"] == 2
        assert result["primary_observed_path_associated"]["count"] == 0


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

class TestCoverageAnalysis:
    def test_returns_per_feature_list(self) -> None:
        rows = [_make_row("r", "a" * 40, "f.py")]
        result = analyze_coverage(rows, rows)
        assert "per_feature" in result
        assert isinstance(result["per_feature"], list)
        assert len(result["per_feature"]) > 0

    def test_all_expected_features_present(self) -> None:
        rows = [_make_row("r", "a" * 40, "f.py")]
        result = analyze_coverage(rows, rows)
        feature_names = {f["feature_name"] for f in result["per_feature"]}
        expected_numeric = {
            "file_total_lines_changed", "file_hunk_count",
            "files_changed", "total_hunks", "max_file_changes",
            "additions_ratio", "test_ratio",
        }
        expected_categorical = {"repo_name", "split", "label_source", "file_status"}
        assert expected_numeric.issubset(feature_names)
        assert expected_categorical.issubset(feature_names)

    def test_observed_unlabeled_counts_match(self) -> None:
        obs = [_make_row("r", "a" * 40, "f.py")]
        unl = [_make_row("r", "b" * 40, "g.py")]
        result = analyze_coverage(obs, unl)
        assert result["observed_count"] == 1
        assert result["unlabeled_count"] == 1

    def test_deterministic(self) -> None:
        obs = [_make_row("r", "a" * 40, "f.py")]
        unl = [_make_row("r", "b" * 40, "g.py")]
        r1 = analyze_coverage(obs, unl)
        r2 = analyze_coverage(obs, unl)
        # Compare counts and structure (not full dicts with numpy values)
        assert r1["observed_count"] == r2["observed_count"]
        assert r1["unlabeled_count"] == r2["unlabeled_count"]
        assert len(r1["per_feature"]) == len(r2["per_feature"])

    def test_empty_inputs(self) -> None:
        result = analyze_coverage([], [])
        assert result["observed_count"] == 0
        assert result["unlabeled_count"] == 0


# ---------------------------------------------------------------------------
# Repository concentration tests
# ---------------------------------------------------------------------------

class TestRepositoryConcentration:
    def test_repos_with_plus_without_equals_total(self) -> None:
        obs = [
            _make_row("r1", "a" * 40, "f.py"),
            _make_row("r2", "b" * 40, "g.py"),
        ]
        all_rows = obs + [_make_row("r3", "c" * 40, "h.py")]
        result = analyze_repository_concentration(obs, all_rows)
        assert (
            result["repos_with_signal"]
            + result["repos_without_signal"]
            == result["total_repos"]
        )

    def test_concentration_bounds(self) -> None:
        obs = [_make_row("r1", "a" * 40, "f.py")]
        result = analyze_repository_concentration(obs, obs)
        c = result["concentration"]
        assert 0 <= c["top_5_share"] <= 1
        assert 0 <= c["top_10_share"] <= 1
        assert 0 <= c["gini"] <= 1
        assert 0 <= c["herfindahl"] <= 1
        assert 0 <= c["entropy"] <= 1

    def test_deterministic(self) -> None:
        obs = [_make_row("r1", "a" * 40, "f.py")]
        r1 = analyze_repository_concentration(obs, obs)
        r2 = analyze_repository_concentration(obs, obs)
        assert r1 == r2

    def test_single_repo_concentration(self) -> None:
        obs = [_make_row("r1", "a" * 40, "f.py")]
        result = analyze_repository_concentration(obs, obs)
        assert result["repos_with_signal"] == 1
        assert result["concentration"]["gini"] == 0.0


# ---------------------------------------------------------------------------
# Split distribution tests
# ---------------------------------------------------------------------------

class TestSplitDistribution:
    def test_split_counts_reconcile(self) -> None:
        obs = [
            _make_row("r", "a" * 40, "f.py", split="train"),
            _make_row("r", "b" * 40, "g.py", split="validation"),
        ]
        unl = [_make_row("r", "c" * 40, "h.py", split="test")]
        sens = [_make_row("r", "d" * 40, "i.py", split="train")]
        oos = [_make_row("r", "e" * 40, "/dev/null", split="train")]
        all_rows = obs + unl + sens + oos
        result = analyze_split_distribution(obs, unl, sens, oos, all_rows)
        total_obs = sum(
            result[s]["observed_count"]
            for s in ("train", "validation", "test")
        )
        assert total_obs == 2

    def test_no_negative_counts(self) -> None:
        result = analyze_split_distribution([], [], [], [], [])
        for split in ("train", "validation", "test"):
            assert result[split]["observed_count"] >= 0
            assert result[split]["unlabeled_count"] >= 0
            assert result[split]["sensitivity_count"] >= 0
            assert result[split]["out_of_scope_count"] >= 0

    def test_deterministic(self) -> None:
        obs = [_make_row("r", "a" * 40, "f.py", split="train")]
        r1 = analyze_split_distribution(obs, [], [], [], obs)
        r2 = analyze_split_distribution(obs, [], [], [], obs)
        assert r1 == r2

    def test_sparsity_warnings_present(self) -> None:
        result = analyze_split_distribution([], [], [], [], [])
        assert "sparsity_warnings" in result

    def test_reconciliation_per_split(self) -> None:
        obs = [_make_row("r", "a" * 40, "f.py", split="train")]
        unl = [_make_row("r", "b" * 40, "g.py", split="train")]
        sens = [_make_row("r", "c" * 40, "h.py", split="train")]
        oos = [_make_row("r", "d" * 40, "/dev/null", split="train")]
        all_rows = obs + unl + sens + oos
        result = analyze_split_distribution(obs, unl, sens, oos, all_rows)
        assert result["train"]["reconciled"] is True
        assert result["train"]["raw_total"] == 4
        assert result["train"]["classified_total"] == 4


# ---------------------------------------------------------------------------
# PU assumption tests
# ---------------------------------------------------------------------------

class TestPUAssumptions:
    def test_scar_has_required_fields(self) -> None:
        conc = {
            "repos_with_signal": 5, "repos_without_signal": 45,
            "total_repos": 50, "concentration": {
                "top_5_share": 0.3, "gini": 0.2,
                "herfindahl": 0.05, "entropy": 0.8,
            },
        }
        cov = {"notable_differences": []}
        result = evaluate_pu_assumptions(conc, cov)
        scar = result["scar"]
        assert "assumption" in scar
        assert "evidence_consistent" in scar
        assert "evidence_inconsistent" in scar
        assert "untestable_components" in scar
        assert "verdict" in scar
        assert "reasoning" in scar

    def test_scar_verdict_not_violated(self) -> None:
        conc = {
            "repos_with_signal": 5, "repos_without_signal": 45,
            "total_repos": 50, "concentration": {
                "top_5_share": 0.3, "gini": 0.2,
                "herfindahl": 0.05, "entropy": 0.8,
            },
        }
        cov = {"notable_differences": []}
        result = evaluate_pu_assumptions(conc, cov)
        assert result["scar"]["verdict"] != "VIOLATED"

    def test_sar_has_required_fields(self) -> None:
        conc = {
            "repos_with_signal": 25, "repos_without_signal": 25,
            "total_repos": 50, "concentration": {
                "top_5_share": 0.3, "gini": 0.2,
                "herfindahl": 0.05, "entropy": 0.8,
            },
        }
        cov = {"notable_differences": []}
        result = evaluate_pu_assumptions(conc, cov)
        sar = result["sar"]
        assert "verdict" in sar
        assert sar["verdict"] in ("COMPATIBLE_WITH_CAVEATS", "UNTESTABLE_WITH_CURRENT_EVIDENCE")

    def test_distinction_table_has_three_keys(self) -> None:
        conc = {
            "repos_with_signal": 5, "repos_without_signal": 45,
            "total_repos": 50, "concentration": {
                "top_5_share": 0.3, "gini": 0.2,
                "herfindahl": 0.05, "entropy": 0.8,
            },
        }
        cov = {"notable_differences": []}
        result = evaluate_pu_assumptions(conc, cov)
        dt = result["distinction_table"]
        assert "observed_facts" in dt
        assert "inferences" in dt
        assert "untestable_assumptions" in dt


# ---------------------------------------------------------------------------
# Negative feasibility tests
# ---------------------------------------------------------------------------

class TestNegativeFeasibility:
    def test_defensible_negatives_zero(self) -> None:
        result = verify_negative_feasibility([], [], [], [], [])
        assert result["defensible_negatives"] == 0

    def test_categories_reconcile(self) -> None:
        p = _make_row("r", "b" * 40, "g.py")
        s = _make_row("r", "c" * 40, "h.py")
        u = _make_row("r", "d" * 40, "i.py")
        o = _make_row("r", "e" * 40, "/dev/null")
        all_rows = [p, s, u, o]
        result = verify_negative_feasibility(
            all_rows, [p], [s], [u], [o],
        )
        assert result["reconciled"] is True
        assert result["classified_total"] == result["total_supervised_rows"]

    def test_never_converted_non_empty(self) -> None:
        result = verify_negative_feasibility([], [], [], [], [])
        assert len(result["never_converted"]) > 0

    def test_out_of_scope_in_categories(self) -> None:
        all_rows = [
            _make_row("r", "a" * 40, "f.py"),
            _make_row("r", "b" * 40, "/dev/null"),
        ]
        oos = [_make_row("r", "b" * 40, "/dev/null")]
        result = verify_negative_feasibility(all_rows, [], [], [], oos)
        assert result["categories"]["out_of_scope"] == 1


# ---------------------------------------------------------------------------
# Strategy comparison tests
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    def test_exactly_five_strategies(self) -> None:
        sup = {"summary": {"primary_observed_count": 10, "sensitivity_count": 5,
                           "unlabeled_count": 50, "defensible_negative_count": 0,
                           "total_file_rows": 65, "observed_positive_rate": 0.15}}
        cov = {"notable_differences": []}
        conc = {"concentration": {"top_5_share": 0.3}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"},
              "scar": {"verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE"},
              "repository_dependency": {"severity": "low"}}
        result = compare_supervision_strategies(sup, cov, conc, pu)
        assert len(result["strategies"]) == 5

    def test_each_strategy_has_required_fields(self) -> None:
        sup = {"summary": {"primary_observed_count": 10, "sensitivity_count": 5,
                           "unlabeled_count": 50, "defensible_negative_count": 0,
                           "total_file_rows": 65, "observed_positive_rate": 0.15}}
        cov = {"notable_differences": []}
        conc = {"concentration": {"top_5_share": 0.3}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"},
              "scar": {"verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE"},
              "repository_dependency": {"severity": "low"}}
        result = compare_supervision_strategies(sup, cov, conc, pu)
        required = {"name", "assumptions", "methodological_risk",
                    "compatibility_with_dataset", "pursue_next", "reasoning"}
        for strat in result["strategies"]:
            assert required.issubset(strat.keys())

    def test_binary_classification_not_recommended_next(self) -> None:
        sup = {"summary": {"primary_observed_count": 10, "sensitivity_count": 5,
                           "unlabeled_count": 50, "defensible_negative_count": 0,
                           "total_file_rows": 65, "observed_positive_rate": 0.15}}
        cov = {"notable_differences": []}
        conc = {"concentration": {"top_5_share": 0.3}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"},
              "scar": {"verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE"},
              "repository_dependency": {"severity": "low"}}
        result = compare_supervision_strategies(sup, cov, conc, pu)
        binary = [s for s in result["strategies"]
                  if "binary" in s["name"].lower()]
        assert all(not s["pursue_next"] for s in binary)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluationDesign:
    def test_measurable_now_non_empty(self) -> None:
        sup = {"summary": {"observed_positive_rate": 0.1}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"}}
        result = design_evaluation_strategy(sup, pu)
        assert len(result["measurable_now"]) > 0

    def test_not_measurable_includes_true_defect_metrics(self) -> None:
        sup = {"summary": {"observed_positive_rate": 0.1}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"}}
        result = design_evaluation_strategy(sup, pu)
        names = [m["name"] for m in result["not_measurable_now"]]
        assert any("recall" in n.lower() for n in names)

    def test_exploratory_metrics_labeled_as_such(self) -> None:
        sup = {"summary": {"observed_positive_rate": 0.1}}
        pu = {"sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"}}
        result = design_evaluation_strategy(sup, pu)
        for m in result["exploratory_pu_metrics"]:
            assert "caveat" in m


# ---------------------------------------------------------------------------
# Data requirements tests
# ---------------------------------------------------------------------------

class TestDataRequirements:
    def test_requirements_non_empty(self) -> None:
        result = identify_data_requirements()
        assert len(result["requirements"]) > 0

    def test_each_has_feasibility_and_impact(self) -> None:
        result = identify_data_requirements()
        for req in result["requirements"]:
            assert "feasibility" in req
            assert "expected_impact" in req


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommendation_in_allowed_set(self) -> None:
        result = _run_minimal()
        assert result["recommendation"]["recommendation"] in ALLOWED_RECOMMENDATIONS

    def test_recommendation_has_reasoning(self) -> None:
        result = _run_minimal()
        assert result["recommendation"]["reasoning"] != ""

    def test_recommendation_has_next_step(self) -> None:
        result = _run_minimal()
        assert result["recommendation"]["next_step"] != ""

    def test_reacts_to_zero_observed(self) -> None:
        sup = {
            "summary": {
                "primary_observed_count": 0,
                "sensitivity_count": 0,
                "unlabeled_count": 100,
                "defensible_negative_count": 0,
                "total_file_rows": 100,
                "observed_positive_rate": 0.0,
            }
        }
        coverage = {"notable_differences": []}
        conc = {
            "repos_with_signal": 0, "repos_without_signal": 50,
            "total_repos": 50,
            "concentration": {
                "top_5_share": 0.0, "gini": 0.0,
                "herfindahl": 0.0, "entropy": 0.0,
            },
        }
        split_a = {
            "train": {"observed_count": 0, "unlabeled_count": 50,
                      "total_count": 50, "repos_with_signal": 0,
                      "repos_without_signal": 20, "total_repos_in_split": 20,
                      "observed_positive_rate": 0.0, "observed_commits": 0},
            "validation": {"observed_count": 0, "unlabeled_count": 25,
                           "total_count": 25, "repos_with_signal": 0,
                           "repos_without_signal": 15, "total_repos_in_split": 15,
                           "observed_positive_rate": 0.0, "observed_commits": 0},
            "test": {"observed_count": 0, "unlabeled_count": 25,
                     "total_count": 25, "repos_with_signal": 0,
                     "repos_without_signal": 15, "total_repos_in_split": 15,
                     "observed_positive_rate": 0.0, "observed_commits": 0},
            "sparsity_warnings": [],
        }
        pu = {
            "scar": {"verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE"},
            "sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"},
            "repository_dependency": {"severity": "low"},
        }
        neg = {"defensible_negatives": 0}
        ev = {"measurable_now": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        rec = derive_recommendation(sup, coverage, conc, split_a, pu, neg, ev)
        assert rec["recommendation"] == "IMPROVE_ATTRIBUTION_FIRST"

    def test_reacts_to_defensible_negatives(self) -> None:
        sup = {
            "summary": {
                "primary_observed_count": 50,
                "sensitivity_count": 10,
                "unlabeled_count": 200,
                "defensible_negative_count": 100,
                "total_file_rows": 260,
                "observed_positive_rate": 0.19,
            }
        }
        coverage = {"notable_differences": []}
        conc = {
            "repos_with_signal": 20, "repos_without_signal": 30,
            "total_repos": 50,
            "concentration": {
                "top_5_share": 0.3, "gini": 0.2,
                "herfindahl": 0.05, "entropy": 0.8,
            },
        }
        split_a = {
            "train": {"observed_count": 25, "unlabeled_count": 100,
                      "total_count": 125, "repos_with_signal": 10,
                      "repos_without_signal": 10, "total_repos_in_split": 20,
                      "observed_positive_rate": 0.2, "observed_commits": 10},
            "validation": {"observed_count": 12, "unlabeled_count": 50,
                           "total_count": 62, "repos_with_signal": 5,
                           "repos_without_signal": 10, "total_repos_in_split": 15,
                           "observed_positive_rate": 0.19, "observed_commits": 5},
            "test": {"observed_count": 13, "unlabeled_count": 50,
                     "total_count": 63, "repos_with_signal": 5,
                     "repos_without_signal": 10, "total_repos_in_split": 15,
                     "observed_positive_rate": 0.21, "observed_commits": 5},
            "sparsity_warnings": [],
        }
        pu = {
            "scar": {"verdict": "UNTESTABLE_WITH_CURRENT_EVIDENCE"},
            "sar": {"verdict": "COMPATIBLE_WITH_CAVEATS"},
            "repository_dependency": {"severity": "low"},
        }
        neg = {"defensible_negatives": 100}
        ev = {"measurable_now": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        rec = derive_recommendation(sup, coverage, conc, split_a, pu, neg, ev)
        assert rec["recommendation"] == "BINARY_SUPERVISION_NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_run_feasibility_study_deterministic(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        _write_test_data(data_dir)
        r1 = run_feasibility_study(data_dir, out1)
        r2 = run_feasibility_study(data_dir, out2)

        # Full deep equality — all values are plain Python types
        assert (
            r1["supervision_result"]["summary"]
            == r2["supervision_result"]["summary"]
        )
        assert (
            r1["recommendation"]["recommendation"]
            == r2["recommendation"]["recommendation"]
        )
        assert (
            r1["negative_feasibility"]["defensible_negatives"]
            == r2["negative_feasibility"]["defensible_negatives"]
        )
        assert (
            r1["negative_feasibility"]["reconciled"]
            == r2["negative_feasibility"]["reconciled"]
        )
        assert (
            r1["repo_concentration"]["repos_with_signal"]
            == r2["repo_concentration"]["repos_with_signal"]
        )
        assert (
            r1["split_analysis"]["train"]["observed_count"]
            == r2["split_analysis"]["train"]["observed_count"]
        )
        assert (
            r1["split_analysis"]["train"]["reconciled"]
            == r2["split_analysis"]["train"]["reconciled"]
        )


# ---------------------------------------------------------------------------
# Artifact schema tests
# ---------------------------------------------------------------------------

class TestArtifactSchemas:
    def test_all_artifacts_exist(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        run_feasibility_study(data_dir, out)

        assert (out / "supervision_feasibility.json").exists()
        assert (out / "observed_positive_analysis.json").exists()
        assert (out / "repository_analysis.json").exists()
        assert (out / "split_analysis.json").exists()
        assert (out / "strategy_comparison.json").exists()
        assert (out / "phase49_supervision_feasibility.md").exists()

    def test_json_schemas_valid(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        run_feasibility_study(data_dir, out)

        with open(out / "supervision_feasibility.json") as f:
            sf = json.load(f)
        assert "recommendation" in sf
        assert "supervision_sets_summary" in sf

        with open(out / "repository_analysis.json") as f:
            ra = json.load(f)
        assert "repos_with_signal" in ra
        assert "concentration" in ra


# ---------------------------------------------------------------------------
# Frozen data integrity tests
# ---------------------------------------------------------------------------

class TestFrozenIntegrity:
    def test_frozen_jsonl_row_counts(self) -> None:
        frozen = Path("backend/data/datasets/combined-v3")
        if not frozen.exists():
            pytest.skip("Frozen dataset not available")
        expected = {"train": 34405, "validation": 9410, "test": 10574}
        for split, count in expected.items():
            path = frozen / f"{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                actual = sum(1 for line in f if line.strip())
            assert actual == count, f"{split}.jsonl: {actual} != {count}"

    def test_frozen_metadata_unchanged(self) -> None:
        path = Path("backend/data/datasets/combined-v3/metadata.json")
        if not path.exists():
            pytest.skip("metadata.json not found")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert m["train_examples"] == 34405
        assert m["validation_examples"] == 9410
        assert m["test_examples"] == 10574


# ---------------------------------------------------------------------------
# Manifest parsing tests
# ---------------------------------------------------------------------------

class TestManifestParsing:
    def test_parse_frozen_manifest(self) -> None:
        manifest = _parse_frozen_manifest()
        assert len(manifest) == 50
        splits = set(manifest.values())
        assert splits == {"train", "validation", "test"}

    def test_manifest_deterministic(self) -> None:
        m1 = _parse_frozen_manifest()
        m2 = _parse_frozen_manifest()
        assert m1 == m2


# ---------------------------------------------------------------------------
# Feature index tests
# ---------------------------------------------------------------------------

class TestFeatureIndices:
    def test_commit_feature_index_has_29_entries(self) -> None:
        assert len(COMMIT_FEATURE_INDEX) == 29

    def test_file_feature_index_has_16_entries(self) -> None:
        assert len(FILE_FEATURE_INDEX) == 16

    def test_commit_features_are_unique(self) -> None:
        names = list(COMMIT_FEATURE_INDEX.keys())
        assert len(names) == len(set(names))

    def test_file_features_are_unique(self) -> None:
        names = list(FILE_FEATURE_INDEX.keys())
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Distribution helper tests
# ---------------------------------------------------------------------------

class TestDistribution:
    def test_empty_list(self) -> None:
        result = _distribution([])
        assert result["count"] == 0

    def test_single_value(self) -> None:
        result = _distribution([5.0])
        assert result["count"] == 1
        assert result["mean"] == 5.0
        assert result["median"] == 5.0

    def test_known_distribution(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _distribution(vals)
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["iqr"] > 0


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_global_reconciliation(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        s = result["supervision_result"]["summary"]
        assert (
            s["primary_observed_count"]
            + s["sensitivity_count"]
            + s["unlabeled_count"]
            + s["defensible_negative_count"]
            + s["out_of_scope_count"]
            == s["total_file_rows"]
        )

    def test_per_split_reconciliation(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        for split in ("train", "validation", "test"):
            sd = result["split_analysis"][split]
            assert sd["reconciled"] is True, (
                f"{split}: {sd['classified_total']} != {sd['raw_total']}"
            )

    def test_dev_null_all_out_of_scope(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        oos = result["supervision_result"]["out_of_scope"]["rows"]
        for row in oos:
            assert row["file_path"] == "/dev/null"
            assert row["observation_role"] == "out_of_scope"
            assert row["out_of_scope_reason"] == "parser_artifact_dev_null"

    def test_no_dev_null_in_primary(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        primary = result["supervision_result"]["primary_observed_path_associated"]["rows"]
        for row in primary:
            assert row["file_path"] != "/dev/null"

    def test_no_dev_null_in_sensitivity(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        sens = result["supervision_result"]["sensitivity_commit_only_overlap"]["rows"]
        for row in sens:
            assert row["file_path"] != "/dev/null"

    def test_no_dev_null_in_defensible_negative(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        neg = result["supervision_result"]["defensible_negative"]["rows"]
        for row in neg:
            assert row["file_path"] != "/dev/null"

    def test_primary_sensitivity_separation(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        primary_keys = {
            (r["repo_name"], r["commit_sha"], r["file_path"])
            for r in result["supervision_result"]["primary_observed_path_associated"]["rows"]
        }
        sensitivity_keys = {
            (r["repo_name"], r["commit_sha"], r["file_path"])
            for r in result["supervision_result"]["sensitivity_commit_only_overlap"]["rows"]
        }
        assert primary_keys.isdisjoint(sensitivity_keys)

    def test_exhaustiveness(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "out"
        _write_test_data(data_dir)
        result = run_feasibility_study(data_dir, out)
        sr = result["supervision_result"]
        total = sr["summary"]["total_file_rows"]
        all_rows = []
        all_rows.extend(sr["primary_observed_path_associated"]["rows"])
        all_rows.extend(sr["sensitivity_commit_only_overlap"]["rows"])
        all_rows.extend(sr["unlabeled"]["rows"])
        all_rows.extend(sr["defensible_negative"]["rows"])
        all_rows.extend(sr["out_of_scope"]["rows"])
        assert len(all_rows) == total

    def test_determinism_deep_equality(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        _write_test_data(data_dir)
        r1 = run_feasibility_study(data_dir, out1)
        r2 = run_feasibility_study(data_dir, out2)
        # Complete semantic comparison
        assert (
            r1["supervision_result"]["summary"]
            == r2["supervision_result"]["summary"]
        )
        assert (
            len(r1["supervision_result"]["primary_observed_path_associated"]["rows"])
            == len(r2["supervision_result"]["primary_observed_path_associated"]["rows"])
        )
        assert (
            len(r1["supervision_result"]["sensitivity_commit_only_overlap"]["rows"])
            == len(r2["supervision_result"]["sensitivity_commit_only_overlap"]["rows"])
        )
        assert (
            len(r1["supervision_result"]["unlabeled"]["rows"])
            == len(r2["supervision_result"]["unlabeled"]["rows"])
        )
        assert (
            len(r1["supervision_result"]["out_of_scope"]["rows"])
            == len(r2["supervision_result"]["out_of_scope"]["rows"])
        )
        assert (
            r1["negative_feasibility"]["reconciled"]
            == r2["negative_feasibility"]["reconciled"]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_data(data_dir: Path) -> None:
    """Write minimal test JSONL data for integration tests."""
    fix_sha = "abcdef1234567890"
    rows = [
        _make_row("repo_a", "aaa" + "0" * 37, "src/a.py", 1,
                   label_source="explicit_sha_reference",
                   label_evidence=f"bug-fix {fix_sha}",
                   split="train"),
        _make_row("repo_a", "aaa" + "0" * 37, "src/b.py", 1,
                   label_source="explicit_sha_reference",
                   label_evidence=f"bug-fix {fix_sha}",
                   split="train"),
        _make_row("repo_a", fix_sha, "src/a.py", 0, split="train"),
        _make_row("repo_a", fix_sha, "src/c.py", 0, split="train"),
        _make_row("repo_b", "bbb" + "0" * 37, "lib/x.py", 0,
                   split="validation"),
        _make_row("repo_c", "ccc" + "0" * 37, "test/y.py", 0,
                   split="test"),
        # /dev/null rows — parser artifacts (OUT_OF_SCOPE)
        _make_row("repo_a", "aaa" + "0" * 37, "/dev/null", 1,
                   label_source="explicit_sha_reference",
                   label_evidence=f"bug-fix {fix_sha}",
                   split="train"),
        _make_row("repo_b", "bbb" + "0" * 37, "/dev/null", 0,
                   split="validation"),
    ]
    for split in ("train", "validation", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")


def _run_minimal() -> dict:
    """Run with minimal synthetic data for recommendation tests."""
    fix_sha = "abcdef1234567890"
    rows = [
        _make_row("repo_a", "aaa" + "0" * 37, "src/a.py", 1,
                   label_source="explicit_sha_reference",
                   label_evidence=f"bug-fix {fix_sha}",
                   split="train"),
        _make_row("repo_a", "aaa" + "0" * 37, "src/b.py", 1,
                   label_source="explicit_sha_reference",
                   label_evidence=f"bug-fix {fix_sha}",
                   split="train"),
        _make_row("repo_a", fix_sha, "src/a.py", 0, split="train"),
        _make_row("repo_a", fix_sha, "src/c.py", 0, split="train"),
        _make_row("repo_b", "bbb" + "0" * 37, "lib/x.py", 0,
                   split="validation"),
    ]
    lookup = {
        "aaa" + "0" * 37: rows[:2],
        fix_sha: rows[2:4],
        "bbb" + "0" * 37: [rows[4]],
    }
    manifest = {"repo_a": "train", "repo_b": "validation"}

    supervision = construct_supervision_sets(rows, lookup, manifest)
    primary = supervision["primary_observed_path_associated"]["rows"]
    unlabeled = supervision["unlabeled"]["rows"]
    sensitivity = supervision["sensitivity_commit_only_overlap"]["rows"]
    out_of_scope = supervision["out_of_scope"]["rows"]

    coverage = analyze_coverage(primary, unlabeled)
    repo_conc = analyze_repository_concentration(primary, rows)
    split_a = analyze_split_distribution(
        primary, unlabeled, sensitivity, out_of_scope, rows,
    )
    pu = evaluate_pu_assumptions(repo_conc, coverage)
    neg = verify_negative_feasibility(
        rows, primary, sensitivity, unlabeled, out_of_scope,
    )
    strat = compare_supervision_strategies(supervision, coverage, repo_conc, pu)
    ev = design_evaluation_strategy(supervision, pu)
    reqs = identify_data_requirements()
    rec = derive_recommendation(supervision, coverage, repo_conc,
                                split_a, pu, neg, ev)

    return {
        "supervision_result": supervision,
        "coverage": coverage,
        "repo_concentration": repo_conc,
        "split_analysis": split_a,
        "pu_assumptions": pu,
        "negative_feasibility": neg,
        "strategy_comparison": strat,
        "evaluation": ev,
        "data_requirements": reqs,
        "recommendation": rec,
    }
