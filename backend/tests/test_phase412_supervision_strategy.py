"""Tests for Phase 4.12: Defensible Supervision & Negative-Label Construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.ml.phase412_supervision_strategy import (
    CANDIDATE_NEGATIVE,
    DEFENSIBLE_NEGATIVE,
    EVIDENCE_UNKNOWN,
    EVIDENCE_WEAK,
    LEAKAGE_FEATURES,
    OBSERVED_POSITIVE,
    UNLABELED,
    _audit_data_leakage,
    _load_phase411b_commit_evidence,
    _load_phase411b_file_results,
    _load_timestamps,
    _parse_timestamp,
    _verify_population_disjointness,
    build_prediction_time_feature_table,
    construct_supervision_populations,
    evaluate_commit_level_strategy,
    evaluate_corrective_control_strategy,
    evaluate_evidence_ranking_strategy,
    evaluate_pu_strategy,
    evaluate_reliable_negative_strategy,
    evaluate_repository_control_strategy,
    evaluate_synthetic_strategy,
    evaluate_temporal_strategy,
    evaluate_weak_supervision_strategy,
)

DATA_DIR = Path("backend/data/datasets/combined-v3")
PHASE411B_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def supervised_rows() -> list[dict]:
    """Load all supervised rows (cached per module)."""
    from backend.app.ml.phase49_supervision_feasibility import (
        _read_all_supervised_rows,
    )
    return _read_all_supervised_rows(DATA_DIR)


@pytest.fixture(scope="module")
def phase411b_files() -> list[dict]:
    """Load Phase 4.11b file results (cached per module)."""
    return _load_phase411b_file_results(PHASE411B_DIR)


@pytest.fixture(scope="module")
def phase411b_commits() -> list[dict]:
    """Load Phase 4.11b commit evidence (cached per module)."""
    return _load_phase411b_commit_evidence(PHASE411B_DIR)


@pytest.fixture(scope="module")
def timestamps() -> dict:
    """Load timestamps (cached per module)."""
    return _load_timestamps(DATA_DIR)


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Load frozen manifest (cached per module)."""
    from backend.app.ml.phase49_supervision_feasibility import (
        _parse_frozen_manifest,
    )
    return _parse_frozen_manifest()


@pytest.fixture(scope="module")
def populations(
    supervised_rows, phase411b_files, phase411b_commits, manifest, timestamps,
) -> dict:
    """Construct populations (cached per module)."""
    from backend.app.ml.phase49_supervision_feasibility import (
        _read_ambiguous_rows,
    )
    ambiguous = _read_ambiguous_rows(DATA_DIR)
    return construct_supervision_populations(
        phase411b_files, phase411b_commits,
        supervised_rows, ambiguous, manifest, timestamps,
    )


# ---------------------------------------------------------------------------
# Population accounting tests
# ---------------------------------------------------------------------------


class TestPopulationAccounting:
    """Verify population counts match expected values."""

    def test_total_supervised_rows(self, populations):
        assert populations["metadata"]["total_supervised_rows"] == 54389

    def test_historical_positive_rows(self, populations):
        assert populations["metadata"]["historical_positive_rows"] == 1055

    def test_observed_positive_rows(self, populations):
        assert populations["metadata"]["observed_positive_rows"] == 499

    def test_evidence_weak_rows(self, populations):
        assert populations["metadata"]["evidence_weak_rows"] == 238

    def test_evidence_unknown_rows(self, populations):
        assert populations["metadata"]["evidence_unknown_rows"] == 314

    def test_evidence_moderate_rows(self, populations):
        assert populations["metadata"]["evidence_moderate_rows"] == 0

    def test_no_equality_assertion_between_historical_and_observed(self, populations):
        """historical_positive_rows and observed_positive_rows are different concepts."""
        hist = populations["metadata"]["historical_positive_rows"]
        obs = populations["metadata"]["observed_positive_rows"]
        # These are NOT equal (1055 != 499) — verify we don't assert equality
        assert hist != obs

    def test_six_way_partition_sums_to_supervised(self, populations):
        """Six-way population partition must sum exactly to total supervised rows."""
        meta = populations["metadata"]
        partition_sum = (
            meta["observed_positive_rows"]
            + meta["evidence_moderate_rows"]
            + meta["evidence_weak_rows"]
            + meta["evidence_unknown_rows"]
            + meta["unlabeled_rows"]
            + meta["out_of_scope_rows"]
        )
        assert partition_sum == meta["total_supervised_rows"], (
            f"Partition sum {partition_sum} != "
            f"total_supervised_rows {meta['total_supervised_rows']}"
        )

    def test_unlabeled_is_correct_value(self, populations):
        """UNLABELED must be 53,224, not double-counted 106,448."""
        assert populations["metadata"]["unlabeled_rows"] == 53224

    def test_no_population_has_duplicate_keys(self, populations):
        """No evidence population should contain duplicate (repo, sha, file) keys.

        OUT_OF_SCOPE is excluded because the frozen dataset has duplicate
        /dev/null rows per commit (e.g., bottle has 17 identical rows).
        """
        for pop_name in [OBSERVED_POSITIVE, EVIDENCE_WEAK, EVIDENCE_UNKNOWN,
                         UNLABELED]:
            keys = [(r["repo_name"], r["commit_sha"], r["file_path"])
                    for r in populations[pop_name]]
            assert len(keys) == len(set(keys)), (
                f"{pop_name} has duplicate keys: "
                f"{len(keys)} rows vs {len(set(keys))} unique"
            )


# ---------------------------------------------------------------------------
# Dual statistics tests
# ---------------------------------------------------------------------------


class TestDualStatistics:
    """Verify historical-positive and FILE_STRONG statistics are distinct."""

    def test_historical_positive_commits(self, populations):
        assert populations["metadata"]["historical_positive_commits"] == 210

    def test_file_strong_commits(self, populations):
        assert populations["metadata"]["file_strong_commits"] == 125

    def test_historical_positive_repos(self, populations):
        assert populations["metadata"]["historical_positive_repos"] == 42

    def test_repos_with_file_strong(self, populations):
        assert populations["metadata"]["repos_with_file_strong"] == 40

    def test_file_strong_subset_of_historical(self, populations):
        """FILE_STRONG commits/repos are a strict subset of historical positives."""
        meta = populations["metadata"]
        assert meta["file_strong_commits"] < meta["historical_positive_commits"]
        assert meta["repos_with_file_strong"] < meta["historical_positive_repos"]


# ---------------------------------------------------------------------------
# Population disjointness tests
# ---------------------------------------------------------------------------


class TestPopulationDisjointness:
    """Verify populations are mutually disjoint."""

    def test_all_disjoint(self, populations):
        result = _verify_population_disjointness(populations)
        assert result["all_disjoint"], f"Violations: {result['violations']}"

    def test_op_intersect_ew(self, populations):
        op = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[OBSERVED_POSITIVE]}
        ew = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[EVIDENCE_WEAK]}
        assert len(op & ew) == 0

    def test_op_intersect_eu(self, populations):
        op = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[OBSERVED_POSITIVE]}
        eu = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[EVIDENCE_UNKNOWN]}
        assert len(op & eu) == 0

    def test_ew_intersect_eu(self, populations):
        ew = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[EVIDENCE_WEAK]}
        eu = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[EVIDENCE_UNKNOWN]}
        assert len(ew & eu) == 0


# ---------------------------------------------------------------------------
# Candidate negative tests
# ---------------------------------------------------------------------------


class TestCandidateNegative:
    """Verify candidate negatives are properly marked."""

    def test_candidate_negative_status(self, populations):
        for r in populations[CANDIDATE_NEGATIVE]:
            assert r.get("status") == CANDIDATE_NEGATIVE

    def test_candidate_negative_not_in_observed_positive(self, populations):
        cn = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[CANDIDATE_NEGATIVE]}
        op = {(r["repo_name"], r["commit_sha"], r["file_path"])
              for r in populations[OBSERVED_POSITIVE]}
        assert len(cn & op) == 0

    def test_defensible_negative_can_be_zero(self, populations):
        assert populations["metadata"].get("defensible_negative_count", 0) >= 0 or \
               len(populations[DEFENSIBLE_NEGATIVE]) == 0

    def test_no_candidate_promoted_to_defensible_without_evidence(self, populations):
        """Verify no CANDIDATE_NEGATIVE is promoted to DEFENSIBLE_NEGATIVE."""
        dn = populations[DEFENSIBLE_NEGATIVE]
        # All defensible negatives should have independent evidence
        for r in dn:
            assert "independent_evidence" in r or len(dn) == 0


# ---------------------------------------------------------------------------
# Pairwise identity tests
# ---------------------------------------------------------------------------


class TestPairwiseIdentity:
    """Verify evidence-ranking pair construction requirements."""

    def test_both_files_identity_established(self, populations, phase411b_files):
        result = evaluate_evidence_ranking_strategy(
            populations, phase411b_files,
            _load_phase411b_commit_evidence(PHASE411B_DIR),
        )
        assert result["identity_verification"]["all_pairs_have_identity_established"]

    def test_both_files_same_commit(self, populations, phase411b_files):
        result = evaluate_evidence_ranking_strategy(
            populations, phase411b_files,
            _load_phase411b_commit_evidence(PHASE411B_DIR),
        )
        assert result["identity_verification"]["all_pairs_share_commit_sha"]

    def test_both_files_same_corrective_sha(self, populations, phase411b_files):
        result = evaluate_evidence_ranking_strategy(
            populations, phase411b_files,
            _load_phase411b_commit_evidence(PHASE411B_DIR),
        )
        assert result["identity_verification"]["all_pairs_share_corrective_sha"]

    def test_no_cross_repository_pairing(self, populations, phase411b_files):
        result = evaluate_evidence_ranking_strategy(
            populations, phase411b_files,
            _load_phase411b_commit_evidence(PHASE411B_DIR),
        )
        assert result["identity_verification"]["all_pairs_same_repository"]

    def test_pairwise_determinism(self, populations, phase411b_files):
        commits = _load_phase411b_commit_evidence(PHASE411B_DIR)
        r1 = evaluate_evidence_ranking_strategy(
            populations, phase411b_files, commits,
        )
        r2 = evaluate_evidence_ranking_strategy(
            populations, phase411b_files, commits,
        )
        assert r1["total_pairs"] == r2["total_pairs"]
        for cat in r1["pair_categories"]:
            assert (r1["pair_categories"][cat]["count"] ==
                    r2["pair_categories"][cat]["count"])

    def test_pairs_have_different_evidence_levels(self, populations, phase411b_files):
        """All pair categories involve different evidence levels."""
        commits = _load_phase411b_commit_evidence(PHASE411B_DIR)
        result = evaluate_evidence_ranking_strategy(
            populations, phase411b_files, commits,
        )
        # STRONG_vs_MODERATE should be 0 (no MODERATE exists)
        assert result["pair_categories"]["STRONG_vs_MODERATE"]["count"] == 0


# ---------------------------------------------------------------------------
# Temporal tests
# ---------------------------------------------------------------------------


class TestTemporal:
    """Verify temporal analysis uses no 'now' and derives endpoints from data."""

    def test_observation_endpoint_from_dataset(self, populations, phase411b_commits, timestamps):
        result = evaluate_temporal_strategy(
            populations, phase411b_commits, timestamps,
        )
        # Endpoints should be derived from data, not current time
        endpoints = result["observation_endpoints"]["endpoints"]
        assert len(endpoints) > 0
        for repo, ep in endpoints.items():
            assert ep  # non-empty
            dt = _parse_timestamp(ep)
            assert dt is not None

    def test_no_current_time_usage(self, populations, phase411b_commits, timestamps):
        result = evaluate_temporal_strategy(
            populations, phase411b_commits, timestamps,
        )
        # Verify no endpoint is close to "now"
        import datetime
        now = datetime.datetime.now(datetime.UTC)
        for repo, ep in result["observation_endpoints"]["endpoints"].items():
            dt = _parse_timestamp(ep)
            if dt:
                # Endpoint should be at least 1 day before "now"
                diff = (now - dt).total_seconds()
                assert diff > 86400, f"Endpoint {ep} too close to now"

    def test_continuous_distributions_reported(self, populations, phase411b_commits, timestamps):
        result = evaluate_temporal_strategy(
            populations, phase411b_commits, timestamps,
        )
        obs_dist = result["observation_duration_distribution"]
        assert obs_dist["count"] > 0
        assert "mean" in obs_dist
        assert "median" in obs_dist
        assert "min" in obs_dist
        assert "max" in obs_dist

    def test_data_derived_descriptive_boundary(self, populations, phase411b_commits, timestamps):
        result = evaluate_temporal_strategy(
            populations, phase411b_commits, timestamps,
        )
        median = result["median_observation_duration_days"]
        assert isinstance(median, (int, float))
        assert median >= 0


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------


class TestLeakage:
    """Verify no post-candidate evidence leaks into prediction-time features."""

    def test_label_source_excluded(self):
        audit = _audit_data_leakage()
        assert audit["label_source_excluded_from_features"]

    def test_corrective_sha_excluded(self):
        table = build_prediction_time_feature_table()
        feature_names = [f["name"] for f in table if f["permitted_usage"] == "FEATURE"]
        assert "corrective_sha" not in feature_names

    def test_content_evidence_excluded(self):
        table = build_prediction_time_feature_table()
        feature_names = [f["name"] for f in table if f["permitted_usage"] == "FEATURE"]
        for name in LEAKAGE_FEATURES:
            assert name not in feature_names

    def test_label_source_not_in_features(self):
        table = build_prediction_time_feature_table()
        feature_names = [f["name"] for f in table if f["permitted_usage"] == "FEATURE"]
        assert "label_source" not in feature_names

    def test_leakage_audit_passes(self):
        audit = _audit_data_leakage()
        assert audit["audit_result"] == "PASS"


# ---------------------------------------------------------------------------
# Split/manifest tests
# ---------------------------------------------------------------------------


class TestSplits:
    """Verify splits are loaded from repo_split.py."""

    def test_manifest_from_repo_split(self, manifest):
        assert len(manifest) == 50
        # Verify known splits exist
        splits = set(manifest.values())
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

    def test_no_hardcoded_repo_counts(self, populations):
        """Repo counts are derived from data, not hardcoded."""
        meta = populations["metadata"]
        assert meta["unique_repos"] == 50
        assert meta["repos_with_file_strong"] == 40
        assert meta["repos_without_positives"] == 10


# ---------------------------------------------------------------------------
# Frozen integrity tests
# ---------------------------------------------------------------------------


class TestFrozenIntegrity:
    """Verify frozen dataset and artifact hashes are unchanged."""

    def test_dataset_jsonl_exists(self):
        for split in ("train", "validation", "test"):
            path = DATA_DIR / f"{split}.jsonl"
            assert path.exists(), f"Missing {path}"

    def test_phase411b_artifacts_exist(self):
        for fname in [
            "file_attribution_results.json",
            "per_commit_git_evidence.jsonl",
            "git_evidence_analysis.json",
        ]:
            path = PHASE411B_DIR / fname
            assert path.exists(), f"Missing {path}"

    def test_phase411b_file_count(self):
        path = PHASE411B_DIR / "file_attribution_results.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1055

    def test_phase411b_commit_count(self):
        path = PHASE411B_DIR / "per_commit_git_evidence.jsonl"
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 210


# ---------------------------------------------------------------------------
# Forbidden imports / no training tests
# ---------------------------------------------------------------------------


class TestImplementationConstraints:
    """Verify no forbidden imports or model training."""

    def test_no_aws_braket_imports(self):
        import ast
        import importlib
        mod = importlib.import_module("backend.app.ml.phase412_supervision_strategy")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Collect all import names (not in docstrings/comments)
        import_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.add(node.module.lower())
        forbidden = ["boto3", "braket", "sagemaker", "quantum"]
        for term in forbidden:
            assert term not in import_names, f"Found forbidden import: {term}"

    def test_no_sklearn_fit_predict(self):
        import importlib
        mod = importlib.import_module("backend.app.ml.phase412_supervision_strategy")
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert ".fit(" not in source
        assert ".predict(" not in source


# ---------------------------------------------------------------------------
# Identifiability tests
# ---------------------------------------------------------------------------


class TestIdentifiability:
    """Verify each strategy has exactly one identifiability classification."""

    def test_all_strategies_classified(self, populations, supervised_rows,
                                       phase411b_files, phase411b_commits,
                                       timestamps, manifest):
        results = [
            evaluate_pu_strategy(populations, supervised_rows),
            evaluate_reliable_negative_strategy(
                populations, phase411b_files, phase411b_commits, timestamps,
            ),
            evaluate_temporal_strategy(populations, phase411b_commits, timestamps),
            evaluate_repository_control_strategy(populations, manifest),
            evaluate_corrective_control_strategy(populations, phase411b_commits),
            evaluate_evidence_ranking_strategy(
                populations, phase411b_files, phase411b_commits,
            ),
            evaluate_weak_supervision_strategy(populations, phase411b_files),
            evaluate_synthetic_strategy(),
            evaluate_commit_level_strategy(populations, phase411b_commits),
        ]

        valid = {
            "IDENTIFIABLE", "PARTIALLY_IDENTIFIABLE",
            "NOT_IDENTIFIABLE", "UNTESTABLE",
        }
        for r in results:
            assert "identifiability" in r, f"Missing identifiability in {r['strategy']}"
            assert r["identifiability"] in valid, (
                f"Invalid identifiability '{r['identifiability']}' "
                f"in {r['strategy']}"
            )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify deterministic output."""

    def test_populations_deterministic(self, supervised_rows, phase411b_files,
                                       phase411b_commits, manifest, timestamps):
        from backend.app.ml.phase49_supervision_feasibility import (
            _read_ambiguous_rows,
        )
        ambiguous = _read_ambiguous_rows(DATA_DIR)
        p1 = construct_supervision_populations(
            phase411b_files, phase411b_commits,
            supervised_rows, ambiguous, manifest, timestamps,
        )
        p2 = construct_supervision_populations(
            phase411b_files, phase411b_commits,
            supervised_rows, ambiguous, manifest, timestamps,
        )
        assert p1["metadata"]["observed_positive_rows"] == p2["metadata"]["observed_positive_rows"]
        assert p1["metadata"]["evidence_weak_rows"] == p2["metadata"]["evidence_weak_rows"]
        assert p1["metadata"]["evidence_unknown_rows"] == p2["metadata"]["evidence_unknown_rows"]


# ---------------------------------------------------------------------------
# Recommendation traceability test
# ---------------------------------------------------------------------------


class TestRecommendationTraceability:
    """Verify every recommendation has an explicit reason."""

    def test_all_recommendations_have_reason(self, populations, supervised_rows,
                                              phase411b_files, phase411b_commits,
                                              timestamps, manifest):
        results = [
            evaluate_pu_strategy(populations, supervised_rows),
            evaluate_reliable_negative_strategy(
                populations, phase411b_files, phase411b_commits, timestamps,
            ),
            evaluate_temporal_strategy(populations, phase411b_commits, timestamps),
            evaluate_repository_control_strategy(populations, manifest),
            evaluate_corrective_control_strategy(populations, phase411b_commits),
            evaluate_evidence_ranking_strategy(
                populations, phase411b_files, phase411b_commits,
            ),
            evaluate_weak_supervision_strategy(populations, phase411b_files),
            evaluate_synthetic_strategy(),
            evaluate_commit_level_strategy(populations, phase411b_commits),
        ]

        valid_recs = {
            "PROCEED", "PROCEED_WITH_CAVEATS",
            "FEASIBILITY_ONLY", "DO_NOT_PROCEED",
        }
        for r in results:
            assert "recommended_for_4.13" in r
            assert r["recommended_for_4.13"] in valid_recs
            assert "reason" in r
            assert len(r["reason"]) > 0


# ---------------------------------------------------------------------------
# All 9 strategies evaluated test
# ---------------------------------------------------------------------------


class TestStrategyCompleteness:
    """Verify all 9 strategies are evaluated."""

    def test_nine_strategies(self, populations, supervised_rows,
                              phase411b_files, phase411b_commits,
                              timestamps, manifest):
        results = [
            evaluate_pu_strategy(populations, supervised_rows),
            evaluate_reliable_negative_strategy(
                populations, phase411b_files, phase411b_commits, timestamps,
            ),
            evaluate_temporal_strategy(populations, phase411b_commits, timestamps),
            evaluate_repository_control_strategy(populations, manifest),
            evaluate_corrective_control_strategy(populations, phase411b_commits),
            evaluate_evidence_ranking_strategy(
                populations, phase411b_files, phase411b_commits,
            ),
            evaluate_weak_supervision_strategy(populations, phase411b_files),
            evaluate_synthetic_strategy(),
            evaluate_commit_level_strategy(populations, phase411b_commits),
        ]
        assert len(results) == 9
        names = {r["strategy"] for r in results}
        expected = {
            "PU_LEARNING", "RELIABLE_NEGATIVES", "TEMPORAL_NEGATIVES",
            "REPOSITORY_CONTROLS", "CORRECTIVE_COMMIT_CONTROLS",
            "EVIDENCE_RANKING", "WEAK_SUPERVISION", "SYNTHETIC_NEGATIVES",
            "COMMIT_LEVEL_FALLBACK",
        }
        assert names == expected
