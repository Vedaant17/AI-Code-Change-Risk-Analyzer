"""Tests for Phase 5.3: Experimental Multi-Signal Investigation Ranking.

Tests cover:
1. C1 equivalence to frozen B1
2. Rank normalization basic behavior
3. Single-file behavior
4. All-identical values (permutation check)
5. S1/S2/S3 computation
6. C2 deterministic
7. C3 deterministic
8. Exact pre-specified weights
9. All scores within [0,1]
10. Input X remains unchanged

Also verifies C1 equivalence on actual Phase 4.13 data.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.experiments.phase53_multi_signal import (
    compute_signal_s1,
    compute_signal_s2,
    compute_signal_s3,
    rank_normalize_within_commit,
    score_c1,
    score_c2,
    score_c3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata_and_features(
    commit_files: dict[tuple[str, str], list[tuple[float, float, float, float, float, float]]],
) -> tuple[list[dict], np.ndarray, list[str]]:
    """Create test metadata and feature matrix.

    commit_files maps (repo, sha) -> list of
    (total_lines_changed, hunk_count, fn_added, fn_deleted, imports_added, imports_deleted).

    Feature order matches E0 indices 34-41.
    """
    feature_names = [f"feat_{i}" for i in range(42)]
    feature_names[34] = "file_total_lines_changed"
    feature_names[35] = "file_hunk_count"
    feature_names[36] = "file_function_declarations_added"
    feature_names[37] = "file_function_declarations_deleted"
    feature_names[40] = "file_imports_added"
    feature_names[41] = "file_imports_deleted"

    metadata = []
    rows = []
    for (repo, sha), files in commit_files.items():
        for vals in files:
            metadata.append({"repo_name": repo, "commit_sha": sha,
                             "file_path": f"{repo}/{sha}/{len(rows)}.py"})
            row = [0.0] * 42
            row[34] = vals[0]
            row[35] = vals[1]
            row[36] = vals[2]
            row[37] = vals[3]
            row[40] = vals[4]
            row[41] = vals[5]
            rows.append(row)

    X = np.array(rows, dtype=np.float64)
    return metadata, X, feature_names


# ---------------------------------------------------------------------------
# Test 1: C1 equivalence to frozen B1
# ---------------------------------------------------------------------------

def test_c1_equivalence_to_frozen_b1():
    """C1 must reproduce frozen change_size_scores exactly."""
    from backend.app.ml.phase413_models import change_size_scores

    commit_files = {
        ("repo_a", "sha1"): [(10, 1, 0, 0, 0, 0), (5, 2, 1, 0, 1, 0), (20, 3, 0, 1, 0, 1)],
        ("repo_a", "sha2"): [(1, 1, 0, 0, 0, 0)],
        ("repo_b", "sha3"): [(8, 1, 2, 0, 0, 0), (8, 1, 0, 0, 0, 0), (15, 2, 0, 0, 1, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    frozen_scores = change_size_scores(metadata, X, feat_names)
    c1_scores = score_c1(metadata, X, feat_names)

    assert len(frozen_scores) == len(c1_scores)
    np.testing.assert_array_equal(
        np.array(frozen_scores), np.array(c1_scores),
        err_msg="C1 scores do not match frozen B1",
    )


def test_c1_equivalence_on_large_data():
    """C1 must match frozen B1 on larger random data."""
    from backend.app.ml.phase413_models import change_size_scores

    rng = np.random.RandomState(99)
    n_commits = 20
    commit_files = {}
    for c in range(n_commits):
        n_files = rng.randint(2, 15)
        files = []
        for _ in range(n_files):
            files.append((
                float(rng.randint(0, 100)),
                float(rng.randint(1, 10)),
                float(rng.randint(0, 5)),
                float(rng.randint(0, 3)),
                float(rng.randint(0, 4)),
                float(rng.randint(0, 4)),
            ))
        commit_files[(f"repo_{c % 5}", f"sha_{c}")] = files

    metadata, X, feat_names = _make_metadata_and_features(commit_files)
    frozen_scores = change_size_scores(metadata, X, feat_names)
    c1_scores = score_c1(metadata, X, feat_names)

    np.testing.assert_allclose(
        np.array(frozen_scores), np.array(c1_scores),
        rtol=1e-12, atol=1e-12,
        err_msg="C1 scores diverge from frozen B1 on larger data",
    )


# ---------------------------------------------------------------------------
# Test 2: Rank normalization basic behavior
# ---------------------------------------------------------------------------

def test_rank_normalization_basic():
    """Double-argsort produces correct descending ranking."""
    metadata = [
        {"repo_name": "r", "commit_sha": "s", "file_path": "a"},
        {"repo_name": "r", "commit_sha": "s", "file_path": "b"},
        {"repo_name": "r", "commit_sha": "s", "file_path": "c"},
    ]
    vals = [10.0, 5.0, 1.0]
    scores = rank_normalize_within_commit(vals, metadata)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: Single-file behavior
# ---------------------------------------------------------------------------

def test_rank_normalization_single_file():
    """Single-file commit produces score 1.0."""
    metadata = [
        {"repo_name": "r", "commit_sha": "s", "file_path": "a"},
    ]
    vals = [42.0]
    scores = rank_normalize_within_commit(vals, metadata)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 4: All-identical values (permutation check)
# ---------------------------------------------------------------------------

def test_rank_normalization_all_identical():
    """All-identical values produce sequential ranks (a permutation of 0..n-1).

    Under quicksort, the exact ordering among ties is not guaranteed.
    We only verify that all ranks are present.
    """
    n = 5
    metadata = [
        {"repo_name": "r", "commit_sha": "s", "file_path": f"f{i}"}
        for i in range(n)
    ]
    vals = [7.0] * n
    scores = rank_normalize_within_commit(vals, metadata)
    assert len(scores) == n

    max_rank = max(n - 1, 1)
    expected_ranks = set(range(n))
    actual_ranks = set()
    for s in scores:
        rank = round((1.0 - s) * max_rank)
        actual_ranks.add(rank)
    assert actual_ranks == expected_ranks, (
        f"Ranks {actual_ranks} are not a permutation of {expected_ranks}"
    )


# ---------------------------------------------------------------------------
# Test 5: S1/S2/S3 computation
# ---------------------------------------------------------------------------

def test_s1_computation():
    """S1 = |fn_added| + |fn_deleted|."""
    commit_files = {
        ("r", "s"): [(10, 1, 3, 1, 0, 0), (5, 1, 0, 0, 2, 1)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)
    s1 = compute_signal_s1(X, feat_names)
    assert s1[0] == pytest.approx(4.0)
    assert s1[1] == pytest.approx(0.0)


def test_s2_computation():
    """S2 = |imports_added| + |imports_deleted|."""
    commit_files = {
        ("r", "s"): [(10, 1, 0, 0, 3, 1), (5, 1, 0, 0, 0, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)
    s2 = compute_signal_s2(X, feat_names)
    assert s2[0] == pytest.approx(4.0)
    assert s2[1] == pytest.approx(0.0)


def test_s3_computation():
    """S3 = file_hunk_count."""
    commit_files = {
        ("r", "s"): [(10, 7, 0, 0, 0, 0), (5, 2, 0, 0, 0, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)
    s3 = compute_signal_s3(X, feat_names)
    assert s3[0] == pytest.approx(7.0)
    assert s3[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 6: C2 deterministic
# ---------------------------------------------------------------------------

def test_c2_deterministic():
    """C2 produces identical scores across multiple runs."""
    commit_files = {
        ("repo_a", "sha1"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1), (20, 3, 1, 0, 0, 0)],
        ("repo_b", "sha2"): [(8, 1, 0, 0, 0, 0), (15, 2, 3, 1, 2, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    c2_run1 = score_c2(metadata, X, feat_names)
    c2_run2 = score_c2(metadata, X, feat_names)
    c2_run3 = score_c2(metadata, X, feat_names)

    np.testing.assert_array_equal(np.array(c2_run1), np.array(c2_run2))
    np.testing.assert_array_equal(np.array(c2_run2), np.array(c2_run3))


# ---------------------------------------------------------------------------
# Test 7: C3 deterministic
# ---------------------------------------------------------------------------

def test_c3_deterministic():
    """C3 produces identical scores across multiple runs."""
    commit_files = {
        ("repo_a", "sha1"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1), (20, 3, 1, 0, 0, 0)],
        ("repo_b", "sha2"): [(8, 1, 0, 0, 0, 0), (15, 2, 3, 1, 2, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    c3_run1 = score_c3(metadata, X, feat_names)
    c3_run2 = score_c3(metadata, X, feat_names)
    c3_run3 = score_c3(metadata, X, feat_names)

    np.testing.assert_array_equal(np.array(c3_run1), np.array(c3_run2))
    np.testing.assert_array_equal(np.array(c3_run2), np.array(c3_run3))


# ---------------------------------------------------------------------------
# Test 8: Exact pre-specified weights
# ---------------------------------------------------------------------------

def test_c2_weights_pre_specified():
    """C2 uses exactly 0.7*B1 + 0.3*S1."""
    commit_files = {
        ("r", "s"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    b1 = np.array(score_c1(metadata, X, feat_names), dtype=np.float64)
    s1_raw = compute_signal_s1(X, feat_names)
    s1 = np.array(rank_normalize_within_commit(s1_raw, metadata), dtype=np.float64)
    expected = (0.7 * b1 + 0.3 * s1).tolist()

    c2 = score_c2(metadata, X, feat_names)
    np.testing.assert_allclose(np.array(c2), np.array(expected), rtol=1e-12)


def test_c3_weights_pre_specified():
    """C3 uses exactly 0.5*B1 + 0.2*S1 + 0.15*S2 + 0.15*S3."""
    commit_files = {
        ("r", "s"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    b1 = np.array(score_c1(metadata, X, feat_names), dtype=np.float64)
    s1_raw = compute_signal_s1(X, feat_names)
    s1 = np.array(rank_normalize_within_commit(s1_raw, metadata), dtype=np.float64)
    s2_raw = compute_signal_s2(X, feat_names)
    s2 = np.array(rank_normalize_within_commit(s2_raw, metadata), dtype=np.float64)
    s3_raw = compute_signal_s3(X, feat_names)
    s3 = np.array(rank_normalize_within_commit(s3_raw, metadata), dtype=np.float64)
    expected = (0.5 * b1 + 0.2 * s1 + 0.15 * s2 + 0.15 * s3).tolist()

    c3 = score_c3(metadata, X, feat_names)
    np.testing.assert_allclose(np.array(c3), np.array(expected), rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 9: All scores within [0,1]
# ---------------------------------------------------------------------------

def test_all_scores_within_0_1():
    """C1, C2, C3 scores are all in [0.0, 1.0]."""
    commit_files = {
        ("repo_a", "sha1"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1), (20, 3, 1, 0, 0, 0)],
        ("repo_b", "sha2"): [(8, 1, 0, 0, 0, 0), (15, 2, 3, 1, 2, 0), (1, 1, 0, 0, 0, 0)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)

    for scorer in [score_c1, score_c2, score_c3]:
        scores = scorer(metadata, X, feat_names)
        arr = np.array(scores)
        assert arr.min() >= 0.0, f"{scorer.__name__} has score < 0: {arr.min()}"
        assert arr.max() <= 1.0, f"{scorer.__name__} has score > 1: {arr.max()}"


# ---------------------------------------------------------------------------
# Test 10: Input X remains unchanged
# ---------------------------------------------------------------------------

def test_input_x_unchanged():
    """Scoring functions do not modify the input feature matrix X."""
    commit_files = {
        ("r", "s"): [(10, 1, 2, 0, 1, 0), (5, 2, 0, 1, 0, 1)],
    }
    metadata, X, feat_names = _make_metadata_and_features(commit_files)
    X_copy = X.copy()

    score_c1(metadata, X, feat_names)
    np.testing.assert_array_equal(X, X_copy)

    score_c2(metadata, X, feat_names)
    np.testing.assert_array_equal(X, X_copy)

    score_c3(metadata, X, feat_names)
    np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# Test: C1 equivalence on actual Phase 4.13 data (integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_c1_equivalence_on_phase413_data():
    """Verify C1 matches frozen B1 on actual Phase 4.13 test data."""
    from backend.app.ml.phase49_supervision_feasibility import (
        _read_all_supervised_rows,
    )
    from backend.app.ml.phase412_supervision_strategy import (
        _load_phase411b_file_results,
    )
    from backend.app.ml.phase413_data import (
        DATA_DIR,
        EXP_DIR,
        OBSERVED_POSITIVE,
        PHASE411B_DIR,
        build_phase413_populations,
    )
    from backend.app.ml.phase413_models import change_size_scores
    from backend.app.ml.repo_split import build_repo_split_dataset

    train_ds, val_ds, test_ds = build_repo_split_dataset(
        DATA_DIR, EXP_DIR, mode="E0"
    )

    supervised_rows = _read_all_supervised_rows(DATA_DIR)
    phase411b_files = _load_phase411b_file_results(PHASE411B_DIR)
    populations = build_phase413_populations(phase411b_files, supervised_rows)

    op_keys: set[tuple[str, str]] = set()
    for r in populations[OBSERVED_POSITIVE]:
        op_keys.add((r["commit_sha"], r["file_path"]))

    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        frozen_scores = change_size_scores(ds.metadata, ds.X, ds.feature_names)
        c1_scores = score_c1(ds.metadata, ds.X, ds.feature_names)
        np.testing.assert_allclose(
            np.array(frozen_scores), np.array(c1_scores),
            rtol=1e-12, atol=1e-12,
            err_msg=f"C1 diverges from frozen B1 on {ds_name} split",
        )
