"""Phase 5.3: Experimental Multi-Signal Investigation Ranking.

Research question: Can existing, already-computed structural/change
features improve the deterministic B1 investigation-priority ranking,
without ML training, new labels, or changes to the production API?

This is an exploratory experiment. The production system remains B1
throughout Phase 5.3. No enhanced scorer is promoted to production
during this phase.

Frozen baseline: backend/app/ml/phase413_models.py change_size_scores
(53-84). Uses np.argsort(np.argsort(-vals)) with default quicksort.
C1 must reproduce this exact behavior.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.app.ml.phase413_evaluation import (
    RECALL_AT_K_VALUES,
    aggregate_per_commit_metrics,
    rank_files_within_commit,
)
from backend.app.ml.repo_split import build_repo_split_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")
PHASE411B_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.11b")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase5.3")

BOOTSTRAP_N = 1000
RANDOM_SEED = 42

# E0 feature indices (45-dim canonical)
IDX_TOTAL_LINES_CHANGED = 34
IDX_HUNK_COUNT = 35
IDX_FN_DECL_ADDED = 36
IDX_FN_DECL_DELETED = 37
IDX_IMPORTS_ADDED = 40
IDX_IMPORTS_DELETED = 41


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signal_s1(
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Compute function-churn signal: |fn_added| + |fn_deleted| per file.

    Uses E0 indices 36 + 37 (file_function_declarations_added,
    file_function_declarations_deleted).
    """
    idx_added = feature_names.index("file_function_declarations_added")
    idx_deleted = feature_names.index("file_function_declarations_deleted")
    vals = np.abs(X[:, idx_added]) + np.abs(X[:, idx_deleted])
    return vals.tolist()


def compute_signal_s2(
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Compute import-churn signal: |imports_added| + |imports_deleted| per file.

    Uses E0 indices 40 + 41 (file_imports_added, file_imports_deleted).
    """
    idx_added = feature_names.index("file_imports_added")
    idx_deleted = feature_names.index("file_imports_deleted")
    vals = np.abs(X[:, idx_added]) + np.abs(X[:, idx_deleted])
    return vals.tolist()


def compute_signal_s3(
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Compute change-scattering signal: file_hunk_count per file.

    Uses E0 index 35 (file_hunk_count).
    """
    idx = feature_names.index("file_hunk_count")
    vals = X[:, idx].copy()
    return vals.tolist()


# ---------------------------------------------------------------------------
# Rank normalization (double-argsort, same as frozen B1)
# ---------------------------------------------------------------------------

def rank_normalize_within_commit(
    vals: list[float],
    metadata: list[dict],
) -> list[float]:
    """Rank-normalize values within each commit using double-argsort.

    Replicates the frozen B1 ranking behavior: np.argsort(np.argsort(-vals))
    with default quicksort, then 1.0 - rank / max(n_files - 1, 1).

    Do not add kind= argument.
    """
    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    scores = np.zeros(len(metadata), dtype=np.float64)
    for indices in by_commit.values():
        arr = np.array([vals[i] for i in indices], dtype=np.float64)
        ranks = np.argsort(np.argsort(-arr)).astype(np.float64)
        max_rank = max(len(indices) - 1, 1)
        scores[indices] = 1.0 - ranks / max_rank
    return scores.tolist()


# ---------------------------------------------------------------------------
# Scoring formulations
# ---------------------------------------------------------------------------

def score_c1(
    metadata: list[dict],
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """B1 baseline: rank by file_total_lines_changed.

    Replicates frozen change_size_scores exactly. Uses default argsort
    (no kind= kwarg) to match the frozen implementation.
    """
    idx = None
    for i, name in enumerate(feature_names):
        if name == "file_total_lines_changed":
            idx = i
            break

    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    scores = np.zeros(len(metadata), dtype=np.float64)
    if idx is None:
        rng = np.random.RandomState(RANDOM_SEED)
        for indices in by_commit.values():
            scores[indices] = rng.uniform(0.0, 1.0, size=len(indices))
        return scores.tolist()

    for indices in by_commit.values():
        vals = X[indices, idx]
        ranks = np.argsort(np.argsort(-vals)).astype(np.float64)
        max_rank = max(len(indices) - 1, 1)
        scores[indices] = 1.0 - ranks / max_rank
    return scores.tolist()


def score_c2(
    metadata: list[dict],
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """C2: 0.7 * B1 + 0.3 * S1 (function churn)."""
    b1 = np.array(score_c1(metadata, X, feature_names), dtype=np.float64)
    s1_raw = compute_signal_s1(X, feature_names)
    s1 = np.array(rank_normalize_within_commit(s1_raw, metadata), dtype=np.float64)
    return (0.7 * b1 + 0.3 * s1).tolist()


def score_c3(
    metadata: list[dict],
    X: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """C3: 0.5 * B1 + 0.2 * S1 + 0.15 * S2 + 0.15 * S3."""
    b1 = np.array(score_c1(metadata, X, feature_names), dtype=np.float64)
    s1_raw = compute_signal_s1(X, feature_names)
    s1 = np.array(rank_normalize_within_commit(s1_raw, metadata), dtype=np.float64)
    s2_raw = compute_signal_s2(X, feature_names)
    s2 = np.array(rank_normalize_within_commit(s2_raw, metadata), dtype=np.float64)
    s3_raw = compute_signal_s3(X, feature_names)
    s3 = np.array(rank_normalize_within_commit(s3_raw, metadata), dtype=np.float64)
    return (0.5 * b1 + 0.2 * s1 + 0.15 * s2 + 0.15 * s3).tolist()


# ---------------------------------------------------------------------------
# Bootstrap comparison
# ---------------------------------------------------------------------------

def compare_combinations(
    model_scores: list[float],
    reference_scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> dict[str, dict]:
    """Paired commit-level bootstrap comparing model vs reference (C1).

    Mirrors the Phase 4.13 paired_commit_level_bootstrap logic but
    compares model_scores vs reference_scores (C1) instead of model
    vs random.

    Returns dict with keys for enrichment@k, recall@k, and mrr.
    Each key contains diff, ci_lower, ci_upper, model metric,
    and reference metric.
    """
    rng = np.random.RandomState(seed)

    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    commit_keys = list(by_commit.keys())
    n_commits = len(commit_keys)

    model_per_commit = rank_files_within_commit(model_scores, metadata, positive_keys)
    ref_per_commit = rank_files_within_commit(reference_scores, metadata, positive_keys)

    def _mean_enrichment(per_commit: dict, k: int) -> float:
        vals = [v.enrichment_at_k[k] for v in per_commit.values() if v.p_positives > 0]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_mrr(per_commit: dict) -> float:
        vals = [v.reciprocal_rank for v in per_commit.values()
                if v.reciprocal_rank is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_recall(per_commit: dict, k: int) -> float:
        vals = [v.recall_at_k[k] for v in per_commit.values() if v.p_positives > 0]
        return float(np.mean(vals)) if vals else 0.0

    bootstrap_diffs_enrichment: dict[int, list[float]] = {k: [] for k in RECALL_AT_K_VALUES}
    bootstrap_diffs_recall: dict[int, list[float]] = {k: [] for k in RECALL_AT_K_VALUES}
    bootstrap_diffs_mrr: list[float] = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(n_commits, size=n_commits, replace=True)
        sampled_keys = [commit_keys[i] for i in sampled]

        sampled_model = {k: model_per_commit[k] for k in sampled_keys if k in model_per_commit}
        sampled_ref = {k: ref_per_commit[k] for k in sampled_keys if k in ref_per_commit}

        for k in RECALL_AT_K_VALUES:
            m_enr = _mean_enrichment(sampled_model, k)
            r_enr = _mean_enrichment(sampled_ref, k)
            bootstrap_diffs_enrichment[k].append(m_enr - r_enr)

            m_rec = _mean_recall(sampled_model, k)
            r_rec = _mean_recall(sampled_ref, k)
            bootstrap_diffs_recall[k].append(m_rec - r_rec)

        m_mrr = _mean_mrr(sampled_model)
        r_mrr = _mean_mrr(sampled_ref)
        bootstrap_diffs_mrr.append(m_mrr - r_mrr)

    results: dict[str, dict] = {}
    for k in RECALL_AT_K_VALUES:
        enr_diffs = bootstrap_diffs_enrichment[k]
        enr_ci_lower = float(np.percentile(enr_diffs, 2.5))
        enr_ci_upper = float(np.percentile(enr_diffs, 97.5))
        results[f"enrichment_at_{k}"] = {
            "model_enrichment": _mean_enrichment(model_per_commit, k),
            "ref_enrichment": _mean_enrichment(ref_per_commit, k),
            "diff": float(np.mean(enr_diffs)),
            "ci_lower": enr_ci_lower,
            "ci_upper": enr_ci_upper,
        }

        rec_diffs = bootstrap_diffs_recall[k]
        rec_ci_lower = float(np.percentile(rec_diffs, 2.5))
        rec_ci_upper = float(np.percentile(rec_diffs, 97.5))
        results[f"recall_at_{k}"] = {
            "model_recall": _mean_recall(model_per_commit, k),
            "ref_recall": _mean_recall(ref_per_commit, k),
            "diff": float(np.mean(rec_diffs)),
            "ci_lower": rec_ci_lower,
            "ci_upper": rec_ci_upper,
        }

    results["mrr"] = {
        "model_mrr": _mean_mrr(model_per_commit),
        "ref_mrr": _mean_mrr(ref_per_commit),
        "diff": float(np.mean(bootstrap_diffs_mrr)),
        "ci_lower": float(np.percentile(bootstrap_diffs_mrr, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_diffs_mrr, 97.5)),
    }

    return results


# ---------------------------------------------------------------------------
# Component sensitivity analysis (diagnostic only)
# ---------------------------------------------------------------------------

def component_sensitivity(
    metadata: list[dict],
    X: np.ndarray,
    feature_names: list[str],
    positive_keys: set[tuple[str, str]],
) -> dict[str, dict]:
    """Diagnostic ablations of C3: remove one component at a time.

    C3     = 0.50*B1 + 0.20*S1 + 0.15*S2 + 0.15*S3
    C3-S1  = 0.50*B1 + 0.00*S1 + 0.15*S2 + 0.15*S3
    C3-S2  = 0.50*B1 + 0.20*S1 + 0.00*S2 + 0.15*S3
    C3-S3  = 0.50*B1 + 0.20*S1 + 0.15*S2 + 0.00*S3

    Weights are NOT renormalized. This is diagnostic only.
    """
    b1 = np.array(score_c1(metadata, X, feature_names), dtype=np.float64)
    s1_raw = compute_signal_s1(X, feature_names)
    s1 = np.array(rank_normalize_within_commit(s1_raw, metadata), dtype=np.float64)
    s2_raw = compute_signal_s2(X, feature_names)
    s2 = np.array(rank_normalize_within_commit(s2_raw, metadata), dtype=np.float64)
    s3_raw = compute_signal_s3(X, feature_names)
    s3 = np.array(rank_normalize_within_commit(s3_raw, metadata), dtype=np.float64)

    c3_full = (0.50 * b1 + 0.20 * s1 + 0.15 * s2 + 0.15 * s3).tolist()
    c3_no_s1 = (0.50 * b1 + 0.00 * s1 + 0.15 * s2 + 0.15 * s3).tolist()
    c3_no_s2 = (0.50 * b1 + 0.20 * s1 + 0.00 * s2 + 0.15 * s3).tolist()
    c3_no_s3 = (0.50 * b1 + 0.20 * s1 + 0.15 * s2 + 0.00 * s3).tolist()

    c3_full_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(c3_full, metadata, positive_keys)
    )
    c3_no_s1_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(c3_no_s1, metadata, positive_keys)
    )
    c3_no_s2_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(c3_no_s2, metadata, positive_keys)
    )
    c3_no_s3_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(c3_no_s3, metadata, positive_keys)
    )

    return {
        "C3_full": c3_full_agg,
        "C3_minus_S1": c3_no_s1_agg,
        "C3_minus_S2": c3_no_s2_agg,
        "C3_minus_S3": c3_no_s3_agg,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """Run the Phase 5.3 experiment.

    Loads frozen data, computes C1/C2/C3, evaluates on all splits,
    runs bootstrap comparison on test set, and reports outcomes.

    Returns dict with all results.
    """
    logger.info("Phase 5.3: Loading data...")
    train_ds, val_ds, test_ds = build_repo_split_dataset(
        DATA_DIR, EXP_DIR, mode="E0"
    )
    logger.info(
        "Phase 5.3: train=%d, val=%d, test=%d (E0, 45 features)",
        len(train_ds), len(val_ds), len(test_ds),
    )

    from backend.app.ml.phase49_supervision_feasibility import (
        _read_all_supervised_rows,
    )
    from backend.app.ml.phase412_supervision_strategy import (
        _load_phase411b_commit_evidence,
        _load_phase411b_file_results,
    )
    from backend.app.ml.phase413_data import (
        DATA_DIR as _DATA_DIR,
    )
    from backend.app.ml.phase413_data import (
        OBSERVED_POSITIVE,
        build_phase413_populations,
    )
    from backend.app.ml.phase413_data import (
        PHASE411B_DIR as _P411B_DIR,
    )

    supervised_rows = _read_all_supervised_rows(_DATA_DIR)
    phase411b_files = _load_phase411b_file_results(_P411B_DIR)
    _ = _load_phase411b_commit_evidence(_P411B_DIR)
    populations = build_phase413_populations(phase411b_files, supervised_rows)

    op_keys: set[tuple[str, str]] = set()
    for r in populations[OBSERVED_POSITIVE]:
        op_keys.add((r["commit_sha"], r["file_path"]))

    logger.info("Phase 5.3: OBSERVED_POSITIVE keys: %d", len(op_keys))

    splits = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }

    all_results: dict[str, dict] = {}

    for split_name, ds in splits.items():
        logger.info("Phase 5.3: Evaluating split=%s (%d rows)", split_name, len(ds))
        meta = ds.metadata
        X = ds.X
        feat_names = ds.feature_names

        c1_scores = score_c1(meta, X, feat_names)
        c2_scores = score_c2(meta, X, feat_names)
        c3_scores = score_c3(meta, X, feat_names)

        c1_agg = aggregate_per_commit_metrics(
            rank_files_within_commit(c1_scores, meta, op_keys)
        )
        c2_agg = aggregate_per_commit_metrics(
            rank_files_within_commit(c2_scores, meta, op_keys)
        )
        c3_agg = aggregate_per_commit_metrics(
            rank_files_within_commit(c3_scores, meta, op_keys)
        )

        all_results[split_name] = {
            "C1": c1_agg,
            "C2": c2_agg,
            "C3": c3_agg,
        }

    logger.info("Phase 5.3: Bootstrap comparison on test set...")
    test_meta = test_ds.metadata
    test_X = test_ds.X
    test_feat_names = test_ds.feature_names

    c1_test = score_c1(test_meta, test_X, test_feat_names)
    c2_test = score_c2(test_meta, test_X, test_feat_names)
    c3_test = score_c3(test_meta, test_X, test_feat_names)

    bootstrap_c2 = compare_combinations(c2_test, c1_test, test_meta, op_keys)
    bootstrap_c3 = compare_combinations(c3_test, c1_test, test_meta, op_keys)

    logger.info("Phase 5.3: Component sensitivity analysis...")
    sensitivity = component_sensitivity(test_meta, test_X, test_feat_names, op_keys)

    all_results["test_bootstrap"] = {
        "C2_vs_C1": bootstrap_c2,
        "C3_vs_C1": bootstrap_c3,
    }
    all_results["test_sensitivity"] = sensitivity

    return all_results


def _format_agg(agg) -> str:
    """Format AggregateMetrics for display."""
    lines = [
        f"  n_commits_with_positives: {agg.n_commits_with_positives}",
        f"  n_commits_total: {agg.n_commits_total}",
        f"  MRR: {agg.mean_reciprocal_rank:.6f}",
        f"  Recall@1: {agg.mean_recall_at_k.get(1, 0.0):.6f}",
        f"  Recall@3: {agg.mean_recall_at_k.get(3, 0.0):.6f}",
        f"  Recall@5: {agg.mean_recall_at_k.get(5, 0.0):.6f}",
        f"  Enrichment@1: {agg.mean_enrichment_at_k.get(1, 0.0):.6f}",
        f"  Enrichment@3: {agg.mean_enrichment_at_k.get(3, 0.0):.6f}",
    ]
    return "\n".join(lines)


def _print_results(results: dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("Phase 5.3: Experimental Multi-Signal Investigation Ranking")
    print("=" * 70)

    for split in ["train", "validation", "test"]:
        if split in results:
            print(f"\n--- {split.upper()} SPLIT ---")
            for combo in ["C1", "C2", "C3"]:
                agg = results[split][combo]
                print(f"\n{combo}:")
                print(_format_agg(agg))

    if "test_bootstrap" in results:
        print("\n--- TEST BOOTSTRAP COMPARISON (95% CI) ---")
        for comparison, data in results["test_bootstrap"].items():
            print(f"\n{comparison}:")
            mrr = data["mrr"]
            print(f"  MRR: diff={mrr['diff']:.6f}, "
                  f"CI=[{mrr['ci_lower']:.6f}, {mrr['ci_upper']:.6f}]")
            for k in [1, 3, 5]:
                enr = data.get(f"enrichment_at_{k}")
                if enr:
                    print(f"  Enrichment@{k}: diff={enr['diff']:.6f}, "
                          f"CI=[{enr['ci_lower']:.6f}, {enr['ci_upper']:.6f}]")
                rec = data.get(f"recall_at_{k}")
                if rec:
                    print(f"  Recall@{k}: diff={rec['diff']:.6f}, "
                          f"CI=[{rec['ci_lower']:.6f}, {rec['ci_upper']:.6f}]")

    if "test_sensitivity" in results:
        print("\n--- COMPONENT SENSITIVITY ANALYSIS (diagnostic only) ---")
        for label, agg in results["test_sensitivity"].items():
            print(f"\n{label}:")
            print(_format_agg(agg))

    print("\n" + "=" * 70)
    print("NOTE: This experiment evaluates investigation ordering against")
    print("available evidence, NOT true defect probability. Scores are")
    print("investigation-priority rankings, NOT calibrated risks.")
    print("=" * 70)


def _save_results(results: dict, output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase53_results.json"

    def _agg_to_dict(agg) -> dict:
        return {
            "n_commits_with_positives": agg.n_commits_with_positives,
            "n_commits_total": agg.n_commits_total,
            "mean_mrr": agg.mean_reciprocal_rank,
            "mean_recall_at_k": {str(k): v for k, v in agg.mean_recall_at_k.items()},
            "mean_enrichment_at_k": {str(k): v for k, v in agg.mean_enrichment_at_k.items()},
        }

    serializable: dict[str, dict] = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable[key] = {}
            for k2, v2 in val.items():
                if hasattr(v2, "mean_reciprocal_rank"):
                    serializable[key][k2] = _agg_to_dict(v2)
                elif isinstance(v2, dict):
                    serializable[key][k2] = v2
                else:
                    serializable[key][k2] = str(v2)
        else:
            serializable[key] = str(val)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Phase 5.3: Results saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_experiment()
    _print_results(results)
    _save_results(results, OUTPUT_DIR)
