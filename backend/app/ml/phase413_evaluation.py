"""Phase 4.13 Evaluation: Within-commit ranking metrics, bootstrap CI.

Primary evaluation unit: within-commit file ranking.
Operational question: Given a candidate commit, which files should
an engineer investigate first?

All primary metrics rank files only against other files in the same
commit. Global ranking is secondary only.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

RECALL_AT_K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]
BOOTSTRAP_N = 1000
RANDOM_SEED = 42


@dataclass
class PerCommitRanking:
    """Within-commit ranking metrics for a single commit."""
    repo_name: str
    commit_sha: str
    n_files: int
    p_positives: int
    recall_at_k: dict[int, float]
    enrichment_at_k: dict[int, float]
    rank_of_first_positive: int | None
    reciprocal_rank: float | None
    all_positives_captured_in_top_k: dict[int, bool]


@dataclass
class AggregateMetrics:
    """Macro-averaged metrics across commits with >=1 observed positive."""
    n_commits_with_positives: int
    n_commits_total: int
    coverage: float
    mean_files_per_commit: float
    mean_recall_at_k: dict[int, float]
    mean_enrichment_at_k: dict[int, float]
    mean_reciprocal_rank: float
    mean_rank_of_first_positive: float
    median_reciprocal_rank: float
    median_rank_of_first_positive: float
    fraction_all_captured_at_k: dict[int, float]


# ---------------------------------------------------------------------------
# Core within-commit evaluation
# ---------------------------------------------------------------------------

def rank_files_within_commit(
    scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], PerCommitRanking]:
    """Compute within-commit ranking metrics for every commit.

    Parameters
    ----------
    scores:
        Risk scores aligned with metadata (higher = investigate first).
    metadata:
        List of dicts with keys: repo_name, commit_sha, file_path, ...
    positive_keys:
        Set of (commit_sha, file_path) for observed-positive files.

    Returns
    -------
    Dict mapping (repo_name, commit_sha) -> PerCommitRanking
    """
    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    results: dict[tuple[str, str], PerCommitRanking] = {}

    for c_key, indices in by_commit.items():
        repo, sha = c_key
        n_c = len(indices)
        c_scores = [scores[i] for i in indices]
        c_metadata = [metadata[i] for i in indices]

        sorted_indices = sorted(range(n_c), key=lambda j: -c_scores[j])

        p_c = sum(
            1 for j in sorted_indices
            if (c_metadata[j]["commit_sha"], c_metadata[j]["file_path"]) in positive_keys
        )

        recall_at_k = {}
        enrichment_at_k = {}
        all_captured_at_k = {}
        first_rank = None
        rr = None

        captured = 0
        first_found = False
        for rank_pos, j in enumerate(sorted_indices):
            key = (c_metadata[j]["commit_sha"], c_metadata[j]["file_path"])
            if key in positive_keys:
                captured += 1
                if not first_found:
                    first_rank = rank_pos + 1
                    rr = 1.0 / first_rank
                    first_found = True

        for k in RECALL_AT_K_VALUES:
            captured_at_k = 0
            for rank_pos, j in enumerate(sorted_indices):
                if rank_pos >= k:
                    break
                key = (c_metadata[j]["commit_sha"], c_metadata[j]["file_path"])
                if key in positive_keys:
                    captured_at_k += 1

            obs_recall = captured_at_k / p_c if p_c > 0 else 0.0
            random_recall = min(k, n_c) / n_c if n_c > 0 else 0.0
            obs_enrichment = obs_recall / random_recall if random_recall > 0 else 1.0

            recall_at_k[k] = obs_recall
            enrichment_at_k[k] = obs_enrichment
            all_captured_at_k[k] = captured_at_k >= p_c if p_c > 0 else True

        results[c_key] = PerCommitRanking(
            repo_name=repo,
            commit_sha=sha,
            n_files=n_c,
            p_positives=p_c,
            recall_at_k=recall_at_k,
            enrichment_at_k=enrichment_at_k,
            rank_of_first_positive=first_rank,
            reciprocal_rank=rr,
            all_positives_captured_in_top_k=all_captured_at_k,
        )

    return results


def aggregate_per_commit_metrics(
    per_commit: dict[tuple[str, str], PerCommitRanking],
) -> AggregateMetrics:
    """Macro-average per-commit metrics across commits with >=1 positive."""
    positive_commits = {
        k: v for k, v in per_commit.items() if v.p_positives > 0
    }

    if not positive_commits:
        return AggregateMetrics(
            n_commits_with_positives=0,
            n_commits_total=len(per_commit),
            coverage=0.0,
            mean_files_per_commit=0.0,
            mean_recall_at_k={k: 0.0 for k in RECALL_AT_K_VALUES},
            mean_enrichment_at_k={k: 0.0 for k in RECALL_AT_K_VALUES},
            mean_reciprocal_rank=0.0,
            mean_rank_of_first_positive=0.0,
            median_reciprocal_rank=0.0,
            median_rank_of_first_positive=0.0,
            fraction_all_captured_at_k={k: 0.0 for k in RECALL_AT_K_VALUES},
        )

    n_pos = len(positive_commits)
    mrr_values = [
        v.reciprocal_rank for v in positive_commits.values()
        if v.reciprocal_rank is not None
    ]
    rank_values = [
        v.rank_of_first_positive for v in positive_commits.values()
        if v.rank_of_first_positive is not None
    ]

    mean_recall = {}
    mean_enrichment = {}
    fraction_captured = {}
    for k in RECALL_AT_K_VALUES:
        mean_recall[k] = np.mean([v.recall_at_k[k] for v in positive_commits.values()])
        mean_enrichment[k] = np.mean([v.enrichment_at_k[k] for v in positive_commits.values()])
        fraction_captured[k] = np.mean([1.0 if v.all_positives_captured_in_top_k[k] else 0.0
                                        for v in positive_commits.values()])

    return AggregateMetrics(
        n_commits_with_positives=n_pos,
        n_commits_total=len(per_commit),
        coverage=n_pos / len(per_commit) if len(per_commit) > 0 else 0.0,
        mean_files_per_commit=np.mean([v.n_files for v in per_commit.values()]),
        mean_recall_at_k=mean_recall,
        mean_enrichment_at_k=mean_enrichment,
        mean_reciprocal_rank=float(np.mean(mrr_values)) if mrr_values else 0.0,
        mean_rank_of_first_positive=float(np.mean(rank_values)) if rank_values else 0.0,
        median_reciprocal_rank=float(np.median(mrr_values)) if mrr_values else 0.0,
        median_rank_of_first_positive=float(np.median(rank_values)) if rank_values else 0.0,
        fraction_all_captured_at_k=fraction_captured,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def paired_commit_level_bootstrap(
    model_scores: list[float],
    random_scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> dict[str, dict]:
    """Compute paired commit-level bootstrap of metric differences.

    Resamples commits (not files), retaining all files per commit.

    Returns dict with keys for each K value:
        {k: {"model_metric": float, "random_metric": float,
             "diff": float, "ci_lower": float, "ci_upper": float}}
    """
    rng = np.random.RandomState(seed)

    by_commit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_commit[(m["repo_name"], m["commit_sha"])].append(i)

    commit_keys = list(by_commit.keys())
    n_commits = len(commit_keys)

    model_per_commit = rank_files_within_commit(model_scores, metadata, positive_keys)
    random_per_commit = rank_files_within_commit(random_scores, metadata, positive_keys)

    def _mean_enrichment(per_commit: dict, k: int) -> float:
        vals = [v.enrichment_at_k[k] for v in per_commit.values() if v.p_positives > 0]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_mrr(per_commit: dict) -> float:
        vals = [v.reciprocal_rank for v in per_commit.values()
                if v.reciprocal_rank is not None]
        return float(np.mean(vals)) if vals else 0.0

    bootstrap_diffs: dict[int, list[float]] = {k: [] for k in RECALL_AT_K_VALUES}
    bootstrap_diffs_mrr: list[float] = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(n_commits, size=n_commits, replace=True)
        sampled_keys = [commit_keys[i] for i in sampled]

        sampled_model = {k: model_per_commit[k] for k in sampled_keys if k in model_per_commit}
        sampled_random = {k: random_per_commit[k] for k in sampled_keys if k in random_per_commit}

        for k in RECALL_AT_K_VALUES:
            m_val = _mean_enrichment(sampled_model, k)
            r_val = _mean_enrichment(sampled_random, k)
            bootstrap_diffs[k].append(m_val - r_val)

        m_mrr = _mean_mrr(sampled_model)
        r_mrr = _mean_mrr(sampled_random)
        bootstrap_diffs_mrr.append(m_mrr - r_mrr)

    results: dict[str, dict] = {}
    for k in RECALL_AT_K_VALUES:
        diffs = bootstrap_diffs[k]
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        results[f"enrichment_at_{k}"] = {
            "model_enrichment": _mean_enrichment(model_per_commit, k),
            "random_enrichment": _mean_enrichment(random_per_commit, k),
            "diff": float(np.mean(diffs)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    results["mrr"] = {
        "model_mrr": _mean_mrr(model_per_commit),
        "random_mrr": _mean_mrr(random_per_commit),
        "diff": float(np.mean(bootstrap_diffs_mrr)),
        "ci_lower": float(np.percentile(bootstrap_diffs_mrr, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_diffs_mrr, 97.5)),
    }

    return results


# ---------------------------------------------------------------------------
# Per-repository analysis
# ---------------------------------------------------------------------------

def per_repository_metrics(
    scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
) -> dict[str, dict]:
    """Compute per-repository within-commit metrics.

    Returns dict mapping repo_name -> {"recall@k": ..., "enrichment@k": ..., "mrr": ...}
    """
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        by_repo[m["repo_name"]].append(i)

    repo_results: dict[str, dict] = {}
    for repo, indices in by_repo.items():
        repo_meta = [metadata[i] for i in indices]
        repo_scores = [scores[i] for i in indices]

        per_commit = rank_files_within_commit(repo_scores, repo_meta, positive_keys)
        agg = aggregate_per_commit_metrics(per_commit)

        repo_results[repo] = {
            "n_files": len(indices),
            "n_commits": len(set(m["commit_sha"] for m in repo_meta)),
            "n_commits_with_positives": agg.n_commits_with_positives,
            "mean_recall_at_k": agg.mean_recall_at_k,
            "mean_enrichment_at_k": agg.mean_enrichment_at_k,
            "mean_mrr": agg.mean_reciprocal_rank,
            "median_mrr": agg.median_reciprocal_rank,
        }

    return repo_results


def summarize_per_repo(repo_metrics: dict[str, dict]) -> dict:
    """Summarize per-repository metrics across test repos."""
    mrr_vals = [v["mean_mrr"] for v in repo_metrics.values() if v["n_commits_with_positives"] > 0]
    enrichment_vals = {k: [] for k in RECALL_AT_K_VALUES}
    for v in repo_metrics.values():
        if v["n_commits_with_positives"] > 0:
            for k in RECALL_AT_K_VALUES:
                enrichment_vals[k].append(v["mean_enrichment_at_k"][k])

    summary: dict[str, dict] = {}
    if mrr_vals:
        summary["mrr"] = {
            "median": float(np.median(mrr_vals)),
            "iqr": float(np.percentile(mrr_vals, 75) - np.percentile(mrr_vals, 25)),
            "min": float(np.min(mrr_vals)),
            "max": float(np.max(mrr_vals)),
        }
    for k in RECALL_AT_K_VALUES:
        vals = enrichment_vals[k]
        if vals:
            summary[f"enrichment_at_{k}"] = {
                "median": float(np.median(vals)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


# ---------------------------------------------------------------------------
# Investigation budget curves
# ---------------------------------------------------------------------------

def investigation_budget_curves(
    scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
) -> dict:
    """Compute within-commit investigation budget curves.

    For each K, report the fraction of observed-positive commits where
    ALL FILE_STRONG files are captured within the top K files.
    """
    per_commit = rank_files_within_commit(scores, metadata, positive_keys)
    agg = aggregate_per_commit_metrics(per_commit)

    curves: dict[str, dict] = {}
    for k in RECALL_AT_K_VALUES:
        curves[str(k)] = {
            "fraction_all_captured": agg.fraction_all_captured_at_k[k],
            "mean_recall": agg.mean_recall_at_k[k],
            "mean_enrichment": agg.mean_enrichment_at_k[k],
        }
    return curves


# ---------------------------------------------------------------------------
# Global secondary metrics
# ---------------------------------------------------------------------------

def global_secondary_metrics(
    scores: list[float],
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
) -> dict:
    """Compute global (cross-commit) secondary metrics.

    Labeled as secondary -- not the primary evaluation.
    """
    n_total = len(metadata)
    n_pos = sum(
        1 for m in metadata
        if (m["commit_sha"], m["file_path"]) in positive_keys
    )

    scored = sorted(range(n_total), key=lambda i: -scores[i])

    results: dict[str, float] = {}
    for k in RECALL_AT_K_VALUES:
        captured = sum(
            1 for i in scored[:k]
            if (metadata[i]["commit_sha"], metadata[i]["file_path"]) in positive_keys
        )
        results[f"recall_at_{k}"] = captured / n_pos if n_pos > 0 else 0.0
        results[f"precision_at_{k}"] = captured / k if k > 0 else 0.0

    return {
        "n_total": n_total,
        "n_observed_positives": n_pos,
        "metrics": results,
    }
