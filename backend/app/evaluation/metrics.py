"""Evaluation metrics and runner (Phase 7).

Computes within-commit ranking metrics, bootstrap CI, and assembles
the canonical EvaluationResult with checksum.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from backend.app.evaluation.population import (
    DATA_DIR,
    load_population,
    score_commit,
)
from backend.app.ml.phase413_evaluation import (
    BOOTSTRAP_N,
    RANDOM_SEED,
    RECALL_AT_K_VALUES,
    AggregateMetrics,
    PerCommitRanking,
    aggregate_per_commit_metrics,
    global_secondary_metrics,
    rank_files_within_commit,
)

logger = logging.getLogger(__name__)

EVALUATION_VERSION = "1.0.0"
OUTPUT_DIR = Path("backend/data/evaluation")


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""

    mean: float
    ci_lower: float
    ci_upper: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with metrics, bootstrap CI, and checksum."""

    evaluation_version: str = EVALUATION_VERSION
    feature_version: str = "v1"
    strategy: str = "B1_CHANGE_SIZE"
    split: str = "test"

    n_commits: int = 0
    n_files: int = 0
    n_positives: int = 0
    positive_rate: float = 0.0
    repos: list[str] = field(default_factory=list)

    aggregate: AggregateMetrics | None = None
    per_commit: list[PerCommitRanking] = field(default_factory=list)
    per_repo: dict[str, dict] = field(default_factory=dict)
    global_secondary: dict | None = None
    bootstrap_ci: dict[str, BootstrapCI] = field(default_factory=dict)

    checksum: str = ""


def _compute_checksum(result_dict: dict) -> str:
    """Compute SHA-256 checksum of canonical JSON payload (excluding checksum)."""
    payload = {k: v for k, v in result_dict.items() if k != "checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bootstrap_aggregate(
    per_commit_list: list[PerCommitRanking],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> dict[str, BootstrapCI]:
    """Compute bootstrap CIs for aggregate metrics.

    Resamples at the commit level, preserving duplicate commits.
    """
    rng = np.random.RandomState(seed)
    n = len(per_commit_list)

    if n == 0:
        return {}

    bootstrap_aggregates: list[AggregateMetrics] = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sampled = [per_commit_list[i] for i in indices]
        sampled_dict = {
            (r.repo_name, r.commit_sha, i): r
            for i, r in enumerate(sampled)
        }
        agg = aggregate_per_commit_metrics(sampled_dict)
        bootstrap_aggregates.append(agg)

    ci_results: dict[str, BootstrapCI] = {}

    # MRR
    mrr_vals = [a.mean_reciprocal_rank for a in bootstrap_aggregates]
    ci_results["mrr"] = BootstrapCI(
        mean=float(np.mean(mrr_vals)),
        ci_lower=float(np.percentile(mrr_vals, 2.5)),
        ci_upper=float(np.percentile(mrr_vals, 97.5)),
    )

    # Mean rank of first positive
    rank_vals = [a.mean_rank_of_first_positive for a in bootstrap_aggregates]
    ci_results["mean_rank_first_positive"] = BootstrapCI(
        mean=float(np.mean(rank_vals)),
        ci_lower=float(np.percentile(rank_vals, 2.5)),
        ci_upper=float(np.percentile(rank_vals, 97.5)),
    )

    # Recall@K and Enrichment@K
    for k in RECALL_AT_K_VALUES:
        recall_vals = [a.mean_recall_at_k[k] for a in bootstrap_aggregates]
        ci_results[f"recall_at_{k}"] = BootstrapCI(
            mean=float(np.mean(recall_vals)),
            ci_lower=float(np.percentile(recall_vals, 2.5)),
            ci_upper=float(np.percentile(recall_vals, 97.5)),
        )

        enrichment_vals = [a.mean_enrichment_at_k[k] for a in bootstrap_aggregates]
        ci_results[f"enrichment_at_{k}"] = BootstrapCI(
            mean=float(np.mean(enrichment_vals)),
            ci_lower=float(np.percentile(enrichment_vals, 2.5)),
            ci_upper=float(np.percentile(enrichment_vals, 97.5)),
        )

        capture_vals = [a.fraction_all_captured_at_k[k] for a in bootstrap_aggregates]
        ci_results[f"fraction_all_captured_at_{k}"] = BootstrapCI(
            mean=float(np.mean(capture_vals)),
            ci_lower=float(np.percentile(capture_vals, 2.5)),
            ci_upper=float(np.percentile(capture_vals, 97.5)),
        )

    return ci_results


def run_evaluation(data_dir: Path = DATA_DIR) -> EvaluationResult:
    """Run the full Phase 7 evaluation.

    Parameters
    ----------
    data_dir:
        Path to the combined-v3 dataset directory.

    Returns
    -------
    EvaluationResult
        Complete evaluation result with metrics and checksum.
    """
    population = load_population(data_dir)

    all_scores: list[float] = []
    all_metadata: list[dict] = []

    for commit_key, indices in population.commit_groups.items():
        repo, sha = commit_key
        commit_rows = [population.rows[i] for i in indices]
        commit_metadata = [population.metadata[i] for i in indices]

        scores = score_commit(commit_rows, commit_metadata, sha)

        all_scores.extend(scores)
        all_metadata.extend(commit_metadata)

    per_commit = rank_files_within_commit(
        all_scores, all_metadata, population.positive_keys
    )

    per_commit_list = list(per_commit.values())

    aggregate = aggregate_per_commit_metrics(per_commit)

    bootstrap_ci = _bootstrap_aggregate(per_commit_list)

    # Per-repository breakdown
    per_repo: dict[str, dict] = {}
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(all_metadata):
        by_repo[m["repo_name"]].append(i)

    for repo, indices in by_repo.items():
        repo_meta = [all_metadata[i] for i in indices]
        repo_scores = [all_scores[i] for i in indices]
        repo_per_commit = rank_files_within_commit(
            repo_scores, repo_meta, population.positive_keys
        )
        repo_agg = aggregate_per_commit_metrics(repo_per_commit)
        per_repo[repo] = {
            "n_files": len(indices),
            "n_commits": len(set(m["commit_sha"] for m in repo_meta)),
            "n_commits_with_positives": repo_agg.n_commits_with_positives,
            "mean_mrr": repo_agg.mean_reciprocal_rank,
            "mean_recall_at_k": repo_agg.mean_recall_at_k,
            "mean_enrichment_at_k": repo_agg.mean_enrichment_at_k,
        }

    # Global secondary metrics (optional)
    global_sec = global_secondary_metrics(
        all_scores, all_metadata, population.positive_keys
    )

    result = EvaluationResult(
        n_commits=population.n_commits,
        n_files=population.n_rows,
        n_positives=population.n_positives,
        positive_rate=population.positive_rate,
        repos=population.repositories,
        aggregate=aggregate,
        per_commit=per_commit_list,
        per_repo=per_repo,
        global_secondary=global_sec,
        bootstrap_ci=bootstrap_ci,
    )

    return result


def save_evaluation(result: EvaluationResult, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save evaluation result to JSON with checksum.

    Parameters
    ----------
    result:
        The evaluation result to save.
    output_dir:
        Directory to write the output file.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase7_evaluation.json"

    result_dict = asdict(result)

    checksum = _compute_checksum(result_dict)
    result_dict["checksum"] = checksum

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=True)

    logger.info("Evaluation saved to %s", output_path)
    return output_path


def load_evaluation(path: Path) -> dict:
    """Load and verify an evaluation result JSON.

    Parameters
    ----------
    path:
        Path to the evaluation JSON file.

    Returns
    -------
    dict
        The loaded and verified evaluation result.

    Raises
    ------
    ValueError
        If the checksum does not match.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stored_checksum = data.get("checksum", "")
    if not stored_checksum:
        raise ValueError("No checksum found in evaluation file")

    payload = {k: v for k, v in data.items() if k != "checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if computed != stored_checksum:
        raise ValueError(
            f"Checksum mismatch: expected {stored_checksum}, got {computed}"
        )

    return data
