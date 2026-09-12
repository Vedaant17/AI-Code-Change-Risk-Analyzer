"""Evaluation metrics for Phase 4.6 unseen-repository ranking.

Primary: per-commit ranking (Precision@K, Recall@K, MRR, positive capture).
Secondary: global classification metrics (ROC-AUC, PR-AUC, etc.).
Tertiary: per-repository analysis.

All metrics are deterministic and reproducible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class PerCommitMetrics:
    """Metrics for a single commit's file ranking."""

    repo_name: str
    commit_sha: str
    total_files: int
    positive_files: int
    precision_at_1: float | None
    precision_at_2: float | None
    precision_at_3: float | None
    precision_at_5: float | None
    recall_at_1: float | None
    recall_at_2: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    rank_of_first_positive: int | None
    reciprocal_rank: float | None
    top_1_capture: int | None
    top_3_capture: int | None
    top_5_capture: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateRankingMetrics:
    """Aggregated ranking metrics across commits with >= 1 positive file."""

    num_commits_with_positives: int
    num_commits_total: int
    num_commits_zero_positives: int
    mean_files_per_commit: float

    mean_precision_at_1: float
    mean_precision_at_2: float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_recall_at_1: float
    mean_recall_at_2: float
    mean_recall_at_3: float
    mean_recall_at_5: float

    median_precision_at_1: float
    median_precision_at_2: float
    median_precision_at_3: float
    median_precision_at_5: float
    median_recall_at_1: float
    median_recall_at_2: float
    median_recall_at_3: float
    median_recall_at_5: float

    mean_rank_of_first_positive: float
    median_rank_of_first_positive: float
    mean_reciprocal_rank: float

    fraction_top_1_capture: float
    fraction_top_3_capture: float
    fraction_top_5_capture: float

    per_commit_details: list[PerCommitMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_commit_details", None)
        return d


@dataclass
class PerRepoMetrics:
    """Metrics for a single repository."""

    repo_name: str
    total_rows: int
    positive_rows: int
    positive_commits: int
    total_commits: int
    prevalence: float
    pr_auc: float | None
    mean_precision_at_1: float | None
    mean_recall_at_1: float | None
    mean_reciprocal_rank: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GlobalClassificationMetrics:
    """Secondary global classification metrics."""

    roc_auc: float
    pr_auc: float
    positive_prevalence: float
    pr_auc_lift: float
    precision: float
    recall: float
    f1: float
    precision_at_5: float
    recall_at_5: float
    precision_at_10: float
    recall_at_10: float
    precision_at_20: float
    recall_at_20: float
    precision_at_50: float
    recall_at_50: float
    precision_at_100: float
    recall_at_100: float
    top_10_capture: int
    top_20_capture: int
    top_50_capture: int
    top_100_capture: int
    total_examples: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_mean(vals: list[float | None]) -> float:
    valid = [v for v in vals if v is not None]
    return float(np.mean(valid)) if valid else 0.0


def _safe_median(vals: list[float | None]) -> float:
    valid = [v for v in vals if v is not None]
    return float(np.median(valid)) if valid else 0.0


def per_commit_ranking(
    metadata: list[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> AggregateRankingMetrics:
    """Compute per-commit ranking metrics.

    For each (repo_name, commit_sha) group:
    1. Rank files by descending risk_score (probability).
    2. Break ties deterministically by file_path lexicographic.
    3. Compute Precision@K, Recall@K, MRR, positive capture.
    """
    commit_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        key = (m["repo_name"], m["commit_sha"])
        commit_groups[key].append(i)

    commit_metrics: list[PerCommitMetrics] = []

    for (repo, sha), indices in commit_groups.items():
        n_files = len(indices)
        probs = y_prob[indices]
        labels = y_true[indices]
        file_paths = [metadata[i]["file_path"] for i in indices]

        ranked_items = sorted(
            zip([-p for p in probs], file_paths, range(n_files)),
            key=lambda x: (x[0], x[1]),
        )
        ranked_local_indices = [item[2] for item in ranked_items]
        ranked_labels = labels[ranked_local_indices]

        n_pos = int(np.sum(labels == 1))

        def _p_at_k(k: int) -> float:
            n = min(k, n_files)
            if n == 0:
                return 0.0
            tp = int(np.sum(ranked_labels[:n] == 1))
            return tp / n

        def _r_at_k(k: int) -> float:
            if n_pos == 0:
                return 0.0
            n = min(k, n_files)
            tp = int(np.sum(ranked_labels[:n] == 1))
            return tp / n_pos

        rank_first_pos = None
        for r, lbl in enumerate(ranked_labels, 1):
            if lbl == 1:
                rank_first_pos = r
                break

        reciprocal_rank = 1.0 / rank_first_pos if rank_first_pos is not None else None

        def _capture(k: int) -> int | None:
            if n_pos == 0:
                return None
            return 1 if any(int(lbl) == 1 for lbl in ranked_labels[:k]) else 0

        if n_pos == 0:
            p1 = p2 = p3 = p5 = None
            r1 = r2 = r3 = r5 = None
            capture_1 = capture_3 = capture_5 = None
        else:
            p1 = _p_at_k(1)
            p2 = _p_at_k(2)
            p3 = _p_at_k(3)
            p5 = _p_at_k(5)
            r1 = _r_at_k(1)
            r2 = _r_at_k(2)
            r3 = _r_at_k(3)
            r5 = _r_at_k(5)
            capture_1 = _capture(1)
            capture_3 = _capture(3)
            capture_5 = _capture(5)

        commit_metrics.append(PerCommitMetrics(
            repo_name=repo,
            commit_sha=sha,
            total_files=n_files,
            positive_files=n_pos,
            precision_at_1=p1,
            precision_at_2=p2,
            precision_at_3=p3,
            precision_at_5=p5,
            recall_at_1=r1,
            recall_at_2=r2,
            recall_at_3=r3,
            recall_at_5=r5,
            rank_of_first_positive=rank_first_pos,
            reciprocal_rank=reciprocal_rank,
            top_1_capture=capture_1,
            top_3_capture=capture_3,
            top_5_capture=capture_5,
        ))

    total_commits = len(commit_metrics)
    pos_commits = [c for c in commit_metrics if c.positive_files > 0]
    zero_pos_commits = [c for c in commit_metrics if c.positive_files == 0]
    n_pos_commits = len(pos_commits)

    mean_files = float(np.mean([c.total_files for c in commit_metrics])) if commit_metrics else 0.0

    n_with_pos = len(pos_commits)
    frac_top1 = (
        sum(1 for c in pos_commits if c.top_1_capture == 1) / n_with_pos
        if n_with_pos > 0 else 0.0
    )
    frac_top3 = (
        sum(1 for c in pos_commits if c.top_3_capture == 1) / n_with_pos
        if n_with_pos > 0 else 0.0
    )
    frac_top5 = (
        sum(1 for c in pos_commits if c.top_5_capture == 1) / n_with_pos
        if n_with_pos > 0 else 0.0
    )

    return AggregateRankingMetrics(
        num_commits_with_positives=n_pos_commits,
        num_commits_total=total_commits,
        num_commits_zero_positives=len(zero_pos_commits),
        mean_files_per_commit=mean_files,

        mean_precision_at_1=_safe_mean([c.precision_at_1 for c in pos_commits]),
        mean_precision_at_2=_safe_mean([c.precision_at_2 for c in pos_commits]),
        mean_precision_at_3=_safe_mean([c.precision_at_3 for c in pos_commits]),
        mean_precision_at_5=_safe_mean([c.precision_at_5 for c in pos_commits]),
        mean_recall_at_1=_safe_mean([c.recall_at_1 for c in pos_commits]),
        mean_recall_at_2=_safe_mean([c.recall_at_2 for c in pos_commits]),
        mean_recall_at_3=_safe_mean([c.recall_at_3 for c in pos_commits]),
        mean_recall_at_5=_safe_mean([c.recall_at_5 for c in pos_commits]),

        median_precision_at_1=_safe_median([c.precision_at_1 for c in pos_commits]),
        median_precision_at_2=_safe_median([c.precision_at_2 for c in pos_commits]),
        median_precision_at_3=_safe_median([c.precision_at_3 for c in pos_commits]),
        median_precision_at_5=_safe_median([c.precision_at_5 for c in pos_commits]),
        median_recall_at_1=_safe_median([c.recall_at_1 for c in pos_commits]),
        median_recall_at_2=_safe_median([c.recall_at_2 for c in pos_commits]),
        median_recall_at_3=_safe_median([c.recall_at_3 for c in pos_commits]),
        median_recall_at_5=_safe_median([c.recall_at_5 for c in pos_commits]),

        mean_rank_of_first_positive=_safe_mean(
            [float(c.rank_of_first_positive) for c in pos_commits]
        ),
        median_rank_of_first_positive=_safe_median(
            [float(c.rank_of_first_positive) for c in pos_commits]
        ),
        mean_reciprocal_rank=_safe_mean([c.reciprocal_rank for c in pos_commits]),

        fraction_top_1_capture=frac_top1,
        fraction_top_3_capture=frac_top3,
        fraction_top_5_capture=frac_top5,

        per_commit_details=commit_metrics,
    )


def global_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> GlobalClassificationMetrics:
    """Compute secondary global classification metrics."""
    total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    prevalence = n_pos / total if total > 0 else 0.0

    if n_pos == 0 or n_pos == total:
        roc_auc_val = 0.0
        pr_auc_val = 0.0
    else:
        roc_auc_val = float(roc_auc_score(y_true, y_prob))
        pr_auc_val = float(average_precision_score(y_true, y_prob))

    pr_auc_lift = pr_auc_val / prevalence if prevalence > 0 else 0.0

    y_pred = (y_prob >= 0.5).astype(int)
    if n_pos == 0:
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0
    else:
        precision_val = float(precision_score(y_true, y_pred, zero_division=0))
        recall_val = float(recall_score(y_true, y_pred, zero_division=0))
        f1_val = (
            2 * precision_val * recall_val / (precision_val + recall_val)
            if (precision_val + recall_val) > 0 else 0.0
        )

    ranked_indices = np.argsort(-y_prob, kind="stable")

    def _pr_at_k(k: int) -> tuple[float, float]:
        n = min(k, total)
        if n == 0 or n_pos == 0:
            return 0.0, 0.0
        tp = int(np.sum(y_true[ranked_indices[:n]] == 1))
        return tp / n, tp / n_pos

    p5, r5 = _pr_at_k(5)
    p10, r10 = _pr_at_k(10)
    p20, r20 = _pr_at_k(20)
    p50, r50 = _pr_at_k(50)
    p100, r100 = _pr_at_k(100)

    top_10 = int(np.sum(y_true[ranked_indices[:min(10, total)]] == 1))
    top_20 = int(np.sum(y_true[ranked_indices[:min(20, total)]] == 1))
    top_50 = int(np.sum(y_true[ranked_indices[:min(50, total)]] == 1))
    top_100 = int(np.sum(y_true[ranked_indices[:min(100, total)]] == 1))

    return GlobalClassificationMetrics(
        roc_auc=roc_auc_val,
        pr_auc=pr_auc_val,
        positive_prevalence=prevalence,
        pr_auc_lift=pr_auc_lift,
        precision=precision_val,
        recall=recall_val,
        f1=f1_val,
        precision_at_5=p5,
        recall_at_5=r5,
        precision_at_10=p10,
        recall_at_10=r10,
        precision_at_20=p20,
        recall_at_20=r20,
        precision_at_50=p50,
        recall_at_50=r50,
        precision_at_100=p100,
        recall_at_100=r100,
        top_10_capture=top_10,
        top_20_capture=top_20,
        top_50_capture=top_50,
        top_100_capture=top_100,
        total_examples=total,
    )


def per_repo_metrics(
    metadata: list[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    expected_repos: list[str] | None = None,
) -> list[PerRepoMetrics]:
    """Compute per-repository metrics."""
    repo_data: dict[str, dict] = {}
    for i, m in enumerate(metadata):
        repo = m["repo_name"]
        if expected_repos is not None and repo not in expected_repos:
            continue
        if repo not in repo_data:
            repo_data[repo] = {
                "indices": [],
                "labels": [],
                "probs": [],
                "commits": set(),
                "pos_commits": set(),
            }
        repo_data[repo]["indices"].append(i)
        repo_data[repo]["labels"].append(int(y_true[i]))
        repo_data[repo]["probs"].append(float(y_prob[i]))
        repo_data[repo]["commits"].add(m["commit_sha"])
        if y_true[i] == 1:
            repo_data[repo]["pos_commits"].add(m["commit_sha"])

    results: list[PerRepoMetrics] = []
    for repo in sorted(repo_data):
        d = repo_data[repo]
        labels = np.array(d["labels"])
        probs = np.array(d["probs"])
        n_pos = int(np.sum(labels == 1))
        n_total = len(labels)
        prevalence = n_pos / n_total if n_total > 0 else 0.0

        pr_auc_val = None
        if n_pos >= 3 and n_pos < n_total:
            pr_auc_val = float(average_precision_score(labels, probs))

        repo_meta = [metadata[i] for i in d["indices"]]
        repo_ranking = per_commit_ranking(repo_meta, labels, probs)

        mean_p1 = None
        mean_r1 = None
        mean_mrr = None
        if repo_ranking.num_commits_with_positives > 0:
            mean_p1 = repo_ranking.mean_precision_at_1
            mean_r1 = repo_ranking.mean_recall_at_1
            mean_mrr = repo_ranking.mean_reciprocal_rank

        results.append(PerRepoMetrics(
            repo_name=repo,
            total_rows=n_total,
            positive_rows=n_pos,
            positive_commits=len(d["pos_commits"]),
            total_commits=len(d["commits"]),
            prevalence=prevalence,
            pr_auc=pr_auc_val,
            mean_precision_at_1=mean_p1,
            mean_recall_at_1=mean_r1,
            mean_reciprocal_rank=mean_mrr,
        ))

    return results


def macro_average_per_repo(repo_metrics_list: list[PerRepoMetrics]) -> dict:
    """Compute unweighted macro average across repositories."""
    if not repo_metrics_list:
        return {}

    return {
        "num_repos": len(repo_metrics_list),
        "mean_prevalence": _safe_mean([r.prevalence for r in repo_metrics_list]),
        "mean_pr_auc": _safe_mean([r.pr_auc for r in repo_metrics_list]),
        "mean_precision_at_1": _safe_mean(
            [r.mean_precision_at_1 for r in repo_metrics_list]
        ),
        "mean_recall_at_1": _safe_mean(
            [r.mean_recall_at_1 for r in repo_metrics_list]
        ),
        "mean_reciprocal_rank": _safe_mean(
            [r.mean_reciprocal_rank for r in repo_metrics_list]
        ),
    }


def dominance_check(
    metadata: list[dict],
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Check if any single repo dominates positive examples."""
    repo_pos: dict[str, int] = defaultdict(int)
    total_pos = 0
    for i, m in enumerate(metadata):
        if y_true[i] == 1:
            repo_pos[m["repo_name"]] += 1
            total_pos += 1

    dominated = {}
    for repo, count in repo_pos.items():
        fraction = count / total_pos if total_pos > 0 else 0.0
        dominated[repo] = {
            "positive_count": count,
            "fraction_of_total": fraction,
            "dominates": fraction > threshold,
        }

    return {
        "total_positives": total_pos,
        "repos": dominated,
        "any_dominates": any(v["dominates"] for v in dominated.values()),
    }
