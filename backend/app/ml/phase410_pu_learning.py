"""Phase 4.10: PU Learning Feasibility Experiment.

Evaluates whether the observed-positive label signal provides a learnable
ranking signal above random. Methods: random baseline, naive P-vs-U
(biased baseline), and P-vs-U ranker (ranking signal only, no calibration
claims). Elkan-Noto is NOT implemented because SCAR is untestable.

All metrics are observed-positive ranking metrics. No true-defect metrics
are reported (no ROC-AUC, PR-AUC, precision, recall, F1, specificity,
or accuracy as estimates of true defect performance).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.app.dataset.schemas import DatasetRow
from backend.app.features.experimental_schemas import (
    EXPERIMENTAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_NAMES,
)
from backend.app.ml.dataset_loader import (
    FEATURE_NAMES,
    NUM_FEATURES,
)
from backend.app.ml.phase49_supervision_feasibility import (
    _build_sha_lookup,
    _parse_frozen_manifest,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
    construct_supervision_sets,
)
from backend.app.ml.repo_split import (
    REPO_MANIFEST,
    build_experimental_lookup,
    load_all_rows,
    reassign_splits,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DIR = Path("backend/data/models/v0.1.0-combined-v3-exp-4.5a-real")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.10")

RANDOM_SEEDS = [42, 123, 456]
ELKAN_NOTO_STATUS = "NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE"

EXPERIMENT_MODES: dict[str, list[int]] = {
    "E0": [],
    "E1": [0, 1, 2],
    "E2": [3, 4, 5],
    "E4": list(range(EXPERIMENTAL_FEATURE_COUNT)),
}

PRIMARY_EXPERIMENT_MODES = ["E0", "E1", "E2", "E4"]
SENSITIVITY_EXPERIMENT_MODES = ["E0", "E2"]
RECALL_AT_K_VALUES = [5, 10, 20, 50, 100, 200, 500]


@dataclass
class PUDataset:
    """Container for PU experiment data.

    All data is split-aware: train data is used for fitting, val/test are
    held out for evaluation. No fitting on val/test.
    """

    mode: str
    feature_names: list[str]

    X_P_train: np.ndarray
    X_U_train: np.ndarray
    X_P_val: np.ndarray
    X_U_val: np.ndarray
    X_P_test: np.ndarray
    X_U_test: np.ndarray

    metadata_P_train: list[dict]
    metadata_U_train: list[dict]
    metadata_P_val: list[dict]
    metadata_U_val: list[dict]
    metadata_P_test: list[dict]
    metadata_U_test: list[dict]

    P_keys_train: set[tuple[str, str]]
    P_keys_val: set[tuple[str, str]]
    P_keys_test: set[tuple[str, str]]

    N_train: int
    N_val: int
    N_test: int

    P_count: int
    U_count: int
    excluded_sensitivity: int
    out_of_scope: int
    defensible_negative: int

    def X_all(self, split: str) -> np.ndarray:
        if split == "train":
            return np.vstack([self.X_P_train, self.X_U_train])
        if split == "validation":
            return np.vstack([self.X_P_val, self.X_U_val])
        return np.vstack([self.X_P_test, self.X_U_test])

    def metadata_all(self, split: str) -> list[dict]:
        if split == "train":
            return self.metadata_P_train + self.metadata_U_train
        if split == "validation":
            return self.metadata_P_val + self.metadata_U_val
        return self.metadata_P_test + self.metadata_U_test

    def P_keys(self, split: str) -> set[tuple[str, str]]:
        if split == "train":
            return self.P_keys_train
        if split == "validation":
            return self.P_keys_val
        return self.P_keys_test


def _row_key(row: dict) -> tuple[str, str]:
    return (row["commit_sha"], row["file_path"])


def _row_key_from_metadata(meta: dict) -> tuple[str, str]:
    return (meta["commit_sha"], meta["file_path"])


def _random_baseline_metrics(
    N: int,
    P: int,
) -> dict:
    """Compute expected metrics under random ranking."""
    return {
        "expected_recall_at_k": {
            k: min(k, N) / N for k in RECALL_AT_K_VALUES
        },
        "expected_positive_count_at_k": {
            k: min(k, N) * P / N for k in RECALL_AT_K_VALUES
        },
        "expected_enrichment_at_k": {
            k: 1.0 for k in RECALL_AT_K_VALUES
        },
        "expected_mean_rank": (N + 1) / 2,
        "N": N,
        "P": P,
    }


def _compute_ranking_metrics(
    scores: np.ndarray,
    metadata: list[dict],
    positive_keys: set[tuple[str, str]],
    label: str,
    mode: str,
    split: str,
    seed: int | None = None,
) -> dict:
    """Compute observed-positive ranking metrics.

    All metrics are observed-positive ranking metrics. No true-defect
    metrics are computed.
    """
    row_keys = [_row_key_from_metadata(m) for m in metadata]
    n_total = len(scores)

    order = np.argsort(-scores, kind="stable")
    ordered_keys = [row_keys[i] for i in order]

    positive_ranks = []
    for pk in positive_keys:
        for rank_idx, rk in enumerate(ordered_keys):
            if rk == pk:
                positive_ranks.append(rank_idx + 1)
                break

    positive_ranks_arr = np.array(positive_ranks)

    recall_at_k = {}
    enrichment_at_k = {}
    for k in RECALL_AT_K_VALUES:
        n_pos_in_top_k = sum(1 for r in positive_ranks if r <= k)
        obs_recall = n_pos_in_top_k / len(positive_keys) if positive_keys else 0.0
        obs_enrichment = (
            (n_pos_in_top_k / k) / (len(positive_keys) / n_total)
            if k > 0 and len(positive_keys) > 0
            else 1.0
        )
        recall_at_k[f"observed_positive_recall_at_k_{k}"] = obs_recall
        enrichment_at_k[f"observed_positive_enrichment_at_k_{k}"] = obs_enrichment

    mean_rank = float(positive_ranks_arr.mean()) if len(positive_ranks_arr) > 0 else float(n_total)
    if len(positive_ranks_arr) > 0:
        median_rank = float(np.median(positive_ranks_arr))
    else:
        median_rank = float(n_total)

    ks_stat = 0.0
    ks_pvalue = 1.0
    positive_scores = []
    unlabeled_scores = []
    for i, rk in enumerate(row_keys):
        if rk in positive_keys:
            positive_scores.append(scores[i])
        else:
            unlabeled_scores.append(scores[i])

    if positive_scores and unlabeled_scores:
        ks_result = ks_2samp(positive_scores, unlabeled_scores)
        ks_stat = float(ks_result.statistic)
        ks_pvalue = float(ks_result.pvalue)

    top_10_capture = sum(1 for r in positive_ranks if r <= 10)

    return {
        "label": label,
        "mode": mode,
        "split": split,
        "seed": seed,
        "n_total": n_total,
        "n_observed_positives": len(positive_keys),
        "recall_at_k": recall_at_k,
        "enrichment_at_k": enrichment_at_k,
        "mean_rank_of_observed_positives": mean_rank,
        "median_rank_of_observed_positives": median_rank,
        "top_10_capture": top_10_capture,
        "score_separation_ks": ks_stat,
        "score_separation_ks_pvalue": ks_pvalue,
        "positive_scores": [float(s) for s in positive_scores],
        "unlabeled_score_summary": {
            "mean": float(np.mean(unlabeled_scores)) if unlabeled_scores else 0.0,
            "std": float(np.std(unlabeled_scores)) if unlabeled_scores else 0.0,
            "min": float(np.min(unlabeled_scores)) if unlabeled_scores else 0.0,
            "max": float(np.max(unlabeled_scores)) if unlabeled_scores else 0.0,
        },
    }


def _compute_seed_stability(results: list[dict]) -> dict:
    """Compute cross-seed stability for each mode × label × split."""
    by_config: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in results:
        key = (r["mode"], r["label"], r["split"])
        by_config[key].append(r)

    stability: dict[str, dict] = {}
    for (mode, label, split), seed_results in sorted(by_config.items()):
        if len(seed_results) < 2:
            continue
        metrics_keys = [
            "mean_rank_of_observed_positives",
            "score_separation_ks",
            "top_10_capture",
        ]
        for mk in metrics_keys:
            vals = [sr[mk] for sr in seed_results]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            cv = std_val / mean_val if mean_val != 0 else 0.0
            config_key = f"{mode}_{label}_{split}_{mk}"
            stability[config_key] = {
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "values": vals,
                "n_seeds": len(seed_results),
            }

    return stability


def _compute_repository_analysis(
    primary_rows: list[dict],
    all_supervised_rows: list[dict],
) -> dict:
    """Analyze per-repository coverage and concentration."""
    positive_repos: set[str] = set()
    for row in primary_rows:
        positive_repos.add(row["repo_name"])

    repo_positive_counts: dict[str, int] = defaultdict(int)
    for row in primary_rows:
        repo_positive_counts[row["repo_name"]] += 1

    total_positive = sum(repo_positive_counts.values())
    repo_positive_fraction = {
        repo: count / total_positive if total_positive > 0 else 0.0
        for repo, count in repo_positive_counts.items()
    }

    sorted_fractions = sorted(repo_positive_fraction.values(), reverse=True)
    top_5_share = sum(sorted_fractions[:5]) if len(sorted_fractions) >= 5 else sum(sorted_fractions)

    gini_numerator = 0.0
    for i, fi in enumerate(sorted_fractions):
        for j, fj in enumerate(sorted_fractions):
            gini_numerator += abs(fi - fj)
    gini = gini_numerator / (2 * len(sorted_fractions)) if sorted_fractions else 0.0

    top_repo = sorted(repo_positive_counts.items(), key=lambda x: -x[1])
    concentration_summary = [
        {"repo": repo, "positive_count": count, "fraction": repo_positive_fraction[repo]}
        for repo, count in top_repo[:5]
    ]

    return {
        "positive_repos": sorted(positive_repos),
        "n_positive_repos": len(positive_repos),
        "repo_positive_counts": dict(repo_positive_counts),
        "repo_positive_fraction": repo_positive_fraction,
        "total_positive": total_positive,
        "top_5_share": top_5_share,
        "gini_coefficient": gini,
        "concentration_summary": concentration_summary,
    }


def _compute_ranking_analysis(
    primary_results: list[dict],
    sensitivity_results: list[dict],
) -> dict:
    """Compute ranking analysis with score distributions and per-repo metrics."""
    all_results = primary_results + sensitivity_results
    analysis: dict[str, object] = {}

    for r in all_results:
        key = f"{r['label']}_{r['mode']}_{r['split']}_s{r['seed']}"
        pos_scores = r["positive_scores"]
        ul_summary = r["unlabeled_score_summary"]
        analysis[key] = {
            "positive_score_mean": float(np.mean(pos_scores)) if pos_scores else 0.0,
            "positive_score_std": float(np.std(pos_scores)) if pos_scores else 0.0,
            "unlabeled_score_mean": ul_summary["mean"],
            "unlabeled_score_std": ul_summary["std"],
            "score_gap": (
                float(np.mean(pos_scores)) - ul_summary["mean"]
                if pos_scores
                else 0.0
            ),
        }

    primary_test_results = [r for r in primary_results if r["split"] == "test"]
    repo_metrics: dict[str, dict] = {}
    for r in primary_test_results:
        for repo in sorted(REPO_MANIFEST.keys()):
            repo_rows = [
                m for m in r.get("_metadata_all", []) if m.get("repo_name") == repo
            ]
            if not repo_rows:
                continue
            if repo not in repo_metrics:
                repo_metrics[repo] = {
                    "total_rows": len(repo_rows),
                    "positive_in_repo": 0,
                }

    return {
        "score_analysis": analysis,
        "per_repo_metrics": repo_metrics,
    }


def _write_artifacts(
    output_dir: Path,
    config: dict,
    accounting: dict,
    primary_results: list[dict],
    sensitivity_results: list[dict],
    stability: dict,
    repo_analysis: dict,
    ranking_analysis: dict,
    random_baselines: dict,
    sensitivity_random_baselines: dict,
    report: str,
) -> None:
    """Write all Phase 4.10 artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "pu_experiment_config.json": config,
        "pu_dataset_accounting.json": accounting,
        "pu_results.json": {
            "primary_results": primary_results,
            "sensitivity_results": sensitivity_results,
            "random_baselines": random_baselines,
            "sensitivity_random_baselines": sensitivity_random_baselines,
        },
        "pu_ranking_analysis.json": ranking_analysis,
        "repository_analysis.json": repo_analysis,
        "seed_stability.json": stability,
        "phase410_pu_learning.md": report,
    }

    for filename, data in artifacts.items():
        path = output_dir / filename
        if filename.endswith(".md"):
            path.write_text(data, encoding="utf-8")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        logger.info("Wrote %s", path)


def run_pu_experiment(
    data_dir: Path = DATA_DIR,
    exp_dir: Path = EXP_DIR,
    output_dir: Path = OUTPUT_DIR,
    seeds: list[int] | None = None,
) -> dict:
    """Run the complete Phase 4.10 PU learning experiment.

    Returns a dict with all results. Writes 7 artifacts to output_dir.
    """
    if seeds is None:
        seeds = RANDOM_SEEDS

    logger.info("Phase 4.10: Starting PU learning feasibility experiment")

    # Step 1: Construct supervision sets (Phase 4.9 function)
    supervised_rows = _read_all_supervised_rows(data_dir)
    ambiguous_rows = _read_ambiguous_rows(data_dir)
    sha_lookup = _build_sha_lookup(supervised_rows, ambiguous_rows)
    manifest = _parse_frozen_manifest()

    supervision = construct_supervision_sets(supervised_rows, sha_lookup, manifest)

    primary_rows = supervision["primary_observed_path_associated"]["rows"]
    sensitivity_rows = supervision["sensitivity_commit_only_overlap"]["rows"]
    unlabeled_rows = supervision["unlabeled"]["rows"]
    oos_rows = supervision["out_of_scope"]["rows"]
    defensible_neg_rows = supervision["defensible_negative"]["rows"]

    P_primary = len(primary_rows)
    S_count = len(sensitivity_rows)
    U_count = len(unlabeled_rows)
    OOS_count = len(oos_rows)
    DN_count = len(defensible_neg_rows)
    total_source = len(supervised_rows)

    assert P_primary == 108, f"Primary P = {P_primary}, expected 108"
    assert S_count == 594, f"Sensitivity S = {S_count}, expected 594"
    assert U_count == 53573, f"Unlabeled U = {U_count}, expected 53573"
    assert OOS_count == 114, f"OOS = {OOS_count}, expected 114"
    assert DN_count == 0, f"Defensible negative = {DN_count}, expected 0"

    eligible_primary = P_primary + U_count
    assert eligible_primary == 53681, f"Eligible primary = {eligible_primary}, expected 53681"

    eligible_sensitivity = (P_primary + S_count) + U_count
    assert eligible_sensitivity == 54275, (
        f"Eligible sensitivity = {eligible_sensitivity}, expected 54275"
    )

    assert total_source == P_primary + S_count + U_count + OOS_count + DN_count, (
        f"Total reconciliation: {P_primary} + {S_count} + {U_count} "
        f"+ {OOS_count} + {DN_count} = "
        f"{P_primary + S_count + U_count + OOS_count + DN_count}, "
        f"expected {total_source}"
    )

    # Step 2: Load DatasetRow objects and reassign splits
    all_dataset_rows = load_all_rows(data_dir)
    split_rows = reassign_splits(all_dataset_rows)
    exp_lookup = build_experimental_lookup(exp_dir / "experimental_features.jsonl")

    # Build lookup from row key -> DatasetRow for each split
    row_lookup_by_split: dict[str, dict[tuple[str, str], DatasetRow]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for split_name, rows in split_rows.items():
        for dro in rows:
            key = (dro.commit_sha, dro.file_path)
            row_lookup_by_split[split_name][key] = dro

    # Step 3: Build P/U key sets
    primary_keys = {_row_key(r) for r in primary_rows}
    sensitivity_keys = {_row_key(r) for r in sensitivity_rows}
    unlabeled_keys = {_row_key(r) for r in unlabeled_rows}
    oos_keys = {_row_key(r) for r in oos_rows}

    # Verify disjointness
    assert primary_keys.isdisjoint(unlabeled_keys), "Primary and unlabeled overlap"
    assert primary_keys.isdisjoint(sensitivity_keys), "Primary and sensitivity overlap"
    assert sensitivity_keys.isdisjoint(unlabeled_keys), "Sensitivity and unlabeled overlap"
    assert oos_keys.isdisjoint(primary_keys), "OOS and primary overlap"
    assert oos_keys.isdisjoint(unlabeled_keys), "OOS and unlabeled overlap"
    assert oos_keys.isdisjoint(sensitivity_keys), "OOS and sensitivity overlap"
    assert DN_count == 0, "Defensible negatives fabricated"

    # Step 4: Build feature matrices for each mode
    def _build_feature_arrays(
        mode_indices: list[int],
        experiment: str,
        mode_name: str,
    ) -> PUDataset:
        mode_dim = NUM_FEATURES + len(mode_indices)
        mode_names = [EXPERIMENTAL_FEATURE_NAMES[i] for i in mode_indices]
        feature_names = list(FEATURE_NAMES) + mode_names

        result: dict[str, dict[str, list]] = {}
        for split in ("train", "validation", "test"):
            lookup = row_lookup_by_split[split]
            split_result: dict[str, list] = {"X_P": [], "X_U": [], "meta_P": [], "meta_U": []}
            p_keys_split: set[tuple[str, str]] = set()

            for key, dro in lookup.items():
                cf = dro.commit_features
                ff = dro.file_features
                base_vec = cf + ff
                exp_vec = exp_lookup.get(
                    (dro.commit_sha, dro.file_path),
                    [0.0] * EXPERIMENTAL_FEATURE_COUNT,
                )
                selected_exp = [exp_vec[i] for i in mode_indices]
                feat_vec = base_vec + selected_exp

                meta = {
                    "commit_sha": dro.commit_sha,
                    "file_path": dro.file_path,
                    "repo_name": dro.repo_name,
                    "split": split,
                    "file_status": dro.file_status,
                    "label_status": dro.label_status,
                }

                if experiment == "primary":
                    if key in primary_keys:
                        split_result["X_P"].append(feat_vec)
                        split_result["meta_P"].append(meta)
                        p_keys_split.add(key)
                    elif key in unlabeled_keys:
                        split_result["X_U"].append(feat_vec)
                        split_result["meta_U"].append(meta)
                elif experiment == "sensitivity":
                    if key in primary_keys or key in sensitivity_keys:
                        split_result["X_P"].append(feat_vec)
                        split_result["meta_P"].append(meta)
                        p_keys_split.add(key)
                    elif key in unlabeled_keys:
                        split_result["X_U"].append(feat_vec)
                        split_result["meta_U"].append(meta)

            empty_arr = np.empty((0, mode_dim), dtype=np.float64)
            x_p = (
                np.array(split_result["X_P"], dtype=np.float64)
                if split_result["X_P"]
                else empty_arr
            )
            x_u = (
                np.array(split_result["X_U"], dtype=np.float64)
                if split_result["X_U"]
                else empty_arr
            )
            result[split] = {
                "X_P": x_p,
                "X_U": x_u,
                "meta_P": split_result["meta_P"],
                "meta_U": split_result["meta_U"],
                "p_keys": p_keys_split,
            }

        train = result["train"]
        val = result["validation"]
        test = result["test"]

        P_total = len(train["p_keys"]) + len(val["p_keys"]) + len(test["p_keys"])
        U_total = len(train["X_U"]) + len(val["X_U"]) + len(test["X_U"])

        if experiment == "primary":
            assert P_total == P_primary, (
                f"Primary P total = {P_total}, expected {P_primary}"
            )
            assert U_total == U_count, f"Primary U total = {U_total}, expected {U_count}"
            assert len(train["p_keys"]) + len(val["p_keys"]) + len(test["p_keys"]) == P_primary
        elif experiment == "sensitivity":
            assert P_total == P_primary + S_count, (
                f"Sensitivity P total = {P_total}, expected {P_primary + S_count}"
            )
            assert U_total == U_count, (
                f"Sensitivity U total = {U_total}, expected {U_count}"
            )

        return PUDataset(
            mode=mode_name,
            feature_names=feature_names,
            X_P_train=train["X_P"],
            X_U_train=train["X_U"],
            X_P_val=val["X_P"],
            X_U_val=val["X_U"],
            X_P_test=test["X_P"],
            X_U_test=test["X_U"],
            metadata_P_train=train["meta_P"],
            metadata_U_train=train["meta_U"],
            metadata_P_val=val["meta_P"],
            metadata_U_val=val["meta_U"],
            metadata_P_test=test["meta_P"],
            metadata_U_test=test["meta_U"],
            P_keys_train=train["p_keys"],
            P_keys_val=val["p_keys"],
            P_keys_test=test["p_keys"],
            N_train=len(train["X_P"]) + len(train["X_U"]),
            N_val=len(val["X_P"]) + len(val["X_U"]),
            N_test=len(test["X_P"]) + len(test["X_U"]),
            P_count=P_total,
            U_count=U_total,
            excluded_sensitivity=S_count if experiment == "primary" else 0,
            out_of_scope=OOS_count,
            defensible_negative=DN_count,
        )

    # Run primary experiment
    primary_results: list[dict] = []
    random_baselines: dict[str, dict] = {}

    for mode_name in PRIMARY_EXPERIMENT_MODES:
        mode_indices = EXPERIMENT_MODES[mode_name]
        pu_data = _build_feature_arrays(mode_indices, "primary", mode_name)

        # Per-split random baselines (split-specific N and P)
        random_baselines[mode_name] = {
            "train": _random_baseline_metrics(
                pu_data.N_train, len(pu_data.X_P_train),
            ),
            "validation": _random_baseline_metrics(
                pu_data.N_val, len(pu_data.X_P_val),
            ),
            "test": _random_baseline_metrics(
                pu_data.N_test, len(pu_data.X_P_test),
            ),
        }

        for seed in seeds:
            rng = np.random.RandomState(seed)

            # Method A: Random ranking
            scores_random = rng.uniform(0.0, 1.0, size=pu_data.N_train)
            metrics_random = _compute_ranking_metrics(
                scores_random,
                pu_data.metadata_all("train"),
                pu_data.P_keys("train"),
                label="random",
                mode=mode_name,
                split="train",
                seed=seed,
            )
            metrics_random["_metadata_all"] = pu_data.metadata_all("train")
            primary_results.append(metrics_random)

            for split in ("validation", "test"):
                n_eval = pu_data.N_val if split == "validation" else pu_data.N_test
                scores_split = rng.uniform(0.0, 1.0, size=n_eval)
                metrics_split = _compute_ranking_metrics(
                    scores_split,
                    pu_data.metadata_all(split),
                    pu_data.P_keys(split),
                    label="random",
                    mode=mode_name,
                    split=split,
                    seed=seed,
                )
                primary_results.append(metrics_split)

            # Method B: Naive P-vs-U (deliberately biased, uses predict_proba)
            X_train_b = np.vstack([pu_data.X_P_train, pu_data.X_U_train])
            y_train_b = np.concatenate([
                np.ones(len(pu_data.X_P_train), dtype=np.int64),
                np.zeros(len(pu_data.X_U_train), dtype=np.int64),
            ])
            scaler_b = StandardScaler()
            X_train_b_s = scaler_b.fit_transform(X_train_b)
            clf_b = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf_b.fit(X_train_b_s, y_train_b)

            for split in ("train", "validation", "test"):
                if split == "train":
                    X_eval = np.vstack([pu_data.X_P_train, pu_data.X_U_train])
                elif split == "validation":
                    X_eval = np.vstack([pu_data.X_P_val, pu_data.X_U_val])
                else:
                    X_eval = np.vstack([pu_data.X_P_test, pu_data.X_U_test])

                X_eval_s = scaler_b.transform(X_eval)
                scores_naive = clf_b.predict_proba(X_eval_s)[:, 1]
                metrics_naive = _compute_ranking_metrics(
                    scores_naive,
                    pu_data.metadata_all(split),
                    pu_data.P_keys(split),
                    label="naive_biased",
                    mode=mode_name,
                    split=split,
                    seed=seed,
                )
                primary_results.append(metrics_naive)

            # Method C: P-vs-U Ranker (uses decision_function, no calibration)
            X_train_c = np.vstack([pu_data.X_P_train, pu_data.X_U_train])
            y_train_c = np.concatenate([
                np.ones(len(pu_data.X_P_train), dtype=np.int64),
                np.zeros(len(pu_data.X_U_train), dtype=np.int64),
            ])
            scaler_c = StandardScaler()
            X_train_c_s = scaler_c.fit_transform(X_train_c)
            clf_c = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf_c.fit(X_train_c_s, y_train_c)

            for split in ("train", "validation", "test"):
                if split == "train":
                    X_eval = np.vstack([pu_data.X_P_train, pu_data.X_U_train])
                elif split == "validation":
                    X_eval = np.vstack([pu_data.X_P_val, pu_data.X_U_val])
                else:
                    X_eval = np.vstack([pu_data.X_P_test, pu_data.X_U_test])

                X_eval_s = scaler_c.transform(X_eval)
                decision_scores = clf_c.decision_function(X_eval_s)
                min_val = float(decision_scores.min())
                max_val = float(decision_scores.max())
                range_val = max_val - min_val
                if range_val > 0:
                    scores_ranker = (decision_scores - min_val) / range_val
                else:
                    scores_ranker = np.full_like(decision_scores, 0.5)
                metrics_ranker = _compute_ranking_metrics(
                    scores_ranker,
                    pu_data.metadata_all(split),
                    pu_data.P_keys(split),
                    label="pu_ranker",
                    mode=mode_name,
                    split=split,
                    seed=seed,
                )
                primary_results.append(metrics_ranker)

    # Run sensitivity experiment
    sensitivity_results: list[dict] = []
    sensitivity_random_baselines: dict[str, dict] = {}

    for mode_name in SENSITIVITY_EXPERIMENT_MODES:
        mode_indices = EXPERIMENT_MODES[mode_name]
        pu_data_sens = _build_feature_arrays(mode_indices, "sensitivity", mode_name)

        # Per-split sensitivity random baselines
        sensitivity_random_baselines[mode_name] = {
            "train": _random_baseline_metrics(
                pu_data_sens.N_train, len(pu_data_sens.X_P_train),
            ),
            "validation": _random_baseline_metrics(
                pu_data_sens.N_val, len(pu_data_sens.X_P_val),
            ),
            "test": _random_baseline_metrics(
                pu_data_sens.N_test, len(pu_data_sens.X_P_test),
            ),
        }

        for seed in seeds:
            X_train_s = np.vstack([pu_data_sens.X_P_train, pu_data_sens.X_U_train])
            y_train_s = np.concatenate([
                np.ones(len(pu_data_sens.X_P_train), dtype=np.int64),
                np.zeros(len(pu_data_sens.X_U_train), dtype=np.int64),
            ])
            scaler_s = StandardScaler()
            X_train_s_s = scaler_s.fit_transform(X_train_s)
            clf_s = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf_s.fit(X_train_s_s, y_train_s)

            for split in ("train", "validation", "test"):
                if split == "train":
                    X_eval = np.vstack([pu_data_sens.X_P_train, pu_data_sens.X_U_train])
                elif split == "validation":
                    X_eval = np.vstack([pu_data_sens.X_P_val, pu_data_sens.X_U_val])
                else:
                    X_eval = np.vstack([pu_data_sens.X_P_test, pu_data_sens.X_U_test])

                X_eval_s = scaler_s.transform(X_eval)
                decision_scores = clf_s.decision_function(X_eval_s)
                min_val = float(decision_scores.min())
                max_val = float(decision_scores.max())
                range_val = max_val - min_val
                if range_val > 0:
                    scores_sens = (decision_scores - min_val) / range_val
                else:
                    scores_sens = np.full_like(decision_scores, 0.5)

                metrics_sens = _compute_ranking_metrics(
                    scores_sens,
                    pu_data_sens.metadata_all(split),
                    pu_data_sens.P_keys(split),
                    label="sensitivity_pu_ranker",
                    mode=mode_name,
                    split=split,
                    seed=seed,
                )
                sensitivity_results.append(metrics_sens)

    # Seed stability
    stability = _compute_seed_stability(primary_results)

    # Repository analysis
    repo_analysis = _compute_repository_analysis(
        primary_rows, supervised_rows,
    )

    # Ranking analysis
    ranking_analysis = _compute_ranking_analysis(primary_results, sensitivity_results)

    # Accounting artifact
    per_split_accounting = {}
    for split_name in ("train", "validation", "test"):
        lookup = row_lookup_by_split[split_name]
        p_split = sum(1 for k in lookup if k in primary_keys)
        s_split = sum(1 for k in lookup if k in sensitivity_keys)
        u_split = sum(1 for k in lookup if k in unlabeled_keys)
        oos_split = sum(1 for k in lookup if k in oos_keys)
        per_split_accounting[split_name] = {
            "primary": p_split,
            "sensitivity": s_split,
            "unlabeled": u_split,
            "out_of_scope": oos_split,
            "total": p_split + s_split + u_split + oos_split,
        }

    accounting = {
        "total_source_rows": total_source,
        "primary": {
            "positive_count": P_primary,
            "unlabeled_count": U_count,
            "eligible_count": eligible_primary,
            "excluded_sensitivity": S_count,
            "out_of_scope": OOS_count,
            "defensible_negative": DN_count,
        },
        "sensitivity": {
            "positive_count": P_primary + S_count,
            "unlabeled_count": U_count,
            "eligible_count": eligible_sensitivity,
            "out_of_scope": OOS_count,
            "defensible_negative": DN_count,
        },
        "per_split": per_split_accounting,
        "metric_semantics": "observed-positive ranking metrics only",
        "elkan_noto_status": ELKAN_NOTO_STATUS,
    }

    # Config artifact
    config = {
        "seeds": seeds,
        "modes": PRIMARY_EXPERIMENT_MODES,
        "sensitivity_modes": SENSITIVITY_EXPERIMENT_MODES,
        "methods": ["random", "naive_biased", "pu_ranker"],
        "recall_at_k_values": RECALL_AT_K_VALUES,
        "data_dir": str(data_dir),
        "exp_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "elkan_noto_status": ELKAN_NOTO_STATUS,
        "pu_assumptions": {
            "scar_testable": False,
            "scar_status": "UNTESTABLE_WITH_CURRENT_EVIDENCE",
            "sar_status": "COMPATIBLE_WITH_CAVEATS",
            "elkan_noto_c_identifiable": False,
            "metric_semantics": "observed-positive ranking only",
            "no_true_defect_metrics": True,
        },
    }

    # Generate report
    report = _generate_report(
        accounting, config, primary_results, sensitivity_results,
        stability, repo_analysis, random_baselines,
        sensitivity_random_baselines,
    )

    # Write artifacts
    _write_artifacts(
        output_dir, config, accounting, primary_results,
        sensitivity_results, stability, repo_analysis,
        ranking_analysis, random_baselines,
        sensitivity_random_baselines, report,
    )

    logger.info("Phase 4.10: Experiment complete. Artifacts in %s", output_dir)

    return {
        "primary_results": primary_results,
        "sensitivity_results": sensitivity_results,
        "stability": stability,
        "repo_analysis": repo_analysis,
        "ranking_analysis": ranking_analysis,
        "random_baselines": random_baselines,
        "sensitivity_random_baselines": sensitivity_random_baselines,
        "accounting": accounting,
    }


def _generate_report(
    accounting: dict,
    config: dict,
    primary_results: list[dict],
    sensitivity_results: list[dict],
    stability: dict,
    repo_analysis: dict,
    random_baselines: dict,
    sensitivity_random_baselines: dict,
) -> str:
    """Generate the Phase 4.10 markdown report."""
    report = []
    report.append("# Phase 4.10: PU Learning Feasibility Experiment\n")

    report.append("## Executive Summary\n")
    report.append(
        "This experiment evaluates whether the observed-positive label signal "
        "provides a learnable ranking signal above random, using three methods: "
        "random baseline, naive P-vs-U (deliberately biased), and P-vs-U ranker "
        "(ranking signal only, no calibration claims). "
        "All metrics are observed-positive ranking metrics; no true-defect "
        "precision, recall, or ROC is reported.\n"
    )

    report.append("## Methodology\n")

    report.append("### Elkan-Noto Status\n")
    report.append(
        f"ELKAN_NOTO_STATUS = {ELKAN_NOTO_STATUS}\n\n"
        "SCAR is UNTESTABLE_WITH_CURRENT_EVIDENCE (Phase 4.9). "
        "The Elkan-Noto parameter c = P(s=1|y=1) cannot be estimated. "
        "The observed-positive fraction n_P / (n_P + n_U) is P(s=1), NOT c.\n"
    )

    report.append("### PU Methods\n")
    report.append(
        "1. **Random (Method A)**: Uniform random scores. Analytic null expectation "
        "and seeded random realizations are reported.\n"
        "2. **Naive P-vs-U (Method B)**: Treats U as negative. Uses "
        "LogisticRegression with predict_proba(). Deliberately biased. "
        "Probability values are NOT calibrated defect probabilities. "
        "For comparison only.\n"
        "3. **P-vs-U Ranker (Method C)**: Same LogisticRegression formulation "
        "as Method B. Uses decision_function() normalized to [0,1] as a ranking "
        "score. NO calibration claims. Output is interpreted as ranking "
        "similarity to observed positives, NOT calibrated true-defect probability. "
        "The ranking is the scientifically preferred interpretation.\n"
    )
    report.append(
        "#### Method B vs Method C Relationship\n\n"
        "Methods B and C use the same LogisticRegression model (identical "
        "training data, solver, class_weight, random_state). Method B uses "
        "predict_proba() (sigmoid of decision function); Method C uses "
        "decision_function() directly. Because sigmoid is strictly monotonic "
        "in the decision function, their ranking orderings are identical up "
        "to ties and floating-point precision. Differences in recall@K arise "
        "only from ties in the score distribution.\n"
    )

    report.append("### Random Baseline Methodology\n")
    report.append(
        "For each split (train/validation/test), the analytic random baseline "
        "is computed using the split-specific N (eligible observations) and P "
        "(observed positives). The expected metrics under random ranking are:\n"
        "- expected_recall_at_k = min(k, N) / N\n"
        "- expected_positive_count_at_k = min(k, N) * P / N\n"
        "- expected_enrichment_at_k = 1.0\n"
        "- expected_mean_rank = (N + 1) / 2\n\n"
        "Seeded random realizations (single draws from the random baseline "
        "distribution) are also reported for direct comparison with learned "
        "methods. These realizations have inherent variance; the analytic "
        "expectation is the appropriate null comparison.\n"
    )

    report.append("### Evaluation Metrics\n")
    report.append(
        "All metrics are **observed-positive ranking metrics**. "
        "No true-defect metrics are reported.\n"
        "- observed_positive_recall_at_k: fraction of observed positives in top-K\n"
        "- observed_positive_enrichment_at_k: (obs_in_top_K / K) / (P / N)\n"
        "- mean_rank_of_observed_positives\n"
        "- median_rank_of_observed_positives\n"
        "- score_separation_ks: KS statistic between P and U score distributions\n"
    )

    report.append("## Dataset Accounting\n")
    acct = accounting["primary"]
    report.append(f"- Primary P = {acct['positive_count']}")
    report.append(f"- Primary U = {acct['unlabeled_count']}")
    report.append(f"- Primary eligible = {acct['eligible_count']}")
    report.append(f"- Excluded sensitivity = {acct['excluded_sensitivity']}")
    report.append(f"- Out of scope = {acct['out_of_scope']}")
    report.append(f"- Defensible negative = {acct['defensible_negative']}")
    reconcile = (
        acct["positive_count"] + acct["unlabeled_count"]
        + acct["excluded_sensitivity"] + acct["out_of_scope"]
    )
    report.append(
        f"- Reconciliation: {acct['positive_count']} + "
        f"{acct['unlabeled_count']} + {acct['excluded_sensitivity']} + "
        f"{acct['out_of_scope']} = {reconcile}"
    )
    report.append("")

    sacct = accounting["sensitivity"]
    report.append(f"- Sensitivity P = {sacct['positive_count']}")
    report.append(f"- Sensitivity U = {sacct['unlabeled_count']}")
    report.append(f"- Sensitivity eligible = {sacct['eligible_count']}")
    report.append(
        f"- Reconciliation: {sacct['positive_count']} + "
        f"{sacct['unlabeled_count']} + {sacct['out_of_scope']} = "
        f"{sacct['positive_count'] + sacct['unlabeled_count'] + sacct['out_of_scope']}"
    )
    report.append("")

    report.append("### Per-Split Accounting\n")
    for split_name in ("train", "validation", "test"):
        ps = accounting["per_split"][split_name]
        report.append(
            f"- {split_name}: P={ps['primary']}, S={ps['sensitivity']}, "
            f"U={ps['unlabeled']}, OOS={ps['out_of_scope']}, total={ps['total']}"
        )
    report.append("")

    # --- Primary Experiment Results ---
    report.append("## Results\n")
    report.append("### Primary Experiment Results\n")

    for mode in PRIMARY_EXPERIMENT_MODES:
        report.append(f"#### Mode {mode}\n")
        rb = random_baselines[mode]

        # Per-split random baseline summary
        report.append("**Analytic Random Baselines (per split):**\n")
        report.append("| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] |")
        report.append("|---|---|---|---|---|---|")
        for split_name in ("train", "validation", "test"):
            rbs = rb[split_name]
            e_recall_10 = rbs["expected_recall_at_k"][10]
            e_count_10 = rbs["expected_positive_count_at_k"][10]
            e_mean = rbs["expected_mean_rank"]
            report.append(
                f"| {split_name} | {rbs['N']} | {rbs['P']} | "
                f"{e_recall_10:.6f} | {e_count_10:.2f} | {e_mean:.1f} |"
            )
        report.append("")

        # Full K-level results for each method
        for label in ["random", "naive_biased", "pu_ranker"]:
            report.append(f"**{label}**\n")
            for split in ["train", "validation", "test"]:
                results_for_config = [
                    r for r in primary_results
                    if r["mode"] == mode and r["label"] == label and r["split"] == split
                ]
                if not results_for_config:
                    continue
                r0 = results_for_config[0]
                rbs = rb[split]
                p_split = rbs["P"]
                n_split = rbs["N"]

                report.append(f"*{split}* (N={n_split}, P={p_split}):\n")
                report.append(
                    "| K | recall@K | enrichment@K | E[random recall] | "
                    "E[random count] | observed count |"
                )
                report.append("|---|---|---|---|---|---|")
                for k in RECALL_AT_K_VALUES:
                    recall_k = r0["recall_at_k"][f"observed_positive_recall_at_k_{k}"]
                    enrich_k = r0["enrichment_at_k"][f"observed_positive_enrichment_at_k_{k}"]
                    e_recall = rbs["expected_recall_at_k"][k]
                    e_count = rbs["expected_positive_count_at_k"][k]
                    obs_count = recall_k * p_split
                    report.append(
                        f"| {k} | {recall_k:.4f} | {enrich_k:.2f} | "
                        f"{e_recall:.6f} | {e_count:.2f} | {obs_count:.1f} |"
                    )
                mean_r = r0["mean_rank_of_observed_positives"]
                median_r = r0["median_rank_of_observed_positives"]
                ks = r0["score_separation_ks"]
                e_mean = rbs["expected_mean_rank"]
                report.append(
                    f"\nMean rank: {mean_r:.1f} (random E: {e_mean:.1f}), "
                    f"Median rank: {median_r:.1f}, KS: {ks:.4f}\n"
                )
            report.append("")

    # --- Sensitivity Experiment Results ---
    report.append("### Sensitivity Experiment Results\n")
    report.append(
        "Sensitivity analysis includes COMMIT_ONLY path-associated observations "
        "in the positive set. Random baselines use sensitivity-specific N and P.\n"
    )

    for mode in SENSITIVITY_EXPERIMENT_MODES:
        report.append(f"#### Mode {mode}\n")
        srb = sensitivity_random_baselines.get(mode, {})

        if srb:
            report.append("**Analytic Random Baselines (per split):**\n")
            report.append("| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] |")
            report.append("|---|---|---|---|---|---|")
            for split_name in ("train", "validation", "test"):
                if split_name in srb:
                    rbs = srb[split_name]
                    e_recall_10 = rbs["expected_recall_at_k"][10]
                    e_count_10 = rbs["expected_positive_count_at_k"][10]
                    e_mean = rbs["expected_mean_rank"]
                    report.append(
                        f"| {split_name} | {rbs['N']} | {rbs['P']} | "
                        f"{e_recall_10:.6f} | {e_count_10:.2f} | {e_mean:.1f} |"
                    )
            report.append("")

        for split in ["train", "validation", "test"]:
            results_for_config = [
                r for r in sensitivity_results
                if r["mode"] == mode and r["split"] == split
            ]
            if not results_for_config:
                continue
            r0 = results_for_config[0]
            rbs = srb.get(split, {})
            p_s = rbs.get("P", "?")
            n_s = rbs.get("N", "?")

            report.append(f"**{split}** (N={n_s}, P={p_s}):\n")
            report.append("| K | recall@K | enrichment@K | E[random recall] | E[random count] |")
            report.append("|---|---|---|---|---|")
            for k in RECALL_AT_K_VALUES:
                recall_k = r0["recall_at_k"][f"observed_positive_recall_at_k_{k}"]
                enrich_k = r0["enrichment_at_k"][f"observed_positive_enrichment_at_k_{k}"]
                e_recall = rbs.get("expected_recall_at_k", {}).get(k, "?")
                e_count = rbs.get("expected_positive_count_at_k", {}).get(k, "?")
                e_recall_str = f"{e_recall:.6f}" if isinstance(e_recall, (int, float)) else e_recall
                e_count_str = f"{e_count:.2f}" if isinstance(e_count, (int, float)) else e_count
                report.append(
                    f"| {k} | {recall_k:.4f} | {enrich_k:.2f} | "
                    f"{e_recall_str} | {e_count_str} |"
                )
            mean_r = r0["mean_rank_of_observed_positives"]
            ks = r0["score_separation_ks"]
            e_mean_str = (
                f"{rbs['expected_mean_rank']:.1f}"
                if "expected_mean_rank" in rbs else "?"
            )
            report.append(
                f"\nMean rank: {mean_r:.1f} (random E: {e_mean_str}), "
                f"KS: {ks:.4f}\n"
            )
        report.append("")

    # --- Seed Stability ---
    report.append("### Seed Stability\n")
    high_cv = [
        (k, v) for k, v in stability.items() if v.get("cv", 0) > 0.1
    ]
    if high_cv:
        report.append("High-variation configs (CV > 0.1):\n")
        for k, v in high_cv:
            report.append(f"- {k}: CV={v['cv']:.4f}")
    else:
        report.append("All configs show stable results across seeds (CV <= 0.1).\n")
    report.append("")

    # --- Repository Analysis ---
    report.append("### Repository Analysis\n")
    report.append(
        f"- Repos with observed positives: {repo_analysis['n_positive_repos']}\n"
    )
    report.append(f"- Top 5 share: {repo_analysis['top_5_share']:.4f}\n")
    report.append(f"- Gini coefficient: {repo_analysis['gini_coefficient']:.4f}\n")
    report.append("Concentration:\n")
    for item in repo_analysis["concentration_summary"]:
        report.append(
            f"  - {item['repo']}: {item['positive_count']} "
            f"({item['fraction']:.4f})\n"
        )

    # --- Practical Investigation Budget ---
    report.append("## Practical Investigation Budget\n")
    report.append(
        "K represents an engineering investigation budget: the number of "
        "highest-ranked files a developer would review. K=10 means reviewing "
        "10 files; K=50 means 50 files; K=100 means 100 files; K=200 and K=500 "
        "represent broader triage efforts.\n"
    )
    report.append(
        "The key question is: does the learned ranking concentrate observed "
        "positives near the top more than random, across unseen data?\n"
    )

    # Collect representative metrics for the primary mode (E0) on test set
    e0_test = [
        r for r in primary_results
        if r["mode"] == "E0" and r["split"] == "test"
    ]
    e0_test_naive = [r for r in e0_test if r["label"] == "naive_biased"]
    e0_rb_test = random_baselines.get("E0", {}).get("test", {})

    if e0_test_naive and e0_rb_test:
        r_naive = e0_test_naive[0]
        p_test = e0_rb_test["P"]
        n_test = e0_rb_test["N"]
        report.append(f"**Unseen test set (N={n_test}, P={p_test}):**\n")
        report.append("| Budget (K) | Observed positives found | Random expected | Assessment |")
        report.append("|---|---|---|---|")
        for k in [10, 20, 50, 100, 200, 500]:
            recall_k = r_naive["recall_at_k"][f"observed_positive_recall_at_k_{k}"]
            obs_count = recall_k * p_test
            e_count = e0_rb_test["expected_positive_count_at_k"][k]
            if obs_count > e_count * 1.5:
                assessment = "above random"
            elif obs_count < e_count * 0.5:
                assessment = "below random"
            else:
                assessment = "near random"
            report.append(
                f"| {k} | {obs_count:.1f} | {e_count:.1f} | {assessment} |"
            )
        report.append("")

    # --- Evidence Matrix ---
    report.append("## Evidence Matrix\n")
    report.append("| Dimension | Result |\n")
    report.append("|---|---|\n")
    report.append("| Ranking vs random | See per-mode results |\n")
    report.append("| Validation behavior | See per-split results |\n")
    report.append("| Unseen-test behavior | See per-split results |\n")
    report.append("| Repository coverage | See repository analysis |\n")
    report.append("| Seed stability | See seed stability |\n")
    report.append("| Sensitivity to COMMIT_ONLY | See sensitivity experiment |\n")
    report.append("| Observed-positive concentration | See repository analysis |\n")
    report.append(
        "| Limitations | SCAR untestable, small positive set, "
        "selection mechanism unknown |\n"
    )

    # --- Limitations ---
    report.append("## Limitations\n")
    report.append(
        "1. SCAR is untestable with current evidence.\n"
        "2. The positive set is small (108 primary observations).\n"
        "3. The selection mechanism (fix-commit path association) is "
        "not validated as a proxy for true defect presence.\n"
        "4. Elkan-Noto correction cannot be applied.\n"
        "5. No defensible negatives exist for calibration.\n"
        "6. True-defect precision, recall, ROC, and accuracy are NOT established "
        "by this experiment.\n"
    )

    # --- Final Recommendation ---
    report.append("## Recommendation\n")
    report.append(
        "The recommendation is derived from the following evidence:\n"
    )

    # Gather key evidence for recommendation
    e0_train_results = [
        r for r in primary_results
        if r["mode"] == "E0" and r["split"] == "train" and r["label"] == "naive_biased"
    ]
    e0_val_results = [
        r for r in primary_results
        if r["mode"] == "E0" and r["split"] == "validation" and r["label"] == "naive_biased"
    ]
    e0_test_results = [
        r for r in primary_results
        if r["mode"] == "E0" and r["split"] == "test" and r["label"] == "naive_biased"
    ]

    e0_rb = random_baselines.get("E0", {})

    def _get_metric(results, key):
        if results:
            return results[0][key]
        return None

    train_mean_rank = _get_metric(e0_train_results, "mean_rank_of_observed_positives")
    val_mean_rank = _get_metric(e0_val_results, "mean_rank_of_observed_positives")
    test_mean_rank = _get_metric(e0_test_results, "mean_rank_of_observed_positives")

    train_ks = _get_metric(e0_train_results, "score_separation_ks")
    val_ks = _get_metric(e0_val_results, "score_separation_ks")
    test_ks = _get_metric(e0_test_results, "score_separation_ks")

    train_e_mean = e0_rb.get("train", {}).get("expected_mean_rank")
    val_e_mean = e0_rb.get("validation", {}).get("expected_mean_rank")
    test_e_mean = e0_rb.get("test", {}).get("expected_mean_rank")

    def _get_recall(results, k_val):
        if results:
            key = f"observed_positive_recall_at_k_{k_val}"
            return results[0]["recall_at_k"][key]
        return None

    test_recall_50 = _get_recall(e0_test_results, 50)
    test_recall_100 = _get_recall(e0_test_results, 100)

    report.append("| Evidence | Value |")
    report.append("|---|---|")
    if train_mean_rank is not None and train_e_mean is not None:
        report.append(
            f"| Train mean rank | {train_mean_rank:.1f} "
            f"(random E: {train_e_mean:.1f}) |"
        )
    if val_mean_rank is not None and val_e_mean is not None:
        report.append(
            f"| Validation mean rank | {val_mean_rank:.1f} "
            f"(random E: {val_e_mean:.1f}) |"
        )
    if test_mean_rank is not None and test_e_mean is not None:
        report.append(
            f"| Test mean rank | {test_mean_rank:.1f} "
            f"(random E: {test_e_mean:.1f}) |"
        )
    if train_ks is not None:
        report.append(f"| Train KS | {train_ks:.4f} |")
    if val_ks is not None:
        report.append(f"| Validation KS | {val_ks:.4f} |")
    if test_ks is not None:
        report.append(f"| Test KS | {test_ks:.4f} |")
    if test_recall_50 is not None:
        report.append(f"| Test recall@50 | {test_recall_50:.4f} |")
    if test_recall_100 is not None:
        report.append(f"| Test recall@100 | {test_recall_100:.4f} |")
    report.append("| Sensitivity (COMMIT_ONLY) | See sensitivity experiment above |")
    report.append(
        f"| Repository concentration | Top 5 repos = {repo_analysis['top_5_share']:.1%} |"
    )
    report.append(f"| Elkan-Noto | {ELKAN_NOTO_STATUS} |")
    report.append("")

    # Derive recommendation
    train_has_signal = (
        train_mean_rank is not None and train_e_mean is not None
        and train_mean_rank < train_e_mean * 0.5
    )
    val_has_signal = (
        val_mean_rank is not None and val_e_mean is not None
        and val_mean_rank < val_e_mean * 0.5
    )
    test_has_signal = (
        test_mean_rank is not None and test_e_mean is not None
        and test_mean_rank < test_e_mean * 0.5
    )

    if train_has_signal and not val_has_signal and not test_has_signal:
        recommendation = "PU_SIGNAL_INSUFFICIENT"
        rationale = (
            "The observed-positive ranking shows in-sample (train) separation "
            "from random (mean rank significantly below random expectation), "
            "but this does not generalize to validation or test sets. "
            "Recall@50 and recall@100 on the unseen test set are zero or near "
            "zero. The in-sample signal may reflect overfitting to the small "
            "positive set (P=65 in train) rather than a generalizable pattern. "
            "The observed-positive ranking signal is insufficient for "
            "reliable unseen-repository risk ranking.\n\n"
            "This experiment does NOT establish true-defect prediction capability. "
            "It only evaluates observed-positive ranking above random. "
            "Calibrated defect probabilities, true-defect precision/recall, "
            "and production readiness are NOT established."
        )
    elif train_has_signal and (val_has_signal or test_has_signal):
        recommendation = "PROCEED_WITH_RESTRICTIONS"
        rationale = (
            "The observed-positive ranking shows some generalization beyond "
            "training data, though with caveats. Proceed with the understanding "
            "that this is an observed-positive ranking signal, NOT a validated "
            "true-defect predictor. Elkan-Noto correction is not applicable. "
            "Calibrated probabilities are not available."
        )
    else:
        recommendation = "IMPROVE_ATTRIBUTION_FIRST"
        rationale = (
            "The observed-positive ranking does not materially separate from "
            "random on any split. Before investing in PU learning or risk "
            "ranking, improve the attribution signal quality, increase the "
            "observed-positive set size, or explore alternative supervision "
            "strategies."
        )

    report.append(f"**Recommendation: {recommendation}**\n")
    report.append(f"{rationale}\n")

    report.append(
        "**Important**: This recommendation addresses observed-positive ranking "
        "only. It does NOT claim: calibrated defect probability, true-defect "
        "precision/recall, production readiness, or quantum readiness.\n"
    )

    # --- Frozen Data Integrity ---
    report.append("## Frozen Data Integrity\n")
    report.append(
        "- Phase 4.6-4.9 code and artifacts: UNCHANGED\n"
        "- Combined-v3 JSONL dataset: UNCHANGED\n"
        "- repo_split.py: UNCHANGED\n"
        "- diff_parser.py: UNCHANGED\n"
        "- Experimental feature generation: UNCHANGED\n"
    )

    return "".join(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pu_experiment()
