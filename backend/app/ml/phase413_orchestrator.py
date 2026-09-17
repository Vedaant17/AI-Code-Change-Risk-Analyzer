"""Phase 4.13 Orchestrator: Run the full file-level investigation-priority experiment.

Runs all baselines (B0-B4), evaluates within-commit ranking, computes
bootstrap CI, and writes all artifacts.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.app.features.experimental_schemas import EXPERIMENTAL_FEATURE_NAMES
from backend.app.ml.dataset_loader import FEATURE_NAMES as BASE_FEATURE_NAMES
from backend.app.ml.phase49_supervision_feasibility import (
    _parse_frozen_manifest,
    _read_all_supervised_rows,
    _read_ambiguous_rows,
    _write_json,
)
from backend.app.ml.phase412_supervision_strategy import (
    _load_phase411b_commit_evidence,
    _load_phase411b_file_results,
)
from backend.app.ml.phase413_data import (
    DATA_DIR,
    EXP_DIR,
    OBSERVED_POSITIVE,
    OUTPUT_DIR,
    PHASE411B_DIR,
    audit_candidate_boundary,
    audit_feature_leakage,
    audit_file_strong_split_distribution,
    audit_pair_split_distribution,
    audit_split_disjointness,
    build_evidence_ranking_pairs,
    build_feature_matrix,
    build_phase413_populations,
    compute_population_metadata,
    split_pairs_by_split,
)
from backend.app.ml.phase413_evaluation import (
    aggregate_per_commit_metrics,
    investigation_budget_curves,
    paired_commit_level_bootstrap,
    per_repository_metrics,
    rank_files_within_commit,
    summarize_per_repo,
)
from backend.app.ml.phase413_models import (
    PairwiseRankingModel,
    change_size_scores,
    predict_biased_pu_scores,
    predict_commit_level_scores,
    random_within_commit_scores,
    train_biased_pu_model,
    train_commit_level_model,
)
from backend.app.ml.repo_split import build_repo_split_dataset

logger = logging.getLogger(__name__)

EXPERIMENT_MODES = ["E0", "E1", "E2", "E4"]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0,1] for presentation.

    This is NOT a probability. It is a normalized ranking score.
    Ordering must be preserved.
    """
    arr = np.array(scores, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-12:
        return [0.5] * len(scores)
    return ((arr - min_val) / (max_val - min_val)).tolist()


def _run_single_experiment(
    mode: str,
    populations: dict[str, list[dict]],
    pop_metadata: dict,
    manifest: dict[str, str],
    phase411b_files: list[dict],
    phase411b_commits: list[dict],
) -> dict:
    """Run a single feature-mode experiment."""
    logger.info("Phase 4.13: Running experiment mode=%s", mode)

    train_ds, val_ds, test_ds = build_repo_split_dataset(DATA_DIR, EXP_DIR, mode=mode)

    X_train, y_train, meta_train, feat_names = build_feature_matrix(train_ds, mode)
    X_val, y_val, meta_val, _ = build_feature_matrix(val_ds, mode)
    X_test, y_test, meta_test, _ = build_feature_matrix(test_ds, mode)

    leakage = audit_feature_leakage(feat_names)

    op_keys = set(
        (r["commit_sha"], r["file_path"])
        for r in populations[OBSERVED_POSITIVE]
    )

    pu_train_y = np.array([
        1 if (m["commit_sha"], m["file_path"]) in op_keys else 0
        for m in meta_train
    ], dtype=np.int64)

    p_indices_train = [i for i, v in enumerate(pu_train_y) if v == 1]
    u_indices_train = [i for i, v in enumerate(pu_train_y) if v == 0]
    X_P_train = X_train[p_indices_train]
    X_U_train = X_train[u_indices_train]
    meta_P_train = [meta_train[i] for i in p_indices_train]
    meta_U_train = [meta_train[i] for i in u_indices_train]

    logger.info(
        "Mode %s: train P=%d U=%d, test rows=%d",
        mode, len(X_P_train), len(X_U_train), len(X_test),
    )

    results: dict[str, dict] = {}

    # B0: Random baseline
    logger.info("Mode %s: B0 random baseline", mode)
    np.random.seed(42)
    b0_test_scores = random_within_commit_scores(meta_test, seed=42)
    b0_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(b0_test_scores, meta_test, op_keys)
    )
    b0_per_repo = per_repository_metrics(b0_test_scores, meta_test, op_keys)
    b0_curves = investigation_budget_curves(b0_test_scores, meta_test, op_keys)
    results["B0_random"] = {
        "agg": b0_agg,
        "per_repo": b0_per_repo,
        "curves": b0_curves,
    }

    # B1: Change-size heuristic
    logger.info("Mode %s: B1 change-size heuristic", mode)
    b1_scores = change_size_scores(meta_test, X_test, feat_names)
    b1_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(b1_scores, meta_test, op_keys)
    )
    b1_per_repo = per_repository_metrics(b1_scores, meta_test, op_keys)
    b1_curves = investigation_budget_curves(b1_scores, meta_test, op_keys)
    results["B1_change_size"] = {
        "agg": b1_agg,
        "per_repo": b1_per_repo,
        "curves": b1_curves,
    }

    # B2: Biased P-vs-U XGBoost
    logger.info("Mode %s: B2 biased P-vs-U", mode)
    b2_model, b2_train_scores, b2_train_meta = train_biased_pu_model(
        X_P_train, X_U_train, meta_P_train, meta_U_train, feat_names,
    )
    b2_test_scores = predict_biased_pu_scores(b2_model, X_test)
    b2_normalized = _normalize_scores(b2_test_scores)
    b2_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(b2_normalized, meta_test, op_keys)
    )
    b2_per_repo = per_repository_metrics(b2_normalized, meta_test, op_keys)
    b2_curves = investigation_budget_curves(b2_normalized, meta_test, op_keys)
    b2_bootstrap = paired_commit_level_bootstrap(
        b2_normalized, b0_test_scores, meta_test, op_keys,
    )
    results["B2_biased_pu"] = {
        "agg": b2_agg,
        "per_repo": b2_per_repo,
        "curves": b2_curves,
        "bootstrap": b2_bootstrap,
    }

    # B3: Evidence-ranking pairwise
    logger.info("Mode %s: B3 evidence-ranking", mode)
    pairs = build_evidence_ranking_pairs(phase411b_files, phase411b_commits, manifest)
    pair_split = split_pairs_by_split(pairs, manifest)
    train_pairs = pair_split["train"]
    test_pairs = pair_split["test"]

    pairwise_result = {"status": "COMPUTED", "train_pairs": len(train_pairs),
                       "test_pairs": len(test_pairs)}

    if len(train_pairs) >= 2:
        feat_idx = {}
        for i, name in enumerate(feat_names):
            feat_idx[name] = i

        def _extract_features(pairs_list: list[dict], X_all: np.ndarray,
                              meta_all: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            meta_lookup = {(m["repo_name"], m["commit_sha"], m["file_path"]): i
                           for i, m in enumerate(meta_all)}
            X_s, X_w = [], []
            for p in pairs_list:
                s_key = (p["repo_name"], p["commit_sha"], p["strong_file"])
                w_key = (p["repo_name"], p["commit_sha"], p["weak_file"])
                s_idx = meta_lookup.get(s_key)
                w_idx = meta_lookup.get(w_key)
                if s_idx is not None and w_idx is not None:
                    X_s.append(X_all[s_idx])
                    X_w.append(X_all[w_idx])
            if not X_s:
                return np.empty((0, X_all.shape[1])), np.empty((0, X_all.shape[1]))
            return np.array(X_s), np.array(X_w)

        X_s_train, X_w_train = _extract_features(train_pairs, X_train, meta_train)
        if len(X_s_train) > 0:
            pairwise_model = PairwiseRankingModel(random_state=42)
            pairwise_model.fit(X_s_train, X_w_train)

            if len(test_pairs) > 0:
                X_s_test, X_w_test = _extract_features(test_pairs, X_test, meta_test)
                if len(X_s_test) > 0:
                    pw_acc = pairwise_model.predict_pairwise_accuracy(X_s_test, X_w_test)
                    pairwise_result["test_pairwise_accuracy"] = pw_acc
                    pairwise_result["test_evaluated_pairs"] = len(X_s_test)
                else:
                    pairwise_result["status"] = "PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT"
            else:
                pairwise_result["status"] = "PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT"

            X_all_scored = pairwise_model.score(X_test)
            b3_scores = _normalize_scores(X_all_scored.tolist())
            b3_agg = aggregate_per_commit_metrics(
                rank_files_within_commit(b3_scores, meta_test, op_keys)
            )
            b3_per_repo = per_repository_metrics(b3_scores, meta_test, op_keys)
            b3_curves = investigation_budget_curves(b3_scores, meta_test, op_keys)
            b3_bootstrap = paired_commit_level_bootstrap(
                b3_scores, b0_test_scores, meta_test, op_keys,
            )
            results["B3_evidence_ranking"] = {
                "agg": b3_agg,
                "per_repo": b3_per_repo,
                "curves": b3_curves,
                "bootstrap": b3_bootstrap,
                "pairwise": pairwise_result,
            }
        else:
            pairwise_result["status"] = "NO_TRAINING_PAIRS"
            results["B3_evidence_ranking"] = {"pairwise": pairwise_result}
    else:
        pairwise_result["status"] = "NO_TRAINING_PAIRS"
        results["B3_evidence_ranking"] = {"pairwise": pairwise_result}

    # B4: Commit-level fallback
    logger.info("Mode %s: B4 commit-level fallback", mode)
    meta_train_with_op = []
    for m in meta_train:
        m_copy = dict(m)
        m_copy["observed_positive"] = (m["commit_sha"], m["file_path"]) in op_keys
        meta_train_with_op.append(m_copy)

    b4_model, b4_commit_scores, b4_commit_meta = train_commit_level_model(
        X_train, meta_train_with_op,
    )
    b4_test_scores = predict_commit_level_scores(b4_model, X_test, meta_test)
    b4_agg = aggregate_per_commit_metrics(
        rank_files_within_commit(b4_test_scores, meta_test, op_keys)
    )
    b4_per_repo = per_repository_metrics(b4_test_scores, meta_test, op_keys)
    b4_curves = investigation_budget_curves(b4_test_scores, meta_test, op_keys)
    b4_bootstrap = paired_commit_level_bootstrap(
        b4_test_scores, b0_test_scores, meta_test, op_keys,
    )
    results["B4_commit_level"] = {
        "agg": b4_agg,
        "per_repo": b4_per_repo,
        "curves": b4_curves,
        "bootstrap": b4_bootstrap,
    }

    # Global secondary
    from backend.app.ml.phase413_evaluation import global_secondary_metrics as _gsm
    _b2_global = _gsm(b2_normalized, meta_test, op_keys)

    # Prediction output
    predictions = []
    for i, m in enumerate(meta_test):
        predictions.append({
            "repo_name": m["repo_name"],
            "commit_sha": m["commit_sha"],
            "file_path": m["file_path"],
            "risk_score": b2_normalized[i],
            "rank_in_commit": 0,
            "commit_total_files": 0,
            "model_version": "4.13.0",
            "feature_version": "v1",
            "supervision_method": "PU_OBSERVED_POSITIVE",
            "fallback_status": "NONE",
        })

    by_commit_pred: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, p in enumerate(predictions):
        by_commit_pred[(p["repo_name"], p["commit_sha"])].append(i)
    for c_key, indices in by_commit_pred.items():
        n_files = len(indices)
        scored = sorted(indices, key=lambda i: -predictions[i]["risk_score"])
        for rank_pos, idx in enumerate(scored):
            predictions[idx]["rank_in_commit"] = rank_pos + 1
            predictions[idx]["commit_total_files"] = n_files

    return {
        "mode": mode,
        "results": results,
        "predictions": predictions,
        "leakage_audit": leakage,
        "feature_names": feat_names,
    }


def run_phase413(
    data_dir: Path = DATA_DIR,
    exp_dir: Path = EXP_DIR,
    phase411b_dir: Path = PHASE411B_DIR,
    output_dir: Path = OUTPUT_DIR,
    modes: list[str] | None = None,
) -> dict:
    """Run the full Phase 4.13 experiment."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    if modes is None:
        modes = EXPERIMENT_MODES

    supervised_rows = _read_all_supervised_rows(data_dir)
    _ = _read_ambiguous_rows(data_dir)
    manifest = _parse_frozen_manifest()
    phase411b_files = _load_phase411b_file_results(phase411b_dir)
    phase411b_commits = _load_phase411b_commit_evidence(phase411b_dir)

    logger.info("Phase 4.13: Constructing populations...")
    populations = build_phase413_populations(phase411b_files, supervised_rows)
    pop_metadata = compute_population_metadata(populations)

    split_audit = audit_file_strong_split_distribution(populations, manifest)
    pairs = build_evidence_ranking_pairs(phase411b_files, phase411b_commits, manifest)
    pair_audit = audit_pair_split_distribution(pairs, manifest)

    all_feature_names = list(BASE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES)
    feature_leakage = audit_feature_leakage(all_feature_names)
    candidate_boundary = audit_candidate_boundary()
    split_disjoint = audit_split_disjointness(manifest)

    config = {
        "phase": "4.13",
        "description": "File-level investigation-priority ranking",
        "modes": modes,
        "data_dir": str(data_dir),
        "exp_dir": str(exp_dir),
        "phase411b_dir": str(phase411b_dir),
        "output_dir": str(output_dir),
        "supervision": {
            "primary": "P = OBSERVED_POSITIVE, U = UNLABELED",
            "sensitivity": "P = OBSERVED_POSITIVE, U = UNLABELED + WEAK + UNKNOWN",
            "excluded": "OUT_OF_SCOPE",
        },
        "prediction_unit": "(repo_name, commit_sha, file_path)",
        "score_semantics": "investigation_priority_score (NOT a probability)",
    }

    _write_json(output_dir / "phase413_config.json", config)
    _write_json(output_dir / "phase413_split_audit.json", split_audit)
    _write_json(output_dir / "phase413_pair_audit.json", pair_audit)
    _write_json(output_dir / "phase413_supervision_audit.json", pop_metadata)
    _write_json(output_dir / "phase413_feature_audit.json", {
        "leakage_audit": feature_leakage,
        "candidate_boundary": candidate_boundary,
        "split_disjointness": split_disjoint,
    })

    mode_results: dict[str, dict] = {}
    all_predictions: list[dict] = []

    for mode in modes:
        mode_result = _run_single_experiment(
            mode, populations, pop_metadata, manifest,
            phase411b_files, phase411b_commits,
        )
        mode_results[mode] = mode_result
        all_predictions.extend(mode_result["predictions"])

    results_summary: dict[str, dict] = {}
    for mode, mr in mode_results.items():
        mode_summary: dict[str, dict] = {}
        for model_name, model_result in mr["results"].items():
            if "agg" in model_result:
                agg = model_result["agg"]
                mode_summary[model_name] = {
                    "n_commits_with_positives": agg.n_commits_with_positives,
                    "n_commits_total": agg.n_commits_total,
                    "coverage": agg.coverage,
                    "mean_mrr": agg.mean_reciprocal_rank,
                    "mean_recall_at_k": {str(k): v for k, v in agg.mean_recall_at_k.items()},
                    "mean_enrichment_at_k": {
                        str(k): v for k, v in agg.mean_enrichment_at_k.items()
                    },
                }
            if "bootstrap" in model_result:
                mode_summary[model_name]["bootstrap"] = model_result["bootstrap"]
            if "pairwise" in model_result:
                mode_summary[model_name]["pairwise"] = model_result["pairwise"]
        results_summary[mode] = mode_summary

    _write_json(output_dir / "phase413_results.json", results_summary)

    eval_report: dict[str, dict] = {}
    for mode, mr in mode_results.items():
        eval_report[mode] = {}
        for model_name, model_result in mr["results"].items():
            if "per_repo" in model_result:
                eval_report[mode][model_name] = {
                    "per_repo": model_result["per_repo"],
                    "summary": summarize_per_repo(model_result["per_repo"]),
                }
    _write_json(output_dir / "phase413_evaluation_report.json", eval_report)

    budget_curves: dict[str, dict] = {}
    for mode, mr in mode_results.items():
        budget_curves[mode] = {}
        for model_name, model_result in mr["results"].items():
            if "curves" in model_result:
                budget_curves[mode][model_name] = model_result["curves"]
    _write_json(output_dir / "phase413_investigation_budget.json", budget_curves)

    pw_results: dict[str, dict] = {}
    for mode, mr in mode_results.items():
        if "B3_evidence_ranking" in mr["results"]:
            pw_results[mode] = mr["results"]["B3_evidence_ranking"].get("pairwise", {})
    _write_json(output_dir / "phase413_pairwise_ranking.json", pw_results)

    with open(output_dir / "phase413_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    model_meta = {
        "model_version": "4.13.0",
        "feature_version": "v1",
        "modes_evaluated": modes,
        "total_predictions": len(all_predictions),
        "elapsed_seconds": time.time() - start_time,
    }
    _write_json(output_dir / "phase413_model_metadata.json", model_meta)

    random_baselines: dict[str, dict] = {}
    for mode, mr in mode_results.items():
        if "B0_random" in mr["results"]:
            agg = mr["results"]["B0_random"]["agg"]
            random_baselines[mode] = {
                "mean_enrichment_at_k": {str(k): v for k, v in agg.mean_enrichment_at_k.items()},
                "mean_recall_at_k": {str(k): v for k, v in agg.mean_recall_at_k.items()},
            }
    _write_json(output_dir / "phase413_random_baselines.json", random_baselines)

    _generate_report(output_dir, results_summary, split_audit, pair_audit,
                     pop_metadata, config, eval_report, budget_curves, pw_results)

    elapsed = time.time() - start_time
    logger.info("Phase 4.13: Complete in %.1f seconds", elapsed)

    return {
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "modes": modes,
        "results_summary": results_summary,
        "pair_audit": pair_audit,
    }


def _generate_report(
    output_dir: Path,
    results_summary: dict,
    split_audit: dict,
    pair_audit: dict,
    pop_metadata: dict,
    config: dict,
    eval_report: dict,
    budget_curves: dict,
    pw_results: dict,
) -> None:
    """Generate the scientific report."""
    lines = [
        "# Phase 4.13: File-Level Investigation-Priority Ranking",
        "",
        "## 1. Research Question",
        "",
        "Given 499 files with direct Git content evidence of defect association",
        "(FILE_STRONG), 238 files with weak evidence (FILE_WEAK), 314 files with",
        "unresolved identity (UNKNOWN), and zero defensible negatives, can we build",
        "a file-level ranking model that improves upon random investigation ordering",
        "when triaging new commits?",
        "",
        "## 2. Hypothesis",
        "",
        "Files in a new commit that share structural/statistical properties with",
        "the 499 FILE_STRONG files will be ranked higher by the model than files",
        "that do not, and this ranking will generalize to unseen repositories",
        "better than random ordering.",
        "",
        "## 3. Supervision Populations",
        "",
        "| Population | Count |",
        "|-----------|-------|",
        f"| OBSERVED_POSITIVE | {pop_metadata['observed_positive_rows']} |",
        f"| EVIDENCE_WEAK | {pop_metadata['evidence_weak_rows']} |",
        f"| EVIDENCE_UNKNOWN | {pop_metadata['evidence_unknown_rows']} |",
        f"| UNLABELED | {pop_metadata['unlabeled_rows']} |",
        f"| OUT_OF_SCOPE | {pop_metadata['out_of_scope_rows']} |",
        f"| TOTAL | {pop_metadata['total_supervised_rows']} |",
        "",
        "## 4. Pre-Experiment Audits",
        "",
        "### 4.1 FILE_STRONG Split Distribution",
        "",
        "| Split | Rows | Commits | Repos |",
        "|-------|------|---------|-------|",
    ]

    for split in ["train", "validation", "test"]:
        s = split_audit.get(split, {})
        lines.append(
            f"| {split} | {s.get('file_strong_rows', 0)} | "
            f"{s.get('unique_commits', 0)} | {s.get('unique_repos', 0)} |"
        )
    t = split_audit.get("total", {})
    lines.append(
        f"| total | {t.get('file_strong_rows', 0)} | "
        f"{t.get('unique_commits', 0)} | {t.get('unique_repos', 0)} |"
    )

    lines.extend([
        "",
        "### 4.2 STRONG_vs_WEAK Pair Audit",
        "",
        "| Split | Pairs | Commits | Repos |",
        "|-------|-------|---------|-------|",
    ])
    for split in ["train", "validation", "test"]:
        pa = pair_audit.get(split, {})
        lines.append(
            f"| {split} | {pa.get('pair_count', 0)} | "
            f"{pa.get('unique_commits', 0)} | {pa.get('unique_repos', 0)} |"
        )

    test_pairs = pair_audit.get("test", {}).get("pair_count", 0)
    if test_pairs == 0:
        lines.extend([
            "",
            "**PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT**",
        ])

    lines.extend([
        "",
        "## 5. Results",
        "",
        "### B1 Behavior",
        "",
        "B1 ranks files by `file_total_lines_changed` (index 34) descending.",
        "This is a simple file-level change-size heuristic.",
        "",
        "### B4 Behavior",
        "",
        "B4 assigns the same commit-level score to all files in a commit. Its",
        "within-commit file ordering is determined entirely by the deterministic",
        "tie-breaking policy. B4 is a commit-level fallback and does not provide",
        "file-level discrimination within a commit.",
        "",
    ])

    for mode, mode_results in results_summary.items():
        lines.append(f"### Mode {mode}")
        lines.append("")
        for model_name, mr in mode_results.items():
            lines.append(f"#### {model_name}")
            if "mean_mrr" in mr:
                lines.append(f"- MRR: {mr['mean_mrr']:.4f}")
                lines.append(f"- Commits with positives: {mr['n_commits_with_positives']}")
                lines.append(f"- Coverage: {mr['coverage']:.4f}")
            if "pairwise" in mr:
                pw = mr["pairwise"]
                if "test_pairwise_accuracy" in pw:
                    lines.append(f"- Pairwise accuracy: {pw['test_pairwise_accuracy']:.4f}")
                elif pw.get("status") == "PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT":
                    lines.append("- Pairwise: PAIRWISE_EVALUATION_UNAVAILABLE_ON_TEST_SPLIT")
            if "bootstrap" in mr:
                for metric, bdata in mr["bootstrap"].items():
                    if isinstance(bdata, dict) and "diff" in bdata:
                        lines.append(
                            f"- {metric}: diff={bdata['diff']:.4f}, "
                            f"CI=[{bdata['ci_lower']:.4f}, {bdata['ci_upper']:.4f}]"
                        )
            lines.append("")

    # --- Bootstrap caveat: compute from results_summary ---
    sig_comparisons = []
    for mode, mode_results in results_summary.items():
        for model_name, mr in mode_results.items():
            bs = mr.get("bootstrap", {})
            for metric, bdata in bs.items():
                if isinstance(bdata, dict) and "ci_lower" in bdata:
                    if bdata["ci_lower"] > 0 or bdata["ci_upper"] < 0:
                        sig_comparisons.append((
                            mode, model_name, metric,
                            bdata["diff"], bdata["ci_lower"], bdata["ci_upper"],
                        ))
    n_sig = len(sig_comparisons)
    n_total = 0
    for mode, mode_results in results_summary.items():
        for model_name, mr in mode_results.items():
            bs = mr.get("bootstrap", {})
            for metric, bdata in bs.items():
                if isinstance(bdata, dict) and "diff" in bdata:
                    n_total += 1

    lines.append("")
    lines.append("### Bootstrap Summary")
    lines.append("")
    lines.append(
        f"Of {n_total} model-vs-random bootstrap comparisons, "
        f"{n_sig} exclude zero:"
    )
    lines.append("")
    for mode, model, metric, diff, lo, hi in sig_comparisons:
        lines.append(
            f"- {mode} {model} {metric}: diff={diff:.4f}, "
            f"CI=[{lo:.4f}, {hi:.4f}]"
        )
    lines.append("")
    lines.append(
        "E2 and E4 produce identical bootstrap results for enrichment@2 "
        "(same underlying test set with shared historical features). "
        "These findings should be treated as **exploratory** (no multiple-"
        "comparison correction). The positive enrichment@2 aggregate is "
        "concentrated in 3 repos: click (+0.650, 5 positive commits), "
        "nox (+0.538, 1 positive commit), and celery (+0.063, 12 positive "
        "commits), which together account for 18/45 positive test commits. "
        "9 of 12 repos with positive commits show zero diff. The result "
        "may not generalize."
    )

    lines.extend([
        "",
        "## 6. Limitations",
        "",
        "1. No defensible negatives exist. The model cannot estimate true defect probability.",
        "2. FILE_STRONG represents labeling pipeline coverage, not confirmed defectives.",
        "3. The dataset only includes commits flagged as potentially bug-fixing.",
        "4. The model ranks files within commits, not across commits.",
        "5. Temporal censoring: observation endpoint is the latest commit per repo.",
        "6. Repository selection bias: 50 non-random repositories.",
        "7. AST features are Python-only.",
        "8. PU class prior is not identifiable (Phase 4.10).",
        "9. SCAR is untestable.",
        "10. Score is not calibrated to any external frequency.",
        "",
        "## 7. Final Scientific Claim",
        "",
        "The model can attempt to rank files according to their similarity to the",
        "observed FILE_STRONG evidence population using pre-candidate features.",
        "It cannot establish true defect probability, defect prevalence, defect-free",
        "probability, or whether a particular file is actually defective.",
        "",
        "The correct framing is investigation prioritization, not defect prediction.",
        "",
        "## 8. SageMaker / Braket Status",
        "",
        "SageMaker: BLOCKED until classical baseline demonstrates useful signal.",
        "Braket: BLOCKED until a specific quantum formulation is justified.",
        "",
        "## 9. Final Status",
        "",
        "NO_DEFENSIBLE_MODEL_SELECTED. No model demonstrates sufficient",
        "signal to justify deployment.",
        "",
        "B1 = simple file-level change-size heuristic (file_total_lines_changed).",
        "B2 = biased P-vs-U XGBoost model.",
        "B3 = evidence-ranking pairwise logistic model.",
        "B4 = commit-level fallback with no genuine within-commit file discrimination.",
        "",
        "B3 achieves the highest MRR in modes E0/E1 (0.9778 vs B1 0.9519),",
        "but B3 E2/E4 MRR (0.9296) falls below B1. The overall comparison",
        "depends on the feature mode. 4 of 27 model-vs-random bootstrap",
        "comparisons exclude zero (see Bootstrap Summary). These findings",
        "should be treated as exploratory (no multiple-comparison correction).",
        "Phase 4.13 is complete but no model is recommended for deployment.",
    ])

    with open(output_dir / "phase413_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase413()
