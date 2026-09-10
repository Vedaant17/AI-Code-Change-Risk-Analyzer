"""Phase 4 real-data training on the combined multi-repository dataset (v3).

Uses the dataset from Phase 3.7 (backend/data/datasets/combined-v3/).
Evaluates MajorityClassBaseline, WeightedLogisticRegression, XGBoostDefectModel.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

import numpy as np  # noqa: E402

from backend.app.ml.artifacts import save_artifacts  # noqa: E402
from backend.app.ml.dataset_loader import (  # noqa: E402
    FEATURE_NAMES,
    load_dataset,
)
from backend.app.ml.importance import extract_importance, format_importance_report  # noqa: E402
from backend.app.ml.metrics import compute_baseline, compute_metrics  # noqa: E402
from backend.app.ml.models import (  # noqa: E402
    MajorityClassBaseline,
    WeightedLogisticRegression,
    XGBoostDefectModel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "backend" / "data" / "datasets" / "combined-v3"
MODELS_DIR = project_root / "backend" / "data" / "models"
MODEL_VERSION = "v0.1.0-combined-v3-real"

# Change-size proxy feature indices (within the 45-feature vector)
CHANGE_SIZE_FEATURES = [
    "files_modified", "max_file_changes", "avg_file_changes",
    "total_lines_changed", "lines_added", "lines_deleted",
    "total_hunks", "avg_hunks_per_file",
]

CHANGE_SIZE_INDICES = [FEATURE_NAMES.index(f) for f in CHANGE_SIZE_FEATURES if f in FEATURE_NAMES]


def _ranking_analysis(
    prob_arr: list[float], ds, label: str,
) -> dict:
    """Compute top-k and top-pct ranking stats for positive examples."""
    total = len(ds)
    pos_indices = [i for i in range(len(ds)) if ds.y[i] == 1]
    if not pos_indices:
        print(f"  No positives in {label}")
        return {}

    indexed = list(enumerate(prob_arr))
    ranked = sorted(indexed, key=lambda x: x[1], reverse=True)
    pos_set = set(pos_indices)

    result = {}
    for cutoff_name, cutoff in [
        ("top_10", 10), ("top_20", 20), ("top_50", 50), ("top_100", 100),
        ("top_10pct", max(1, total // 10)), ("top_20pct", max(1, total // 5)),
    ]:
        n = min(cutoff, total)
        top_n = [idx for idx, _ in ranked[:n]]
        count = len(pos_set & set(top_n))
        pct = 100.0 * count / len(pos_indices)
        result[cutoff_name] = {"count": count, "total": len(pos_indices), "pct": pct}
        print(f"    {cutoff_name:>10}: {count}/{len(pos_indices)} positives ({pct:.1f}%)")

    return result


def _precision_recall_at_k(
    prob_arr: list[float], ds, k: int,
) -> tuple[float, float]:
    """Compute Precision@K and Recall@K."""
    total = len(ds)
    pos_count = sum(1 for y in ds.y if y == 1)
    if pos_count == 0 or k == 0:
        return 0.0, 0.0

    indexed = list(enumerate(prob_arr))
    ranked = sorted(indexed, key=lambda x: x[1], reverse=True)
    n = min(k, total)
    top_indices = [idx for idx, _ in ranked[:n]]
    tp_at_k = sum(1 for i in top_indices if ds.y[i] == 1)

    precision_at_k = tp_at_k / n if n > 0 else 0.0
    recall_at_k = tp_at_k / pos_count
    return precision_at_k, recall_at_k


def _change_size_analysis(fitted_models: dict, train_ds, test_ds) -> None:
    """Analyze whether the model relies disproportionately on change-size features."""
    print(f"\n{'=' * 70}")
    print("CHANGE-SIZE FEATURE ANALYSIS")
    print("=" * 70)

    xgb_model = fitted_models.get("xgboost")
    if xgb_model is None:
        print("  No XGBoost model available.")
        return

    importances = extract_importance(xgb_model, FEATURE_NAMES)
    imp_map = {fi.feature_name: fi.importance for fi in importances}

    change_size_imp = sum(imp_map.get(f, 0.0) for f in CHANGE_SIZE_FEATURES)
    total_imp = sum(fi.importance for fi in importances) or 1.0

    print(f"  Change-size features: {CHANGE_SIZE_FEATURES}")
    print(f"  Change-size total importance: {change_size_imp:.4f} / {total_imp:.4f} "
          f"({100.0 * change_size_imp / total_imp:.1f}%)")
    print()

    for feat in CHANGE_SIZE_FEATURES:
        imp = imp_map.get(feat, 0.0)
        rank = next((fi.rank for fi in importances if fi.feature_name == feat), "N/A")
        print(f"    {feat:<35} importance={imp:.4f}  rank={rank}")

    # Check if top-5 are dominated by change-size
    top5_names = [fi.feature_name for fi in importances[:5]]
    cs_in_top5 = sum(1 for f in top5_names if f in CHANGE_SIZE_FEATURES)
    print(f"\n  Change-size features in top 5: {cs_in_top5}/5")
    print(f"  Top 5 features: {top5_names}")

    # Correlation: change-size features vs labels
    print("\n  Mean feature values by class (test set):")
    for feat in CHANGE_SIZE_FEATURES[:5]:
        idx = FEATURE_NAMES.index(feat)
        mean_pos = test_ds.X[test_ds.y == 1, idx].mean() if sum(test_ds.y == 1) > 0 else 0
        mean_neg = test_ds.X[test_ds.y == 0, idx].mean() if sum(test_ds.y == 0) > 0 else 0
        ratio = mean_pos / mean_neg if mean_neg > 0 else float("inf")
        print(f"    {feat:<35} pos={mean_pos:.4f}  neg={mean_neg:.4f}  ratio={ratio:.2f}")


def main() -> None:
    """Run Phase 4 training on the combined multi-repo v3 dataset."""
    # -- Load combined dataset --------------------------------------------
    logger.info("Loading combined-v3 dataset from %s", DATA_DIR)
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    print("\n" + "=" * 70)
    print("PHASE 4: Real ML Training -- Combined Multi-Repository Dataset (v3)")
    print("=" * 70)

    # -- Dataset Summary --------------------------------------------------
    tp = train_ds.positive_count
    tn = train_ds.negative_count
    vp = val_ds.positive_count
    vn = val_ds.negative_count
    tsp = test_ds.positive_count
    tsn = test_ds.negative_count
    total = len(train_ds) + len(val_ds) + len(test_ds)

    def _pos_shas(ds):
        return set(ds.metadata[i]["commit_sha"] for i in range(len(ds)) if ds.y[i] == 1)

    def _pos_repos(ds):
        return set(ds.metadata[i]["repo_name"] for i in range(len(ds)) if ds.y[i] == 1)

    train_pos_shas = _pos_shas(train_ds)
    val_pos_shas = _pos_shas(val_ds)
    test_pos_shas = _pos_shas(test_ds)
    all_pos_shas = train_pos_shas | val_pos_shas | test_pos_shas
    all_pos_repos = _pos_repos(train_ds) | _pos_repos(val_ds) | _pos_repos(test_ds)

    print("\nDataset Summary:")
    print(f"  TRAIN:      {len(train_ds):>5} files  (pos={tp}, neg={tn})  "
          f"{len(train_pos_shas)} pos commits  repos={len(_pos_repos(train_ds))}")
    print(f"  VALIDATION: {len(val_ds):>5} files  (pos={vp}, neg={vn})  "
          f"{len(val_pos_shas)} pos commits  repos={len(_pos_repos(val_ds))}")
    print(f"  TEST:       {len(test_ds):>5} files  (pos={tsp}, neg={tsn})  "
          f"{len(test_pos_shas)} pos commits  repos={len(_pos_repos(test_ds))}")
    print(f"  TOTAL:      {total:>5} files  pos={tp+vp+tsp}  "
          f"distinct commits={len(all_pos_shas)}  repos={sorted(all_pos_repos)}")

    # Positive prevalence per split
    train_prev = tp / len(train_ds) if len(train_ds) > 0 else 0.0
    val_prev = vp / len(val_ds) if len(val_ds) > 0 else 0.0
    test_prev = tsp / len(test_ds) if len(test_ds) > 0 else 0.0
    overall_prev = (tp + vp + tsp) / total if total > 0 else 0.0
    print(f"\n  Positive prevalence:  train={train_prev:.4f}  val={val_prev:.4f}  "
          f"test={test_prev:.4f}  overall={overall_prev:.4f}")
    print(f"  Random PR-AUC baseline ~ prevalence: {overall_prev:.4f}")

    # -- Feature Contract Verification ------------------------------------
    print("\n--- Feature Contract Verification ---")
    commit_features = [f for f in FEATURE_NAMES if not f.startswith("file_")]
    file_features = [f for f in FEATURE_NAMES if f.startswith("file_")]
    print(f"  Commit features: {len(commit_features)}")
    print(f"  File features:   {len(file_features)} (prefixed with 'file_')")
    print(f"  Total:           {len(FEATURE_NAMES)}")
    assert train_ds.X.shape[1] == 45, f"Expected 45 features, got {train_ds.X.shape[1]}"
    assert val_ds.X.shape[1] == 45
    assert test_ds.X.shape[1] == 45
    print("  [OK] All splits have 45 features")

    # -- Leakage Verification ---------------------------------------------
    print("\n--- Leakage Verification ---")
    train_shas = set(m["commit_sha"] for m in train_ds.metadata)
    val_shas = set(m["commit_sha"] for m in val_ds.metadata)
    test_shas = set(m["commit_sha"] for m in test_ds.metadata)
    overlap_tv = train_shas & val_shas
    overlap_tt = train_shas & test_shas
    overlap_vt = val_shas & test_shas
    print(f"  Train-Val commit overlap:   {len(overlap_tv)}")
    print(f"  Train-Test commit overlap:  {len(overlap_tt)}")
    print(f"  Val-Test commit overlap:    {len(overlap_vt)}")
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("  [OK] No commit leakage across splits")
    else:
        print("  [FAIL] Commit leakage detected!")
        pairs = [("Train-Val", overlap_tv), ("Train-Test", overlap_tt),
                 ("Val-Test", overlap_vt)]
        for lbl, ovl in pairs:
            if ovl:
                print(f"    {lbl}: {ovl}")

    print("  Checking for ambiguous rows in supervised splits...")
    all_clean = True
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for i, y in enumerate(ds.y):
            if y == -1:
                print(f"  [FAIL] Ambiguous row in {split_name}: {ds.metadata[i]}")
                all_clean = False
    if all_clean:
        print("  [OK] No ambiguous rows in supervised splits")

    # -- Baselines --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("BASELINES")
    print("=" * 70)
    baseline = compute_baseline(train_ds.y)
    print("Majority-class baseline (always predicts class 0):")
    print(f"  accuracy={baseline.accuracy:.4f}  majority_class={baseline.majority_class}")
    print(f"  Random PR-AUC ~ positive prevalence = {overall_prev:.4f}")

    # -- Models -----------------------------------------------------------
    models = [
        ("majority_class", MajorityClassBaseline()),
        ("weighted_logistic_regression", WeightedLogisticRegression()),
        ("xgboost", XGBoostDefectModel(n_estimators=200, max_depth=6, learning_rate=0.1)),
    ]

    all_results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    split_probs: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for name, model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {name}")
        print("=" * 70)

        if train_ds.positive_count == 0 and name != "majority_class":
            print("  Skipping: training set has 0 positive examples.")
            all_results[name] = {"skipped": True, "reason": "0 training positives"}
            continue

        model.fit(train_ds.X, train_ds.y, X_val=val_ds.X, y_val=val_ds.y)
        fitted_models[name] = model

        for split_name, ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
            y_prob = model.predict_proba(ds.X)
            metrics = compute_metrics(ds.y, y_prob)
            prevalence = metrics.actual_positives / metrics.total if metrics.total > 0 else 0.0

            print(f"\n--- {split_name.upper()} ---")
            print(f"  Examples:          {metrics.total}")
            print(f"  Actual positives:  {metrics.actual_positives}")
            print(f"  Actual negatives:  {metrics.actual_negatives}")
            print(f"  Positive predictions: {metrics.positive_predictions}")
            print(f"  Positive prevalence: {prevalence:.4f}")

            def _fmt(v):
                return f"{v:.4f}" if v != 0 or metrics.actual_positives > 0 else "undefined"

            print(f"  ROC-AUC:           {_fmt(metrics.roc_auc)}")
            print(f"  PR-AUC (AP):       {_fmt(metrics.pr_auc)}")
            print(f"  Precision:         {_fmt(metrics.precision)}")
            print(f"  Recall:            {_fmt(metrics.recall)}")
            print(f"  F1:                {_fmt(metrics.f1)}")
            print(f"  Confusion matrix:  {metrics.confusion_matrix}")

            if prevalence > 0 and metrics.pr_auc > 0:
                lift = metrics.pr_auc / prevalence
                print(f"  PR-AUC lift:       {lift:.2f}x (vs random baseline {prevalence:.4f})")

            all_results[f"{name}_{split_name}"] = metrics.to_dict()
            all_results[f"{name}_{split_name}"]["positive_prevalence"] = prevalence

            # Store probabilities for ranking analysis
            if name not in split_probs:
                split_probs[name] = {}
            split_probs[name][split_name] = [(i, y_prob[i]) for i in range(len(ds))]

    # -- Ranking Analysis (Test Set) -------------------------------------
    print(f"\n{'=' * 70}")
    print("RANKING ANALYSIS (Test Set)")
    print("=" * 70)

    for model_name in ["xgboost", "weighted_logistic_regression"]:
        if model_name not in split_probs:
            continue
        probs = split_probs[model_name]["test"]
        prob_arr = [p for _, p in probs]
        print(f"\n--- {model_name.upper()} ---")

        pos_count = sum(1 for y in test_ds.y if y == 1)
        print(f"  Positives in test set: {pos_count}")

        print("\n  Top-K ranking:")
        result = _ranking_analysis(prob_arr, test_ds, "test")

        print("\n  Precision@K / Recall@K:")
        for k in [10, 20, 50, 100]:
            p_k, r_k = _precision_recall_at_k(prob_arr, test_ds, k)
            print(f"    P@{k:<3} = {p_k:.4f}    R@{k:<3} = {r_k:.4f}")

        # Store ranking in results
        all_results[f"{model_name}_test_ranking"] = result

    # -- Feature Importance -----------------------------------------------
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    xgb_model = fitted_models.get("xgboost")
    if xgb_model is not None:
        importances = extract_importance(xgb_model, FEATURE_NAMES)
        print("\nXGBoost Feature Importance (top 15):")
        print(format_importance_report(importances, top_n=15))

    lr_model = fitted_models.get("weighted_logistic_regression")
    if lr_model is not None:
        lr_importances = extract_importance(lr_model, FEATURE_NAMES)
        print("\nLogistic Regression Coefficients (top 15):")
        print(format_importance_report(lr_importances, top_n=15))

    # -- Change-Size Feature Analysis -------------------------------------
    _change_size_analysis(fitted_models, train_ds, test_ds)

    # -- Error Analysis --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("ERROR ANALYSIS")
    print("=" * 70)

    for model_name in ["xgboost", "weighted_logistic_regression"]:
        if model_name not in fitted_models:
            continue
        print(f"\n--- {model_name.upper()} ---")

        for split_name, ds in [("test", test_ds), ("validation", val_ds)]:
            probs = split_probs[model_name][split_name]
            prob_arr = [p for _, p in probs]
            y_pred = (np.array(prob_arr) >= 0.5).astype(int)

            fp_indices = [i for i in range(len(ds)) if ds.y[i] == 0 and y_pred[i] == 1]
            fn_indices = [i for i in range(len(ds)) if ds.y[i] == 1 and y_pred[i] == 0]
            tp_indices = [i for i in range(len(ds)) if ds.y[i] == 1 and y_pred[i] == 1]
            tn_indices = [i for i in range(len(ds)) if ds.y[i] == 0 and y_pred[i] == 0]

            print(f"\n  {split_name.upper()}:")
            print(f"    True Positives:  {len(tp_indices)}")
            print(f"    True Negatives:  {len(tn_indices)}")
            print(f"    False Positives: {len(fp_indices)}")
            print(f"    False Negatives: {len(fn_indices)}")

            # Highest-confidence false positives
            if fp_indices:
                fp_probs = [(i, prob_arr[i]) for i in fp_indices]
                fp_probs.sort(key=lambda x: x[1], reverse=True)
                print("    Top 5 highest-confidence false positives:")
                for idx, prob in fp_probs[:5]:
                    meta = ds.metadata[idx]
                    print(f"      {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                          f"{meta['file_path']:<50} prob={prob:.4f}")

            # Lowest-confidence false negatives (missed positives)
            if fn_indices:
                fn_probs = [(i, prob_arr[i]) for i in fn_indices]
                fn_probs.sort(key=lambda x: x[1], reverse=True)
                print("    Top 5 highest-confidence false negatives (closest to being caught):")
                for idx, prob in fn_probs[:5]:
                    meta = ds.metadata[idx]
                    print(f"      {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                          f"{meta['file_path']:<50} prob={prob:.4f}")

            # Lowest-confidence true positives (caught with least margin)
            if tp_indices:
                tp_probs = [(i, prob_arr[i]) for i in tp_indices]
                tp_probs.sort(key=lambda x: x[1])
                print("    Lowest-confidence true positives (caught with least margin):")
                for idx, prob in tp_probs[:3]:
                    meta = ds.metadata[idx]
                    print(f"      {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                          f"{meta['file_path']:<50} prob={prob:.4f}")

    # -- Per-Repo Performance --------------------------------------------
    print(f"\n{'=' * 70}")
    print("PER-REPO PERFORMANCE (XGBoost, Test Set)")
    print("=" * 70)

    if xgb_model is not None:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        repos: dict[str, dict] = {}
        for i in range(len(test_ds)):
            repo = test_ds.metadata[i]["repo_name"]
            if repo not in repos:
                repos[repo] = {
                    "pos": 0, "neg": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "probs": [], "labels": [],
                }
            repos[repo]["probs"].append(prob_arr[i])
            repos[repo]["labels"].append(int(test_ds.y[i]))
            if test_ds.y[i] == 1:
                repos[repo]["pos"] += 1
                if prob_arr[i] >= 0.5:
                    repos[repo]["tp"] += 1
                else:
                    repos[repo]["fn"] += 1
            else:
                repos[repo]["neg"] += 1
                if prob_arr[i] >= 0.5:
                    repos[repo]["fp"] += 1
                else:
                    repos[repo]["tn"] += 1

        header = (f"\n{'Repo':<20} {'Pos':>4} {'Neg':>4} {'TP':>3} "
                  f"{'FP':>3} {'FN':>3} {'Recall':>8} {'MeanProb':>8}")
        print(header)
        print("-" * 70)
        for repo in sorted(repos):
            r = repos[repo]
            recall = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0
            mean_prob = sum(r["probs"]) / len(r["probs"]) if r["probs"] else 0
            row = (f"  {repo:<20} {r['pos']:>4} {r['neg']:>4} {r['tp']:>3} "
                   f"{r['fp']:>3} {r['fn']:>3} {recall:>8.4f} {mean_prob:>8.4f}")
            print(row)

        # Check if positives are concentrated in a few repos
        pos_repos = {repo: r["pos"] for repo, r in repos.items() if r["pos"] > 0}
        if pos_repos:
            total_pos_in_test = sum(pos_repos.values())
            print(f"\n  Repos with positives in test: {len(pos_repos)}")
            print("  Concentration:")
            for repo in sorted(pos_repos, key=pos_repos.get, reverse=True):
                pct = 100.0 * pos_repos[repo] / total_pos_in_test
                print(f"    {repo:<20} {pos_repos[repo]:>3} positives ({pct:.1f}%)")

            # Check if performance is dominated by one repo
            repo_recalls = {}
            for repo, r in repos.items():
                if r["pos"] > 0:
                    repo_recalls[repo] = r["tp"] / (r["tp"] + r["fn"])
            if repo_recalls:
                max_recall_repo = max(repo_recalls, key=repo_recalls.get)
                min_recall_repo = min(repo_recalls, key=repo_recalls.get)
                print(f"\n  Best recall:  {max_recall_repo} = {repo_recalls[max_recall_repo]:.4f}")
                print(f"  Worst recall: {min_recall_repo} = {repo_recalls[min_recall_repo]:.4f}")

    # -- Generalization Analysis ------------------------------------------
    print(f"\n{'=' * 70}")
    print("GENERALIZATION ANALYSIS")
    print("=" * 70)

    for model_name in ["xgboost", "weighted_logistic_regression"]:
        if model_name not in fitted_models:
            continue
        print(f"\n--- {model_name.upper()} ---")

        train_r = all_results.get(f"{model_name}_train", {})
        val_r = all_results.get(f"{model_name}_validation", {})
        test_r = all_results.get(f"{model_name}_test", {})

        train_roc = train_r.get("roc_auc", 0)
        val_roc = val_r.get("roc_auc", 0)
        test_roc = test_r.get("roc_auc", 0)
        train_pr = train_r.get("pr_auc", 0)
        val_pr = val_r.get("pr_auc", 0)
        test_pr = test_r.get("pr_auc", 0)

        print(f"  ROC-AUC:  train={train_roc:.4f}  val={val_roc:.4f}  test={test_roc:.4f}")
        print(f"  PR-AUC:   train={train_pr:.4f}  val={val_pr:.4f}  test={test_pr:.4f}")

        # Overfitting signals
        if train_roc > 0 and val_roc > 0:
            gap_tv = train_roc - val_roc
            gap_vt = val_roc - test_roc
            print(f"  Train-Val gap:  {gap_tv:.4f}")
            print(f"  Val-Test gap:   {gap_vt:.4f}")

            if gap_tv > 0.15:
                print("  [!] Large train-val gap suggests overfitting")
            elif gap_tv < 0.05:
                print("  [OK] Small train-val gap -- good generalization")

            if abs(gap_vt) < 0.05:
                print("  [OK] Val-test performance is stable")
            elif gap_vt > 0.1:
                print("  [!] Val performance much better than test -- possible distribution shift")
            elif gap_vt < -0.1:
                print("  [!] Test performance much better than val -- unusual, check data")

        # Check if model exploits change-size
        print("\n  Correlation: top features vs change-size")
        if model_name == "xgboost":
            importances = extract_importance(xgb_model, FEATURE_NAMES)
            top10_names = [fi.feature_name for fi in importances[:10]]
            cs_in_top10 = sum(1 for f in top10_names if f in CHANGE_SIZE_FEATURES)
            print(f"  Change-size features in top 10: {cs_in_top10}/10")
            print(f"  Top 10: {top10_names}")

    # -- Save Artifacts ---------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SAVING ARTIFACTS")
    print("=" * 70)

    metrics_data = {
        "baseline": baseline.to_dict(),
        "models": all_results,
        "random_pr_auc_baseline": overall_prev,
    }

    if xgb_model is not None:
        importances = extract_importance(xgb_model, FEATURE_NAMES)
        metrics_data["feature_importance_top15"] = [
            {"rank": fi.rank, "feature": fi.feature_name, "importance": fi.importance}
            for fi in importances[:15]
        ]

    # Build predictions.jsonl for test set
    predictions = []
    if xgb_model is not None:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        y_pred = (np.array(prob_arr) >= 0.5).astype(int)
        ranked = sorted(
            [(i, prob_arr[i]) for i in range(len(test_ds))],
            key=lambda x: x[1], reverse=True,
        )
        rank_map = {idx: r for r, (idx, _) in enumerate(ranked, 1)}
        for i in range(len(test_ds)):
            meta = test_ds.metadata[i]
            predictions.append({
                "commit_sha": meta["commit_sha"],
                "file_path": meta["file_path"],
                "repo_name": meta["repo_name"],
                "label": int(test_ds.y[i]),
                "probability": prob_arr[i],
                "prediction": int(y_pred[i]),
                "rank": rank_map[i],
                "total_test_examples": len(test_ds),
            })

    model_dir = save_artifacts(
        output_dir=MODELS_DIR,
        model_version=MODEL_VERSION,
        model=xgb_model,
        feature_names=FEATURE_NAMES,
        dataset_version="v3-multi-phase3.7",
        feature_version="v1",
        train_count=len(train_ds),
        val_count=len(val_ds),
        test_count=len(test_ds),
        train_pos=tp,
        train_neg=tn,
        val_pos=vp,
        val_neg=vn,
        test_pos=tsp,
        test_neg=tsn,
        metrics=metrics_data,
        predictions=predictions,
    )

    print(f"Artifacts saved to: {model_dir}")

    # -- Final Verdict ---------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PHASE 4 EVALUATION VERDICT")
    print("=" * 70)

    print("\n  Dataset:          combined-v3 (v3-multi-phase3.7)")
    print(f"  Total files:      {total}")
    print(f"  Positive files:   {tp+vp+tsp}")
    print(f"  Positive commits: {len(all_pos_shas)}")
    print(f"  Repos:            {len(all_pos_repos)}")
    print(f"  Positive prevalence: {overall_prev:.4f}")
    print(f"  Random PR-AUC baseline: {overall_prev:.4f}")

    val_metrics = all_results.get("xgboost_validation", {})
    test_metrics = all_results.get("xgboost_test", {})
    lr_val = all_results.get("weighted_logistic_regression_validation", {})
    lr_test = all_results.get("weighted_logistic_regression_test", {})

    print(f"\n  XGBoost Validation ROC-AUC: {val_metrics.get('roc_auc', 'N/A')}")
    print(f"  XGBoost Test ROC-AUC:       {test_metrics.get('roc_auc', 'N/A')}")
    print(f"  XGBoost Validation PR-AUC:  {val_metrics.get('pr_auc', 'N/A')}")
    print(f"  XGBoost Test PR-AUC:        {test_metrics.get('pr_auc', 'N/A')}")
    print(f"  LR Validation ROC-AUC:      {lr_val.get('roc_auc', 'N/A')}")
    print(f"  LR Test ROC-AUC:            {lr_test.get('roc_auc', 'N/A')}")
    print(f"  LR Validation PR-AUC:       {lr_val.get('pr_auc', 'N/A')}")
    print(f"  LR Test PR-AUC:             {lr_test.get('pr_auc', 'N/A')}")

    # PR-AUC lift
    xgb_val_pr = val_metrics.get("pr_auc", 0)
    xgb_test_pr = test_metrics.get("pr_auc", 0)
    if overall_prev > 0:
        val_lift = xgb_val_pr / overall_prev
        test_lift = xgb_test_pr / overall_prev
        print(f"\n  XGBoost Val PR-AUC lift:  {val_lift:.2f}x")
        print(f"  XGBoost Test PR-AUC lift: {test_lift:.2f}x")

    # Ranking summary
    if xgb_model is not None and test_ds.positive_count > 0:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        ranked = sorted(probs, key=lambda x: x[1], reverse=True)
        pos_indices = set(i for i in range(len(test_ds)) if test_ds.y[i] == 1)
        total_test = len(test_ds)

        top10 = len(pos_indices & set(i for i, _ in ranked[:10]))
        top20 = len(pos_indices & set(i for i, _ in ranked[:20]))
        top50 = len(pos_indices & set(i for i, _ in ranked[:50]))
        top100 = len(pos_indices & set(i for i, _ in ranked[:100]))
        top10pct = len(pos_indices & set(i for i, _ in ranked[:max(1, total_test // 10)]))
        top20pct = len(pos_indices & set(i for i, _ in ranked[:max(1, total_test // 5)]))

        print(f"\n  XGBoost Test Ranking (out of {tsp} positives):")
        print(f"    Top 10:   {top10}/{tsp}")
        print(f"    Top 20:   {top20}/{tsp}")
        print(f"    Top 50:   {top50}/{tsp}")
        print(f"    Top 100:  {top100}/{tsp}")
        print(f"    Top 10%:  {top10pct}/{tsp}")
        print(f"    Top 20%:  {top20pct}/{tsp}")

    # Data-driven assessment
    print("\n  Assessment:")
    val_roc = val_metrics.get("roc_auc", 0)
    test_roc = test_metrics.get("roc_auc", 0)
    val_pr = val_metrics.get("pr_auc", 0)
    test_pr = test_metrics.get("pr_auc", 0)

    signals = []
    if val_roc > 0.5 + 0.01:
        signals.append("val ROC-AUC > 0.51")
    if test_roc > 0.5 + 0.01:
        signals.append("test ROC-AUC > 0.51")
    if val_pr > overall_prev * 1.5:
        signals.append("val PR-AUC lift > 1.5x")
    if test_pr > overall_prev * 1.5:
        signals.append("test PR-AUC lift > 1.5x")
    if len(all_pos_shas) >= 20:
        signals.append("sufficient positive commits (>=20)")

    # Check ranking signal
    if xgb_model is not None and test_ds.positive_count > 0:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        ranked = sorted(probs, key=lambda x: x[1], reverse=True)
        pos_indices = set(i for i in range(len(test_ds)) if test_ds.y[i] == 1)
        top100_count = len(pos_indices & set(i for i, _ in ranked[:100]))
        if top100_count > 0:
            signals.append(f"ranking: {top100_count}/{tsp} positives in top 100")

    # Overfitting check
    if val_roc > 0 and test_roc > 0:
        gap = val_roc - test_roc
        if abs(gap) < 0.1:
            signals.append("val-test stable (gap < 0.1)")
        elif gap > 0.15:
            signals.append("WARNING: val-test gap > 0.15 (possible overfitting)")

    if signals:
        print("  Positive signals:")
        for s in signals:
            print(f"    + {s}")
    else:
        print("  No strong positive signals detected.")

    print()
    print("=" * 70)
    print("STOP -- Phase 4 evaluation complete. Do not proceed to SageMaker.")
    print("=" * 70)


if __name__ == "__main__":
    main()
