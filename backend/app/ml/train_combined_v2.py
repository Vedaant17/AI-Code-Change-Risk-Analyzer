"""Phase 4 real-data training on the combined multi-repository dataset (v2).

Uses the dataset from Phase 3.6 (backend/data/datasets/combined-v2/).
Evaluates MajorityClassBaseline, WeightedLogisticRegression, XGBoostDefectModel.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

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

DATA_DIR = project_root / "backend" / "data" / "datasets" / "combined-v2"
MODELS_DIR = project_root / "backend" / "data" / "models"
MODEL_VERSION = "v0.1.0-combined-v2-real"


def _ranking_analysis(probs: list[tuple[int, float]], ds, label: str) -> dict:
    """Compute top-k and top-pct ranking stats for positive examples."""
    total = len(ds)
    pos_indices = [i for i in range(len(ds)) if ds.y[i] == 1]
    if not pos_indices:
        print(f"  No positives in {label}")
        return {}

    ranked = sorted(
        [(i, p) for i, p in enumerate(probs)],
        key=lambda x: x[1], reverse=True,
    )
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


def main() -> None:
    """Run Phase 4 training on the combined multi-repo v2 dataset."""
    # ── Load combined dataset ────────────────────────────────────────────
    logger.info("Loading combined-v2 dataset from %s", DATA_DIR)
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    print("\n" + "=" * 70)
    print("PHASE 4: Real ML Training — Combined Multi-Repository Dataset (v2)")
    print("=" * 70)

    # ── Dataset Summary ──────────────────────────────────────────────────
    tp = train_ds.positive_count
    tn = train_ds.negative_count
    vp = val_ds.positive_count
    vn = val_ds.negative_count
    tsp = test_ds.positive_count
    tsn = test_ds.negative_count
    total = len(train_ds) + len(val_ds) + len(test_ds)

    # Distinct positive commits per split
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

    # ── Feature Contract Verification ────────────────────────────────────
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

    # ── Leakage Verification ─────────────────────────────────────────────
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
        for label, ovl in pairs:
            if ovl:
                print(f"    {label}: {ovl}")

    print("  Checking for ambiguous rows in supervised splits...")
    all_clean = True
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for i, y in enumerate(ds.y):
            if y == -1:
                print(f"  [FAIL] Ambiguous row in {split_name}: {ds.metadata[i]}")
                all_clean = False
    if all_clean:
        print("  [OK] No ambiguous rows in supervised splits")

    # ── Baselines ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("BASELINES")
    print("=" * 70)
    baseline = compute_baseline(train_ds.y)
    print("Majority-class baseline (always predicts class 0):")
    print(f"  accuracy={baseline.accuracy:.4f}  majority_class={baseline.majority_class}")

    # ── Models ───────────────────────────────────────────────────────────
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

            print(f"\n--- {split_name.upper()} ---")
            print(f"  Examples:          {metrics.total}")
            print(f"  Actual positives:  {metrics.actual_positives}")
            print(f"  Actual negatives:  {metrics.actual_negatives}")
            print(f"  Positive predictions: {metrics.positive_predictions}")

            def _fmt(v):
                return f"{v:.4f}" if v != 0 or metrics.actual_positives > 0 else "undefined"

            print(f"  ROC-AUC:           {_fmt(metrics.roc_auc)}")
            print(f"  PR-AUC:            {_fmt(metrics.pr_auc)}")
            print(f"  Precision:         {_fmt(metrics.precision)}")
            print(f"  Recall:            {_fmt(metrics.recall)}")
            print(f"  F1:                {_fmt(metrics.f1)}")
            print(f"  Confusion matrix:  {metrics.confusion_matrix}")

            all_results[f"{name}_{split_name}"] = metrics.to_dict()

            # Store probabilities for ranking analysis
            if name not in split_probs:
                split_probs[name] = {}
            split_probs[name][split_name] = [(i, y_prob[i]) for i in range(len(ds))]

    # ── Feature Importance ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    xgb_model = fitted_models.get("xgboost")
    if xgb_model is not None:
        importances = extract_importance(xgb_model, FEATURE_NAMES)
        print(format_importance_report(importances, top_n=15))

    lr_model = fitted_models.get("weighted_logistic_regression")
    if lr_model is not None:
        lr_importances = extract_importance(lr_model, FEATURE_NAMES)
        print("\nLogistic Regression Coefficients (top 15):")
        print(format_importance_report(lr_importances, top_n=15))

    # ── Positive-Case Test Analysis (XGBoost, primary) ──────────────────
    print(f"\n{'=' * 70}")
    print("POSITIVE-CASE ANALYSIS (Test Set — XGBoost)")
    print("=" * 70)

    if xgb_model is not None and test_ds.positive_count > 0:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        ranked = sorted(probs, key=lambda x: x[1], reverse=True)
        pos_indices = set(i for i in range(len(test_ds)) if test_ds.y[i] == 1)

        print(f"\n{'SHA':<14} {'Repo':<12} {'File':<50} {'Prob':>8} {'Rank':>6}")
        print("-" * 95)
        for idx in range(len(test_ds)):
            if test_ds.y[idx] == 1:
                meta = test_ds.metadata[idx]
                prob = prob_arr[idx]
                rank = next(r for r, (i, _) in enumerate(ranked, 1) if i == idx)
                print(f"  {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                      f"{meta['file_path']:<50} {prob:.4f} {rank:>4}/{len(test_ds)}")

        print("\n--- Ranking Analysis ---")
        _ranking_analysis(prob_arr, test_ds, "test")

    # ── Positive-Case Validation Analysis (XGBoost) ────────────────────
    print(f"\n{'=' * 70}")
    print("POSITIVE-CASE ANALYSIS (Validation Set — XGBoost)")
    print("=" * 70)

    if xgb_model is not None and val_ds.positive_count > 0:
        probs = split_probs["xgboost"]["validation"]
        prob_arr = [p for _, p in probs]
        ranked = sorted(probs, key=lambda x: x[1], reverse=True)

        print(f"\n{'SHA':<14} {'Repo':<12} {'File':<50} {'Prob':>8} {'Rank':>6}")
        print("-" * 95)
        for idx in range(len(val_ds)):
            if val_ds.y[idx] == 1:
                meta = val_ds.metadata[idx]
                prob = prob_arr[idx]
                rank = next(r for r, (i, _) in enumerate(ranked, 1) if i == idx)
                print(f"  {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                      f"{meta['file_path']:<50} {prob:.4f} {rank:>4}/{len(val_ds)}")

        print("\n--- Ranking Analysis ---")
        _ranking_analysis(prob_arr, val_ds, "validation")

    # ── Error Analysis ──────────────────────────────────────────────────
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

            # False positives
            fp_indices = [i for i in range(len(ds)) if ds.y[i] == 0 and y_pred[i] == 1]
            # False negatives
            fn_indices = [i for i in range(len(ds)) if ds.y[i] == 1 and y_pred[i] == 0]

            print(f"\n  {split_name.upper()}:")
            print(f"    False Positives:  {len(fp_indices)}")
            print(f"    False Negatives:  {len(fn_indices)}")

            # Highest-risk false positives
            if fp_indices:
                fp_probs = [(i, prob_arr[i]) for i in fp_indices]
                fp_probs.sort(key=lambda x: x[1], reverse=True)
                print("    Top 5 highest-risk false positives:")
                for idx, prob in fp_probs[:5]:
                    meta = ds.metadata[idx]
                    print(f"      {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                          f"{meta['file_path']:<50} prob={prob:.4f}")

            # Lowest-risk true positives (closest to threshold)
            tp_indices = [i for i in range(len(ds)) if ds.y[i] == 1 and y_pred[i] == 1]
            if tp_indices:
                tp_probs = [(i, prob_arr[i]) for i in tp_indices]
                tp_probs.sort(key=lambda x: x[1])
                print("    Lowest-risk true positives (closest to threshold):")
                for idx, prob in tp_probs[:3]:
                    meta = ds.metadata[idx]
                    print(f"      {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                          f"{meta['file_path']:<50} prob={prob:.4f}")

    # ── Per-Repo Performance ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PER-REPO PERFORMANCE (XGBoost, Test Set)")
    print("=" * 70)

    if xgb_model is not None:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        repos = {}
        for i in range(len(test_ds)):
            repo = test_ds.metadata[i]["repo_name"]
            if repo not in repos:
                repos[repo] = {"pos": 0, "neg": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "probs": []}
            repos[repo]["probs"].append(prob_arr[i])
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

    # ── Confusion Analysis at threshold 0.5 (XGBoost, Test) ─────────────
    print(f"\n{'=' * 70}")
    print("CONFUSION ANALYSIS (XGBoost, Test Set, threshold=0.5)")
    print("=" * 70)

    if xgb_model is not None:
        probs = split_probs["xgboost"]["test"]
        prob_arr = np.array([p for _, p in probs])
        y_pred = (prob_arr >= 0.5).astype(int)
        y_true = test_ds.y

        fp_indices = [i for i in range(len(test_ds)) if y_true[i] == 0 and y_pred[i] == 1]
        fn_indices = [i for i in range(len(test_ds)) if y_true[i] == 1 and y_pred[i] == 0]

        print(f"\n  False Positives: {len(fp_indices)}")
        for idx in fp_indices:
            meta = test_ds.metadata[idx]
            print(f"    {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                  f"{meta['file_path']:<50} prob={prob_arr[idx]:.4f}")

        print(f"\n  False Negatives: {len(fn_indices)}")
        for idx in fn_indices:
            meta = test_ds.metadata[idx]
            print(f"    {meta['commit_sha'][:12]}... {meta['repo_name']:<12} "
                  f"{meta['file_path']:<50} prob={prob_arr[idx]:.4f}")

    # ── Statistical Validity Assessment ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STATISTICAL VALIDITY ASSESSMENT")
    print("=" * 70)

    print(f"  Total positive file examples: {tp+vp+tsp}")
    print(f"  Distinct positive commits:    {len(all_pos_shas)}")
    print(f"  Repos with positives:         {sorted(all_pos_repos)}")
    print(f"  Train positives:              {tp} files in {len(train_pos_shas)} commits")
    print(f"  Val positives:                {vp} files in {len(val_pos_shas)} commits")
    print(f"  Test positives:               {tsp} files in {len(test_pos_shas)} commits")

    if tp < 50:
        print(f"\n  [!] {tp} training positives is below 50+ threshold.")
    if len(all_pos_shas) < 10:
        print(f"  [!] {len(all_pos_shas)} distinct positive commits is below 10+ threshold.")

    # ── Save Artifacts ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SAVING ARTIFACTS")
    print("=" * 70)

    metrics_data = {
        "baseline": baseline.to_dict(),
        "models": all_results,
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
        dataset_version="v2-multi-phase3.6",
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

    # ── Final Verdict ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL VERDICT")
    print("=" * 70)

    val_metrics = all_results.get("xgboost_validation", {})
    test_metrics = all_results.get("xgboost_test", {})
    lr_val = all_results.get("weighted_logistic_regression_validation", {})
    lr_test = all_results.get("weighted_logistic_regression_test", {})

    print("\n  Dataset:          combined-v2 (v2-multi-phase3.6)")
    print(f"  Total files:      {total}")
    print(f"  Positive files:   {tp+vp+tsp}")
    print(f"  Positive commits: {len(all_pos_shas)}")
    print(f"  Repos:            {len(all_pos_repos)}")

    print(f"\n  XGBoost Validation ROC-AUC: {val_metrics.get('roc_auc', 'N/A')}")
    print(f"  XGBoost Test ROC-AUC:       {test_metrics.get('roc_auc', 'N/A')}")
    print(f"  LR Validation ROC-AUC:      {lr_val.get('roc_auc', 'N/A')}")
    print(f"  LR Test ROC-AUC:            {lr_test.get('roc_auc', 'N/A')}")

    # Ranking analysis summary
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
        top10pct = len(pos_indices & set(i for i, _ in ranked[:max(1, total_test//10)]))
        top20pct = len(pos_indices & set(i for i, _ in ranked[:max(1, total_test//5)]))

        print(f"\n  XGBoost Test Ranking (out of {tsp} positives):")
        print(f"    Top 10:   {top10}/{tsp}")
        print(f"    Top 20:   {top20}/{tsp}")
        print(f"    Top 50:   {top50}/{tsp}")
        print(f"    Top 100:  {top100}/{tsp}")
        print(f"    Top 10%:  {top10pct}/{tsp}")
        print(f"    Top 20%:  {top20pct}/{tsp}")

    print("\n  Assessment:")
    roc_auc = test_metrics.get("roc_auc", 0)
    val_roc = val_metrics.get("roc_auc", 0)
    baseline_acc = baseline.accuracy
    xgb_above_baseline = roc_auc > baseline_acc
    val_above_baseline = val_roc > baseline_acc

    print(f"  Baseline accuracy (majority class): {baseline_acc:.4f}")
    val_sign = ">" if val_above_baseline else "<="
    print(f"  XGBoost val ROC-AUC:  {val_roc:.4f}  "
          f"{val_sign} baseline? {val_above_baseline}")
    test_sign = ">" if xgb_above_baseline else "<="
    print(f"  XGBoost test ROC-AUC: {roc_auc:.4f}  "
          f"{test_sign} baseline? {xgb_above_baseline}")

    # Check if positives are ranked highly
    if xgb_model is not None and test_ds.positive_count > 0:
        probs = split_probs["xgboost"]["test"]
        prob_arr = [p for _, p in probs]
        ranked = sorted(probs, key=lambda x: x[1], reverse=True)
        pos_indices = set(i for i in range(len(test_ds)) if test_ds.y[i] == 1)
        top100_count = len(pos_indices & set(i for i, _ in ranked[:100]))
        ranking_signal = top100_count >= tsp * 0.3

        print(f"\n  Ranking signal: {top100_count}/{tsp} positives in top 100 "
              f"({'YES' if ranking_signal else 'NO'})")

    print()
    if (val_roc > baseline_acc + 0.01 or xgb_above_baseline) and len(all_pos_shas) >= 10:
        print("  VERDICT: A. MODEL SHOWS PROMISING OUT-OF-SAMPLE SIGNAL")
        print("  RECOMMEND: Proceed to Phase 5 SageMaker.")
    elif len(all_pos_shas) < 10:
        print("  VERDICT: B. INSUFFICIENT POSITIVE COMMITS")
        print("  RECOMMEND: Expand dataset further.")
    else:
        print("  VERDICT: B. PIPELINE WORKS BUT MODEL QUALITY IS INSUFFICIENT")
        print("  RECOMMEND: Expand dataset before SageMaker.")
    print("=" * 70)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    main()
