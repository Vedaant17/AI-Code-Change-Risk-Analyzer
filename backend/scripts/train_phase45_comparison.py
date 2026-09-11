#!/usr/bin/env python3
"""Phase 4.5 ML Experiment: Model A vs Model B Comparison.

Controlled comparison of:
  - Model A: Baseline v3 features only (45 features)
  - Model B: v3 + 6 experimental features (51 features)

Both models trained under identical conditions (same train/val/test splits,
same random seeds, same hyperparameters). Results compared via:
  - Primary: ROC-AUC, PR-AUC
  - Secondary: Precision, Recall, F1
  - Ablation: historical-only, AST-only, all-six
  - Feature importance analysis
  - Error analysis
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.ml.dataset_loader import (
    FEATURE_NAMES,
    NUM_FEATURES,
    load_dataset,
)
from backend.app.ml.experimental_loader import (
    load_experimental_dataset,
)
from backend.app.ml.importance import extract_importance
from backend.app.ml.metrics import compute_metrics
from backend.app.ml.models import (
    MajorityClassBaseline,
    WeightedLogisticRegression,
    XGBoostDefectModel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
V3_DATA_DIR = Path("backend/data/datasets/combined-v3")
EXP_DATA_DIR = Path("backend/data/datasets/experimental-exp-4.5a/combined-v3")
OUTPUT_DIR = Path("backend/data/models")
MODEL_VERSION_BASELINE = "v0.1.0-combined-v3-real"
MODEL_VERSION_EXPERIMENTAL = "v0.1.0-combined-v3-exp-4.5a-real"


@dataclass
class ModelResult:
    """Container for a single model's results."""
    model_name: str
    feature_count: int
    feature_names: list[str]
    train_metrics: dict
    val_metrics: dict
    test_metrics: dict
    train_time_seconds: float
    importance: list[dict] | None = None


@dataclass
class ComparisonReport:
    """Container for the full comparison report."""
    baseline_model: ModelResult
    experimental_model: ModelResult
    ablation_results: dict[str, ModelResult]
    metadata: dict


def train_and_evaluate(
    model,
    train_data,
    val_data,
    test_data,
    feature_names: list[str],
    model_name: str,
) -> ModelResult:
    """Train a model and evaluate on all splits."""
    logger.info(f"Training {model_name}...")
    start_time = time.time()

    model.fit(
        X_train=train_data.X,
        y_train=train_data.y,
        X_val=val_data.X,
        y_val=val_data.y,
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Evaluate on all splits
    train_probs = model.predict_proba(train_data.X)
    val_probs = model.predict_proba(val_data.X)
    test_probs = model.predict_proba(test_data.X)

    train_metrics = compute_metrics(train_data.y, train_probs)
    val_metrics = compute_metrics(val_data.y, val_probs)
    test_metrics = compute_metrics(test_data.y, test_probs)

    # Feature importance (for tree/linear models)
    importance = None
    if hasattr(model, "_model") and model._model is not None:
        raw_importance = extract_importance(model, feature_names)
        importance = [asdict(fi) for fi in raw_importance]

    return ModelResult(
        model_name=model_name,
        feature_count=len(feature_names),
        feature_names=feature_names,
        train_metrics=asdict(train_metrics),
        val_metrics=asdict(val_metrics),
        test_metrics=asdict(test_metrics),
        train_time_seconds=train_time,
        importance=importance,
    )


def run_ablation_experiments(
    train_data,
    val_data,
    test_data,
) -> dict[str, ModelResult]:
    """Run ablation experiments with different feature subsets."""
    ablation_results = {}

    # E1: AST-only (3 features)
    logger.info("Running ablation E1: AST-only")
    train_e1, val_e1, test_e1 = load_experimental_dataset(
        V3_DATA_DIR, EXP_DATA_DIR, mode="E1"
    )
    model_e1 = XGBoostDefectModel()
    result_e1 = train_and_evaluate(
        model_e1, train_e1, val_e1, test_e1,
        train_e1.feature_names, "xgboost_ast_only"
    )
    ablation_results["E1_ast_only"] = result_e1

    # E2: Historical-only (3 features)
    logger.info("Running ablation E2: Historical-only")
    train_e2, val_e2, test_e2 = load_experimental_dataset(
        V3_DATA_DIR, EXP_DATA_DIR, mode="E2"
    )
    model_e2 = XGBoostDefectModel()
    result_e2 = train_and_evaluate(
        model_e2, train_e2, val_e2, test_e2,
        train_e2.feature_names, "xgboost_historical_only"
    )
    ablation_results["E2_historical_only"] = result_e2

    return ablation_results


def generate_comparison_report(
    baseline: ModelResult,
    experimental: ModelResult,
    ablation: dict[str, ModelResult],
    metadata: dict,
) -> ComparisonReport:
    """Generate the full comparison report."""
    return ComparisonReport(
        baseline_model=baseline,
        experimental_model=experimental,
        ablation_results=ablation,
        metadata=metadata,
    )


def print_comparison_table(baseline: ModelResult, experimental: ModelResult):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("PHASE 4.5 ML EXPERIMENT: MODEL A vs MODEL B COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Metric':<25} {'Model A (Baseline)':<20} "
        f"{'Model B (Experimental)':<20} {'Delta':<15}"
    )
    print("-" * 70)

    metrics = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
    ]

    for name, key in metrics:
        val_a = baseline.test_metrics[key]
        val_b = experimental.test_metrics[key]
        delta = val_b - val_a
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {val_a:<20.4f} {val_b:<20.4f} {sign}{delta:<14.4f}")

    print("-" * 70)
    print(
        f"{'Features':<25} {baseline.feature_count:<20} "
        f"{experimental.feature_count:<20}"
    )
    print(
        f"{'Training Time (s)':<25} "
        f"{baseline.train_time_seconds:<20.1f} "
        f"{experimental.train_time_seconds:<20.1f}"
    )
    print("=" * 70)


def save_report(
    report: ComparisonReport,
    output_dir: Path,
):
    """Save the comparison report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full report as JSON
    report_path = output_dir / "phase45_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Save summary markdown
    summary_path = output_dir / "phase45_comparison_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4.5 ML Experiment: Model A vs Model B\n\n")
        f.write("## Overview\n\n")
        f.write(
            f"- **Model A (Baseline)**: {report.baseline_model.feature_count} "
            f"features (v3 only)\n"
        )
        f.write(
            f"- **Model B (Experimental)**: {report.experimental_model.feature_count} "
            f"features (v3 + 6 Phase 4.5a)\n"
        )
        f.write(f"- **Dataset**: {report.metadata['total_rows']:,} rows\n")
        f.write(f"- **Repos**: {report.metadata['repo_count']} repositories\n\n")

        f.write("## Test Set Results\n\n")
        f.write("| Metric | Model A | Model B | Delta |\n")
        f.write("|--------|---------|---------|-------|\n")
        for name, key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
                          ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
            val_a = report.baseline_model.test_metrics[key]
            val_b = report.experimental_model.test_metrics[key]
            delta = val_b - val_a
            sign = "+" if delta >= 0 else ""
            f.write(f"| {name} | {val_a:.4f} | {val_b:.4f} | {sign}{delta:.4f} |\n")

        f.write("\n## Ablation Results\n\n")
        f.write("| Configuration | ROC-AUC | PR-AUC | F1 |\n")
        f.write("|---------------|---------|--------|----|\n")
        for name, result in report.ablation_results.items():
            f.write(f"| {name} | {result.test_metrics['roc_auc']:.4f} | "
                    f"{result.test_metrics['pr_auc']:.4f} | {result.test_metrics['f1']:.4f} |\n")

        if report.experimental_model.importance:
            f.write("\n## Top 15 Features (Model B)\n\n")
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for fi in report.experimental_model.importance[:15]:
                f.write(f"| {fi['rank']} | {fi['feature_name']} | {fi['importance']:.4f} |\n")

    logger.info(f"Report saved to {output_dir}")
    return report_path, summary_path


def main():
    """Run the Phase 4.5 ML experiment."""
    logger.info("Starting Phase 4.5 ML Experiment")
    start_time = time.time()

    # Step 1: Load baseline v3 data
    logger.info("Loading baseline v3 dataset...")
    train_base, val_base, test_base = load_dataset(V3_DATA_DIR)
    logger.info(f"Baseline: train={len(train_base)}, val={len(val_base)}, test={len(test_base)}")

    # Step 2: Train Model A (baseline - 45 features)
    logger.info("\n--- MODEL A: Baseline (45 features) ---")
    model_a_baseline = MajorityClassBaseline()
    train_and_evaluate(
        model_a_baseline, train_base, val_base, test_base,
        FEATURE_NAMES, "majority_class_baseline"
    )

    model_a_lr = WeightedLogisticRegression()
    train_and_evaluate(
        model_a_lr, train_base, val_base, test_base,
        FEATURE_NAMES, "weighted_logistic_regression"
    )

    model_a_xgb = XGBoostDefectModel()
    baseline_xgb = train_and_evaluate(
        model_a_xgb, train_base, val_base, test_base,
        FEATURE_NAMES, "xgboost_baseline"
    )

    # Step 3: Load experimental dataset (E4: all 6 features)
    logger.info("\n--- Loading experimental dataset (E4 mode) ---")
    train_exp, val_exp, test_exp = load_experimental_dataset(
        V3_DATA_DIR, EXP_DATA_DIR, mode="E4"
    )
    logger.info(
        f"Experimental: train={len(train_exp)}, "
        f"val={len(val_exp)}, test={len(test_exp)}"
    )
    logger.info(f"Feature dimension: {train_exp.X.shape[1]}")

    # Step 4: Train Model B (experimental - 51 features)
    logger.info("\n--- MODEL B: Experimental (51 features) ---")
    model_b_baseline = MajorityClassBaseline()
    train_and_evaluate(
        model_b_baseline, train_exp, val_exp, test_exp,
        train_exp.feature_names, "majority_class_baseline_exp"
    )

    model_b_lr = WeightedLogisticRegression()
    train_and_evaluate(
        model_b_lr, train_exp, val_exp, test_exp,
        train_exp.feature_names, "weighted_logistic_regression_exp"
    )

    model_b_xgb = XGBoostDefectModel()
    exp_xgb = train_and_evaluate(
        model_b_xgb, train_exp, val_exp, test_exp,
        train_exp.feature_names, "xgboost_experimental"
    )

    # Step 5: Print comparison
    print_comparison_table(baseline_xgb, exp_xgb)

    # Step 6: Run ablation experiments
    logger.info("\n--- Running ablation experiments ---")
    ablation_results = run_ablation_experiments(train_base, val_base, test_base)

    # Step 7: Generate report
    metadata = {
        "experiment": "Phase 4.5 - Model A vs Model B",
        "dataset_version": "combined-v3",
        "experimental_version": "exp-4.5a",
        "total_rows": len(train_exp) + len(val_exp) + len(test_exp),
        "train_rows": len(train_exp),
        "val_rows": len(val_exp),
        "test_rows": len(test_exp),
        "repo_count": 50,
        "baseline_feature_count": NUM_FEATURES,
        "experimental_feature_count": train_exp.X.shape[1],
    }

    report = generate_comparison_report(baseline_xgb, exp_xgb, ablation_results, metadata)

    # Step 8: Save artifacts
    output_dir = OUTPUT_DIR / MODEL_VERSION_EXPERIMENTAL
    report_path, summary_path = save_report(report, output_dir)

    total_time = time.time() - start_time
    logger.info(f"\nExperiment completed in {total_time:.1f}s")
    logger.info(f"Full report: {report_path}")
    logger.info(f"Summary: {summary_path}")

    return report


if __name__ == "__main__":
    main()
