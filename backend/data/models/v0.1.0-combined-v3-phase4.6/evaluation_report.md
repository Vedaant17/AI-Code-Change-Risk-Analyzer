# Phase 4.6 Evaluation Report

## Objective

Determine whether the defect-risk model generalizes to completely unseen repositories and ranks risky files near the top of changed commits/PRs.

## Methodology

- Dataset: v3-multi-phase3.7 (54,389 rows, 50 repos)
- Split: Repo-level round-robin (17/17/16 = TRAIN/VAL/TEST)
- Model: XGBoost (n_estimators=200, max_depth=6, lr=0.1, random_state=42)
- scale_pos_weight: auto-computed (neg_count / pos_count)
- Primary evaluation: Per-commit ranking (Precision@K, Recall@K, MRR)
- Secondary evaluation: Global classification (ROC-AUC, PR-AUC, F1)

## Model Variants

| Variant | Features | Description |
|---------|----------|-------------|
| A | 45 | Baseline v3 features |
| B | 48 | +3 historical features |
| C | 51 | +3 historical +3 AST features |

## Variant A (45 features)

### Global Classification (Secondary)

- ROC-AUC: 0.5443
- PR-AUC: 0.0210 (lift 1.08x over prevalence 0.0195)
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

### Per-Commit Ranking (Primary)

- Commits with positives: 55/6968
- Mean files per commit: 2.3

| Metric | Mean | Median |
|--------|------|--------|
| Precision@1 | 1.0000 | 1.0000 |
| Precision@3 | 1.0000 | 1.0000 |
| Precision@5 | 1.0000 | 1.0000 |
| Recall@1 | 0.5745 | 0.5000 |
| Recall@3 | 0.8418 | 1.0000 |
| Recall@5 | 0.9076 | 1.0000 |
| Rank of first positive | 1.0 | 1.0 |
| MRR | 1.0000 | - |

### Positive Capture

- Top-1: 100.0% of commits
- Top-3: 100.0% of commits
- Top-5: 100.0% of commits

## Variant B (48 features)

### Global Classification (Secondary)

- ROC-AUC: 0.5950
- PR-AUC: 0.0288 (lift 1.48x over prevalence 0.0195)
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

### Per-Commit Ranking (Primary)

- Commits with positives: 55/6968
- Mean files per commit: 2.3

| Metric | Mean | Median |
|--------|------|--------|
| Precision@1 | 1.0000 | 1.0000 |
| Precision@3 | 1.0000 | 1.0000 |
| Precision@5 | 1.0000 | 1.0000 |
| Recall@1 | 0.5745 | 0.5000 |
| Recall@3 | 0.8418 | 1.0000 |
| Recall@5 | 0.9076 | 1.0000 |
| Rank of first positive | 1.0 | 1.0 |
| MRR | 1.0000 | - |

### Positive Capture

- Top-1: 100.0% of commits
- Top-3: 100.0% of commits
- Top-5: 100.0% of commits

## Variant C (51 features)

### Global Classification (Secondary)

- ROC-AUC: 0.5950
- PR-AUC: 0.0288 (lift 1.48x over prevalence 0.0195)
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

### Per-Commit Ranking (Primary)

- Commits with positives: 55/6968
- Mean files per commit: 2.3

| Metric | Mean | Median |
|--------|------|--------|
| Precision@1 | 1.0000 | 1.0000 |
| Precision@3 | 1.0000 | 1.0000 |
| Precision@5 | 1.0000 | 1.0000 |
| Recall@1 | 0.5745 | 0.5000 |
| Recall@3 | 0.8418 | 1.0000 |
| Recall@5 | 0.9076 | 1.0000 |
| Rank of first positive | 1.0 | 1.0 |
| MRR | 1.0000 | - |

### Positive Capture

- Top-1: 100.0% of commits
- Top-3: 100.0% of commits
- Top-5: 100.0% of commits

## Ablation Results

### A vs B (Historical Feature Contribution)

- mean_precision_at_1: 1.0000 -> 1.0000 (delta=+0.0000)
- mean_recall_at_1: 0.5745 -> 0.5745 (delta=+0.0000)
- mean_reciprocal_rank: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_1_capture: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_3_capture: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_5_capture: 1.0000 -> 1.0000 (delta=+0.0000)

### B vs C (AST Feature Contribution)

- mean_precision_at_1: 1.0000 -> 1.0000 (delta=+0.0000)
- mean_recall_at_1: 0.5745 -> 0.5745 (delta=+0.0000)
- mean_reciprocal_rank: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_1_capture: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_3_capture: 1.0000 -> 1.0000 (delta=+0.0000)
- fraction_top_5_capture: 1.0000 -> 1.0000 (delta=+0.0000)

## Per-Repository Analysis (Variant A, Test Set)

Macro-averaged across 16 test repos:
- Mean prevalence: 0.0200
- Mean PR-AUC: 0.0308
- Mean Precision@1: 1.0000
- Mean MRR: 1.0000

### Dominance Check

- Total test positives: 319
- Any repo dominates (>50%): False

## Historical Leakage Sanity Checks

Phase 4.6 reads precomputed experimental features from Phase 4.5. The authoritative git commands use C^ boundary (excluding C itself). No post-commit information is introduced. The following are sanity checks, NOT proof of leakage safety.

### Variant A

### Variant B

### Variant C


## Qualitative Assessment

- MRR=1.0000 > 0.5 — model ranks positives near the top
- Top-1 capture=100.0% — frequently catches a positive at rank 1
- Top-3 capture=100.0% — frequently catches a positive in top 3

## Conclusion

This evaluation determines whether the model is ready for downstream quantum/classical optimization (Phase 6).

**Assessment: NOT READY — FURTHER ML WORK REQUIRED**

---

## Methodological Notes

- Validation metrics are descriptive only and do not influence model selection
- No early stopping, no hyperparameter tuning, no threshold optimization
- Feature selection frozen: A=45, B=48, C=51
- All statistics derived programmatically from actual dataset
