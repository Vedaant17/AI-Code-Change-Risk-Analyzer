# Phase 4.10: PU Learning Feasibility Experiment
## Executive Summary
This experiment evaluates whether the observed-positive label signal provides a learnable ranking signal above random, using three methods: random baseline, naive P-vs-U (deliberately biased), and P-vs-U ranker (ranking signal only, no calibration claims). All metrics are observed-positive ranking metrics; no true-defect precision, recall, or ROC is reported.
## Methodology
### Elkan-Noto Status
ELKAN_NOTO_STATUS = NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE

SCAR is UNTESTABLE_WITH_CURRENT_EVIDENCE (Phase 4.9). The Elkan-Noto parameter c = P(s=1|y=1) cannot be estimated. The observed-positive fraction n_P / (n_P + n_U) is P(s=1), NOT c.
### PU Methods
1. **Random (Method A)**: Uniform random scores. Analytic null expectation and seeded random realizations are reported.
2. **Naive P-vs-U (Method B)**: Treats U as negative. Uses LogisticRegression with predict_proba(). Deliberately biased. Probability values are NOT calibrated defect probabilities. For comparison only.
3. **P-vs-U Ranker (Method C)**: Same LogisticRegression formulation as Method B. Uses decision_function() normalized to [0,1] as a ranking score. NO calibration claims. Output is interpreted as ranking similarity to observed positives, NOT calibrated true-defect probability. The ranking is the scientifically preferred interpretation.
#### Method B vs Method C Relationship

Methods B and C use the same LogisticRegression model (identical training data, solver, class_weight, random_state). Method B uses predict_proba() (sigmoid of decision function); Method C uses decision_function() directly. Because sigmoid is strictly monotonic in the decision function, their ranking orderings are identical up to ties and floating-point precision. Differences in recall@K arise only from ties in the score distribution.
### Random Baseline Methodology
For each split (train/validation/test), the analytic random baseline is computed using the split-specific N (eligible observations) and P (observed positives). The expected metrics under random ranking are:
- expected_recall_at_k = min(k, N) / N
- expected_positive_count_at_k = min(k, N) * P / N
- expected_enrichment_at_k = 1.0
- expected_mean_rank = (N + 1) / 2

Seeded random realizations (single draws from the random baseline distribution) are also reported for direct comparison with learned methods. These realizations have inherent variance; the analytic expectation is the appropriate null comparison.
### Evaluation Metrics
All metrics are **observed-positive ranking metrics**. No true-defect metrics are reported.
- observed_positive_recall_at_k: fraction of observed positives in top-K
- observed_positive_enrichment_at_k: (obs_in_top_K / K) / (P / N)
- mean_rank_of_observed_positives
- median_rank_of_observed_positives
- score_separation_ks: KS statistic between P and U score distributions
## Dataset Accounting
- Primary P = 108- Primary U = 53573- Primary eligible = 53681- Excluded sensitivity = 594- Out of scope = 114- Defensible negative = 0- Reconciliation: 108 + 53573 + 594 + 114 = 54389- Sensitivity P = 702- Sensitivity U = 53573- Sensitivity eligible = 54275- Reconciliation: 702 + 53573 + 114 = 54389### Per-Split Accounting
- train: P=65, S=279, U=15473, OOS=7, total=15824- validation: P=22, S=106, U=22048, OOS=26, total=22202- test: P=21, S=209, U=16052, OOS=21, total=16303## Results
### Primary Experiment Results
#### Mode E0
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15538 | 65 | 0.000644 | 0.04 | 7769.5 || validation | 22070 | 22 | 0.000453 | 0.01 | 11035.5 || test | 16073 | 21 | 0.000622 | 0.01 | 8037.0 |**random**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000322 | 0.02 | 0.0 || 10 | 0.0000 | 0.00 | 0.000644 | 0.04 | 0.0 || 20 | 0.0000 | 0.00 | 0.001287 | 0.08 | 0.0 || 50 | 0.0000 | 0.00 | 0.003218 | 0.21 | 0.0 || 100 | 0.0000 | 0.00 | 0.006436 | 0.42 | 0.0 || 200 | 0.0000 | 0.00 | 0.012872 | 0.84 | 0.0 || 500 | 0.0308 | 0.96 | 0.032179 | 2.09 | 2.0 |
Mean rank: 8272.2 (random E: 7769.5), Median rank: 8748.0, KS: 0.1030
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 10525.8 (random E: 11035.5), Median rank: 9681.0, KS: 0.1101
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0476 | 1.53 | 0.031108 | 0.65 | 1.0 |
Mean rank: 7791.6 (random E: 8037.0), Median rank: 7703.0, KS: 0.1157
**naive_biased**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8142
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.1 (random E: 11035.5), Median rank: 7030.5, KS: 0.4866
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6711.5 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
**pu_ranker**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8142
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.1 (random E: 11035.5), Median rank: 7030.5, KS: 0.4866
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6711.5 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
#### Mode E1
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15538 | 65 | 0.000644 | 0.04 | 7769.5 || validation | 22070 | 22 | 0.000453 | 0.01 | 11035.5 || test | 16073 | 21 | 0.000622 | 0.01 | 8037.0 |**random**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000322 | 0.02 | 0.0 || 10 | 0.0000 | 0.00 | 0.000644 | 0.04 | 0.0 || 20 | 0.0000 | 0.00 | 0.001287 | 0.08 | 0.0 || 50 | 0.0000 | 0.00 | 0.003218 | 0.21 | 0.0 || 100 | 0.0000 | 0.00 | 0.006436 | 0.42 | 0.0 || 200 | 0.0000 | 0.00 | 0.012872 | 0.84 | 0.0 || 500 | 0.0308 | 0.96 | 0.032179 | 2.09 | 2.0 |
Mean rank: 8272.2 (random E: 7769.5), Median rank: 8748.0, KS: 0.1030
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 10525.8 (random E: 11035.5), Median rank: 9681.0, KS: 0.1101
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0476 | 1.53 | 0.031108 | 0.65 | 1.0 |
Mean rank: 7791.6 (random E: 8037.0), Median rank: 7703.0, KS: 0.1157
**naive_biased**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8143
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.0 (random E: 11035.5), Median rank: 7030.5, KS: 0.4867
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6714.0 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
**pu_ranker**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8143
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.0 (random E: 11035.5), Median rank: 7030.5, KS: 0.4867
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6714.0 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
#### Mode E2
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15538 | 65 | 0.000644 | 0.04 | 7769.5 || validation | 22070 | 22 | 0.000453 | 0.01 | 11035.5 || test | 16073 | 21 | 0.000622 | 0.01 | 8037.0 |**random**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000322 | 0.02 | 0.0 || 10 | 0.0000 | 0.00 | 0.000644 | 0.04 | 0.0 || 20 | 0.0000 | 0.00 | 0.001287 | 0.08 | 0.0 || 50 | 0.0000 | 0.00 | 0.003218 | 0.21 | 0.0 || 100 | 0.0000 | 0.00 | 0.006436 | 0.42 | 0.0 || 200 | 0.0000 | 0.00 | 0.012872 | 0.84 | 0.0 || 500 | 0.0308 | 0.96 | 0.032179 | 2.09 | 2.0 |
Mean rank: 8272.2 (random E: 7769.5), Median rank: 8748.0, KS: 0.1030
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 10525.8 (random E: 11035.5), Median rank: 9681.0, KS: 0.1101
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0476 | 1.53 | 0.031108 | 0.65 | 1.0 |
Mean rank: 7791.6 (random E: 8037.0), Median rank: 7703.0, KS: 0.1157
**naive_biased**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8143
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.0 (random E: 11035.5), Median rank: 7030.5, KS: 0.4867
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6714.0 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
**pu_ranker**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.5 (random E: 7769.5), Median rank: 89.0, KS: 0.8143
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6525.0 (random E: 11035.5), Median rank: 7030.5, KS: 0.4867
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6714.0 (random E: 8037.0), Median rank: 5058.0, KS: 0.2168
#### Mode E4
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15538 | 65 | 0.000644 | 0.04 | 7769.5 || validation | 22070 | 22 | 0.000453 | 0.01 | 11035.5 || test | 16073 | 21 | 0.000622 | 0.01 | 8037.0 |**random**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000322 | 0.02 | 0.0 || 10 | 0.0000 | 0.00 | 0.000644 | 0.04 | 0.0 || 20 | 0.0000 | 0.00 | 0.001287 | 0.08 | 0.0 || 50 | 0.0000 | 0.00 | 0.003218 | 0.21 | 0.0 || 100 | 0.0000 | 0.00 | 0.006436 | 0.42 | 0.0 || 200 | 0.0000 | 0.00 | 0.012872 | 0.84 | 0.0 || 500 | 0.0308 | 0.96 | 0.032179 | 2.09 | 2.0 |
Mean rank: 8272.2 (random E: 7769.5), Median rank: 8748.0, KS: 0.1030
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 10525.8 (random E: 11035.5), Median rank: 9681.0, KS: 0.1101
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0476 | 1.53 | 0.031108 | 0.65 | 1.0 |
Mean rank: 7791.6 (random E: 8037.0), Median rank: 7703.0, KS: 0.1157
**naive_biased**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.4 (random E: 7769.5), Median rank: 89.0, KS: 0.8142
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6526.0 (random E: 11035.5), Median rank: 7027.0, KS: 0.4874
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6721.0 (random E: 8037.0), Median rank: 5056.0, KS: 0.2161
**pu_ranker**
*train* (N=15538, P=65):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0154 | 47.81 | 0.000322 | 0.02 | 1.0 || 10 | 0.0154 | 23.90 | 0.000644 | 0.04 | 1.0 || 20 | 0.0154 | 11.95 | 0.001287 | 0.08 | 1.0 || 50 | 0.2615 | 81.28 | 0.003218 | 0.21 | 17.0 || 100 | 0.5538 | 86.06 | 0.006436 | 0.42 | 36.0 || 200 | 0.6769 | 52.59 | 0.012872 | 0.84 | 44.0 || 500 | 0.7385 | 22.95 | 0.032179 | 2.09 | 48.0 |
Mean rank: 611.4 (random E: 7769.5), Median rank: 89.0, KS: 0.8142
*validation* (N=22070, P=22):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000227 | 0.00 | 0.0 || 10 | 0.0000 | 0.00 | 0.000453 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.000906 | 0.02 | 0.0 || 50 | 0.0000 | 0.00 | 0.002266 | 0.05 | 0.0 || 100 | 0.0000 | 0.00 | 0.004531 | 0.10 | 0.0 || 200 | 0.0000 | 0.00 | 0.009062 | 0.20 | 0.0 || 500 | 0.0000 | 0.00 | 0.022655 | 0.50 | 0.0 |
Mean rank: 6526.0 (random E: 11035.5), Median rank: 7027.0, KS: 0.4874
*test* (N=16073, P=21):
| K | recall@K | enrichment@K | E[random recall] | E[random count] | observed count ||---|---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000311 | 0.01 | 0.0 || 10 | 0.0000 | 0.00 | 0.000622 | 0.01 | 0.0 || 20 | 0.0000 | 0.00 | 0.001244 | 0.03 | 0.0 || 50 | 0.0000 | 0.00 | 0.003111 | 0.07 | 0.0 || 100 | 0.0000 | 0.00 | 0.006222 | 0.13 | 0.0 || 200 | 0.0000 | 0.00 | 0.012443 | 0.26 | 0.0 || 500 | 0.0000 | 0.00 | 0.031108 | 0.65 | 0.0 |
Mean rank: 6721.0 (random E: 8037.0), Median rank: 5056.0, KS: 0.2161
### Sensitivity Experiment Results
Sensitivity analysis includes COMMIT_ONLY path-associated observations in the positive set. Random baselines use sensitivity-specific N and P.
#### Mode E0
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15817 | 344 | 0.000632 | 0.22 | 7909.0 || validation | 22176 | 128 | 0.000451 | 0.06 | 11088.5 || test | 16282 | 230 | 0.000614 | 0.14 | 8141.5 |**train** (N=15817, P=344):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000316 | 0.11 || 10 | 0.0000 | 0.00 | 0.000632 | 0.22 || 20 | 0.0000 | 0.00 | 0.001264 | 0.43 || 50 | 0.0000 | 0.00 | 0.003161 | 1.09 || 100 | 0.0116 | 1.84 | 0.006322 | 2.17 || 200 | 0.2413 | 19.08 | 0.012645 | 4.35 || 500 | 0.3285 | 10.39 | 0.031612 | 10.87 |
Mean rank: 2128.7 (random E: 7909.0), KS: 0.6021
**validation** (N=22176, P=128):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000225 | 0.03 || 10 | 0.0000 | 0.00 | 0.000451 | 0.06 || 20 | 0.0000 | 0.00 | 0.000902 | 0.12 || 50 | 0.0000 | 0.00 | 0.002255 | 0.29 || 100 | 0.0000 | 0.00 | 0.004509 | 0.58 || 200 | 0.0000 | 0.00 | 0.009019 | 1.15 || 500 | 0.0000 | 0.00 | 0.022547 | 2.89 |
Mean rank: 10418.0 (random E: 11088.5), KS: 0.1943
**test** (N=16282, P=230):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000307 | 0.07 || 10 | 0.0000 | 0.00 | 0.000614 | 0.14 || 20 | 0.0000 | 0.00 | 0.001228 | 0.28 || 50 | 0.0000 | 0.00 | 0.003071 | 0.71 || 100 | 0.0000 | 0.00 | 0.006142 | 1.41 || 200 | 0.0000 | 0.00 | 0.012284 | 2.83 || 500 | 0.0000 | 0.00 | 0.030709 | 7.06 |
Mean rank: 10524.2 (random E: 8141.5), KS: 0.4213
#### Mode E2
**Analytic Random Baselines (per split):**
| Split | N | P | E[recall@10] | E[count@10] | E[mean_rank] ||---|---|---|---|---|---|| train | 15817 | 344 | 0.000632 | 0.22 | 7909.0 || validation | 22176 | 128 | 0.000451 | 0.06 | 11088.5 || test | 16282 | 230 | 0.000614 | 0.14 | 8141.5 |**train** (N=15817, P=344):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000316 | 0.11 || 10 | 0.0000 | 0.00 | 0.000632 | 0.22 || 20 | 0.0000 | 0.00 | 0.001264 | 0.43 || 50 | 0.0000 | 0.00 | 0.003161 | 1.09 || 100 | 0.0058 | 0.92 | 0.006322 | 2.17 || 200 | 0.2413 | 19.08 | 0.012645 | 4.35 || 500 | 0.3285 | 10.39 | 0.031612 | 10.87 |
Mean rank: 2128.0 (random E: 7909.0), KS: 0.6019
**validation** (N=22176, P=128):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000225 | 0.03 || 10 | 0.0000 | 0.00 | 0.000451 | 0.06 || 20 | 0.0000 | 0.00 | 0.000902 | 0.12 || 50 | 0.0000 | 0.00 | 0.002255 | 0.29 || 100 | 0.0000 | 0.00 | 0.004509 | 0.58 || 200 | 0.0000 | 0.00 | 0.009019 | 1.15 || 500 | 0.0000 | 0.00 | 0.022547 | 2.89 |
Mean rank: 10422.8 (random E: 11088.5), KS: 0.1953
**test** (N=16282, P=230):
| K | recall@K | enrichment@K | E[random recall] | E[random count] ||---|---|---|---|---|| 5 | 0.0000 | 0.00 | 0.000307 | 0.07 || 10 | 0.0000 | 0.00 | 0.000614 | 0.14 || 20 | 0.0000 | 0.00 | 0.001228 | 0.28 || 50 | 0.0000 | 0.00 | 0.003071 | 0.71 || 100 | 0.0000 | 0.00 | 0.006142 | 1.41 || 200 | 0.0000 | 0.00 | 0.012284 | 2.83 || 500 | 0.0000 | 0.00 | 0.030709 | 7.06 |
Mean rank: 10522.4 (random E: 8141.5), KS: 0.4208
### Seed Stability
High-variation configs (CV > 0.1):
- E0_random_train_score_separation_ks: CV=0.2703- E0_random_validation_mean_rank_of_observed_positives: CV=0.1346- E0_random_validation_score_separation_ks: CV=0.3427- E1_random_train_score_separation_ks: CV=0.2703- E1_random_validation_mean_rank_of_observed_positives: CV=0.1346- E1_random_validation_score_separation_ks: CV=0.3427- E2_random_train_score_separation_ks: CV=0.2703- E2_random_validation_mean_rank_of_observed_positives: CV=0.1346- E2_random_validation_score_separation_ks: CV=0.3427- E4_random_train_score_separation_ks: CV=0.2703- E4_random_validation_mean_rank_of_observed_positives: CV=0.1346- E4_random_validation_score_separation_ks: CV=0.3427### Repository Analysis
- Repos with observed positives: 19
- Top 5 share: 0.6852
- Gini coefficient: 0.5624
Concentration:
  - networkx: 38 (0.3519)
  - uvicorn: 13 (0.1204)
  - jinja2: 9 (0.0833)
  - sqlalchemy: 8 (0.0741)
  - click: 6 (0.0556)
## Practical Investigation Budget
K represents an engineering investigation budget: the number of highest-ranked files a developer would review. K=10 means reviewing 10 files; K=50 means 50 files; K=100 means 100 files; K=200 and K=500 represent broader triage efforts.
The key question is: does the learned ranking concentrate observed positives near the top more than random, across unseen data?
**Unseen test set (N=16073, P=21):**
| Budget (K) | Observed positives found | Random expected | Assessment ||---|---|---|---|| 10 | 0.0 | 0.0 | below random || 20 | 0.0 | 0.0 | below random || 50 | 0.0 | 0.1 | below random || 100 | 0.0 | 0.1 | below random || 200 | 0.0 | 0.3 | below random || 500 | 0.0 | 0.7 | below random |## Evidence Matrix
| Dimension | Result |
|---|---|
| Ranking vs random | See per-mode results |
| Validation behavior | See per-split results |
| Unseen-test behavior | See per-split results |
| Repository coverage | See repository analysis |
| Seed stability | See seed stability |
| Sensitivity to COMMIT_ONLY | See sensitivity experiment |
| Observed-positive concentration | See repository analysis |
| Limitations | SCAR untestable, small positive set, selection mechanism unknown |
## Limitations
1. SCAR is untestable with current evidence.
2. The positive set is small (108 primary observations).
3. The selection mechanism (fix-commit path association) is not validated as a proxy for true defect presence.
4. Elkan-Noto correction cannot be applied.
5. No defensible negatives exist for calibration.
6. True-defect precision, recall, ROC, and accuracy are NOT established by this experiment.
## Recommendation
The recommendation is derived from the following evidence:
| Evidence | Value ||---|---|| Train mean rank | 611.5 (random E: 7769.5) || Validation mean rank | 6525.1 (random E: 11035.5) || Test mean rank | 6711.5 (random E: 8037.0) || Train KS | 0.8142 || Validation KS | 0.4866 || Test KS | 0.2168 || Test recall@50 | 0.0000 || Test recall@100 | 0.0000 || Sensitivity (COMMIT_ONLY) | See sensitivity experiment above || Repository concentration | Top 5 repos = 68.5% || Elkan-Noto | NOT_IDENTIFIABLE_WITH_CURRENT_EVIDENCE |**Recommendation: PU_SIGNAL_INSUFFICIENT**
The observed-positive ranking shows in-sample (train) separation from random (mean rank significantly below random expectation), but this does not generalize to validation or test sets. Recall@50 and recall@100 on the unseen test set are zero or near zero. The in-sample signal may reflect overfitting to the small positive set (P=65 in train) rather than a generalizable pattern. The observed-positive ranking signal is insufficient for reliable unseen-repository risk ranking.

This experiment does NOT establish true-defect prediction capability. It only evaluates observed-positive ranking above random. Calibrated defect probabilities, true-defect precision/recall, and production readiness are NOT established.
**Important**: This recommendation addresses observed-positive ranking only. It does NOT claim: calibrated defect probability, true-defect precision/recall, production readiness, or quantum readiness.
## Frozen Data Integrity
- Phase 4.6-4.9 code and artifacts: UNCHANGED
- Combined-v3 JSONL dataset: UNCHANGED
- repo_split.py: UNCHANGED
- diff_parser.py: UNCHANGED
- Experimental feature generation: UNCHANGED
