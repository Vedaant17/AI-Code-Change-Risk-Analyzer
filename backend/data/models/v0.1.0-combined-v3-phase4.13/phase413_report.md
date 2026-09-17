# Phase 4.13: File-Level Investigation-Priority Ranking

## 1. Research Question

Given 499 files with direct Git content evidence of defect association
(FILE_STRONG), 238 files with weak evidence (FILE_WEAK), 314 files with
unresolved identity (UNKNOWN), and zero defensible negatives, can we build
a file-level ranking model that improves upon random investigation ordering
when triaging new commits?

## 2. Hypothesis

Files in a new commit that share structural/statistical properties with
the 499 FILE_STRONG files will be ranked higher by the model than files
that do not, and this ranking will generalize to unseen repositories
better than random ordering.

## 3. Supervision Populations

| Population | Count |
|-----------|-------|
| OBSERVED_POSITIVE | 499 |
| EVIDENCE_WEAK | 238 |
| EVIDENCE_UNKNOWN | 314 |
| UNLABELED | 53224 |
| OUT_OF_SCOPE | 114 |
| TOTAL | 54389 |

## 4. Pre-Experiment Audits

### 4.1 FILE_STRONG Split Distribution

| Split | Rows | Commits | Repos |
|-------|------|---------|-------|
| train | 187 | 37 | 14 |
| validation | 102 | 43 | 14 |
| test | 210 | 45 | 12 |
| total | 499 | 125 | 40 |

### 4.2 STRONG_vs_WEAK Pair Audit

| Split | Pairs | Commits | Repos |
|-------|-------|---------|-------|
| train | 50 | 9 | 6 |
| validation | 112 | 11 | 7 |
| test | 28 | 9 | 8 |

## 5. Results

### B1 Behavior

B1 ranks files by `file_total_lines_changed` (index 34) descending.
This is a simple file-level change-size heuristic.

### B4 Behavior

B4 assigns the same commit-level score to all files in a commit. Its
within-commit file ordering is determined entirely by the deterministic
tie-breaking policy. B4 is a commit-level fallback and does not provide
file-level discrimination within a commit.

### Mode E0

#### B0_random
- MRR: 0.9204
- Commits with positives: 45
- Coverage: 0.0065

#### B1_change_size
- MRR: 0.9519
- Commits with positives: 45
- Coverage: 0.0065

#### B2_biased_pu
- MRR: 0.9365
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=-0.0124, CI=[-0.2529, 0.2500]
- enrichment_at_10: diff=-0.0240, CI=[-0.0464, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0684, CI=[-0.0851, 0.2190]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0347, CI=[-0.1419, 0.0815]
- enrichment_at_5: diff=-0.0674, CI=[-0.1333, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0149, CI=[-0.0382, 0.0718]

#### B3_evidence_ranking
- MRR: 0.9778
- Commits with positives: 45
- Coverage: 0.0065
- Pairwise accuracy: 0.5000
- enrichment_at_1: diff=0.1827, CI=[-0.0923, 0.4643]
- enrichment_at_10: diff=-0.0120, CI=[-0.0232, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.1187, CI=[0.0192, 0.2037]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0568, CI=[0.0000, 0.1090]

#### B4_commit_level
- MRR: 0.9407
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=0.0491, CI=[-0.0741, 0.1849]
- enrichment_at_10: diff=-0.0097, CI=[-0.0222, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0754, CI=[-0.0030, 0.1423]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0201, CI=[-0.0172, 0.0595]

### Mode E1

#### B0_random
- MRR: 0.9204
- Commits with positives: 45
- Coverage: 0.0065

#### B1_change_size
- MRR: 0.9519
- Commits with positives: 45
- Coverage: 0.0065

#### B2_biased_pu
- MRR: 0.9365
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=-0.0124, CI=[-0.2529, 0.2500]
- enrichment_at_10: diff=-0.0240, CI=[-0.0464, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0684, CI=[-0.0851, 0.2190]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0347, CI=[-0.1419, 0.0815]
- enrichment_at_5: diff=-0.0674, CI=[-0.1333, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0149, CI=[-0.0382, 0.0718]

#### B3_evidence_ranking
- MRR: 0.9778
- Commits with positives: 45
- Coverage: 0.0065
- Pairwise accuracy: 0.5357
- enrichment_at_1: diff=0.1659, CI=[-0.1013, 0.4483]
- enrichment_at_10: diff=-0.0120, CI=[-0.0232, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.1018, CI=[-0.0030, 0.1875]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0567, CI=[0.0055, 0.1063]

#### B4_commit_level
- MRR: 0.9407
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=0.0491, CI=[-0.0741, 0.1849]
- enrichment_at_10: diff=-0.0097, CI=[-0.0222, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0754, CI=[-0.0030, 0.1423]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0201, CI=[-0.0172, 0.0595]

### Mode E2

#### B0_random
- MRR: 0.9204
- Commits with positives: 45
- Coverage: 0.0065

#### B1_change_size
- MRR: 0.9519
- Commits with positives: 45
- Coverage: 0.0065

#### B2_biased_pu
- MRR: 0.9378
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=-0.0124, CI=[-0.2529, 0.2500]
- enrichment_at_10: diff=-0.0120, CI=[-0.0232, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0603, CI=[-0.0661, 0.1964]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0513, CI=[-0.1422, 0.0556]
- enrichment_at_5: diff=-0.0481, CI=[-0.0928, 0.0000]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0162, CI=[-0.0362, 0.0718]

#### B3_evidence_ranking
- MRR: 0.9296
- Commits with positives: 45
- Coverage: 0.0065
- Pairwise accuracy: 0.7143
- enrichment_at_1: diff=-0.0878, CI=[-0.2619, 0.1296]
- enrichment_at_10: diff=-0.0217, CI=[-0.0444, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0986, CI=[0.0186, 0.1681]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0034, CI=[-0.0684, 0.0651]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0086, CI=[-0.0278, 0.0449]

#### B4_commit_level
- MRR: 0.9407
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=0.0491, CI=[-0.0741, 0.1849]
- enrichment_at_10: diff=-0.0097, CI=[-0.0222, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0754, CI=[-0.0030, 0.1423]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0201, CI=[-0.0172, 0.0595]

### Mode E4

#### B0_random
- MRR: 0.9204
- Commits with positives: 45
- Coverage: 0.0065

#### B1_change_size
- MRR: 0.9519
- Commits with positives: 45
- Coverage: 0.0065

#### B2_biased_pu
- MRR: 0.9378
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=-0.0124, CI=[-0.2529, 0.2500]
- enrichment_at_10: diff=-0.0120, CI=[-0.0232, 0.0000]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0603, CI=[-0.0661, 0.1964]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0513, CI=[-0.1422, 0.0556]
- enrichment_at_5: diff=-0.0481, CI=[-0.0928, 0.0000]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0162, CI=[-0.0362, 0.0718]

#### B3_evidence_ranking
- MRR: 0.9296
- Commits with positives: 45
- Coverage: 0.0065
- Pairwise accuracy: 0.7143
- enrichment_at_1: diff=-0.0878, CI=[-0.2619, 0.1296]
- enrichment_at_10: diff=-0.0217, CI=[-0.0444, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0986, CI=[0.0186, 0.1681]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=-0.0034, CI=[-0.0684, 0.0651]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0086, CI=[-0.0278, 0.0449]

#### B4_commit_level
- MRR: 0.9407
- Commits with positives: 45
- Coverage: 0.0065
- enrichment_at_1: diff=0.0491, CI=[-0.0741, 0.1849]
- enrichment_at_10: diff=-0.0097, CI=[-0.0222, 0.0045]
- enrichment_at_100: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_2: diff=0.0754, CI=[-0.0030, 0.1423]
- enrichment_at_20: diff=0.0000, CI=[0.0000, 0.0000]
- enrichment_at_3: diff=0.0054, CI=[-0.0757, 0.0815]
- enrichment_at_5: diff=-0.0434, CI=[-0.0889, 0.0090]
- enrichment_at_50: diff=0.0000, CI=[0.0000, 0.0000]
- mrr: diff=0.0201, CI=[-0.0172, 0.0595]


### Bootstrap Summary

Of 108 model-vs-random bootstrap comparisons, 4 exclude zero:

- E0 B3_evidence_ranking enrichment_at_2: diff=0.1187, CI=[0.0192, 0.2037]
- E1 B3_evidence_ranking mrr: diff=0.0567, CI=[0.0055, 0.1063]
- E2 B3_evidence_ranking enrichment_at_2: diff=0.0986, CI=[0.0186, 0.1681]
- E4 B3_evidence_ranking enrichment_at_2: diff=0.0986, CI=[0.0186, 0.1681]

E2 and E4 produce identical bootstrap results for enrichment@2 (same underlying test set with shared historical features). These findings should be treated as **exploratory** (no multiple-comparison correction). The positive enrichment@2 aggregate is concentrated in 3 repos: click (+0.650, 5 positive commits), nox (+0.538, 1 positive commit), and celery (+0.063, 12 positive commits), which together account for 18/45 positive test commits. 9 of 12 repos with positive commits show zero diff. The result may not generalize.

## 6. Limitations

1. No defensible negatives exist. The model cannot estimate true defect probability.
2. FILE_STRONG represents labeling pipeline coverage, not confirmed defectives.
3. The dataset only includes commits flagged as potentially bug-fixing.
4. The model ranks files within commits, not across commits.
5. Temporal censoring: observation endpoint is the latest commit per repo.
6. Repository selection bias: 50 non-random repositories.
7. AST features are Python-only.
8. PU class prior is not identifiable (Phase 4.10).
9. SCAR is untestable.
10. Score is not calibrated to any external frequency.

## 7. Final Scientific Claim

The model can attempt to rank files according to their similarity to the
observed FILE_STRONG evidence population using pre-candidate features.
It cannot establish true defect probability, defect prevalence, defect-free
probability, or whether a particular file is actually defective.

The correct framing is investigation prioritization, not defect prediction.

## 8. SageMaker / Braket Status

SageMaker: BLOCKED until classical baseline demonstrates useful signal.
Braket: BLOCKED until a specific quantum formulation is justified.

## 9. Final Status

NO_DEFENSIBLE_MODEL_SELECTED. No model demonstrates sufficient
signal to justify deployment.

B1 = simple file-level change-size heuristic (file_total_lines_changed).
B2 = biased P-vs-U XGBoost model.
B3 = evidence-ranking pairwise logistic model.
B4 = commit-level fallback with no genuine within-commit file discrimination.

B3 achieves the highest MRR in modes E0/E1 (0.9778 vs B1 0.9519),
but B3 E2/E4 MRR (0.9296) falls below B1. The overall comparison
depends on the feature mode. 4 of 27 model-vs-random bootstrap
comparisons exclude zero (see Bootstrap Summary). These findings
should be treated as exploratory (no multiple-comparison correction).
Phase 4.13 is complete but no model is recommended for deployment.
