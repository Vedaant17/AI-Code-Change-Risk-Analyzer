# Phase 4.9: Supervision Strategy & PU Learning Feasibility Study

## Executive Summary

This study evaluates whether the frozen Phase 4.8 file-level evidence supports a future Positive-Unlabeled (PU) learning experiment. The analysis identifies 108 observed-path-associated rows from FILE_PARTIAL commits, with 594 sensitivity rows, 53573 unlabeled rows, 0 defensible negatives, and 114 out-of-scope parser artifacts.

**Primary Recommendation: PU_LEARNING_FEASIBLE_WITH_RESTRICTIONS**

Some concerns identified: Many feature distribution differences (14); SCAR not established. PU learning is feasible with appropriate restrictions and sensitivity analyses.

---

## OBSERVED FACTS

### Dataset Scope

- Total supervised file rows: 54389
- Observed path-associated rows: 108
- Sensitivity (COMMIT_ONLY overlap) rows: 594
- Unlabeled rows: 53573
- Defensible negatives: 0
- Out-of-scope (parser artifacts): 114
- Observed positive rate: 0.2%

### Evidence Populations

- **OBSERVED_PATH_ASSOCIATED**: FILE_PARTIAL path-associated rows
  only. Candidate positives for PU formulation.
- **UNLABELED**: Legitimate file rows without file-level defect
  association evidence.
- **SENSITIVITY (COMMIT_ONLY)**: Path-overlap rows from COMMIT_ONLY
  commits. Weaker evidence. Not included in primary PU population.
- **DEFENSIBLE_NEGATIVE**: None. No independent negative evidence.
- **OUT_OF_SCOPE**: Parser artifacts (file_path == '/dev/null').
  Binary file additions where the diff parser captured the source
  side instead of the destination path. Not usable for supervision.

---

## IMPLEMENTATION FACTS

The analysis:

1. Reads frozen JSONL files (train/validation/test/ambiguous).
2. Parses REPO_MANIFEST from frozen repo_split.py source.
3. Classifies all 210 positive commits by evidence category.
4. Constructs supervision sets from classification results.
5. Analyzes coverage across feature dimensions.
6. Measures repository concentration.
7. Reports split distribution.
8. Evaluates SCAR/SAR assumptions cautiously.
9. Verifies negative-label feasibility.
10. Compares five supervision strategies.
11. Designs evaluation protocol.
12. Identifies data requirements.
13. Derives recommendation from evidence matrix.

---

## INFERENCES

### Coverage Analysis

- files_changed: observed median 14.5000 vs unlabeled median 4.0000
- total_hunks: observed median 29.0000 vs unlabeled median 6.0000
- max_file_changes: observed median 66.0000 vs unlabeled median 23.0000
- additions_ratio: observed median 0.7009 vs unlabeled median 0.5455
- test_ratio: observed median 0.3158 vs unlabeled median 0.0000
- repo_name: click: observed share 0.056 vs unlabeled share 0.014
- repo_name: sqlalchemy: observed share 0.074 vs unlabeled share 0.017
- repo_name: uvicorn: observed share 0.120 vs unlabeled share 0.020
- repo_name: networkx: observed share 0.352 vs unlabeled share 0.022
- repo_name: jinja2: observed share 0.083 vs unlabeled share 0.022

### Repository Concentration

- Repos with signal: 19/50
- Top-5 share: 68.5%
- Gini: 0.834
- Herfindahl: 0.1636
- Entropy: 0.789

### Split Distribution

| Split | Obs | Sens | Unl | OOS | Raw | Class | Rec | Rate |
|-------|-----|------|-----|-----|-----|-------|-----|------|
| train | 86 | 350 | 33874 | 95 | 34405 | 34405 | Yes | 0.2% |
| validation | 12 | 80 | 9310 | 8 | 9410 | 9410 | Yes | 0.1% |
| test | 10 | 164 | 10389 | 11 | 10574 | 10574 | Yes | 0.1% |

---

## UNTESTABLE ASSUMPTIONS

### SCAR

**Verdict: UNTESTABLE_WITH_CURRENT_EVIDENCE**

The deterministic construction of FILE_PARTIAL establishes that selection is mechanism-dependent on corrective-commit scope. However, this does not by itself prove that SCAR is violated with respect to the latent defect outcome. The relevant components remain untestable without external ground truth. SCAR is not established from current evidence, but formal violation cannot be claimed.

### SAR

**Verdict: COMPATIBLE_WITH_CAVEATS**

SAR is the weaker assumption needed for PU learning. The current evidence does not establish strong violation, but repository concentration and feature distribution differences are relevant indicators of possible selection dependence. If pursued, the PU experiment should include sensitivity analyses that test robustness to these concerns.

---

## SUPERVISION OPTIONS

### Standard binary supervised classification

- **Compatibility**: NOT_COMPATIBLE
- **Pursue next**: No
- **Risk**: Zero defensible negatives. Treating unlabeled as negative would introduce massive label noise. Model would learn the labeling mechanism, not defects.
- **Reasoning**: The dataset contains zero defensible negatives. Standard binary classification requires reliable negative labels. Not viable with current evidence.

### PU learning (positive-unlabeled)

- **Compatibility**: CONDITIONALLY_COMPATIBLE
- **Pursue next**: Yes
- **Risk**: Small observed-positive set. Possible repository-dependent selection. Path-associated evidence is not ground-truth defect labeling. PU assumptions may not hold.
- **Reasoning**: PU learning is the only viable approach given zero defensible negatives. The primary observed population is small (10.2% rate) but non-trivial. SAR is the working assumption with caveats. Restrictions apply.

### Weak supervision / heuristic labeling

- **Compatibility**: PARTIALLY_COMPATIBLE
- **Pursue next**: No
- **Risk**: Heuristic labels may correlate with commit patterns rather than defects. May amplify existing selection bias.
- **Reasoning**: Weak supervision could complement PU learning but should not be the primary approach. Could be explored as a secondary signal source.

### Commit-level prediction

- **Compatibility**: COMPATIBLE
- **Pursue next**: Yes
- **Risk**: Loses file-level granularity. Not directly useful for targeted code review. Commit-level prediction may not translate to file-level utility.
- **Reasoning**: Commit-level prediction is viable with current data. Binary labels are clear. Could serve as an intermediate step before file-level PU learning.

### Improved file-level attribution followed by supervised learning

- **Compatibility**: NOT_CURRENTLY_COMPATIBLE
- **Pursue next**: No
- **Risk**: High implementation cost. May not be feasible for all repos. External data access required.
- **Reasoning**: Improved attribution would provide stronger supervision but requires capabilities beyond the frozen JSONL data. This is a future-work direction, not the next step.

---

## EVALUATION FEASIBILITY

### Measurable Now

- Observed-positive enrichment: Compare the distribution of model scores for observed-path-associated rows vs unlabeled rows.
- Observed-positive precision@K: Among the top-K highest-scored files, what fraction are observed-path-associated. This is an exploratory metric against observed evidence, not true defect ground truth.
- Ranking stability: Assess whether model rankings are stable across random seeds and data perturbations.
- Score distribution diagnostics: Examine the distribution of model scores for observed vs unlabeled populations. Bimodality, separation, etc.
- Repository-held-out ranking diagnostics: Train on N-1 repos, evaluate on held-out repo. Assess cross-repository generalization.
- Split-level stability: Compare observed-positive enrichment across train/validation/test splits.
- Sensitivity analysis: Vary the observed-positive population definition (e.g., include/exclude sensitivity population) and measure stability of conclusions.

### Not Currently Measurable

- True recall: No defensible negatives or independent defect ground truth.
- True specificity: No defensible negatives available.
- True binary precision: No defensible negatives for binary evaluation.
- True defect PR-AUC: Only observed-path-associated and unlabeled available.
- True defect ROC-AUC: Only observed-path-associated and unlabeled available.
- Complete defect-ranking recall: No independent defect inventory exists.

### Exploratory PU Metrics (clearly labeled)

- PU-adjusted ROC-AUC (exploratory): This metric conflates true positive rate with selection mechanism. It should NOT be interpreted as defect classification performance.
- PU-adjusted PR-AUC (exploratory): Precision is inflated because unlabeled includes unknown positives. Should NOT be interpreted as true defect precision.

---

## ADDITIONAL DATA REQUIREMENTS

### Line-level blame/history

- **Feasibility**: FEASIBLE
- **Impact**: HIGH
- **Limitations**: Requires repository checkouts. Blame accuracy depends on merge history. May not work for squash-merged commits.

### Issue-to-file linkage

- **Feasibility**: FEASIBLE
- **Impact**: HIGH
- **Limitations**: Not all bugs have linked issues. Issue quality varies. May require API access to issue trackers.

### Patch-level causal evidence

- **Feasibility**: FEASIBLE
- **Impact**: HIGH
- **Limitations**: Requires access to git diffs. Patches may be incomplete or span multiple concerns.

### Independently reviewed labels

- **Feasibility**: FEASIBLE
- **Impact**: VERY HIGH
- **Limitations**: Requires manual effort. Inter-annotator agreement may vary. Expensive to scale.

### Structured bug-fix/file mappings

- **Feasibility**: FEASIBLE
- **Impact**: MEDIUM
- **Limitations**: Quality depends on commit message conventions. Not uniformly available across repos.

### Manually validated samples

- **Feasibility**: FEASIBLE
- **Impact**: HIGH
- **Limitations**: Small sample size limits statistical power. Validation is retrospective, not prospective.

---

## EVIDENCE MATRIX

- **observed_positive_quantity**: 108 observed-path-associated rows (0.2% of total file rows).
- **repository_diversity**: 19/50 repos contain signal. Top-5 share: 68.5%. Gini: 0.834.
- **split_coverage**: 3/3 splits contain observed signal.
- **feature_distribution**: 14 feature distribution differences noted.
- **selection_dependence**: SCAR: UNTESTABLE_WITH_CURRENT_EVIDENCE. SAR: COMPATIBLE_WITH_CAVEATS. Repository-dependence: medium.
- **attribution_strength**: Path-level evidence only. No content-level or line-level attribution available.
- **negative_availability**: 0 defensible negatives available.
- **evaluation_feasibility**: 7 measurable-now diagnostics identified. True defect metrics not measurable.
- **reproducibility**: Analysis is deterministic and auditable.

---

## RECOMMENDATION

**PU_LEARNING_FEASIBLE_WITH_RESTRICTIONS**

Some concerns identified: Many feature distribution differences (14); SCAR not established. PU learning is feasible with appropriate restrictions and sensitivity analyses.

### Restrictions

- Many feature distribution differences (14)
- SCAR not established

---

## LIMITATIONS

1. Path-level evidence only. No content-level analysis.
2. 108 observed-path-associated rows is a small sample.
3. No defensible negatives. PU learning required.
4. SCAR not established. SAR is working assumption with caveats.
5. Possible repository-dependent selection.
6. Evaluation limited to observed-positive diagnostics.
7. True defect performance cannot be measured.

---

## NEXT STEP

Proceed with PU learning experiment with explicit restrictions. Include sensitivity analysis across repository subsets and population definitions.

---

## Frozen Data Integrity

- No frozen dataset files were modified.
- No frozen labeling logic was modified.
- No frozen Phase 4.6/4.7/4.8 artifacts were modified.
- No dataset regeneration was performed.
