# Phase 4.12: Defensible Supervision & Negative-Label Construction Feasibility

## 1. Research Question

Given Phase 4.11b results (499 OBSERVED_POSITIVE, 238 EVIDENCE_WEAK, 318 EVIDENCE_UNKNOWN, 0 defensible negatives), determine which supervision strategy can produce a scientifically defensible file-level risk model.

## 2. Frozen Inputs

- Frozen supervised JSONL (train/validation/test)
- Phase 4.6 REPO_MANIFEST from repo_split.py
- Phase 4.11b file_attribution_results.json
- Phase 4.11b per_commit_git_evidence.jsonl
- Raw commit_timestamps from dataset JSONL

## 3. Population Accounting

| Population | Count |
|------------|-------|
| Total supervised rows | 54389 |
| Historical defect_label=1 | 1055 |
| OBSERVED_POSITIVE | 499 |
| EVIDENCE_MODERATE | 0 |
| EVIDENCE_WEAK | 238 |
| EVIDENCE_UNKNOWN | 314 |
| UNLABELED | 53224 |
| OUT_OF_SCOPE | 114 |
| Date range | ['2006-03-19T15:42:44Z', '2026-09-07T07:24:57Z'] |

**Note:** historical_positive_rows (defect_label=1) and observed_positive_rows (FILE_STRONG) are different concepts and are NOT asserted to be equal.

### Positive Commit / Repository Coverage

| Metric | Count |
|--------|-------|
| Historical positive commits (defect_label=1) | 210 |
| FILE_STRONG commits | 125 |
| Historical positive repos | 42 |
| Repos with FILE_STRONG | 40 |

**Disjointness:** PASS

## 4. Leakage Audit

| Check | Result |
|-------|--------|
| Leakage-free features | True |
| label_source excluded | True |
| Overall | PASS |

## 5. Strategy Assessments

### PU_LEARNING
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: PROCEED_WITH_CAVEATS
- Reason: PU ranking is identifiable but class prior and SCAR are not. Proceed with observed-positive ranking only, not prevalence estimation.

### RELIABLE_NEGATIVES
- Identifiability: NOT_IDENTIFIABLE
- Recommendation: DO_NOT_PROCEED
- Reason: No criterion produces defensible negatives with available evidence. All produce candidate negatives only.

### TEMPORAL_NEGATIVES
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: FEASIBILITY_ONLY
- Reason: Observation windows are computable from available timestamps. However, censoring is informative and survival analysis is confounded by selection bias. Temporal survival is exposure/censoring analysis only, not negative-label evidence.

### REPOSITORY_CONTROLS
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: PROCEED_WITH_CAVEATS
- Reason: Matched controls reduce confounding but are NOT negative labels. They may serve as ranking references or evaluation strata. Selection bias and residual confounding remain.

### CORRECTIVE_COMMIT_CONTROLS
- Identifiability: NOT_IDENTIFIABLE
- Recommendation: DO_NOT_PROCEED
- Reason: Files in corrective commits without strong attribution are candidate controls/unlabeled, NOT negative labels. Absence of strong attribution evidence does not establish absence of defect.

### EVIDENCE_RANKING
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: PROCEED_WITH_CAVEATS
- Reason: Evidence-ranking pairs are constructible from available data with verified identity. The partial ordering is evidence-based, not true-defect-based. Proceed with explicit caveats about evidence semantics.

### WEAK_SUPERVISION
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: FEASIBILITY_ONLY
- Reason: Weak signals exist but are correlated. Dependency-aware label modeling is possible in principle but requires explicit implementation of the implication structure.

### SYNTHETIC_NEGATIVES
- Identifiability: NOT_IDENTIFIABLE
- Recommendation: DO_NOT_PROCEED
- Reason: Domain shift between synthetic and real examples makes synthetic negatives unreliable for training. No synthetic dataset should be generated.

### COMMIT_LEVEL_FALLBACK
- Identifiability: PARTIALLY_IDENTIFIABLE
- Recommendation: PROCEED_WITH_CAVEATS
- Reason: Commit-level labels are more reliable than file-level (less attribution ambiguity). However, commit risk is not file risk. A commit may introduce a defect in one file while others are clean. This is a valid fallback but does not solve file-level attribution.

## 6. Decision Matrix

| Strategy | Identifiable | Negative Labels | Leakage Risk | Recommendation |
|----------|-------------|----------------|-------------|----------------|
| PU_LEARNING | PARTIALLY_IDENTIFIABLE | 0 | NONE | PROCEED_WITH_CAVEATS |
| RELIABLE_NEGATIVES | NOT_IDENTIFIABLE | 0 | NONE | DO_NOT_PROCEED |
| TEMPORAL_NEGATIVES | PARTIALLY_IDENTIFIABLE | 0 | NONE | FEASIBILITY_ONLY |
| REPOSITORY_CONTROLS | PARTIALLY_IDENTIFIABLE | 0 | NONE | PROCEED_WITH_CAVEATS |
| CORRECTIVE_COMMIT_CONTROLS | NOT_IDENTIFIABLE | 0 | NONE | DO_NOT_PROCEED |
| EVIDENCE_RANKING | PARTIALLY_IDENTIFIABLE | 0 | NONE | PROCEED_WITH_CAVEATS |
| WEAK_SUPERVISION | PARTIALLY_IDENTIFIABLE | 0 | NONE | FEASIBILITY_ONLY |
| SYNTHETIC_NEGATIVES | NOT_IDENTIFIABLE | 0 | NONE | DO_NOT_PROCEED |
| COMMIT_LEVEL_FALLBACK | PARTIALLY_IDENTIFIABLE | 0 | NONE | PROCEED_WITH_CAVEATS |

## 7. Limitations

- No defensible negatives exist with current evidence
- Evidence-level labels (FILE_STRONG, FILE_WEAK, UNKNOWN) are outcome-derived and excluded from prediction-time features
- PU class prior is not identifiable
- Temporal survival is confounded by informative censoring
- Candidate negatives are NOT true negatives

## 8. Final Recommendation

The decision matrix determines the recommended Phase 4.13 strategy.
If all strategies are insufficient, DO_NOT_PROCEED with a list of what additional evidence or data collection is needed.

## 9. Braket Status

Braket remains blocked until a scientifically defensible risk-ranking formulation exists.