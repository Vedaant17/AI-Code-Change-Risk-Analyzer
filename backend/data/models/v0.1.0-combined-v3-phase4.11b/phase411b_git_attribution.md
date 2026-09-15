# Phase 4.11b: Repository-Backed Git-History Attribution Feasibility

## Executive Summary

This study investigates whether repository-level Git content evidence materially improves file-level defect attribution beyond the Phase 4.11a JSONL-only baseline.

**Result**: 499 FILE_STRONG, 0 FILE_MODERATE, 238 FILE_WEAK, 318 UNKNOWN across 1055 candidate files.

## Methodology

Git objects accessed via immutable inspection (git show, git diff). No checkout used for core analysis.

### Evidence Hierarchy

| Level | Name | Requirement |
|---|---|---|
| COMMIT_STRONG | Strong commit-level | Corrective SHA resolved |
| FILE_STRONG | Strong file-level | Exact content removal or restoration |
| FILE_MODERATE | Moderate file-level | Content correspondence + structural relationship |
| FILE_WEAK | Weak file-level | Path overlap or file association only |
| UNKNOWN | Unknown | File identity not established |

## A. Commit-Level Evidence

- COMMIT_STRONG: 210 (100.0%)
- COMMIT_WEAK: 0 (0.0%)
- COMMIT_NONE: 0 (0.0%)

## B. File-Level Evidence

| Evidence Level | Files | Percentage |
|---|---|---|
| FILE_STRONG | 499 | 47.3% |
| FILE_MODERATE | 0 | 0.0% |
| FILE_WEAK | 238 | 22.6% |
| UNKNOWN | 318 | 30.1% |

### Baseline Comparison

- Phase 4.8: 38 FILE_PARTIAL, 0 FILE_EXACT
- Phase 4.11a: 185 FILE_WEAK, 0 FILE_MODERATE, 0 FILE_STRONG
- Phase 4.11b: 499 FILE_STRONG, 0 FILE_MODERATE, 238 FILE_WEAK

### Evidence Type Distribution

- CONTENT_RESTORATION: 420
- EXACT_CANDIDATE_CONTENT_REMOVAL: 479
- SAME_FUNCTION_CANDIDATE_CONTENT: 95
- SAME_FUNCTION_ONLY: 421
- SAME_REGION_OVERLAP: 620

## C. Function/Hunk Feasibility

- Python files analyzed: 516
- Function-attributable: Requires AST parse + content correspondence

## D. Negative Feasibility

- Candidate negatives: 107
- Defensible negatives: 0
- Conclusion: No defensible negatives constructed. All proposed negatives remain CANDIDATE_NEGATIVE requiring independent justification.

## E. Prediction Unit Analysis

### Commit Level

- Label availability: high
- Attribution quality: STRONG commit-level
- Practical usefulness: Limited — identifies risky commits but not which files within the commit

### File Level

- Label availability: medium
- Attribution quality: STRONG=499, MODERATE=0, WEAK=238
- Practical usefulness: High potential if MODERATE/STRONG evidence exists

### Function Level

- Label availability: limited
- Attribution quality: Requires AST parse + content correspondence
- Practical usefulness: Most actionable for developers if available

### Hunk Level

- Label availability: unavailable
- Attribution quality: UNAVAILABLE without diff analysis
- Practical usefulness: Most precise but hardest to act on

## F. Repository Execution

- Repos required: 42
- Repos cloned: 42
- Object failures: 0

## Final Decision

### MATERIAL_IMPROVEMENT_OBSERVED

Git content evidence produced 499 FILE_STRONG and 0 FILE_MODERATE cases, materially improving beyond Phase 4.11a's 0/0 baseline.

### Key Distinction

**Git evidence exists** does not mean **Git evidence is strong enough to justify file-level supervision.**

Content similarity between candidate and corrective changes is reported as evidence. It is not equated with causal attribution.

## Limitations

1. All file-level evidence is based on Git content analysis. No independent validation.

2. Function attribution is limited to Python files with successful AST parsing.

3. No defensible negatives were constructed.

4. Similarity ratios are descriptive, not causal.

## Frozen Data Integrity

- Phase 4.6–4.11a code and artifacts: UNCHANGED

- Combined-v3 JSONL dataset: UNCHANGED

- No repositories were modified
