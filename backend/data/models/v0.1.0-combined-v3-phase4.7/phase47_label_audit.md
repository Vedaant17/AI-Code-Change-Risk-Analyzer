# Phase 4.7: Label & Attribution Distribution Audit

## 1. Executive Finding

The frozen combined-v3 dataset contains **210 positive commits** and **0 mixed commits**.

**100% of positive commits** use commit-level attribution (explicit SHA reference or revert), which labels **every changed file** as positive.

**0 commits** use file-level attribution (line overlap), which can distinguish implicated files.

**The dataset provides no supervision for file-level risk ranking within commits.** The product objective requires mixed commits where some files are positive and others are negative.

**Assessment: NOT SUITABLE** for training/evaluating file-level risk ranking.

---

## 2. Positive Commit Composition

### Overall

| Metric | Value |
|--------|-------|
| Total commits | 21,791 |
| Positive commits | 210 |
| Negative commits | 21,581 |
| Ambiguous commits | 0 |
| All-positive commits | 210 (100.0%) |
| Mixed commits | 0 (0.0%) |
| Total rows | 54,389 |
| Positive rows | 1,055 |
| Negative rows | 53,334 |
| Ambiguous rows | 0 |

### Positive Commit File Distribution

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Total files | 5.0 | 2.0 | 1 | 110 |
| Positive files | 5.0 | 2.0 | 1 | 110 |
| Positive ratio | 1.000 | 1.000 | 1.000 | 1.000 |

### Per-Split Breakdown

| Split | Commits | Pos Commits | All-Pos | Mixed | All-Pos % | Mixed % |
|-------|---------|-------------|---------|-------|-----------|---------|
| train | 15,670 | 111 | 111 | 0 | 100.0% | 0.0% |
| validation | 2,952 | 42 | 42 | 0 | 100.0% | 0.0% |
| test | 3,169 | 57 | 57 | 0 | 100.0% | 0.0% |

---

## 3. Attribution-Source Analysis

| Source | Pos Commits | Pos File Rows | All-Pos | Mixed | Granularity |
|--------|-------------|---------------|---------|-------|-------------|
| explicit_sha_reference | 114 | 516 | 114 | 0 | commit |
| revert | 96 | 539 | 96 | 0 | commit |
| line_overlap | 0 | 0 | 0 | 0 | file |
| none | 0 | 0 | 0 | 0 | n/a |
| ambiguous | 0 | 0 | 0 | 0 | n/a |

**Granularity summary:** 210/210 positive commits (100%) use commit-level attribution.

---

## 4. File-Level Label Construction Analysis

### A. Implementation Fact: What the Frozen Code Does

The labeling is implemented in `labeling.py` (`DefectLabeler.attribute_defects()`) and resolved in `builder.py` (`CombinedDatasetBuilder._build_single_repo()`).

**Commit-level attribution (high confidence):**
- `explicit_sha_reference`: A bug-fix commit message contains an explicit SHA reference to the candidate commit. The entire commit is labeled positive. Evidence: "commit-level attribution propagated to all changed files."
- `revert`: The candidate commit was reverted. The entire commit is labeled positive. Evidence: "commit-level attribution propagated to all changed files."

**File-level attribution (medium confidence):**
- `line_overlap`: Line-level overlap between a bug-fix and the candidate commit. Only the specific matching file is labeled positive.

**Builder resolution (builder.py lines 125-137):**
```python
if ff.file_path in commit_file_attrs:
    # File has its own medium-confidence (line-overlap) attribution
    fa = commit_file_attrs[ff.file_path]
    file_defect_label = fa[1]  # = 1
else:
    # Fall back to commit-level attribution
    file_defect_label = defect_label  # inherited from results[c.sha]
```

### B. Observed Dataset Fact: What the Frozen Data Contains

- Total positive commits: 210
- All-positive commits: 210 (100.0%)
- Mixed commits: 0 (0.0%)
- File-level attributed commits: 0

### C. Inference

The observed data is fully consistent with the implementation: all positive labels come from commit-level attribution, and no file-level attribution was applied. The labeling methodology cannot distinguish implicated files from unrelated files within the same commit.

Files unrelated to the defect (refactors, formatting, tests, docs) that happen to be changed in the same commit receive `defect_label=1` with the same confidence as the actually implicated file.

---

## 5. Repository-Level Distribution

| Repo | Pos Commits | All-Pos | Mixed | Pos Files | Total Files | Pos Ratio | Sources |
|------|-------------|---------|-------|-----------|-------------|-----------|---------|
| aiohttp | 43 | 43 | 0 | 136 | 773 | 0.176 | explicit_sha_reference, revert |
| pytest | 17 | 17 | 0 | 47 | 439 | 0.107 | explicit_sha_reference, revert |
| celery | 14 | 14 | 0 | 32 | 871 | 0.037 | explicit_sha_reference, revert |
| sqlalchemy | 13 | 13 | 0 | 148 | 968 | 0.153 | explicit_sha_reference, revert |
| uvicorn | 13 | 13 | 0 | 44 | 1123 | 0.039 | explicit_sha_reference, revert |
| cherrypy | 11 | 11 | 0 | 122 | 1094 | 0.112 | explicit_sha_reference, revert |
| jinja2 | 7 | 7 | 0 | 143 | 1249 | 0.114 | explicit_sha_reference, revert |
| requests | 7 | 7 | 0 | 15 | 838 | 0.018 | revert |
| Pillow | 5 | 5 | 0 | 7 | 1329 | 0.005 | explicit_sha_reference, revert |
| bandit | 5 | 5 | 0 | 33 | 1025 | 0.032 | explicit_sha_reference, revert |
| bottle | 5 | 5 | 0 | 7 | 1004 | 0.007 | explicit_sha_reference |
| click | 5 | 5 | 0 | 27 | 786 | 0.034 | explicit_sha_reference, revert |
| networkx | 5 | 5 | 0 | 44 | 1224 | 0.036 | explicit_sha_reference, revert |
| starlette | 5 | 5 | 0 | 9 | 1291 | 0.007 | explicit_sha_reference, revert |
| mypy | 4 | 4 | 0 | 56 | 1354 | 0.041 | explicit_sha_reference |
| tornado | 4 | 4 | 0 | 6 | 1112 | 0.005 | explicit_sha_reference, revert |
| urllib3 | 4 | 4 | 0 | 11 | 1169 | 0.009 | explicit_sha_reference, revert |
| black | 3 | 3 | 0 | 4 | 361 | 0.011 | revert |
| marshmallow | 3 | 3 | 0 | 4 | 579 | 0.007 | revert |
| pymemcache | 3 | 3 | 0 | 6 | 777 | 0.008 | explicit_sha_reference, revert |
| treq | 3 | 3 | 0 | 4 | 1250 | 0.003 | revert |
| flask-sqlalchemy | 2 | 2 | 0 | 3 | 1244 | 0.002 | explicit_sha_reference, revert |
| httpie | 2 | 2 | 0 | 3 | 785 | 0.004 | revert |
| httpx | 2 | 2 | 0 | 7 | 635 | 0.011 | revert |
| invoke | 2 | 2 | 0 | 5 | 872 | 0.006 | explicit_sha_reference, revert |
| isort | 2 | 2 | 0 | 30 | 674 | 0.045 | revert |
| nox | 2 | 2 | 0 | 18 | 712 | 0.025 | explicit_sha_reference, revert |
| paramiko | 2 | 2 | 0 | 2 | 1121 | 0.002 | revert |
| pre-commit | 2 | 2 | 0 | 3 | 1466 | 0.002 | revert |
| pydantic | 2 | 2 | 0 | 7 | 1171 | 0.006 | revert |
| tox | 2 | 2 | 0 | 9 | 867 | 0.010 | explicit_sha_reference |
| fabric | 1 | 1 | 0 | 8 | 877 | 0.009 | revert |
| flask | 1 | 1 | 0 | 24 | 1265 | 0.019 | explicit_sha_reference |
| flask-testing | 1 | 1 | 0 | 3 | 122 | 0.025 | revert |
| flask-wtf | 1 | 1 | 0 | 1 | 1078 | 0.001 | revert |
| itsdangerous | 1 | 1 | 0 | 2 | 1235 | 0.002 | revert |
| pymongo | 1 | 1 | 0 | 13 | 2005 | 0.006 | revert |
| pyyaml | 1 | 1 | 0 | 1 | 1822 | 0.001 | explicit_sha_reference |
| redis | 1 | 1 | 0 | 6 | 508 | 0.012 | explicit_sha_reference |
| rich | 1 | 1 | 0 | 1 | 463 | 0.002 | explicit_sha_reference |
| sympy | 1 | 1 | 0 | 1 | 752 | 0.001 | explicit_sha_reference |
| werkzeug | 1 | 1 | 0 | 3 | 1384 | 0.002 | explicit_sha_reference |

**Repos with mixed commits: 0**

---

## 6. Commit-Size Analysis

| Bucket | Pos Commits | All-Pos | Mixed | Mean Ratio |
|--------|-------------|---------|-------|------------|
| 1 | 67 | 67 | 0 | 1.000 |
| 2-3 | 77 | 77 | 0 | 1.000 |
| 4-5 | 30 | 30 | 0 | 1.000 |
| 6-10 | 17 | 17 | 0 | 1.000 |
| 11-20 | 8 | 8 | 0 | 1.000 |
| >20 | 11 | 11 | 0 | 1.000 |

---

## 7. Why TEST Has Zero Mixed Commits

The zero-mixed-commits finding is **systemic**, not a repository-split artifact:

1. **Root cause:** No file-level attribution (line overlap) was successfully applied in the frozen dataset.
2. **Mechanism:** All positive labels come from `explicit_sha_reference` and/or `revert`, both of which are commit-level and label every file as positive.
3. **Evidence:** The same all-positive pattern appears in TRAIN, VALIDATION, and TEST. The repository split is irrelevant.
4. **Quantification:** See per-split breakdown above and repository table above.

---

## 8. Ranking-Supervision Assessment

**Question A (Classification):** The model can distinguish positive from negative file rows globally.

**Question B (Ranking):** The dataset provides NO supervision for ranking files within a commit.

**Root cause:** The labeling methodology uses commit-level attribution for all positive labels in the dataset. No file-level attribution (line overlap) was successfully applied.

**Suitability: NOT_SUITABLE**

**Reasoning:** All 210 positive commits are all-positive (100.0%). 210/210 positive commits use commit-level attribution (explicit SHA reference or revert). 0 commits use file-level attribution (line overlap). The labeling methodology cannot distinguish implicated files from unrelated files within the same commit. File-level risk ranking requires mixed commits where some files are positive and others are negative.

---

## 9. Recommended Next Methodological Directions

### Direction 1: Construct mixed-commit evaluation subset

- **Evidence needed:** Commits where bug-fixes touch a subset of changed files, with reliable file-level attribution.
- **Assumptions:** That some bug-fix commits exist where only specific files are implicated and others are collateral.
- **Changes frozen semantics:** False
- **Requires regeneration:** True
- **Supports ranking objective:** True

### Direction 2: Improve file-level defect attribution

- **Evidence needed:** Implementation of line-overlap or other file-level attribution that can identify specific files implicated by a defect.
- **Assumptions:** That git diff hunk analysis can reliably identify which files in a bug-fix commit are directly related to the defect.
- **Changes frozen semantics:** True
- **Requires regeneration:** True
- **Supports ranking objective:** True

### Direction 3: Use post-hoc bug-fix evidence for labeling

- **Evidence needed:** Later commits that fix bugs introduced by earlier commits, where the fix touches specific files.
- **Assumptions:** That bug-introducing commits can be identified through follow-up bug-fix analysis.
- **Changes frozen semantics:** True
- **Requires regeneration:** True
- **Supports ranking objective:** True


---

## 10. Frozen Data Integrity

- No frozen dataset files (JSONL) were modified.
- No frozen labeling logic (`labeling.py`) was modified.
- No frozen builder logic (`builder.py`) was modified.
- No frozen schema definitions (`schemas.py`) were modified.
- No Phase 4/4.5/4.6 model artifacts were modified.
- No dataset regeneration was performed.
- All analyses were performed by reading frozen JSONL files.
