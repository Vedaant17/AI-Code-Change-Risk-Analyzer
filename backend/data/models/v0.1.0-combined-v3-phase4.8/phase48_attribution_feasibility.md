# Phase 4.8: File-Level Attribution Feasibility Study

## A. OBSERVED FACTS

### Dataset Overview

- Total positive commits analyzed: 210
- Attribution sources: {
  "revert": {
    "count": 96,
    "COMMIT_ONLY": 87,
    "FILE_PARTIAL": 9
  },
  "explicit_sha_reference": {
    "count": 114,
    "COMMIT_ONLY": 85,
    "FILE_PARTIAL": 29
  }
}

### Evidence Category Distribution

| Category | Count | % |
|----------|-------|---|
| FILE_PARTIAL | 38 | 18.1% |
| COMMIT_ONLY | 172 | 81.9% |
| NO_ATTRIBUTION | 0 | 0.0% |
| UNUSABLE | 0 | 0.0% |

### File-Level Yield

| Metric | Count |
|--------|-------|
| Total candidate files | 1055 |
| Path-overlap files (COMMIT_ONLY) | 594 |
| Path-associated files (FILE_PARTIAL) | 108 |
| Unknown files | 353 |
| Path transform candidates | 60 |

---

## B. IMPLEMENTATION FACTS

### What the Analysis Does

1. Reads all rows from frozen train/validation/test JSONL files.
2. Reads all rows from ambiguous.jsonl for SHA resolution.
3. For each positive commit, extracts the corrective SHA from `label_evidence` text.
4. Resolves the full SHA across all JSONL data.
5. Extracts file paths from the corrective commit's rows.
6. Computes exact path intersection with candidate files.
7. Classifies evidence category.

### What the Analysis Does NOT Do

- Does NOT access repository checkouts or raw diffs.
- Does NOT perform line-level or content-level analysis.
- Does NOT normalize or transform paths.
- Does NOT modify any frozen files.
- Does NOT train ML models.
- Does NOT use corrective commit data as features.

---

## C. INFERENCES

### Evidence Category: FILE_PARTIAL

- 38 commits where the corrective change touches a strict subset of candidate files.
- 108 path-associated file rows.
- 236 unknown file rows.

These represent the primary potentially useful file-level signal. The intersected files have path-level association with the corrective change, but causality is not uniquely established.

### Evidence Category: COMMIT_ONLY

- 172 commits where path intersection does not narrow the candidate set.
  - 147 with full overlap (all candidate files in corrective).
  - 25 with zero overlap (no candidate files in corrective).
- 594 path-overlap files (stored separately, NOT labeled positive).
- 117 unknown files.

Path overlap in COMMIT_ONLY cases establishes commit-level association but does NOT provide file-level narrowing.

### Evidence Category: NO_ATTRIBUTION

- 0 commits where no corrective commit was found or resolved.

### Evidence Category: UNUSABLE

- 0 commits with inconsistent or unavailable data.

---

## D. LIMITATIONS

1. **No content-level analysis**: The frozen JSONL contains only file paths, not diffs or patch contents. Line-level attribution is impossible without repository checkouts.
2. **Path overlap ≠ defect attribution**: Sharing file paths between candidate and corrective commits establishes association, not causation.
3. **No defensible negatives**: Files absent from the corrective commit cannot be labeled negative (partial fixes, renames, interaction effects).
4. **Path transform ambiguity**: Some 'no overlap' cases may be renames or layout changes, but these are not automatically resolved.
5. **Positive-unlabeled supervision**: The resulting dataset has POSITIVE and UNKNOWN labels, not binary POSITIVE/NEGATIVE.

---

## E. RECOMMENDATION

**PATH_LEVEL_EVIDENCE_ONLY**

38 FILE_PARTIAL cases exist (18.1%) with 108 path-associated file rows and 236 unknown file rows. However, no defensible negative labels can be constructed. The supervision is positive/unknown (positive-unlabeled), not binary positive/negative. This requires a PU learning approach rather than standard binary classification.

### Decision Factors

- **file_partial_commits**: 38
- **file_partial_pct**: 18.0952
- **commit_only_commits**: 172
- **no_attribution_commits**: 0
- **unusable_commits**: 0
- **total_positive_commits**: 210
- **path_associated_file_rows**: 108
- **unknown_file_rows**: 236
- **total_candidate_file_rows**: 1055
- **unknown_rate**: 0.3346
- **defensible_negatives**: 0
- **supervision_type**: positive/unknown (PU)

---

## Per-Split Breakdown

| Split | Commits | FILE_PARTIAL | COMMIT_ONLY | NO_ATTRIBUTION | UNUSABLE | Pos Files | Unknown |
|-------|---------|-------------|-----------|---------------|----------|-----------|---------|
| train | 111 | 26 | 85 | 0 | 0 | 86 | 142 |
| validation | 42 | 6 | 36 | 0 | 0 | 12 | 73 |
| test | 57 | 6 | 51 | 0 | 0 | 10 | 21 |

---

## Per-Repository Summary

| Repo | Commits | FILE_PARTIAL | COMMIT_ONLY | Pos Files |
|------|---------|-------------|-------------|-----------|
| aiohttp | 43 | 1 | 42 | 2 |
| pytest | 17 | 3 | 14 | 5 |
| celery | 14 | 3 | 11 | 4 |
| sqlalchemy | 13 | 6 | 7 | 8 |
| uvicorn | 13 | 5 | 8 | 13 |
| cherrypy | 11 | 0 | 11 | 0 |
| jinja2 | 7 | 3 | 4 | 9 |
| requests | 7 | 0 | 7 | 0 |
| Pillow | 5 | 0 | 5 | 0 |
| bandit | 5 | 2 | 3 | 3 |
| bottle | 5 | 1 | 4 | 1 |
| click | 5 | 2 | 3 | 6 |
| networkx | 5 | 2 | 3 | 38 |
| starlette | 5 | 0 | 5 | 0 |
| mypy | 4 | 2 | 2 | 6 |
| tornado | 4 | 1 | 3 | 1 |
| urllib3 | 4 | 1 | 3 | 1 |
| black | 3 | 0 | 3 | 0 |
| marshmallow | 3 | 0 | 3 | 0 |
| pymemcache | 3 | 1 | 2 | 2 |
| treq | 3 | 0 | 3 | 0 |
| flask-sqlalchemy | 2 | 0 | 2 | 0 |
| httpie | 2 | 0 | 2 | 0 |
| httpx | 2 | 0 | 2 | 0 |
| invoke | 2 | 1 | 1 | 2 |
| isort | 2 | 0 | 2 | 0 |
| nox | 2 | 0 | 2 | 0 |
| paramiko | 2 | 0 | 2 | 0 |
| pre-commit | 2 | 0 | 2 | 0 |
| pydantic | 2 | 0 | 2 | 0 |
| tox | 2 | 1 | 1 | 2 |
| fabric | 1 | 0 | 1 | 0 |
| flask | 1 | 1 | 0 | 1 |
| flask-testing | 1 | 0 | 1 | 0 |
| flask-wtf | 1 | 0 | 1 | 0 |
| itsdangerous | 1 | 0 | 1 | 0 |
| pymongo | 1 | 0 | 1 | 0 |
| pyyaml | 1 | 0 | 1 | 0 |
| redis | 1 | 1 | 0 | 2 |
| rich | 1 | 0 | 1 | 0 |
| sympy | 1 | 0 | 1 | 0 |
| werkzeug | 1 | 1 | 0 | 2 |

---

## Frozen Data Integrity

- No frozen dataset files (JSONL) were modified.
- No frozen labeling logic was modified.
- No frozen builder logic was modified.
- No Phase 4/4.5/4.6/4.7 artifacts were modified.
- No dataset regeneration was performed.
- All analyses read frozen JSONL files and ambiguous.jsonl.
