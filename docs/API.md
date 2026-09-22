# API Reference

The server runs on `localhost:8000` by default. Interactive Swagger documentation is available at `/docs`.

## Endpoints

| Method | Path | Request Body | Response | Purpose |
|--------|------|-------------|----------|---------|
| GET | `/health` | — | `HealthResponse` | Health check |
| POST | `/analysis/commit` | `AnalyzeCommitRequest` | `CommitAnalysisResponse` | Single commit diff analysis |
| POST | `/analysis/pull-request` | `AnalyzePRRequest` | `CommitAnalysisResponse` | Branch comparison diff analysis |
| POST | `/analysis/risk` | `AnalyzeRiskRequest` | `AnalyzeRiskResponse` | B1 investigation-priority ranking |
| POST | `/analysis/investigate` | `InvestigateRequest` | `InvestigationResult` | B1 ranking + evidence collection |
| POST | `/analysis/decision` | `InvestigateRequest` | `DecisionResponse` | Full deterministic presentation |

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Repository access failure or invalid input |
| 404 | Commit not found in the repository |
| 422 | Repository inaccessible or invalid request body |
| 500 | Feature extraction or inference failure |

---

## GET /health

Returns a simple health status.

**Response**: `HealthResponse`

```json
{
  "status": "ok"
}
```

---

## POST /analysis/commit

Analyzes a single commit and returns structured diff information. Does not perform risk ranking.

**Request**: `AnalyzeCommitRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_url` | string | yes | HTTPS URL or local path to the repository (min 1 char) |
| `commit_sha` | string | yes | Full or abbreviated commit SHA (min 1 char) |

**Response**: `CommitAnalysisResponse`

| Field | Type | Description |
|-------|------|-------------|
| `sha` | string | Full commit SHA |
| `short_sha` | string | Abbreviated SHA |
| `author` | string | Commit author |
| `author_date` | datetime \| null | Author timestamp |
| `message` | string | Commit message |
| `stats` | `DiffStatsResponse` | Aggregate diff statistics |
| `files` | `FileDiffResponse[]` | Per-file diff information |

**`DiffStatsResponse`**:

| Field | Type | Description |
|-------|------|-------------|
| `total_files` | int | Total files in the diff |
| `total_lines_added` | int | Lines added |
| `total_lines_deleted` | int | Lines deleted |
| `files_added` | int | New files |
| `files_deleted` | int | Deleted files |
| `files_modified` | int | Modified files |
| `files_renamed` | int | Renamed files |
| `files_binary` | int | Binary files |

**`FileDiffResponse`**:

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | File path |
| `status` | string | One of: `added`, `modified`, `deleted`, `renamed`, `binary` |
| `lines_added` | int | Lines added |
| `lines_deleted` | int | Lines deleted |
| `is_binary` | bool | Whether the file is binary |
| `old_path` | string \| null | Original path (renames only) |

**Errors**: 400 on repository access failure or `DiffAnalyzerError`.

---

## POST /analysis/pull-request

Analyzes the diff between two branches, simulating a pull request analysis.

**Request**: `AnalyzePRRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_url` | string | yes | HTTPS URL or local path to the repository (min 1 char) |
| `base_branch` | string | yes | Target / base branch (min 1 char) |
| `head_branch` | string | yes | Source / feature branch (min 1 char) |

**Response**: `CommitAnalysisResponse` — same schema as `/analysis/commit`.

**Errors**: 400 on repository access failure or `DiffAnalyzerError`.

---

## POST /analysis/risk

Ranks files within a commit by investigation priority using the B1 change-size heuristic. Scores are rank-normalized signals in [0, 1] — not probabilities, not calibrated confidence, not defect likelihood.

**Request**: `AnalyzeRiskRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_url` | string | yes | HTTPS URL or local filesystem path (min 1 char) |
| `commit_sha` | string | yes | Full or abbreviated commit SHA (min 1 char) |

**Response**: `AnalyzeRiskResponse`

| Field | Type | Description |
|-------|------|-------------|
| `repo_url` | string | Source repository URL |
| `commit_sha` | string | Full commit SHA |
| `short_sha` | string | Abbreviated SHA |
| `strategy` | string | Always `"B1_CHANGE_SIZE"` |
| `strategy_version` | string | Strategy version |
| `feature_version` | string | Feature schema version |
| `analyzed_at` | string | ISO 8601 UTC timestamp |
| `elapsed_ms` | float | Wall-clock time in milliseconds |
| `total_files` | int | Total files in commit |
| `files_analyzed` | int | Files that received a ranking score |
| `files_skipped` | int | Files excluded from ranking (binary, etc.) |
| `files` | `FileRiskResult[]` | Per-file ranking results |
| `warnings` | string[] | Analysis warnings |
| `score_semantics` | string | Always `"investigation_priority_ranking"` |
| `limitations` | string[] | Known limitations |

**`FileRiskResult`**:

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | File path relative to repository root |
| `status` | string | File status: `added`, `modified`, `deleted`, `renamed`, or `binary` |
| `investigation_priority_score` | float | Ranking score in [0, 1]. Higher = investigate first. Not a probability. |
| `rank` | int | 1-indexed rank within the commit (0 = not scored) |
| `total_files_in_commit` | int | Total files in this commit |
| `lines_added` | int | Lines added |
| `lines_deleted` | int | Lines deleted |
| `is_binary` | bool | Whether the file is binary |
| `language` | string | Detected language name |
| `is_test_file` | bool | Whether the file is a test file |

**Errors**:
- 404: Commit not found
- 422: Repository inaccessible or invalid request
- 500: Feature extraction or inference failure

---

## POST /analysis/investigate

Same B1 ranking as `/analysis/risk`, plus observable evidence collection for each file. Evidence items are factual and provenance-tracked — not risk prediction.

**Request**: `InvestigateRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `repo_url` | string | yes | — | HTTPS URL or local filesystem path (min 1 char) |
| `commit_sha` | string | yes | — | Full or abbreviated commit SHA (min 1 char) |
| `top_k` | int | no | 10 | Max eligible files for historical evidence (1–100) |

**Response**: `InvestigationResult`

| Field | Type | Description |
|-------|------|-------------|
| `repo_url` | string | Source repository URL or path |
| `commit_sha` | string | Full commit SHA |
| `short_sha` | string | Abbreviated SHA |
| `strategy` | string | Always `"B1_CHANGE_SIZE"` |
| `total_files` | int | Total files in commit |
| `files` | `FileEvidence[]` | Per-file results with evidence |
| `warnings` | string[] | Analysis warnings |
| `evidence_version` | string | Evidence engine version |
| `top_k_configured` | int | Requested max eligible files |
| `top_k_analyzed` | int | Actual files selected for historical analysis |
| `git_subprocess_count` | int | Number of git subprocess invocations attempted |

**`FileEvidence`**:

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | File path relative to repository root |
| `score` | float | B1 score in [0, 1] — copied verbatim from B1 output |
| `position` | int | 0-indexed position in B1 output |
| `total_lines_changed` | int | Raw B1 signal |
| `evidence` | `EvidenceItem[]` | All evidence items for this file |

**`EvidenceItem`**:

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | Category: `change`, `structural`, `test`, `historical`, `context` |
| `evidence_type` | string | Type, e.g. `lines_changed`, `recent_commits` |
| `description` | string | Observable, source-backed fact |
| `source` | string | Data source: `diff`, `feature_extraction`, `git_log`, `path_convention`, `b1_ranking` |
| `provenance` | string | Whether directly observed or derived: `direct` or `derived` |
| `ref_commit` | string | Related commit SHA (historical evidence only) |
| `ref_file` | string | Related file path (cross-file evidence) |

**Errors**:
- 404: Commit not found
- 422: Repository inaccessible or invalid request
- 500: Feature extraction or inference failure

---

## POST /analysis/decision

The complete ranking → evidence → presentation flow. B1 ranking, Phase 6 evidence, priority bands, deterministic explanations, and evidence gaps — all in one call.

**This is the primary endpoint for downstream consumers.**

**Request**: `InvestigateRequest` (same as `/analysis/investigate`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `repo_url` | string | yes | — | HTTPS URL or local filesystem path (min 1 char) |
| `commit_sha` | string | yes | — | Full or abbreviated commit SHA (min 1 char) |
| `top_k` | int | no | 10 | Max eligible files for historical evidence (1–100) |

**Response**: `DecisionResponse`

The response is an envelope with deterministic semantic content and runtime metadata.

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `DecisionResult` | Deterministic investigation-priority presentation |
| `analyzed_at` | string | ISO 8601 UTC timestamp (runtime only, not deterministic) |
| `elapsed_ms` | float | Wall-clock time in milliseconds (runtime only, not deterministic) |

### DecisionResult

`DecisionResult` is deterministic: given identical `InvestigationResult` input, repeated runs produce byte-for-byte identical content. Runtime metadata (`analyzed_at`, `elapsed_ms`) lives in the `DecisionResponse` envelope and does not affect `DecisionResult`.

| Field | Type | Description |
|-------|------|-------------|
| `repo_url` | string | Repository URL |
| `commit_sha` | string | Full commit SHA |
| `short_sha` | string | Abbreviated SHA |
| `strategy` | string | Always `"B1_CHANGE_SIZE"` |
| `strategy_version` | string | Strategy version |
| `feature_version` | string | Feature schema version |
| `evidence_version` | string | Evidence collection version |
| `decision_version` | string | Decision layer version |
| `summary` | `PrioritySummary` | Commit-level summary counts |
| `files` | `FileDecision[]` | Per-file investigation-priority decisions |
| `limitations` | string[] | Known limitations |
| `warnings` | string[] | Pipeline warnings |

### PrioritySummary

| Field | Type | Description |
|-------|------|-------------|
| `total_files` | int | Total files analyzed |
| `files_ranked` | int | Non-binary files |
| `files_binary` | int | Binary files |
| `highest_count` | int | Files in HIGHEST band |
| `high_count` | int | Files in HIGH band |
| `medium_count` | int | Files in MEDIUM band |
| `low_count` | int | Files in LOW band |
| `evidence_available` | int | Files with at least one evidence item |
| `evidence_unavailable` | int | Files with zero evidence items |

### FileDecision

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | File path relative to repository root |
| `priority_band` | string | One of: `highest`, `high`, `medium`, `low` |
| `b1_score` | float | B1 investigation-priority score in [0, 1] |
| `b1_position` | int | 0-indexed position in B1 output |
| `total_lines_changed` | int | Raw B1 signal |
| `is_binary` | bool | Whether the file is binary |
| `is_test_file` | bool | Whether the file is a test file |
| `evidence_count` | int | Number of evidence items |
| `evidence_summaries` | `EvidenceSummary[]` | Projected evidence items |
| `explanation` | string | Deterministic text composed from evidence + B1 metadata |
| `evidence_gaps` | string[] | What evidence is absent or limited |

### EvidenceSummary

Field-by-field projection of a Phase 6 `EvidenceItem`, with `ref_commit` and `ref_file` dropped.

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | Evidence category (verbatim from `EvidenceItem`) |
| `evidence_type` | string | Evidence type (verbatim from `EvidenceItem`) |
| `description` | string | Observable fact (verbatim from `EvidenceItem`) |
| `source` | string | Data source (verbatim from `EvidenceItem`) |
| `provenance` | string | `direct` or `derived` (verbatim from `EvidenceItem`) |

### Priority Bands

Priority bands are rank-position groupings derived from B1 output order. They are **not** risk categories, severity levels, or defect classifications.

| Band | Meaning |
|------|---------|
| `highest` | File at the top of the B1 ranking — investigate first |
| `high` | File near the top of the B1 ranking |
| `medium` | File in the middle of the B1 ranking |
| `low` | File at the bottom of the B1 ranking (includes binary files) |

Band assignment logic:
1. If the file is binary → LOW
2. If the file is the first non-binary file → HIGHEST
3. If the file is in the first half of the commit → HIGH
4. Otherwise → MEDIUM

### Errors

- 404: Commit not found
- 422: Repository inaccessible or invalid request
- 500: Feature extraction or inference failure

### Illustrative Abbreviated Response

> This is an illustrative example, not a measured production result. Values are invented to show structure.

```json
{
  "decision": {
    "repo_url": "https://github.com/example/repo",
    "commit_sha": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "short_sha": "da39a3ee",
    "strategy": "B1_CHANGE_SIZE",
    "strategy_version": "1.0.0",
    "feature_version": "v1",
    "evidence_version": "1.0.0",
    "decision_version": "1.0.0",
    "summary": {
      "total_files": 4,
      "files_ranked": 3,
      "files_binary": 1,
      "highest_count": 1,
      "high_count": 1,
      "medium_count": 1,
      "low_count": 1,
      "evidence_available": 3,
      "evidence_unavailable": 1
    },
    "files": [
      {
        "path": "src/main.py",
        "priority_band": "highest",
        "b1_score": 1.0,
        "b1_position": 0,
        "total_lines_changed": 120,
        "is_binary": false,
        "is_test_file": false,
        "evidence_count": 7,
        "evidence_summaries": [
          {
            "category": "change",
            "evidence_type": "lines_changed",
            "description": "120 lines changed",
            "source": "feature_extraction",
            "provenance": "direct"
          }
        ],
        "explanation": "Highest-priority file with 120 line(s) changed; 7 evidence item(s) collected across 5 categories.",
        "evidence_gaps": []
      },
      {
        "path": "tests/test_main.py",
        "priority_band": "medium",
        "b1_score": 0.333,
        "b1_position": 1,
        "total_lines_changed": 40,
        "is_binary": false,
        "is_test_file": true,
        "evidence_count": 5,
        "evidence_summaries": [],
        "explanation": "Medium-priority file with 40 line(s) changed; test file detected.",
        "evidence_gaps": []
      },
      {
        "path": "data/defaults.bin",
        "priority_band": "low",
        "b1_score": 0.0,
        "b1_position": 0,
        "total_lines_changed": 0,
        "is_binary": true,
        "is_test_file": false,
        "evidence_count": 1,
        "evidence_summaries": [],
        "explanation": "Binary file excluded from B1 ranking.",
        "evidence_gaps": [
          "Binary files have limited evidence available."
        ]
      }
    ],
    "limitations": [
      "Priority bands are rank-position groupings, not risk categories.",
      "B1 scores are investigation-priority ranking signals, not probabilities."
    ],
    "warnings": []
  },
  "analyzed_at": "2026-09-22T12:00:00+00:00",
  "elapsed_ms": 285.3
}
```
