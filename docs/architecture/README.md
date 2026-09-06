# Architecture — Phase 1: Repository & Git Diff Ingestion

## Overview

Phase 1 implements the data-ingestion foundation of the AI Code Change-Risk Analyzer.
It takes a Git commit or branch diff and produces structured, typed output ready for
feature extraction in Phase 2.

## Data Flow (Phase 1)

```
GitHub Commit / PR
        │
        ▼
  ┌─────────────┐
  │  FastAPI     │  POST /analysis/commit  or  /analysis/pull-request
  │  (main.py)   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────┐
  │  DiffAnalyzer    │  Orchestration layer
  │  (diff_analyzer) │
  └──────┬──────────┘
         │
         ▼
  ┌─────────────────┐         ┌──────────────────┐
  │  GitService      │────────▶│  diff_parser      │
  │  (git_service)   │         │  (diff_parser.py) │
  └──────┬──────────┘         └──────┬───────────┘
         │                           │
         ▼                           ▼
  Clone repo, run           Parse unified diff
  git diff                  into FileDiff list
         │                           │
         └──────────┬────────────────┘
                    ▼
            CommitInfo + DiffStats
            (schemas/diff.py)
```

## Modules

| Module | Responsibility |
|---|---|
| `backend/app/schemas/diff.py` | Pydantic domain models: `FileDiff`, `CommitInfo`, `DiffStats`, `Hunk`, `FileStatus` |
| `backend/app/services/diff_parser.py` | Lightweight regex/string unified-diff parser with clean `parse_unified_diff()` interface |
| `backend/app/services/git_service.py` | GitPython wrapper: clone, commit diff, branch diff |
| `backend/app/analyzers/diff_analyzer.py` | Thin orchestration: repo URL → clone → diff → `CommitInfo` |
| `backend/app/api/main.py` | FastAPI application with health + analysis endpoints |
| `backend/app/api/schemas.py` | Request/response Pydantic models for the API |

## Diff Parser Design

The parser exposes a single public function:

```python
def parse_unified_diff(raw_diff: str) -> list[FileDiff]:
```

It splits on `diff --git` headers, then for each file section:
1. Detects file status from `---`/`+++` and `/dev/null` markers
2. Detects renames via `rename from`/`rename to` lines
3. Detects binary files from the `Binary files` marker
4. Extracts hunk headers via regex: `@@ -x,y +x,y @@`
5. Counts `+`/`-` lines in each hunk body

**Swap path**: Replace the body of `parse_unified_diff` with `unidiff` without
changing any callers.

## Phase 1 Scope

- Ingest Git diffs (commit and branch comparison)
- Parse into structured per-file data with line counts and hunk information
- Expose via REST API
- Run locally with no AWS dependencies

## Phase 1 Limitations (addressed later)

- No feature extraction (Phase 2)
- No ML prediction (Phase 4+)
- No QUBO/optimization (Phase 8+)
- No persistent storage
- No authentication
