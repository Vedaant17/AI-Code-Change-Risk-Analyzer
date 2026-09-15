# Phase 4.11a: Attribution Feasibility Study (JSONL-only)
## Executive Summary
This study investigates whether existing JSONL dataset artifacts can improve file-level defect attribution without repository access. It operates exclusively on the 210 positive commits, their candidate files, corrective commit references, and Phase 4.8/4.9 artifacts.
**Key finding**: All file-level evidence remains WEAK (path overlap only). No STRONG or MODERATE file-level attribution can be established from JSONL data alone. Repository-level git evidence is required for stronger attribution.
## Methodology
### Evidence Hierarchy
| Level | Name | Definition |
|---|---|---|
| COMMIT_STRONG | Strong commit-level | Candidate commit has resolved corrective SHA |
| COMMIT_WEAK | Weak commit-level | Corrective SHA prefix found but unresolvable |
| COMMIT_NONE | No commit evidence | No corrective reference found |
| FILE_STRONG | Strong file-level | Direct content/line evidence identifies specific file |
| FILE_MODERATE | Moderate file-level | Evidence narrows to file but not causal |
| FILE_WEAK | Weak file-level | Path/message/structural association only |
| UNKNOWN | Unknown | Insufficient evidence |
**Important**: Contextual signals (feature proximity, file status patterns, test separation, message mentions) are reported separately and must NOT be treated as attribution evidence.
### Dataset
- Total positive commits: 210- Source: revert=96, explicit_sha=114- Repositories with positives: 42## A. Commit-Level Evidence
Commit-level evidence establishes that a candidate commit is associated with a corrective commit. This is STRONG for all 210 commits where a corrective SHA can be resolved.
| Evidence Level | Count | Percentage ||---|---|---|| COMMIT_STRONG | 210 | 100.0% || COMMIT_WEAK | 0 | 0.0% || COMMIT_NONE | 0 | 0.0% |**Interpretation**: All 210 commits have strong commit-level evidence (corrective SHA resolved). This is unchanged from Phase 4.8 — no new commits were discovered.
## B. File-Level Evidence
File-level evidence identifies specific candidate files as affected by the defect. Current evidence is WEAK (path overlap with corrective commit only).
| Evidence Level | Commits | Pct | Attributed Files ||---|---|---|---|| FILE_STRONG | 0 | 0.0% | — || FILE_MODERATE | 0 | 0.0% | — || FILE_WEAK | 185 | 88.1% | — || UNKNOWN | 25 | 11.9% | — |### Baseline Comparison (Phase 4.8)
- Phase 4.8 FILE_PARTIAL: 38 commits (18.1%)- Phase 4.11a FILE_WEAK: 185 commits (88.1%)- Change: +147 commits**Interpretation**: File-level evidence has not materially improved beyond Phase 4.8. The same 38 FILE_PARTIAL commits produce WEAK file-level evidence via path overlap. No new file-level attribution strategies succeeded from JSONL data alone.
## C. Contextual Signals (NOT Attribution Evidence)
The following signals are potentially useful features for future model building but must NOT be treated as attribution evidence.
### Message File Signal
- Commits with file/function mentions: 99/210- Note: Contextual signal only — NOT causal attribution### File Status Pattern
- Commits with status overlap: 208/210- Note: Descriptive pattern — NOT causal attribution### Feature Proximity
- Commits with feature data: 210- Mean cosine similarity: 0.886927- Note: Contextual predictor signal — NOT attribution evidence### Test/Production Separation
- Mixed test/prod commits: 101/210- Note: Contextual signal — NOT attribution evidence### Corrective Exclusivity
- Commits with exclusivity data: 210- Mean overlap ratio: 0.8264- Note: Descriptive statistic — NOT attribution evidence## D. Negative Feasibility Audit
This is an audit of whether defensible negatives could be constructed. It does NOT create candidate negative labels.
- Defensible negatives: 0- Candidate negatives: 0- Requires repo access: True- Conclusion: No defensible negatives can be constructed from existing JSONL data alone. All four investigated strategies require repository history access (commit timestamps, subsequent commits, neighboring changes). This is consistent with Phase 4.8 and Phase 4.9 findings.### Strategies Investigated
- **strategy_1_survival**: Long post-change survival without defect attribution  - Feasible without repos: False  - Reason: Requires git history to determine observation windows and subsequent commits- **strategy_2_subsequent_commits**: Subsequent commits without defect attribution  - Feasible without repos: False  - Reason: Requires git log to find subsequent commits to same file- **strategy_3_neighboring_correction**: Independent corrective commits affecting neighboring code  - Feasible without repos: False  - Reason: Requires git history to identify neighboring changes- **strategy_4_stable_files**: Files with sufficient observation windows and no defect evidence  - Feasible without repos: False  - Reason: Requires commit timestamps and window duration analysis## E. Prediction Unit Analysis
Distinguishes LABEL AVAILABILITY from ATTRIBUTION QUALITY.
### Commit Level
- Label availability: high- Label count: 210- Attribution quality: STRONG commit-level, but all files in commit labeled equally- Coverage: 210/210 commits have evidence- Practical usefulness: Limited — identifies which commits are risky but not which files within the commit. Entire commit treated as positive.- Investigation budget: Commit-level budget means reviewing ALL files in a commit. For commits touching 30+ files (e.g., jinja2), this is impractical.### File Level
- Label availability: low- Label count: 702- Attribution quality: WEAK — path intersection only. No content, line, or function-level evidence available without repositories.- Coverage: 702/1055 candidate files attributed (66.5%)- Practical usefulness: High potential but currently limited by WEAK attribution. If repository-level evidence improves attribution, this becomes the most practically useful unit.- Investigation budget: File-level budget means reviewing specific files. Current 108 attributed files across 38 commits is manageable.### Function Level
- Label availability: none- Label count: 0- Attribution quality: UNAVAILABLE without repository access- Coverage: 0% — not measurable from JSONL data- Practical usefulness: Highest potential practical usefulness — functions are the natural unit of code review. But requires repo access.- Investigation budget: Function-level budget is the most actionable for developers. Currently unavailable.### Hunk Level
- Label availability: none- Label count: 0- Attribution quality: UNAVAILABLE without repository access- Coverage: 0% — not measurable from JSONL data- Practical usefulness: Most precise but hardest to act on. Requires repo access.- Investigation budget: Hunk-level budget would be extremely targeted but requires infrastructure not available here.**Recommendation**: FILE_LEVEL has the strongest methodological foundation among available units. It has low label coverage (108/1055 = 10.2%) and WEAK attribution, but it is the only unit with any file-level evidence at all. Commit-level has strong labels but poor practical usefulness (too many files per commit). Function and hunk levels require repository access.
## F. Repository Concentration
- Repos with positives: 42- Total positive commits: 210- Gini coefficient: 0.5442- Top 5 share: 0.4762- Repos with strong commit evidence: 42- Repos with file attribution: 42| Repo | Commits | Files | Attributed | Unknown ||---|---|---|---|---|| aiohttp | 43 | 136 | 135 | 1 || pytest | 17 | 47 | 10 | 37 || celery | 14 | 32 | 24 | 8 || sqlalchemy | 13 | 148 | 44 | 104 || uvicorn | 13 | 44 | 33 | 11 || cherrypy | 11 | 122 | 122 | 0 || jinja2 | 7 | 143 | 67 | 76 || requests | 7 | 15 | 14 | 1 || Pillow | 5 | 7 | 7 | 0 || bandit | 5 | 33 | 22 | 11 || bottle | 5 | 7 | 4 | 3 || click | 5 | 27 | 24 | 3 || networkx | 5 | 44 | 40 | 4 || starlette | 5 | 9 | 9 | 0 || mypy | 4 | 56 | 10 | 46 || ... (27 more repos) | | | | |## Final Decision
### PROCEED_TO_4.11B
File-level evidence improved from 38 to 185 commits. Repository-level git evidence (diffs, content comparison, AST parsing) has reasonable prospect of materially improving attribution further.
### Evidence Summary
- Total commits: 210- Commit STRONG: 210- File WEAK: 185- File UNKNOWN: 25- Defensible negatives: 0## Limitations
1. All file-level evidence is WEAK (path overlap only). No content, line, or function-level evidence is available from JSONL data.
2. Contextual signals (feature proximity, message mentions, file status patterns) are NOT attribution evidence.
3. No defensible negatives can be constructed without repository history access.
4. Function-level and hunk-level prediction units are currently unavailable.
5. This analysis cannot establish causal attribution — only provenance and association.
## Frozen Data Integrity
- Phase 4.6–4.10 code and artifacts: UNCHANGED
- Combined-v3 JSONL dataset: UNCHANGED
- repo_split.py: UNCHANGED
- diff_parser.py: UNCHANGED
- No repositories cloned
- No network access
