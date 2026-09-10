# Phase 4.5: Feature Engineering Investigation

## 1. Current 45-Feature Diagnosis

### 1.1 What the model learned

XGBoost on combined-v3 achieves:
- Train ROC-AUC: 0.9997 (near-perfect memorization)
- Val ROC-AUC: 0.4920 (below random)
- Test ROC-AUC: 0.6173 (weak signal)
- Test PR-AUC: 0.0249 vs 0.0194 prevalence (1.28x lift)
- 0/214 test positives in top 100 ranking

The model overfits catastrophically. The train-val ROC-AUC gap is 0.51. The features do not generalize.

### 1.2 Feature importance diagnosis

XGBoost top 5 features by importance:

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | `prod_files_changed` | 1.0000 | test-related |
| 2 | `files_renamed` | 0.2796 | metadata |
| 3 | `files_changed` | 0.2469 | change-size |
| 4 | `test_prod_coupling` | 0.1757 | test-related |
| 5 | `files_modified` | 0.1558 | change-size |

The model primarily learns: "commits that touch many production files are risky." This is a repository-specific pattern (large commits in aiohttp/sqlalchemy correlate with bug-fixes in training), not a universal defect signal.

### 1.3 Feature category breakdown

| Category | Count | % of 45 | Examples |
|----------|-------|---------|----------|
| Change-size | 14 | 31% | lines_added, files_changed, max_file_changes |
| Structural/code (regex) | 15 | 33% | total_hunks, function_declarations_changed, indent stats |
| Test-related | 7 | 16% | test_files_changed, test_prod_coupling |
| Metadata | 6 | 13% | language, primary_language, files_renamed |
| **Historical** | **0** | **0%** | **NONE** |
| **AST-based** | **0** | **0%** | **NONE** |

### 1.4 Root cause

The 45 features are computed purely from a single commit's diff. They contain zero information about:
- How often the file has changed before
- Whether the author has introduced bugs before
- The file's position in the dependency graph
- The actual code structure (only regex proxies)

The model therefore learns commit-size artifacts that correlate with bug-fix labels in training but do not causally predict defects.

---

## 2. Current Feature Analysis

### 2.1 Change-size features (14 features)

| Feature | Level | Likely value | Redundancy | Memorization risk | Recommendation |
|---------|-------|-------------|-----------|-------------------|----------------|
| `lines_added` | commit | Low — size alone is noisy | High with `total_lines_changed` | High | Keep, normalize |
| `lines_deleted` | commit | Low | High | High | Keep, normalize |
| `total_lines_changed` | commit | Low | Redundant with above | High | Keep, normalize |
| `files_changed` | commit | Low | High with `files_modified` | High | Keep, normalize |
| `files_added` | commit | Low | Medium | High | Keep |
| `files_deleted` | commit | Low | Medium | Medium | Keep |
| `files_modified` | commit | Medium — top 5 XGB | High | High | Keep, normalize |
| `additions_ratio` | commit | Low — ratio is noisy | Inverse of deletions_ratio | Medium | Keep |
| `deletions_ratio` | commit | Low | Inverse of additions_ratio | Medium | Keep |
| `avg_file_changes` | commit | Low | Medium | High | Keep, normalize |
| `max_file_changes` | commit | Low-Medium | Medium | High | Keep, normalize |
| `file_lines_added` | file | Low | High with commit-level | High | Keep |
| `file_lines_deleted` | file | Low | High | High | Keep |
| `file_total_lines_changed` | file | Low | High | High | Keep |

**Diagnosis:** Change-size features are the dominant learned signal but the least transferable. A 500-line change in aiohttp means something different from a 500-line change in flask. Raw counts need normalization.

### 2.2 Structural/code features (15 features, all regex-based)

| Feature | Level | Likely value | Redundancy | Memorization risk | Recommendation |
|---------|-------|-------------|-----------|-------------------|----------------|
| `total_hunks` | commit | Low-Medium | Correlates with files_changed | Medium | Keep |
| `avg_hunks_per_file` | commit | Low | Low | Low | Keep |
| `total_function_declarations_changed` | commit | Medium | Regex imprecise | Low | Keep for ablation |
| `total_class_declarations_changed` | commit | Medium | Regex imprecise | Low | Keep for ablation |
| `total_imports_changed` | commit | Low-Medium | Low | Low | Keep |
| `avg_changed_line_indent` | commit | Low | Low | Low | Keep |
| `max_changed_line_indent` | commit | Low | Low | Low | Keep |
| `change_entropy` | commit | Low | Unique | Low | Keep |
| `file_hunk_count` | file | Low | Redundant with commit-level | Low | Keep |
| `file_function_declarations_added` | file | Medium | Regex imprecise | Low | Keep for ablation |
| `file_function_declarations_deleted` | file | Medium | Regex imprecise | Low | Keep for ablation |
| `file_class_declarations_added` | file | Low | Regex imprecise | Low | Keep for ablation |
| `file_class_declarations_deleted` | file | Low | Regex imprecise | Low | Keep for ablation |
| `file_imports_added` | file | Low-Medium | Low | Low | Keep |
| `file_imports_deleted` | file | Low-Medium | Low | Low | Keep |
| `file_avg_changed_line_indent` | file | Low | Redundant with commit-level | Low | Keep |
| `file_max_changed_line_indent` | file | Low | Redundant with commit-level | Low | Keep |
| `file_avg_hunk_size` | file | Low | Low | Low | Keep |

**Diagnosis:** Regex-based declaration counting is imprecise. Python `def foo():` works, but `async def`, nested functions, lambdas, and decorators are missed. AST parsing would be strictly better for Python files.

### 2.3 Test-related features (7 features)

| Feature | Level | Likely value | Redundancy | Recommendation |
|---------|-------|-------------|-----------|----------------|
| `test_files_changed` | commit | Medium — top 14 XGB | Low | Keep |
| `prod_files_changed` | commit | **Highest importance** | Medium | Keep |
| `test_ratio` | commit | Low | Redundant with above | Keep |
| `has_test_changes` | commit | Low | Binary, redundant | Keep |
| `has_prod_changes` | commit | Low | Binary, redundant | Keep |
| `test_prod_coupling` | commit | Medium — top 4 XGB | Binary, redundant | Keep |
| `file_is_test_file` | file | Low | Binary, redundant | Keep |

**Diagnosis:** Test/prod coupling is the strongest learned signal. This makes intuitive sense: changes that touch both test and production code are more likely to be bug-fixes. But the model overfits this pattern to training repos.

### 2.4 Metadata features (6 features)

| Feature | Level | Likely value | Recommendation |
|---------|-------|-------------|----------------|
| `file_language` | file | Low-Medium — XGB rank 9 | Keep |
| `file_is_binary` | file | Low | Keep |
| `primary_language` | commit | Low-Medium — XGB rank 7 | Keep |
| `languages_touched` | commit | Low-Medium — XGB rank 10 | Keep |
| `files_renamed` | commit | Medium — XGB rank 2 | Keep |
| `files_binary` | commit | Low | Keep |

**Diagnosis:** Language features encode repository identity. The model learns "Python commits are different from JavaScript commits" — useful but risks memorizing repo-specific patterns.

---

## 3. Feature Family Definitions

Each feature family is explicitly distinguished. Features are classified as **commit-level** (same value for every file in a commit) or **file-level** (varies per changed file within a commit).

### A. AST Structural Features (changed code only)

**Principle:** Analyze properties of the ACTUALLY CHANGED code, not the entire file. The diff provides `+`/`-` lines with full text. For Python files, parse added/removed lines as AST fragments. For non-Python files, these features are 0.

**Extraction method:** Python stdlib `ast.parse()` on stripped diff lines. If parsing fails (incomplete statements), fall back to regex. Non-Python files always get 0.

**Leakage boundary:** All AST features are computed from the diff text of commit C itself. No information from before or after C is used. Zero leakage risk.

| Feature | Level | What it measures | Extraction |
|---------|-------|-----------------|-----------|
| `file_ast_functions_added` | file | FunctionDef/AsyncFunctionDef nodes in added lines | AST parse `+` lines |
| `file_ast_functions_deleted` | file | FunctionDef/AsyncFunctionDef nodes in removed lines | AST parse `-` lines |
| `file_ast_classes_added` | file | ClassDef nodes in added lines | AST parse `+` lines |
| `file_ast_classes_deleted` | file | ClassDef nodes in removed lines | AST parse `-` lines |
| `file_ast_try_except_changed` | file | Try nodes added + removed | AST parse both |
| `file_ast_if_changed` | file | If nodes added + removed | AST parse both |
| `file_ast_for_changed` | file | For/While nodes added + removed | AST parse both |
| `file_ast_return_changed` | file | Return nodes added + removed | AST parse both |
| `file_ast_yield_changed` | file | Yield/YieldFrom nodes added + removed | AST parse both |
| `file_ast_decorator_count` | file | Decorated functions/classes in diff | AST parse both |
| `file_ast_max_nesting_depth` | file | Max nesting depth in changed code | AST walk |
| `file_ast_avg_function_length` | file | Avg lines per function in diff | Line counting |

**Why changed-code only:** Full-file features (total functions, cyclomatic complexity) require file checkout and provide signal about the file's static state, not the change. Changed-code features directly characterize what the commit does structurally. This is more predictive and cheaper to compute.

### B. Historical File Features

**Information boundary:** For candidate commit C, historical features use ONLY information from commits strictly before C. Specifically:
- `git log` uses `C^` (parent of C) as revision, ensuring C itself is excluded
- `--before` timestamps use C's parent commit timestamp
- Bug-fix/revert detection uses commit messages from prior commits only
- No labels, evidence, or information from the observation window is used

**Leakage safeguards:**
1. Every `git log` command uses `--before=<parent_timestamp>` and revision `C^`
2. Bug-fix detection in history uses the same regex as labeling (fix/bug/resolved/patch/hotfix/regression) but ONLY on commits before C
3. The bug-fix status of prior commits is used as a feature, not as a label for C
4. Revert detection uses `^Revert` prefix on commit messages before C

| Feature | Level | What it measures | Information boundary |
|---------|-------|-----------------|---------------------|
| `file_age_days` | file | Days between file creation and C | `git log --diff-filter=A --format=%at -- file` gives creation timestamp. Computed as `(C.timestamp - creation_time).days`. Creation is always before C. |
| `file_commit_count` | file | Number of commits touching file before C | `git rev-list --count C^ -- file`. Uses C^ revision, so only commits before C are counted. |
| `file_days_since_last_change` | file | Days since file was last modified before C | `git log -1 --format=%at C^ -- file` gives most recent prior commit timestamp. Always before C. |
| `file_historical_bug_fixes` | file | Bug-fix commits touching file before C | `git log --grep="fix\|bug\|..." --format=oneline C^ -- file`. Only messages from commits before C. The detection regex is the same as labeling, but applied to OTHER commits' messages, not C's. This is safe because we are not using the labels of those commits as features for C — we are counting a historical pattern. |
| `file_historical_reverts` | file | Revert commits touching file before C | `git log --grep="^Revert" --format=oneline C^ -- file`. Same boundary as above. |
| `file_defect_rate` | file | `bug_fixes / commit_count` | Derived from above. Both numerator and denominator use only pre-C data. |
| `file_churn` | file | Total lines changed across all historical commits | `git log --numstat C^ -- file` sum of `|added - deleted|` per commit. All commits are before C. |
| `file_authors_count` | file | Distinct authors before C | `git log --format=%aE C^ -- file`. Only authors of pre-C commits. |

**Subtle risk — bug-fix regex in history:** The same regex used for labeling (fix/bug/resolved/patch/hotfix/regression) is applied to historical commit messages. This could introduce a mild form of proxy leakage if the regex is too broad and catches non-bug-fix commits. Mitigation: the historical count is a continuous feature (not a binary label), and the regex is applied to OTHER commits, not C. This is standard practice in defect prediction literature.

### C. Author Features (commit-level)

**Information boundary:** Author features use ONLY information about the author's activity before commit C. Since the author is identified by commit C, their identity is known, but their historical statistics are computed from prior commits only.

**Leakage safeguards:**
1. All `git log` commands use `--before=C.timestamp` or revision `C^`
2. Author bug-fix rate uses commit messages from prior commits only
3. No information about C's own label is used

| Feature | Level | What it measures | Information boundary |
|---------|-------|-----------------|---------------------|
| `author_commit_count` | commit | Author's total commits before C | `git rev-list --count C^ --author="<author>"`. Only commits before C. Same for all files in C. |
| `author_bug_fix_rate` | commit | `bug_fix_commits / total_commits` by author before C | `git log --grep="fix\|bug\|..." C^ --author="<author>"` count / total. All commits are before C. Same for all files in C. |
| `author_file_familiarity` | file | Author's prior commits to this specific file before C | `git rev-list --count C^ --author="<author>" -- file`. Only commits before C. Varies per file. |
| `author_days_active` | commit | Days between author's first and last commit before C | `git log --reverse --format=%at C^ --author="<author>"` (first) and `git log -1 --format=%at C^ --author="<author>"` (last). Both before C. Same for all files in C. |

**Note on `author_file_familiarity`:** This feature varies per changed file within a commit (an author may have touched file A many times but file B never), so it is classified as **file-level**, not commit-level.

### D. Dependency Features

**Current state:** Import counting already exists via regex (`file_imports_added`, `file_imports_deleted`, `total_imports_changed`).

**Proposed addition:**

| Feature | Level | What it measures | Extraction |
|---------|-------|-----------------|-----------|
| `external_imports_added` | commit | Non-stdlib, non-local imports in added lines | Regex: `from X import` where X is not a known stdlib or local module. Requires a list of stdlib modules (Python: use `sys.stdlib_module_names` from Python 3.10+). |

**Not implementing (dependency graph):**

| Feature | Why not |
|---------|---------|
| `file_fan_out` | Requires full file AST for all files. Marginal value over import count. |
| `file_fan_in` | Requires repo-wide import graph construction. Very expensive. |
| `changed_files_fan_in` | Same as above. |
| `dependency_centrality` | Requires full graph. Overkill for v2. |

**Recommendation:** Skip full dependency graph for v2. Import-based features plus `external_imports_added` are sufficient.

### E. Semantic Change Features (via AST)

These overlap significantly with the AST structural features in family A. The structural features already capture try/except, if, for, return, yield changes. The following are distinctive:

| Feature | Level | What it measures | Distinctive from A? |
|---------|-------|-----------------|-------------------|
| `function_signature_changes` | file | Whether function arguments/defaults changed | Yes — requires comparing old/new signatures |
| `decorator_changes` | file | Decorators added/removed | Already in A as `file_ast_decorator_count` |
| `class_inheritance_changes` | file | Base classes changed | Yes — AST comparison |

**Recommendation:** `function_signature_changes` and `class_inheritance_changes` are the only semantically distinctive features not already covered by family A. Defer to a future phase — they require comparing AST before/after, which adds complexity.

### F. Normalization / Repository Effects

**Problem:** Raw change-size features encode repository identity. A 100-line change in a small repo (bottle) is huge; in aiohttp it's routine.

| Feature | Level | What it measures | Formula |
|---------|-------|-----------------|---------|
| `file_change_ratio_vs_history` | file | Change size relative to file's historical average | `log(1 + file_total_lines_changed) / log(1 + file_avg_historical_change)` where avg is from pre-C commits |
| `file_author_experience_pctile` | file | Author's experience percentile across all authors | `rank(author_commit_count) / total_distinct_authors` where all counts are pre-C |
| `file_activity_pctile` | file | File's activity percentile across all files | `rank(file_commit_count) / total_distinct_files` where all counts are pre-C |

**Information boundary:** All normalization inputs (author_commit_count, file_commit_count, total authors/files) are computed from pre-C data only.

---

## 4. Leakage Analysis

### 4.1 Information boundary definition

For candidate commit C with parent C^, the following information is available:

**Available (safe):**
- The diff of C itself (what changed): `git diff C^ C`
- All commits before C (history up to C^): `git log C^`
- File content at C^ (parent commit): `git show C^:file`
- Repository structure at C^
- Author identity in C (from the commit itself)
- Commit message of C
- Timestamp of C

**NOT available (leaks):**
- Any commit after C
- The label of C (whether C is a bug-fix)
- Future file content (C, C+1, C+2, ...)
- Future author behavior
- Any information from the observation window
- Labels or evidence from the labeling process applied to C

### 4.2 Feature-by-feature leakage audit

| Feature | Information used | Boundary verified? |
|---------|-----------------|-------------------|
| All 45 v1 features | C's diff only | YES |
| All 12 AST features | C's diff text, parsed with `ast` | YES — no git history needed |
| `file_age_days` | File creation timestamp (always before C) | YES |
| `file_commit_count` | `git rev-list C^` | YES — C^ excludes C |
| `file_days_since_last_change` | `git log -1 C^` | YES |
| `file_historical_bug_fixes` | `git log --grep C^` — messages only, not labels | YES — uses commit messages of other commits, not labels of C |
| `file_historical_reverts` | `git log --grep C^` | YES |
| `file_defect_rate` | Derived from above two | YES |
| `file_churn` | `git log --numstat C^` | YES |
| `file_authors_count` | `git log --format=%aE C^` | YES |
| `author_commit_count` | `git rev-list C^ --author` | YES |
| `author_bug_fix_rate` | `git log --grep C^ --author` — messages only | YES |
| `author_file_familiarity` | `git rev-list C^ --author -- file` | YES |
| `author_days_active` | `git log --reverse C^` + `git log -1 C^` | YES |
| `external_imports_added` | C's diff text only | YES |
| Normalized features | Derived from pre-C data | YES |

### 4.3 Residual risks and mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Bug-fix regex applied to historical commits may over-count | Low — affects feature quality, not leakage | Use strict regex (fix/bug/resolved/patch/hotfix/regression). The feature is a continuous count, not a binary label. |
| Author identity in C reveals C is a commit | None — identity is from C itself, which is the observation point | N/A |
| `file_historical_bug_fixes` uses detection heuristic similar to labeling | Low — same regex, different commits | The labels are for OTHER commits, not C. This is standard in defect prediction. The feature counts a historical pattern, not a label. |
| `file_defect_rate` could amplify noise from regex | Medium — affects feature quality | Cap at reasonable range [0, 1]. |

---

## 5. Proposed Feature Schema: v2

### 5.1 Design principles

1. **All 45 v1 features retained** for backward compatibility and ablation
2. **New features are additive** — v2 = v1 + new features
3. **AST features supplement (not replace) regex features** for ablation flexibility
4. **Historical features require git history** — computed during dataset generation, not at inference
5. **Author features are commit-level** except `author_file_familiarity` (file-level)
6. **Normalization reduces repository memorization**

### 5.2 Feature dimensions

| Category | v1 count | v2 additions | v2 total |
|----------|---------|-------------|---------|
| Change-size (commit) | 11 | 0 | 11 |
| Change-size (file) | 3 | 0 | 3 |
| Structural/code (regex, commit) | 7 | 0 | 7 |
| Structural/code (regex, file) | 8 | 0 | 8 |
| Test-related (commit) | 7 | 0 | 7 |
| Metadata (commit+file) | 6 | 0 | 6 |
| **AST structural (file)** | **0** | **12** | **12** |
| **Historical (file)** | **0** | **8** | **8** |
| **Author (commit-level)** | **0** | **3** | **3** |
| **Author (file-level)** | **0** | **1** | **1** |
| **Dependency (commit)** | **0** | **1** | **1** |
| **Normalized (file)** | **0** | **3** | **3** |
| **Total** | **45** | **28** | **73** |

### 5.3 v2 feature names and ordering

**Commit-level features (v1: 29, v2 additions: 4, total: 33):**

```
# v1 commit features (indices 0-28, unchanged)
lines_added                          # 0
lines_deleted                        # 1
total_lines_changed                  # 2
files_changed                        # 3
files_added                          # 4
files_deleted                        # 5
files_modified                       # 6
files_renamed                        # 7
files_binary                         # 8
total_hunks                          # 9
avg_hunks_per_file                   # 10
additions_ratio                      # 11
deletions_ratio                      # 12
test_files_changed                   # 13
prod_files_changed                   # 14
test_ratio                           # 15
has_test_changes                     # 16
has_prod_changes                     # 17
test_prod_coupling                   # 18
languages_touched                    # 19
primary_language                     # 20
avg_file_changes                     # 21
max_file_changes                     # 22
total_function_declarations_changed  # 23
total_class_declarations_changed     # 24
total_imports_changed                # 25
avg_changed_line_indent              # 26
max_changed_line_indent              # 27
change_entropy                       # 28

# v2 commit-level additions (indices 29-32)
author_commit_count                  # 29 - author's total commits before C
author_bug_fix_rate                  # 30 - author's historical bug-fix rate before C
author_days_active                   # 31 - days between author's first and last commit before C
external_imports_added               # 32 - non-stdlib imports added in C
```

**File-level features (v1: 16, v2 additions: 24, total: 40):**

```
# v1 file features (indices 0-15, unchanged)
file_language                        # 0
file_is_binary                       # 1
file_is_test_file                    # 2
file_lines_added                     # 3
file_lines_deleted                   # 4
file_total_lines_changed             # 5
file_hunk_count                      # 6
file_function_declarations_added     # 7
file_function_declarations_deleted   # 8
file_class_declarations_added        # 9
file_class_declarations_deleted      # 10
file_imports_added                   # 11
file_imports_deleted                 # 12
file_avg_changed_line_indent         # 13
file_max_changed_line_indent         # 14
file_avg_hunk_size                   # 15

# v2 AST structural file features (indices 16-27)
file_ast_functions_added             # 16 - FunctionDef/AsyncFunctionDef in added lines
file_ast_functions_deleted           # 17 - FunctionDef/AsyncFunctionDef in removed lines
file_ast_classes_added               # 18 - ClassDef in added lines
file_ast_classes_deleted             # 19 - ClassDef in removed lines
file_ast_try_except_changed          # 20 - Try nodes added + removed
file_ast_if_changed                  # 21 - If nodes added + removed
file_ast_for_changed                 # 22 - For/While nodes added + removed
file_ast_return_changed              # 23 - Return nodes added + removed
file_ast_yield_changed               # 24 - Yield/YieldFrom nodes added + removed
file_ast_decorator_count             # 25 - decorated functions/classes in diff
file_ast_max_nesting_depth           # 26 - max nesting depth in changed code
file_ast_avg_function_length         # 27 - avg lines per function in diff

# v2 historical file features (indices 28-35)
file_age_days                        # 28 - days since file creation
file_commit_count                    # 29 - total commits touching file before C
file_days_since_last_change          # 30 - days since last modification before C
file_historical_bug_fixes            # 31 - bug-fix commits touching file before C
file_historical_reverts              # 32 - revert commits touching file before C
file_defect_rate                     # 33 - bug_fixes / commit_count (capped at 1.0)
file_churn                           # 34 - total lines changed across all historical commits
file_authors_count                   # 35 - distinct authors before C

# v2 author file-level feature (index 36)
file_author_familiarity              # 36 - author's prior commits to this specific file

# v2 normalized file features (indices 37-39)
file_change_ratio_vs_history         # 37 - log(1 + lines_changed) / log(1 + avg_historical_change)
file_author_experience_pctile        # 38 - author's percentile by commit count across all authors
file_activity_pctile                 # 39 - file's percentile by commit count across all files
```

### 5.4 Feature types and missing values

| Type | Handling |
|------|---------|
| int counts | 0 for missing/empty |
| float ratios | 0.0 for division by zero |
| float normalized | 0.0 for missing |
| AST features (non-Python) | 0.0 — AST parsing only works for Python |
| Historical features (new files) | 0 for counts, 0.0 for rates, 0 for age |
| Author features (unknown author) | 0 for counts, 0.0 for rates |
| Defect rate | Capped at 1.0 |

### 5.5 Normalization strategy

- `file_defect_rate`: capped at [0, 1]
- `file_change_ratio_vs_history`: log-scaled, clipped to [0, 10]
- `file_author_experience_pctile`: [0, 1]
- `file_activity_pctile`: [0, 1]
- `author_bug_fix_rate`: already [0, 1]

---

## 6. Ablation Experiment Design

### 6.1 Experiments

All experiments use the exact same combined-v3 chronological splits (train=34,405, val=9,410, test=10,574). No data modification. No split alteration.

| Experiment | Feature set | Dimension | Purpose |
|-----------|------------|-----------|---------|
| **E0** | Current 45 features | 45 | Baseline (already completed) |
| **E1** | 45 + AST structural (12) | 57 | Test whether accurate structural signal helps |
| **E2** | 45 + Historical file (8) | 53 | Test whether file history provides transferable signal |
| **E3** | 45 + Author (4) | 49 | Test whether author behavior is predictive |
| **E4** | 45 + AST (12) + Historical (8) | 65 | Test combined structural + historical signal |
| **E5** | 45 + AST (12) + Historical (8) + Author (4) + Normalized (3) + External imports (1) | 73 | Full proposed v2 representation |

### 6.2 Evaluation protocol

1. **Validation set** may be used for model selection (threshold tuning, early stopping)
2. **Test set** must remain untouched until final evaluation of the selected configuration
3. For each experiment, train both LR and XGBoost with identical hyperparameters as v1
4. Report all metrics on train, validation, and test

### 6.3 Metrics per experiment

For each model (LR + XGBoost), on each split:
- ROC-AUC
- PR-AUC (Average Precision)
- PR-AUC lift vs prevalence baseline
- Precision, Recall, F1 at threshold 0.5
- Confusion matrix
- Positive predictions count

**Ranking metrics (test set):**
- Positives in top 10/20/50/100
- Positives in top 10% / 20%
- Precision@10, Precision@20, Precision@50, Precision@100
- Recall@10, Recall@20, Recall@50, Recall@100

**Generalization metrics:**
- Train-val ROC-AUC gap (overfitting indicator)
- Val-test ROC-AUC gap (distribution shift indicator)
- Per-repo performance consistency (recall per repo, variance across repos)

**Reproducibility:**
- Run each experiment twice with `random_state=42`
- Verify metrics are identical (determinism check)

---

## 7. Acceptance Criteria

### 7.1 Evidence thresholds (not hard gates)

The final decision to adopt v2 features is based on the totality of evidence across these dimensions:

| Dimension | Evidence target | How evaluated |
|-----------|----------------|---------------|
| **PR-AUC lift** | Meaningful lift over prevalence baseline (0.0194) | Compare PR-AUC to prevalence. A model with no signal achieves lift = 1.0x. Target: > 1.5x on test. |
| **Average Precision** | Improves over E0 baseline | Compare test AP across experiments. E0 = 0.0249. |
| **Ranking quality** | Positives appear in top-K | E0 has 0 positives in top 100. Any positives in top 100 is progress. |
| **Precision@K / Recall@K** | Non-zero at meaningful K | P@100, R@100 should be > 0. |
| **Val-test stability** | Val and test metrics within 0.1 ROC-AUC | Large gaps indicate overfitting or distribution shift. |
| **Per-repo robustness** | Performance not dominated by one repo | Recall should not be 0 for all repos except one. |
| **Overfitting** | Train-val gap < 0.3 | E0 gap is 0.51. Reduction indicates better generalization. |
| **Reproducibility** | Identical results across 2 runs | Determinism check must pass. |

### 7.2 Final decision framework

The decision is NOT a simple pass/fail. Consider:

1. **If E1 (AST) improves over E0:** AST features provide value. Proceed to E2.
2. **If E2 (History) improves over E0:** Historical features provide value. This is the most important signal — it indicates the model can learn cross-repository patterns.
3. **If E4 (AST+History) improves over E1 and E2 individually:** The features are complementary.
4. **If E5 (full v2) shows diminishing returns vs E4:** Do not add author/normalized features yet. Ship E4 as v2.
5. **If NO experiment shows meaningful improvement:** The feature set is not the bottleneck. Investigate labeling quality, model architecture, or positive-class definition.

### 7.3 Phase 4.5a (AST features) checklist

- [ ] 12 new AST features defined in schemas.py
- [ ] All existing tests pass
- [ ] New tests for AST extraction pass
- [ ] Feature extraction is deterministic
- [ ] Non-Python files gracefully return 0 (no crashes)
- [ ] Lint passes

### 7.4 Phase 4.5b (Historical features) checklist

- [ ] 8 new historical features defined
- [ ] Causality verified: every git command uses `C^` or `--before`
- [ ] No future information leaks into any feature
- [ ] Caching works (no duplicate git calls per file)
- [ ] All existing tests pass
- [ ] New tests for history extraction pass

### 7.5 Phase 4.5c (Author features) checklist

- [ ] 4 new author features defined (3 commit-level + 1 file-level)
- [ ] Causality verified
- [ ] All tests pass

### 7.6 Phase 4.5d (Normalization) checklist

- [ ] 3 new normalized features defined
- [ ] Division-by-zero handled
- [ ] All tests pass

### 7.7 Full v2 evaluation checklist

- [ ] Combined-v3 dataset regenerated with v2 features
- [ ] All 6 ablation experiments (E0-E5) completed
- [ ] LR and XGBoost trained and evaluated for each
- [ ] All metrics reported in comparison table
- [ ] Reproducibility check passes
- [ ] No frozen files modified
- [ ] Decision documented based on evidence framework

---

## 8. Computational Cost Analysis

### 8.1 Per-feature extraction cost

| Feature family | Time per commit | Storage | Needs git history? | Needs AST? |
|---------------|----------------|---------|-------------------|-----------|
| v1 features (existing) | ~1ms | 45 floats | No | No |
| AST diff features (12) | ~5ms (Python only) | 12 floats | No | Yes (stdlib) |
| Historical file features (8) | ~50ms (git calls) | 8 floats | Yes | No |
| Author features (4) | ~30ms (git calls) | 4 floats | Yes | No |
| Normalized features (3) | ~1ms (derived) | 3 floats | Yes (precomputed) | No |

### 8.2 Dataset generation cost

Current combined-v3: 46,888 commits, ~171K file examples.

| Feature family | Total extraction time | Parallelizable? |
|---------------|----------------------|-----------------|
| v1 features | ~3 minutes | Yes |
| AST diff features | ~15 minutes | Yes |
| Historical file features | ~4 hours (git calls per file) | Yes (per-file caching) |
| Author features | ~2 hours (git calls per author) | Yes (per-author caching) |

**Key optimization:** Historical features can be precomputed once per (repo, file, commit) tuple and cached. During dataset generation, batch all `git log` calls per repo, not per file.

### 8.3 Inference cost (new PR analysis)

| Feature family | Inference time | Needs full clone? |
|---------------|---------------|-------------------|
| v1 features | ~1ms | No (diff only) |
| AST diff features | ~5ms | No (diff only) |
| Historical file features | ~100ms | Yes (need `git log`) |
| Author features | ~50ms | Yes (need `git log`) |

**Important:** For inference on a new PR, historical features require repository access. This is fine for local analysis but adds complexity for SageMaker deployment. Historical features should be computed by the collector (pre-SageMaker), not the model.

---

## 9. Implementation Order

### Recommended first implementation: Phase 4.5a + 4.5b

**Phase 4.5a: AST Diff Features** (lowest risk, no git history needed)
- 12 new file-level features
- ~2-3 hours implementation
- Can be computed from existing data (diffs only)
- Zero leakage risk

**Phase 4.5b: Historical File Features** (highest expected value)
- 8 new file-level features
- ~4-6 hours implementation
- Requires git history access during dataset generation
- Addresses root cause of overfitting (no cross-repository transfer)

### After E1/E2 results: decide on 4.5c + 4.5d

Only implement author features and normalization if E1/E2 show improvement.

### Do NOT implement yet

| Feature family | Why not |
|---------------|---------|
| Full dependency graph (fan-in/fan-out) | Too expensive to compute. Unclear value over import count. Requires repo-wide analysis. |
| Full-file AST complexity (cyclomatic, total functions) | Requires file checkout infrastructure. Changed-code features are more relevant and cheaper. |
| Tree-sitter / multi-language AST | Too complex. Python-only AST covers the majority of the codebase. Other languages get 0. |
| `function_signature_changes` | Requires comparing AST before/after. Adds complexity. Defer to phase 3. |
| `class_inheritance_changes` | Same as above. |
| Repository-level features | Too coarse. Risks memorization. File-level normalization is preferred. |
| `file_fan_in` (who imports this file) | Requires full dependency graph. Very expensive. |
| `complexity_delta` (before vs after) | Requires two full-file AST parses. Marginal over changed-code features. |
| Cyclomatic complexity of changed code | Can be added later as a refinement of AST features. Not needed for v2. |

---

## 10. Recommended First Implementation Subset

The minimum feature subset worth implementing first is:

**6 features (not 28):**

| # | Feature | Level | Family | Why first |
|---|---------|-------|--------|-----------|
| 1 | `file_ast_functions_added` | file | AST | Most accurate structural signal |
| 2 | `file_ast_functions_deleted` | file | AST | Complements above |
| 3 | `file_ast_try_except_changed` | file | AST | Exception handling is defect-correlated |
| 4 | `file_commit_count` | file | Historical | Most basic file history feature |
| 5 | `file_historical_bug_fixes` | file | Historical | Direct defect-history signal |
| 6 | `file_days_since_last_change` | file | Historical | Recency signal |

These 6 features cover the two highest-value families (AST + Historical) with the minimum viable set. If they show improvement, expand to the full 28. If they don't, the problem is not feature quality.

**Why these 6:**
- AST functions are the most reliable structural feature (regex is imprecise)
- File commit count is the most basic historical signal
- Historical bug-fixes directly encode defect history
- Days since last change captures recency (files changed recently may be riskier)
- All are file-level (vary per file, which is where the model operates)
- All are leakage-safe
- All can be implemented in ~4-6 hours total

---

## 11. Evaluation Methodology Summary

1. **E0** = baseline (already done)
2. Implement Phase 4.5a (AST features) → run **E1**
3. Implement Phase 4.5b (Historical features) → run **E2**
4. If both E1 and E2 improve, run **E4** (combined)
5. If E4 improves, implement Phase 4.5c (Author) → run **E3** and **E5**
6. Select best configuration based on evidence framework (section 7.2)
7. Final evaluation on test set
8. Document results and decision
