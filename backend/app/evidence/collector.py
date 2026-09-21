"""Evidence engine (Phase 6).

Collects deterministic, provenance-tracked investigation evidence from
already-computed pipeline data and optional git history.

The EvidenceEngine is independent of the B1 scorer. It receives B1 context
as optional input and never imports, calls, or depends on the B1 implementation.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime

from backend.app.evidence.schemas import EvidenceItem
from backend.app.features.schemas import FeatureExtractionResult
from backend.app.inference.feature_pipeline import PipelineResult
from backend.app.schemas.diff import CommitInfo, FileStatus

logger = logging.getLogger(__name__)


# -- Git helper (same pattern as backend/app/features/historical.py) ----------


def _run_git(repo_path: str, args: list[str], timeout: int = 30) -> str:
    """Run a git command and return stripped stdout.

    Returns empty string on any failure.
    """
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("git command failed: %s: %s", cmd, exc)
        return ""


# -- EvidenceEngine -----------------------------------------------------------


class EvidenceEngine:
    """Deterministic evidence engine for investigation context.

    Provides two collection methods:

    - ``collect_cheap``: No I/O. Uses already-computed pipeline data.
    - ``collect_historical``: Git subprocess for top-K eligible files.
    """

    def collect_cheap(
        self,
        pipeline_result: PipelineResult,
        b1_context: list[tuple[str, float, int]] | None = None,
    ) -> dict[str, list[EvidenceItem]]:
        """Collect cheap evidence for every file using already-computed data.

        Parameters
        ----------
        pipeline_result:
            Output of the frozen feature pipeline.
        b1_context:
            Optional list of (path, score, position) for every file in the
            frozen B1 output. When provided, a ``b1_ranking`` evidence item
            is produced for each file.

        Returns
        -------
        dict
            Mapping from file path to list of evidence items.
        """
        ci: CommitInfo = pipeline_result.commit_info
        ext: FeatureExtractionResult = pipeline_result.extraction_result
        stats = ci.stats

        # Build lookup for b1_context
        b1_lookup: dict[str, tuple[float, int]] = {}
        if b1_context is not None:
            for path, score, position in b1_context:
                b1_lookup[path] = (score, position)

        # Build lookup for full 16-field features (indexed by file_path)
        full_features_lookup = {ff.file_path: ff for ff in ext.file_features}

        result: dict[str, list[EvidenceItem]] = {}

        for file_diff in ci.files:
            items: list[EvidenceItem] = []
            path = file_diff.path
            full_feat = full_features_lookup.get(path)

            # -- Change evidence --
            items.append(EvidenceItem(
                category="change",
                evidence_type="lines_changed",
                description=(
                    f"{file_diff.lines_added} lines added, "
                    f"{file_diff.lines_deleted} deleted"
                ),
                source="diff",
                provenance="direct",
            ))

            if full_feat is not None and full_feat.hunk_count > 1:
                items.append(EvidenceItem(
                    category="change",
                    evidence_type="hunk_count",
                    description=f"Changed across {full_feat.hunk_count} hunks",
                    source="feature_extraction",
                    provenance="direct",
                ))

            if full_feat is not None:
                func_add = full_feat.function_declarations_added
                func_del = full_feat.function_declarations_deleted
                if func_add > 0 or func_del > 0:
                    items.append(EvidenceItem(
                        category="change",
                        evidence_type="function_churn",
                        description=(
                            f"{func_add} function declaration(s) added, "
                            f"{func_del} deleted"
                        ),
                        source="feature_extraction",
                        provenance="direct",
                    ))

                cls_add = full_feat.class_declarations_added
                cls_del = full_feat.class_declarations_deleted
                if cls_add > 0 or cls_del > 0:
                    items.append(EvidenceItem(
                        category="change",
                        evidence_type="class_churn",
                        description=(
                            f"{cls_add} class declaration(s) added, "
                            f"{cls_del} deleted"
                        ),
                        source="feature_extraction",
                        provenance="direct",
                    ))

                imp_add = full_feat.imports_added
                imp_del = full_feat.imports_deleted
                if imp_add > 0 or imp_del > 0:
                    items.append(EvidenceItem(
                        category="change",
                        evidence_type="import_churn",
                        description=(
                            f"{imp_add} import(s) added, {imp_del} deleted"
                        ),
                        source="feature_extraction",
                        provenance="direct",
                    ))

            if file_diff.status != FileStatus.MODIFIED:
                items.append(EvidenceItem(
                    category="change",
                    evidence_type="file_status",
                    description=f"File status: {file_diff.status.value}",
                    source="diff",
                    provenance="direct",
                ))

            if file_diff.is_binary:
                items.append(EvidenceItem(
                    category="change",
                    evidence_type="binary_file",
                    description="Binary file",
                    source="diff",
                    provenance="direct",
                ))

            # -- Structural evidence --
            if full_feat is not None:
                if full_feat.max_changed_line_indent > 0:
                    items.append(EvidenceItem(
                        category="structural",
                        evidence_type="indentation",
                        description=(
                            f"Average changed-line indentation: "
                            f"{full_feat.avg_changed_line_indent:.1f} spaces, "
                            f"max: {full_feat.max_changed_line_indent} spaces"
                        ),
                        source="feature_extraction",
                        provenance="derived",
                    ))

                if full_feat.avg_hunk_size > 0:
                    items.append(EvidenceItem(
                        category="structural",
                        evidence_type="hunk_size",
                        description=(
                            f"Average hunk size: {full_feat.avg_hunk_size:.1f} lines"
                        ),
                        source="feature_extraction",
                        provenance="derived",
                    ))

            # -- Test evidence --
            if full_feat is not None and full_feat.is_test_file:
                items.append(EvidenceItem(
                    category="test",
                    evidence_type="test_file",
                    description="This is a test file",
                    source="path_convention",
                    provenance="direct",
                ))

            if ext.commit_features.test_prod_coupling:
                items.append(EvidenceItem(
                    category="test",
                    evidence_type="test_coupling",
                    description=(
                        "Both test and production files changed in this commit"
                    ),
                    source="feature_extraction",
                    provenance="derived",
                ))

            if not ext.commit_features.has_test_changes:
                items.append(EvidenceItem(
                    category="test",
                    evidence_type="no_test_changes",
                    description="No test files changed in this commit",
                    source="feature_extraction",
                    provenance="derived",
                ))

            # -- Context evidence --
            total_commit_lines = max(
                stats.total_lines_added + stats.total_lines_deleted, 1
            )
            file_lines = file_diff.lines_added + file_diff.lines_deleted
            concentration = file_lines / total_commit_lines
            if concentration > 0.0:
                items.append(EvidenceItem(
                    category="context",
                    evidence_type="change_concentration",
                    description=(
                        f"File accounts for {concentration:.0%} of total "
                        f"commit changes ({file_lines} of "
                        f"{total_commit_lines} lines)"
                    ),
                    source="diff",
                    provenance="derived",
                ))

            # Language evidence
            if full_feat is not None and full_feat.language > 0.0:
                from backend.app.features.language import LANGUAGE_LIST

                idx = int(full_feat.language * (len(LANGUAGE_LIST) - 1) + 0.5)
                idx = max(0, min(idx, len(LANGUAGE_LIST) - 1))
                lang_name = LANGUAGE_LIST[idx]
                if lang_name:
                    items.append(EvidenceItem(
                        category="context",
                        evidence_type="language",
                        description=f"File language: {lang_name}",
                        source="feature_extraction",
                        provenance="direct",
                    ))

            # -- B1 ranking evidence --
            if path in b1_lookup:
                score, position = b1_lookup[path]
                items.append(EvidenceItem(
                    category="context",
                    evidence_type="b1_ranking",
                    description=(
                        f"Ranked at position {position} in B1 output "
                        f"({file_diff.total_lines_changed} lines changed)"
                    ),
                    source="b1_ranking",
                    provenance="derived",
                ))

            result[path] = items

        return result

    def collect_historical(
        self,
        pipeline_result: PipelineResult,
        local_path: str,
        top_k: int = 10,
    ) -> tuple[dict[str, list[EvidenceItem]], int, int]:
        """Collect historical evidence for top-K eligible files.

        Eligibility: non-binary, not deleted. Binary and deleted files do
        not consume top_k slots.

        Parameters
        ----------
        pipeline_result:
            Output of the frozen feature pipeline.
        local_path:
            Path to the local git repository.
        top_k:
            Maximum number of eligible files for historical analysis.

        Returns
        -------
        tuple
            (evidence_by_path, git_subprocess_count, top_k_analyzed)
        """
        ci: CommitInfo = pipeline_result.commit_info

        # Determine analyzed commit timestamp
        analyzed_ts: datetime | None = ci.author_date

        # Resolve first parent for revision boundary
        parent_ref = self._resolve_parent_ref(local_path, ci.sha)

        # Select top-K eligible files from B1 output order
        # B1 output order is the order of pipeline_result.file_features
        # (which matches b1_scored order since score_files preserves order)
        selected = self._select_top_k_eligible(
            pipeline_result, ci, top_k
        )

        if not selected or parent_ref is None:
            # Root commit or no eligible files
            evidence: dict[str, list[EvidenceItem]] = {}
            git_count = 0
            analyzed_count = 0

            if parent_ref is None and ci.sha:
                # Root commit — produce root_commit evidence for all eligible
                for file_feat in selected:
                    items = evidence.get(file_feat.path, [])
                    items.append(EvidenceItem(
                        category="historical",
                        evidence_type="root_commit",
                        description=(
                            "Analyzed commit is a root commit "
                            "(no parent history available)"
                        ),
                        source="git_log",
                        provenance="direct",
                    ))
                    evidence[file_feat.path] = items

            return evidence, git_count, analyzed_count

        # Collect historical evidence for selected files
        evidence_by_path: dict[str, list[EvidenceItem]] = {}
        git_count = 0
        analyzed_count = 0

        for file_feat in selected:
            path = file_feat.path
            items = self._collect_file_historical(
                local_path=local_path,
                commit_sha=ci.sha,
                parent_ref=parent_ref,
                file_path=path,
                analyzed_ts=analyzed_ts,
            )
            git_count += 1  # One git log call per file
            analyzed_count += 1
            evidence_by_path[path] = items

        return evidence_by_path, git_count, analyzed_count

    def _resolve_parent_ref(
        self, repo_path: str, commit_sha: str
    ) -> str | None:
        """Resolve the first parent of the analyzed commit.

        Returns None for root commits (no parent).
        """
        # Try to get parent SHA via git rev-parse
        parent_sha = _run_git(
            repo_path,
            ["rev-parse", f"{commit_sha}^"],
        )
        if parent_sha:
            return parent_sha
        return None

    def _select_top_k_eligible(
        self,
        pipeline_result: PipelineResult,
        ci: CommitInfo,
        top_k: int,
    ) -> list:
        """Select first top_k eligible files from B1 output order.

        Eligibility: non-binary, not deleted. Binary and deleted files
        do not consume top_k slots.
        """
        # Build lookup for file status from CommitInfo
        status_lookup = {fd.path: fd.status for fd in ci.files}

        selected = []
        for file_feat in pipeline_result.file_features:
            if file_feat.is_binary:
                continue
            status = status_lookup.get(file_feat.path)
            if status == FileStatus.DELETED:
                continue

            selected.append(file_feat)

            if len(selected) == top_k:
                break

        return selected

    def _collect_file_historical(
        self,
        local_path: str,
        commit_sha: str,
        parent_ref: str,
        file_path: str,
        analyzed_ts: datetime | None,
    ) -> list[EvidenceItem]:
        """Collect historical evidence for a single file.

        Uses git log starting from parent_ref, with strict timestamp
        validation: candidate_timestamp < analyzed_commit_timestamp.
        """
        items: list[EvidenceItem] = []

        # Get recent commits touching this file before analyzed commit
        before_flag = ""
        if analyzed_ts is not None:
            before_flag = f"--before={analyzed_ts.isoformat()}"

        log_args = ["log", "-n", "5", "--format=%H|%aI|%s"]
        if before_flag:
            log_args.append(before_flag)
        log_args.extend([parent_ref, "--", file_path])

        log_output = _run_git(local_path, log_args)

        if not log_output:
            items.append(EvidenceItem(
                category="historical",
                evidence_type="history_unavailable",
                description="No prior history found for this file",
                source="git_log",
                provenance="direct",
            ))
            return items

        commits = []
        for line in log_output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            sha, date_str, msg = parts[0], parts[1], parts[2]

            # Strict timestamp validation: candidate < analyzed
            if analyzed_ts is not None:
                try:
                    candidate_ts = datetime.fromisoformat(date_str)
                    if candidate_ts >= analyzed_ts:
                        continue
                except (ValueError, TypeError):
                    continue

            commits.append((sha, msg))

        if not commits:
            items.append(EvidenceItem(
                category="historical",
                evidence_type="history_unavailable",
                description="No prior history found for this file",
                source="git_log",
                provenance="direct",
            ))
            return items

        # Recent commits evidence
        items.append(EvidenceItem(
            category="historical",
            evidence_type="recent_commits",
            description=f"File changed in {len(commits)} recent commit(s)",
            source="git_log",
            provenance="derived",
        ))

        # Bug-fix count
        bug_fix_patterns = [
            "fix", "bug", "patch", "hotfix", "resolve", "correct",
        ]
        bug_fix_count = sum(
            1 for _, msg in commits
            if any(p in msg.lower() for p in bug_fix_patterns)
        )
        if bug_fix_count > 0:
            items.append(EvidenceItem(
                category="historical",
                evidence_type="recent_bug_fixes",
                description=f"File has {bug_fix_count} prior bug-fix commit(s)",
                source="git_log",
                provenance="derived",
            ))

        # Days since last change
        if analyzed_ts is not None and commits:
            last_sha, _ = commits[0]  # Most recent (first in log)
            last_ts_str = _run_git(
                local_path,
                ["log", "-1", "--format=%aI", last_sha],
            )
            if last_ts_str:
                try:
                    last_ts = datetime.fromisoformat(last_ts_str)
                    delta = analyzed_ts - last_ts
                    days = max(0, delta.days)
                    items.append(EvidenceItem(
                        category="historical",
                        evidence_type="days_since_last_change",
                        description=f"File last changed {days} day(s) ago",
                        source="git_log",
                        provenance="derived",
                    ))
                except (ValueError, TypeError):
                    pass

        return items

    def collect(
        self,
        pipeline_result: PipelineResult,
        b1_context: list[tuple[str, float, int]] | None = None,
        local_path: str | None = None,
        top_k: int = 10,
    ) -> tuple[dict[str, list[EvidenceItem]], int, int]:
        """Collect all evidence (cheap + historical).

        Parameters
        ----------
        pipeline_result:
            Output of the frozen feature pipeline.
        b1_context:
            Optional B1 ranking context.
        local_path:
            Path to local git repo. Required for historical evidence.
        top_k:
            Maximum eligible files for historical analysis.

        Returns
        -------
        tuple
            (evidence_by_path, git_subprocess_count, top_k_analyzed)
        """
        cheap = self.collect_cheap(pipeline_result, b1_context)

        if local_path is not None:
            historical, git_count, analyzed = self.collect_historical(
                pipeline_result, local_path, top_k
            )
        else:
            historical = {}
            git_count = 0
            analyzed = 0

        # Merge
        merged: dict[str, list[EvidenceItem]] = {}
        all_paths = set(cheap.keys()) | set(historical.keys())
        for path in all_paths:
            merged[path] = cheap.get(path, []) + historical.get(path, [])

        return merged, git_count, analyzed
