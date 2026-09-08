"""Combined multi-repository dataset builder (Phase 3.5).

Orchestrates per-repo dataset generation, then merges all rows into a
single combined dataset with global chronological split assignment.

Split strategy:
  1. Build per-repo datasets with the existing Phase 3 builder
  2. Collect ALL rows (supervised + ambiguous) from all repos
  3. Sort globally by (commit_timestamp, repo_name, commit_sha)
  4. Re-assign splits at commit level using global ordering
  5. Write combined output files

This ensures:
  - Training examples come from earlier history (any repo)
  - Validation from later history
  - Test from latest history
  - All file rows from one commit stay in one split
  - Ambiguous rows remain excluded from supervised splits
  - No random splitting
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import Counter
from datetime import UTC
from pathlib import Path

import git

from backend.app.dataset.builder import DatasetBuilder
from backend.app.dataset.config import DatasetConfig
from backend.app.dataset.multi_repo_config import MultiRepoConfig, RepoConfig
from backend.app.dataset.schemas import DatasetManifest, DatasetRow
from backend.app.schemas.diff import CommitInfo
from backend.app.services.diff_parser import parse_unified_diff

logger = logging.getLogger(__name__)


class CombinedDatasetBuilder:
    """Builds a combined multi-repository dataset with global chronological splits."""

    def __init__(
        self,
        multi_config: MultiRepoConfig | None = None,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.multi_config = multi_config or MultiRepoConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self._extractor = None  # Lazy import to avoid circular

    def build_all(
        self,
        output_base: Path,
    ) -> dict[str, DatasetManifest]:
        """Build per-repo datasets and the combined dataset.

        Parameters
        ----------
        output_base : Path
            Base directory. Per-repo datasets go under output_base/<repo_name>/.
            Combined dataset goes under output_base/combined/.

        Returns
        -------
        dict mapping repo_name -> DatasetManifest (plus "combined" key)
        """
        output_base.mkdir(parents=True, exist_ok=True)
        manifests: dict[str, DatasetManifest] = {}
        all_rows: list[DatasetRow] = []

        for repo_cfg in self.multi_config.repositories:
            logger.info("Processing repo: %s", repo_cfg.repo_name)
            repo_dir = output_base / repo_cfg.repo_name
            repo_dir.mkdir(parents=True, exist_ok=True)

            try:
                rows, manifest = self._build_single_repo(repo_cfg, repo_dir)
                manifests[repo_cfg.repo_name] = manifest
                all_rows.extend(rows)
                logger.info(
                    "  %s: %d rows (%d pos, %d neg, %d amb)",
                    repo_cfg.repo_name,
                    len(rows),
                    sum(1 for r in rows if r.label_status == "positive"),
                    sum(1 for r in rows if r.label_status == "negative"),
                    sum(1 for r in rows if r.label_status == "ambiguous"),
                )
            except Exception as e:
                logger.error("Failed to process %s: %s", repo_cfg.repo_name, e)
                continue

        if not all_rows:
            logger.error("No rows collected from any repository")
            return manifests

        # ── Combine and re-split globally ──
        logger.info("Combining %d total rows from %d repos",
                     len(all_rows), len(manifests))
        combined_dir = output_base / self.multi_config.combined_dir_name
        combined_manifest = self._build_combined(
            all_rows, combined_dir, manifests,
        )
        manifests["combined"] = combined_manifest

        return manifests

    def _build_single_repo(
        self,
        repo_cfg: RepoConfig,
        output_dir: Path,
    ) -> tuple[list[DatasetRow], DatasetManifest]:
        """Clone a repo, collect commits, build per-repo dataset."""

        # Clone to a temp directory, but write output to the permanent path
        clone_dir = tempfile.mkdtemp(prefix=f"{repo_cfg.repo_name}_")
        try:
            repo_path = Path(clone_dir) / repo_cfg.repo_name
            logger.info("  Cloning %s ...", repo_cfg.repo_name)
            repo = git.Repo.clone_from(repo_cfg.repo_url, str(repo_path))

            # Ensure full history is available
            try:
                repo.git.fetch("--unshallow")
            except git.GitCommandError:
                pass  # Already a full clone

            repo.git.checkout(repo_cfg.repo_revision)

            # Collect commits
            commits_iter = repo.iter_commits(
                rev=repo_cfg.repo_revision,
                max_count=repo_cfg.max_commits,
            )
            git_commits = list(commits_iter)
            git_commits.reverse()  # oldest first
            logger.info("  Collected %d commits", len(git_commits))

            # Build CommitInfo objects
            commit_infos = self._build_commit_infos(git_commits, repo)
            logger.info("  Built %d CommitInfo objects", len(commit_infos))

            # Run existing Phase 3 builder (per-repo splits)
            builder = DatasetBuilder(config=self.dataset_config)
            manifest = builder.build(
                commits=commit_infos,
                output_dir=output_dir,
                repo_url=repo_cfg.repo_url,
                repo_name=repo_cfg.repo_name,
                repo_revision=repo_cfg.repo_revision,
            )

            # Read back all rows (including ambiguous) for combining
            all_rows = self._read_all_rows(output_dir)

            return all_rows, manifest
        finally:
            # Clean up clone directory (may fail on Windows, that's OK)
            try:
                import shutil
                shutil.rmtree(clone_dir, ignore_errors=True)
            except Exception:
                pass

    def _build_commit_infos(
        self,
        git_commits: list[git.Commit],
        repo: git.Repo,
    ) -> list[CommitInfo]:
        """Build CommitInfo objects from git commits."""
        from datetime import datetime

        infos: list[CommitInfo] = []
        for gc in git_commits:
            try:
                parent = gc.parents[0] if gc.parents else None
                if parent:
                    raw_diff = repo.git.diff(parent.hexsha, gc.hexsha)
                else:
                    raw_diff = repo.git.diff("--root", gc.hexsha)

                file_diffs = parse_unified_diff(raw_diff)

                author_date: datetime | None = None
                try:
                    author_date = datetime.fromtimestamp(
                        gc.committed_date, tz=UTC
                    )
                except (OSError, ValueError):
                    pass

                info = CommitInfo(
                    sha=gc.hexsha,
                    short_sha=gc.hexsha[:8],
                    author=str(gc.author),
                    author_date=author_date,
                    message=gc.message.strip(),
                    files=file_diffs,
                )
                info.compute_stats()
                infos.append(info)
            except Exception as e:
                logger.warning("  Failed commit %s: %s", gc.hexsha[:12], e)
                continue

        return infos

    def _read_all_rows(self, repo_dir: Path) -> list[DatasetRow]:
        """Read all rows (supervised + ambiguous) from a per-repo dataset."""
        rows: list[DatasetRow] = []
        for split_file in ["train.jsonl", "validation.jsonl", "test.jsonl", "ambiguous.jsonl"]:
            path = repo_dir / split_file
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                obj = json.loads(line)
                row = DatasetRow.model_validate(obj)
                rows.append(row)
        return rows

    def _build_combined(
        self,
        all_rows: list[DatasetRow],
        combined_dir: Path,
        per_repo_manifests: dict[str, DatasetManifest],
    ) -> DatasetManifest:
        """Combine rows from all repos with global chronological split."""
        combined_dir.mkdir(parents=True, exist_ok=True)

        # ── Group rows by commit (all files from one commit stay together) ──
        commit_groups: dict[str, list[DatasetRow]] = {}
        for row in all_rows:
            key = f"{row.repo_name}:{row.commit_sha}"
            commit_groups.setdefault(key, []).append(row)

        # ── Sort commits globally by (timestamp, repo_name, commit_sha) ──
        def sort_key(item: tuple[str, list[DatasetRow]]) -> tuple:
            rows = item[1]
            ts = rows[0].commit_timestamp
            repo = rows[0].repo_name
            sha = rows[0].commit_sha
            # Handle None timestamps by putting them at the start
            if ts is None:
                return ("0000-01-01T00:00:00", repo, sha)
            return (ts.isoformat(), repo, sha)

        sorted_groups = sorted(commit_groups.items(), key=sort_key)

        # ── Assign global splits at commit level ──
        n_commits = len(sorted_groups)
        train_end = int(n_commits * self.dataset_config.train_ratio)
        val_end = train_end + int(n_commits * self.dataset_config.val_ratio)

        global_rows: list[DatasetRow] = []
        for idx, (key, group) in enumerate(sorted_groups):
            if idx < train_end:
                split = "train"
            elif idx < val_end:
                split = "validation"
            else:
                split = "test"

            for row in group:
                row.split = split
                global_rows.append(row)

        # ── Separate supervised and ambiguous ──
        supervised = [r for r in global_rows if r.label_status != "ambiguous"]
        ambiguous = [r for r in global_rows if r.label_status == "ambiguous"]

        train_rows = [r for r in supervised if r.split == "train"]
        val_rows = [r for r in supervised if r.split == "validation"]
        test_rows = [r for r in supervised if r.split == "test"]

        # ── Write combined output ──
        self._write_jsonl(combined_dir / "train.jsonl", train_rows)
        self._write_jsonl(combined_dir / "validation.jsonl", val_rows)
        self._write_jsonl(combined_dir / "test.jsonl", test_rows)
        self._write_jsonl(combined_dir / "ambiguous.jsonl", ambiguous)

        # ── Build combined manifest ──
        manifest = self._build_combined_manifest(
            global_rows, supervised, ambiguous,
            train_rows, val_rows, test_rows,
            sorted_groups, per_repo_manifests,
        )

        combined_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            manifest.model_dump(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        (combined_dir / "metadata.json").write_text(content, encoding="utf-8")

        return manifest

    def _build_combined_manifest(
        self,
        all_rows: list[DatasetRow],
        supervised: list[DatasetRow],
        ambiguous: list[DatasetRow],
        train_rows: list[DatasetRow],
        val_rows: list[DatasetRow],
        test_rows: list[DatasetRow],
        sorted_groups: list[tuple],
        per_repo_manifests: dict[str, DatasetManifest],
    ) -> DatasetManifest:
        """Build manifest for the combined dataset."""
        # Count per-repo stats
        repo_counts: dict[str, int] = Counter()
        repo_pos: dict[str, int] = Counter()
        for row in all_rows:
            repo_counts[row.repo_name] += 1
            if row.label_status == "positive":
                repo_pos[row.repo_name] += 1

        # Count attribution sources across all repos
        explicit_sha = sum(
            m.explicit_sha_attributions for m in per_repo_manifests.values()
        )
        revert = sum(
            m.revert_attributions for m in per_repo_manifests.values()
        )
        line_overlap = sum(
            m.line_overlap_attributions for m in per_repo_manifests.values()
        )
        ambiguous_att = sum(
            m.ambiguous_attributions for m in per_repo_manifests.values()
        )
        bug_fix = sum(
            m.bug_fix_commits_found for m in per_repo_manifests.values()
        )
        revert_commits = sum(
            m.revert_commits_found for m in per_repo_manifests.values()
        )

        # Timestamps
        timestamps = [
            r.commit_timestamp for r in all_rows if r.commit_timestamp is not None
        ]

        # Positive commits per split (computed inline below)

        return DatasetManifest(
            dataset_version=self.multi_config.dataset_version,
            feature_version="v1",
            labeling_version=self.dataset_config.labeling_version,
            config=self.dataset_config.model_dump(),
            repo_url="combined",
            repo_name="combined",
            repo_revision="+".join(
                f"{m.repo_name}:{m.repo_revision[:12]}"
                for m in per_repo_manifests.values()
            ),
            total_commits_processed=sum(
                m.total_commits_processed for m in per_repo_manifests.values()
            ),
            total_commits_included=len(sorted_groups),
            total_commits_excluded=0,
            total_file_examples=len(all_rows),
            positive_examples=sum(1 for r in all_rows if r.label_status == "positive"),
            negative_examples=sum(1 for r in all_rows if r.label_status == "negative"),
            ambiguous_examples=len(ambiguous),
            positive_rate=(
                sum(1 for r in all_rows if r.label_status == "positive") / len(all_rows)
                if all_rows else 0.0
            ),
            train_commits=len({
                r.commit_sha for r in train_rows
            }),
            validation_commits=len({
                r.commit_sha for r in val_rows
            }),
            test_commits=len({
                r.commit_sha for r in test_rows
            }),
            train_examples=len(train_rows),
            validation_examples=len(val_rows),
            test_examples=len(test_rows),
            commits_per_language={},
            examples_per_language={},
            time_range_start=min(timestamps).isoformat() if timestamps else None,
            time_range_end=max(timestamps).isoformat() if timestamps else None,
            bug_fix_commits_found=bug_fix,
            revert_commits_found=revert_commits,
            explicit_sha_attributions=explicit_sha,
            revert_attributions=revert,
            line_overlap_attributions=line_overlap,
            ambiguous_attributions=ambiguous_att,
            exclusion_counts={},
        )

    def _write_jsonl(self, path: Path, rows: list[DatasetRow]) -> None:
        """Write rows to JSONL in deterministic order."""
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")
