"""Dataset builder (Phase 3).

Assembles the supervised dataset from historical commits.  The pipeline:

  1. Accept a list of CommitInfo objects (chronologically ordered)
  2. Detect bug-fixes and reverts
  3. Attribute defect evidence using the strict confidence hierarchy
  4. Extract features for each commit (Phase 2)
  5. Assign chronological splits at the commit level
  6. Expand commits to file rows, inheriting the commit's split
  7. Write output files in deterministic, byte-identical order

Output files:
  metadata.json     — DatasetManifest (no volatile values)
  train.jsonl       — supervised training rows
  validation.jsonl  — supervised validation rows
  test.jsonl        — supervised test rows
  ambiguous.jsonl   — ambiguous rows (excluded from supervised splits)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from backend.app.dataset.config import DatasetConfig
from backend.app.dataset.labeling import DefectLabeler
from backend.app.dataset.schemas import DatasetManifest, DatasetRow
from backend.app.features.extractor import FeatureExtractor
from backend.app.schemas.diff import CommitInfo

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Assembles a supervised dataset from historical commits.

    Usage::

        builder = DatasetBuilder(config=DatasetConfig())
        manifest = builder.build(
            commits=[commit1, commit2, ...],
            output_dir=Path("dataset/"),
            repo_url="https://github.com/org/repo",
            repo_name="repo",
            repo_revision="abc123...",
        )
    """

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self._extractor = FeatureExtractor()
        self._labeler = DefectLabeler(self.config)

    def build(
        self,
        commits: list[CommitInfo],
        output_dir: Path,
        repo_url: str = "",
        repo_name: str = "",
        repo_revision: str = "",
    ) -> DatasetManifest:
        """Build the dataset and write output files.

        Parameters
        ----------
        commits:
            Chronologically ordered list of CommitInfo objects.
        output_dir:
            Directory to write dataset files.
        repo_url:
            Source repository URL.
        repo_name:
            Repository name.
        repo_revision:
            HEAD SHA at processing time.

        Returns
        -------
        DatasetManifest
            Summary statistics about the assembled dataset.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if not commits:
            return self._write_empty_dataset(
                output_dir, repo_url, repo_name, repo_revision,
            )

        # ── Pass 1: Detect bug-fixes and reverts ──
        bug_fix_shas = self._labeler.identify_bug_fixes(commits)
        revert_map = self._labeler.identify_reverts(commits)

        # ── Pass 2: Attribute defects ──
        self._labeler._file_attributions = {}
        attributions = self._labeler.attribute_defects(
            commits, bug_fix_shas, revert_map,
        )

        # ── Compute split ranges (commit-level, chronological) ──
        split_ranges = self._determine_split(len(commits))

        # ── Pass 3: Extract features and build rows ──
        rows: list[DatasetRow] = []
        commit_shas: list[str] = []

        for idx, c in enumerate(commits):
            label_status, defect_label, confidence, source, evidence = (
                attributions.get(c.sha, ("negative", 0, 0.0, "none", ""))
            )
            result = self._extractor.extract(c)
            commit_vec = result.commit_features.to_feature_vector()

            split = self._assign_split_from_ranges(idx, split_ranges)
            commit_shas.append(c.sha)

            file_attributions = self._labeler.get_file_attributions()
            commit_file_attrs = file_attributions.get(c.sha, {})

            for i, ff in enumerate(result.file_features):
                file_vec = ff.to_feature_vector()

                if ff.file_path in commit_file_attrs:
                    fa = commit_file_attrs[ff.file_path]
                    file_label_status = fa[0]
                    file_defect_label = fa[1]
                    file_confidence = fa[2]
                    file_source = fa[3]
                    file_evidence = fa[4]
                else:
                    file_label_status = label_status
                    file_defect_label = defect_label
                    file_confidence = confidence
                    file_source = source
                    file_evidence = evidence

                row = DatasetRow(
                    dataset_version=self.config.dataset_version,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    commit_sha=c.sha,
                    commit_timestamp=c.author_date,
                    commit_message=c.message,
                    author=c.author,
                    file_path=ff.file_path,
                    file_status=c.files[i].status.value
                    if i < len(c.files)
                    else "modified",
                    feature_version=result.feature_version,
                    commit_features=commit_vec,
                    file_features=file_vec,
                    label_status=file_label_status,
                    defect_label=file_defect_label,
                    label_confidence=file_confidence,
                    label_source=file_source,
                    label_evidence=file_evidence,
                    split=split,
                )
                rows.append(row)

        # ── Split into supervised and ambiguous ──
        supervised_rows = [r for r in rows if r.label_status != "ambiguous"]
        ambiguous_rows = [r for r in rows if r.label_status == "ambiguous"]

        train_rows = [r for r in supervised_rows if r.split == "train"]
        val_rows = [r for r in supervised_rows if r.split == "validation"]
        test_rows = [r for r in supervised_rows if r.split == "test"]

        # ── Write output files (deterministic order) ──
        self._write_jsonl(output_dir / "train.jsonl", train_rows)
        self._write_jsonl(output_dir / "validation.jsonl", val_rows)
        self._write_jsonl(output_dir / "test.jsonl", test_rows)
        self._write_jsonl(output_dir / "ambiguous.jsonl", ambiguous_rows)

        # ── Build manifest ──
        manifest = self._build_manifest(
            commits=commits,
            rows=rows,
            supervised_rows=supervised_rows,
            ambiguous_rows=ambiguous_rows,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            bug_fix_shas=bug_fix_shas,
            revert_map=revert_map,
            attributions=attributions,
            repo_url=repo_url,
            repo_name=repo_name,
            repo_revision=repo_revision,
        )

        self._write_manifest(output_dir / "metadata.json", manifest)
        return manifest

    # ── Internal helpers ───────────────────────────────────────────────────

    def _assign_split(self, commit_index: int) -> str:
        """Assign a split based on chronological position.

        The split is deterministic: position-based, no randomness.
        """
        return "train"

    def _determine_split(self, total_commits: int) -> dict[str, tuple[int, int]]:
        """Determine the commit index ranges for each split.

        Returns {split_name: (start_index, end_index)}.
        """
        n = total_commits
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        return {
            "train": (0, train_end),
            "validation": (train_end, val_end),
            "test": (val_end, n),
        }

    def _assign_split_from_ranges(
        self, commit_index: int, split_ranges: dict[str, tuple[int, int]],
    ) -> str:
        """Assign split from precomputed ranges."""
        for split_name, (start, end) in split_ranges.items():
            if start <= commit_index < end:
                return split_name
        return "test"  # fallback

    def _write_jsonl(self, path: Path, rows: list[DatasetRow]) -> None:
        """Write rows to a JSONL file in deterministic order."""
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                line = row.model_dump_json()
                f.write(line + "\n")

    def _write_manifest(self, path: Path, manifest: DatasetManifest) -> None:
        """Write the manifest as formatted JSON."""
        content = json.dumps(
            manifest.model_dump(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        path.write_text(content, encoding="utf-8")

    def _build_manifest(
        self,
        commits: list[CommitInfo],
        rows: list[DatasetRow],
        supervised_rows: list[DatasetRow],
        ambiguous_rows: list[DatasetRow],
        train_rows: list[DatasetRow],
        val_rows: list[DatasetRow],
        test_rows: list[DatasetRow],
        bug_fix_shas: set[str],
        revert_map: dict[str, str],
        attributions: dict[str, tuple[str, int, float, str, str]],
        repo_url: str,
        repo_name: str,
        repo_revision: str,
    ) -> DatasetManifest:
        """Build the dataset manifest with statistics."""
        # Count attributions by source
        explicit_sha_count = sum(
            1 for v in attributions.values() if v[3] == "explicit_sha_reference"
        )
        revert_count = sum(
            1 for v in attributions.values() if v[3] == "revert"
        )
        line_overlap_count = sum(
            1 for v in attributions.values() if v[3] == "line_overlap"
        )
        ambiguous_count = sum(
            1 for v in attributions.values() if v[3] == "ambiguous"
        )

        # Count labels
        positive_count = sum(1 for r in rows if r.label_status == "positive")
        negative_count = sum(1 for r in rows if r.label_status == "negative")
        ambiguous_file_count = len(ambiguous_rows)
        total_count = len(rows)
        positive_rate = positive_count / total_count if total_count > 0 else 0.0

        # Time range
        timestamps = [
            c.author_date
            for c in commits
            if c.author_date is not None
        ]
        time_start = min(timestamps).isoformat() if timestamps else None
        time_end = max(timestamps).isoformat() if timestamps else None

        # Language distributions (placeholder — based on file extensions)
        commits_per_lang: dict[str, int] = Counter()
        examples_per_lang: dict[str, int] = Counter()
        for c in commits:
            for f in c.files:
                ext = Path(f.path).suffix or "unknown"
                commits_per_lang[ext] += 1
                examples_per_lang[ext] += 1

        # Split ranges
        split_ranges = self._determine_split(len(commits))
        train_commits = sum(
            1 for i in range(len(commits))
            if self._assign_split_from_ranges(i, split_ranges) == "train"
        )
        val_commits = sum(
            1 for i in range(len(commits))
            if self._assign_split_from_ranges(i, split_ranges) == "validation"
        )
        test_commits = sum(
            1 for i in range(len(commits))
            if self._assign_split_from_ranges(i, split_ranges) == "test"
        )

        return DatasetManifest(
            dataset_version=self.config.dataset_version,
            feature_version="v1",
            labeling_version=self.config.labeling_version,
            config=self.config.model_dump(),
            repo_url=repo_url,
            repo_name=repo_name,
            repo_revision=repo_revision,
            total_commits_processed=len(commits),
            total_commits_included=len(commits),
            total_commits_excluded=0,
            total_file_examples=total_count,
            positive_examples=positive_count,
            negative_examples=negative_count,
            ambiguous_examples=ambiguous_file_count,
            positive_rate=positive_rate,
            train_commits=train_commits,
            validation_commits=val_commits,
            test_commits=test_commits,
            train_examples=len(train_rows),
            validation_examples=len(val_rows),
            test_examples=len(test_rows),
            commits_per_language=dict(commits_per_lang),
            examples_per_language=dict(examples_per_lang),
            time_range_start=time_start,
            time_range_end=time_end,
            bug_fix_commits_found=len(bug_fix_shas),
            revert_commits_found=len(revert_map),
            explicit_sha_attributions=explicit_sha_count,
            revert_attributions=revert_count,
            line_overlap_attributions=line_overlap_count,
            ambiguous_attributions=ambiguous_count,
            exclusion_counts={},
        )

    def _write_empty_dataset(
        self,
        output_dir: Path,
        repo_url: str,
        repo_name: str,
        repo_revision: str,
    ) -> DatasetManifest:
        """Write an empty dataset with a valid manifest."""
        manifest = DatasetManifest(
            dataset_version=self.config.dataset_version,
            feature_version="v1",
            labeling_version=self.config.labeling_version,
            config=self.config.model_dump(),
            repo_url=repo_url,
            repo_name=repo_name,
            repo_revision=repo_revision,
            total_commits_processed=0,
            total_commits_included=0,
            total_commits_excluded=0,
            total_file_examples=0,
            positive_examples=0,
            negative_examples=0,
            ambiguous_examples=0,
            positive_rate=0.0,
            train_commits=0,
            validation_commits=0,
            test_commits=0,
            train_examples=0,
            validation_examples=0,
            test_examples=0,
            commits_per_language={},
            examples_per_language={},
            time_range_start=None,
            time_range_end=None,
            bug_fix_commits_found=0,
            revert_commits_found=0,
            explicit_sha_attributions=0,
            revert_attributions=0,
            line_overlap_attributions=0,
            ambiguous_attributions=0,
            exclusion_counts={},
        )

        self._write_jsonl(output_dir / "train.jsonl", [])
        self._write_jsonl(output_dir / "validation.jsonl", [])
        self._write_jsonl(output_dir / "test.jsonl", [])
        self._write_jsonl(output_dir / "ambiguous.jsonl", [])
        self._write_manifest(output_dir / "metadata.json", manifest)
        return manifest
