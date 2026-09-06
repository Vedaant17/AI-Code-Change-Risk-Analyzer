"""Feature schemas for the ML pipeline (Phase 2).

Defines deterministic, versioned Pydantic models for commit-level and
file-level features.  Each model exposes an ``to_feature_vector()`` method
that returns values in a **stable, documented order** — the order is the
canonical contract for any downstream ML training or inference.

The feature vector is a ``list[float]`` where every element corresponds
to a named field listed in ``FEATURE_NAMES``.  The ML layer must use
``FEATURE_NAMES`` to convert between dicts and ordered vectors.

Version history
---------------
v1 — 2026-09-06 — Initial feature set for Phase 2 MVP.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

# ── Version contract ──────────────────────────────────────────────────────

FEATURE_VERSION = "v1"

# ───────────────────────────────────────────────────────────────────────────
# Commit-level features
# ───────────────────────────────────────────────────────────────────────────

COMMIT_FEATURE_NAMES: list[str] = [
    "lines_added",
    "lines_deleted",
    "total_lines_changed",
    "files_changed",
    "files_added",
    "files_deleted",
    "files_modified",
    "files_renamed",
    "files_binary",
    "total_hunks",
    "avg_hunks_per_file",
    "additions_ratio",
    "deletions_ratio",
    "test_files_changed",
    "prod_files_changed",
    "test_ratio",
    "has_test_changes",
    "has_prod_changes",
    "test_prod_coupling",
    "languages_touched",
    "primary_language",
    "avg_file_changes",
    "max_file_changes",
    "total_function_declarations_changed",
    "total_class_declarations_changed",
    "total_imports_changed",
    "avg_changed_line_indent",
    "max_changed_line_indent",
    "change_entropy",
]


class CommitFeatures(BaseModel):
    """Deterministic feature vector for an entire commit / PR diff.

    Every field (except ``commit_sha``) maps 1-to-1 to an entry in
    ``COMMIT_FEATURE_NAMES``.  ``commit_sha`` is metadata only and is
    **not** included in the feature vector.
    """

    commit_sha: str = Field(default="", description="Traceability — not a feature")

    # ── Change volume ──
    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)
    total_lines_changed: int = Field(default=0, ge=0)

    # ── File counts ──
    files_changed: int = Field(default=0, ge=0)
    files_added: int = Field(default=0, ge=0)
    files_deleted: int = Field(default=0, ge=0)
    files_modified: int = Field(default=0, ge=0)
    files_renamed: int = Field(default=0, ge=0)
    files_binary: int = Field(default=0, ge=0)

    # ── Hunk density ──
    total_hunks: int = Field(default=0, ge=0)
    avg_hunks_per_file: float = Field(default=0.0, ge=0.0)

    # ── Ratios ──
    additions_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    deletions_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Test / production coupling ──
    test_files_changed: int = Field(default=0, ge=0)
    prod_files_changed: int = Field(default=0, ge=0)
    test_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    has_test_changes: bool = Field(default=False)
    has_prod_changes: bool = Field(default=False)
    test_prod_coupling: bool = Field(default=False)

    # ── Language diversity ──
    languages_touched: int = Field(default=0, ge=0)
    primary_language: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Index of the primary language in the canonical language list (0-1)",
    )

    # ── Change distribution ──
    avg_file_changes: float = Field(default=0.0, ge=0.0)
    max_file_changes: int = Field(default=0, ge=0)

    # ── Heuristic code-level signals (from hunk content) ──
    total_function_declarations_changed: int = Field(default=0, ge=0)
    total_class_declarations_changed: int = Field(default=0, ge=0)
    total_imports_changed: int = Field(default=0, ge=0)

    # ── Structural proxy: indentation of changed lines ──
    avg_changed_line_indent: float = Field(default=0.0, ge=0.0)
    max_changed_line_indent: int = Field(default=0, ge=0)

    # ── Distribution shape ──
    change_entropy: float = Field(default=0.0, ge=0.0)

    def to_feature_vector(self) -> list[float]:
        """Return values in the canonical ``COMMIT_FEATURE_NAMES`` order."""
        d = self.model_dump()
        return [float(d[name]) for name in COMMIT_FEATURE_NAMES]


# ───────────────────────────────────────────────────────────────────────────
# File-level features
# ───────────────────────────────────────────────────────────────────────────

FILE_FEATURE_NAMES: list[str] = [
    "language",
    "is_binary",
    "is_test_file",
    "lines_added",
    "lines_deleted",
    "total_lines_changed",
    "hunk_count",
    "function_declarations_added",
    "function_declarations_deleted",
    "class_declarations_added",
    "class_declarations_deleted",
    "imports_added",
    "imports_deleted",
    "avg_changed_line_indent",
    "max_changed_line_indent",
    "avg_hunk_size",
]


class FileFeatures(BaseModel):
    """Deterministic feature vector for a single changed file.

    ``file_path`` is metadata only.  ``to_feature_vector()`` returns
    values in the canonical ``FILE_FEATURE_NAMES`` order.
    """

    file_path: str = Field(default="", description="Traceability — not a feature")

    # ── Metadata ──
    language: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Index of the language in the canonical language list (0-1)",
    )
    is_binary: bool = Field(default=False)
    is_test_file: bool = Field(default=False)

    # ── Change volume ──
    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)
    total_lines_changed: int = Field(default=0, ge=0)
    hunk_count: int = Field(default=0, ge=0)

    # ── Heuristic declaration detections (from hunk +/- lines) ──
    function_declarations_added: int = Field(default=0, ge=0)
    function_declarations_deleted: int = Field(default=0, ge=0)
    class_declarations_added: int = Field(default=0, ge=0)
    class_declarations_deleted: int = Field(default=0, ge=0)

    # ── Import changes ──
    imports_added: int = Field(default=0, ge=0)
    imports_deleted: int = Field(default=0, ge=0)

    # ── Structural proxy: indentation of changed lines ──
    avg_changed_line_indent: float = Field(default=0.0, ge=0.0)
    max_changed_line_indent: int = Field(default=0, ge=0)

    # ── Hunk size distribution ──
    avg_hunk_size: float = Field(default=0.0, ge=0.0)

    def to_feature_vector(self) -> list[float]:
        """Return values in the canonical ``FILE_FEATURE_NAMES`` order."""
        d = self.model_dump()
        return [float(d[name]) for name in FILE_FEATURE_NAMES]


# ───────────────────────────────────────────────────────────────────────────
# Combined extraction result
# ───────────────────────────────────────────────────────────────────────────


class FeatureExtractionResult(BaseModel):
    """Container returned by the public ``FeatureExtractor``."""

    commit_features: CommitFeatures
    file_features: list[FileFeatures]
    feature_version: str = FEATURE_VERSION
