"""Experimental feature schema for Phase 4.5a.

Defines a versioned, standalone feature representation for experimental
features that extend the frozen v1 45-feature baseline.  This schema is
independent of ``DatasetRow`` and the v1 feature contract.

The experimental features are combined with v1 features at load time
by ``backend.app.ml.experimental_loader``, enabling ablation experiments
(E0/E1/E2/E4) without modifying the serialized dataset.

Version history
---------------
exp-4.5a -- 2026-09-08 -- Initial 6-feature subset (3 AST + 3 historical).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# -- Version contract --------------------------------------------------------

EXPERIMENTAL_FEATURE_VERSION = "exp-4.5a"

# -- Feature names (canonical ordering) -------------------------------------

EXPERIMENTAL_FEATURE_NAMES: list[str] = [
    # AST structural features (changed-code analysis)
    "file_ast_functions_added",      # 0: FunctionDef/AsyncFunctionDef in C
                                     #    whose source range overlaps added lines
    "file_ast_functions_deleted",    # 1: FunctionDef/AsyncFunctionDef in C^
                                     #    whose source range overlaps deleted lines
    "file_ast_try_except_changed",   # 2: Try nodes in C or C^ whose source
                                     #    range overlaps any changed lines

    # Historical file features
    "file_commit_count",             # 3: commits touching file before C
    "file_historical_bug_fixes",     # 4: bug-fix commits touching file before C
    "file_days_since_last_change",   # 5: days between last prior modification
                                     #    and C
]

EXPERIMENTAL_FEATURE_COUNT: int = len(EXPERIMENTAL_FEATURE_NAMES)

assert EXPERIMENTAL_FEATURE_COUNT == 6, (
    f"Expected 6 experimental features, got {EXPERIMENTAL_FEATURE_COUNT}"
)


# -- Feature model ----------------------------------------------------------


class ExperimentalFileFeatures(BaseModel):
    """Experimental per-file features for Phase 4.5a.

    ``file_path`` is metadata only and is not included in the feature
    vector.  ``to_feature_vector()`` returns values in the canonical
    ``EXPERIMENTAL_FEATURE_NAMES`` order.

    Feature semantics
    -----------------
    file_ast_functions_added:
        Count of FunctionDef/AsyncFunctionDef nodes in the new-file AST
        (parsed from full file content at C) whose source line range
        overlaps at least one line added in the diff.  This measures
        *function regions affected by added lines*, not literal function
        declaration additions.

    file_ast_functions_deleted:
        Count of FunctionDef/AsyncFunctionDef nodes in the old-file AST
        (parsed from full file content at C^) whose source line range
        overlaps at least one line deleted in the diff.  This measures
        *function regions affected by deleted lines*, not literal function
        declaration deletions.

    file_ast_try_except_changed:
        Count of Try nodes in either the old-file or new-file AST whose
        source line range overlaps any changed lines (added or deleted).
        This is a count, not a boolean flag.

    file_commit_count:
        Number of commits that touched this file before C, counted via
        ``git rev-list C^ -- file``.  C itself is excluded.

    file_historical_bug_fixes:
        Number of commits touching this file before C whose messages match
        the frozen bug-fix patterns from ``labeling.py``.

    file_days_since_last_change:
        Days between the most recent commit touching this file before C
        and C's timestamp.  0 for newly created files.
    """

    file_path: str = Field(default="", description="Traceability -- not a feature")

    # -- AST structural features --
    file_ast_functions_added: float = Field(
        default=0.0, ge=0.0,
        description="FunctionDef/AsyncFunctionDef in C overlapping added lines",
    )
    file_ast_functions_deleted: float = Field(
        default=0.0, ge=0.0,
        description="FunctionDef/AsyncFunctionDef in C^ overlapping deleted lines",
    )
    file_ast_try_except_changed: float = Field(
        default=0.0, ge=0.0,
        description="Try nodes overlapping any changed lines (count)",
    )

    # -- Historical file features --
    file_commit_count: float = Field(
        default=0.0, ge=0.0,
        description="Commits touching file before C",
    )
    file_historical_bug_fixes: float = Field(
        default=0.0, ge=0.0,
        description="Bug-fix commits touching file before C",
    )
    file_days_since_last_change: float = Field(
        default=0.0, ge=0.0,
        description="Days since last modification before C",
    )

    def to_feature_vector(self) -> list[float]:
        """Return values in the canonical ``EXPERIMENTAL_FEATURE_NAMES`` order."""
        d = self.model_dump()
        return [float(d[name]) for name in EXPERIMENTAL_FEATURE_NAMES]
