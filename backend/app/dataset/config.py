"""Configuration for historical dataset construction (Phase 3).

All fields are deterministic and reproducible.  No volatile values
(datetime.now, random IDs) are permitted in this model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Tunable parameters for dataset construction.

    Changing these parameters changes the dataset.  The config is
    serialized into the dataset manifest for reproducibility.
    """

    # ── Observation window (forward scan for defect evidence) ──
    observation_window_commits: int = Field(
        default=200,
        description="Max subsequent commits examined for defect evidence",
    )
    observation_window_days: int = Field(
        default=90,
        description="Max days forward to look for defect evidence",
    )

    # ── Lookback window (backward scan for attribution) ──
    lookback_commits: int = Field(
        default=100,
        description="Max preceding commits examined when attributing a bug-fix",
    )

    # ── Medium-confidence attribution thresholds ──
    max_files_for_medium_attribution: int = Field(
        default=5,
        description="Bug-fixes with more files cannot qualify for medium line-overlap",
    )
    max_lines_for_medium_attribution: int = Field(
        default=200,
        description="Bug-fixes touching more lines cannot qualify for medium-confidence",
    )

    # ── Full exclusion threshold ──
    max_files_for_exclusion: int = Field(
        default=20,
        description="Bug-fixes touching more files are excluded from ALL attribution",
    )

    # ── Chronological split ratios ──
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # ── Schema versioning ──
    dataset_version: str = Field(default="v1")
    labeling_version: str = Field(default="v1")
