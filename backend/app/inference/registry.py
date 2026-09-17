"""Production strategy registry (Phase 5.0).

Defines the available production ranking strategies and their metadata.
Only strategies validated for production use are selectable.  Experimental
strategies (B0, B2, B3, B4) are recorded for traceability but rejected
at runtime if requested.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyMeta:
    """Immutable metadata for a ranking strategy."""

    name: str
    display_name: str
    description: str
    score_semantics: str
    requires_training: bool
    requires_model_artifact: bool
    feature_modes: tuple[str, ...]
    status: str  # "PRODUCTION_SAFE" or "RESEARCH_ONLY"
    scientific_rationale: str
    strategy_version: str = "1.0.0"
    feature_version: str = "v1"


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

_B1 = StrategyMeta(
    name="B1_CHANGE_SIZE",
    display_name="Change-Size Investigation Heuristic",
    description=(
        "Ranks files within a commit by file_total_lines_changed (descending). "
        "A simple, deterministic, explainable heuristic."
    ),
    score_semantics="investigation_priority_ranking",
    requires_training=False,
    requires_model_artifact=False,
    feature_modes=("E0",),
    status="PRODUCTION_SAFE",
    scientific_rationale=(
        "B1 is a deterministic file-level change-size heuristic. "
        "Phase 4.13 showed B1 MRR=0.9519, comparable to ML baselines "
        "(B2 MRR=0.9378, B3 E2/E4 MRR=0.9296). It requires no training "
        "and no model artifacts. It is NOT a defect classifier."
    ),
)

_B0 = StrategyMeta(
    name="B0_RANDOM",
    display_name="Random Baseline",
    description="Random scores within each commit (seeded, deterministic).",
    score_semantics="investigation_priority_ranking",
    requires_training=False,
    requires_model_artifact=False,
    feature_modes=("E0",),
    status="RESEARCH_ONLY",
    scientific_rationale="Seeded random baseline for comparison only.",
    strategy_version="4.13.0",
)

_B2 = StrategyMeta(
    name="B2_BIASED_PVU",
    display_name="Biased P-vs-U XGBoost (Experimental)",
    description="XGBoost trained on P=observed-positive, U=unlabeled.",
    score_semantics="investigation_priority_ranking",
    requires_training=True,
    requires_model_artifact=True,
    feature_modes=("E0", "E1", "E2", "E4"),
    status="RESEARCH_ONLY",
    scientific_rationale=(
        "NOT legitimate PU learning. Biased baseline. "
        "Trained on frozen dataset. Not validated as defect classifier."
    ),
    strategy_version="4.13.0",
)

_B3 = StrategyMeta(
    name="B3_EVIDENCE_RANKING",
    display_name="Evidence-Ranking Pairwise Logistic (Experimental)",
    description="Pairwise logistic trained on STRONG vs WEAK evidence pairs.",
    score_semantics="investigation_priority_ranking",
    requires_training=True,
    requires_model_artifact=True,
    feature_modes=("E0", "E1", "E2", "E4"),
    status="RESEARCH_ONLY",
    scientific_rationale=(
        "Pairwise logistic on evidence-ranking pairs. "
        "Accuracy near chance (0.5-0.7). Exploratory."
    ),
    strategy_version="4.13.0",
)

_B4 = StrategyMeta(
    name="B4_COMMIT_LEVEL",
    display_name="Commit-Level Fallback (Experimental)",
    description="XGBoost at commit level, same score for all files.",
    score_semantics="investigation_priority_ranking",
    requires_training=True,
    requires_model_artifact=True,
    feature_modes=("E0", "E1", "E2", "E4"),
    status="RESEARCH_ONLY",
    scientific_rationale=(
        "Commit-level fallback. No within-commit file discrimination."
    ),
    strategy_version="4.13.0",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_STRATEGIES: dict[str, StrategyMeta] = {
    s.name: s for s in (_B1, _B0, _B2, _B3, _B4)
}

_PRODUCTION_STRATEGIES: dict[str, StrategyMeta] = {
    k: v for k, v in _ALL_STRATEGIES.items() if v.status == "PRODUCTION_SAFE"
}

DEFAULT_STRATEGY: str = "B1_CHANGE_SIZE"


def get_strategy(name: str) -> StrategyMeta:
    """Return strategy metadata by name. Raises ``KeyError`` if unknown."""
    return _ALL_STRATEGIES[name]


def get_production_strategy(name: str) -> StrategyMeta:
    """Return strategy metadata, rejecting non-production strategies.

    Raises ``UnsupportedStrategyError`` if the strategy is unknown or
    not approved for production use.
    """
    from backend.app.inference.errors import UnsupportedStrategyError

    if name not in _ALL_STRATEGIES:
        raise UnsupportedStrategyError(
            f"Unknown strategy: {name!r}. "
            f"Available: {sorted(_ALL_STRATEGIES)}"
        )
    meta = _ALL_STRATEGIES[name]
    if meta.status != "PRODUCTION_SAFE":
        raise UnsupportedStrategyError(
            f"Strategy {name!r} is research-only and not approved for "
            f"production use.  Use {DEFAULT_STRATEGY!r} for production."
        )
    return meta


def list_production_strategies() -> list[StrategyMeta]:
    """Return all production-safe strategies."""
    return list(_PRODUCTION_STRATEGIES.values())
