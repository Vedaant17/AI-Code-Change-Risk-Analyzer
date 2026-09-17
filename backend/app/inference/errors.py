"""Inference-layer error hierarchy (Phase 5.0).

All inference errors inherit from ``InferenceError``.  HTTP mapping
belongs to the API layer (Phase 5.1) and is NOT defined here.
"""
from __future__ import annotations


class InferenceError(Exception):
    """Base for all inference-layer errors."""


class RepositoryAccessError(InferenceError):
    """Cannot clone or open the target repository."""


class CommitNotFoundError(InferenceError):
    """The requested commit SHA does not exist in the repository."""


class FeatureExtractionError(InferenceError):
    """Feature extraction failed for a file or commit."""


class UnsupportedStrategyError(InferenceError):
    """The requested ranking strategy is not available for production use."""


class StrategyArtifactMissingError(InferenceError):
    """A strategy requires a trained model artifact that is not available."""
