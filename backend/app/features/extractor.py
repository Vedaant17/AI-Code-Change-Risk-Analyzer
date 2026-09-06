"""Public feature extraction API (Phase 2).

Single entry point: ``CommitInfo`` → ``FeatureExtractionResult``.
"""

from __future__ import annotations

from backend.app.features.commit_features import extract_commit_features
from backend.app.features.file_features import extract_file_features
from backend.app.features.schemas import (
    COMMIT_FEATURE_NAMES,
    FEATURE_VERSION,
    FILE_FEATURE_NAMES,
    CommitFeatures,
    FeatureExtractionResult,
    FileFeatures,
)
from backend.app.schemas.diff import CommitInfo


class FeatureExtractor:
    """Extracts deterministic feature vectors from a ``CommitInfo``.

    Usage::

        extractor = FeatureExtractor()
        result = extractor.extract(commit_info)

        # Ordered numerical vectors for ML
        commit_vec = result.commit_features.to_feature_vector()
        file_vecs = [ff.to_feature_vector() for ff in result.file_features]
    """

    def extract(self, commit_info: CommitInfo) -> FeatureExtractionResult:
        """Extract all features from a structured commit.

        Parameters
        ----------
        commit_info:
            Phase 1 structured commit output.

        Returns
        -------
        FeatureExtractionResult
            Contains ``commit_features``, ``file_features``, and
            ``feature_version``.
        """
        file_features = [extract_file_features(f) for f in commit_info.files]
        commit_features = extract_commit_features(commit_info, file_features)

        return FeatureExtractionResult(
            commit_features=commit_features,
            file_features=file_features,
            feature_version=FEATURE_VERSION,
        )
