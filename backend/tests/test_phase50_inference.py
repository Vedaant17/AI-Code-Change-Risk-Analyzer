"""Tests for Phase 5.0: Production Inference Layer (B1_CHANGE_SIZE).

Tests the inference service, B1 scoring, feature pipeline, strategy
registry, and error hierarchy.  Uses mock/fake GitService to avoid
cloning real repositories.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.app.features.schemas import FeatureExtractionResult
from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
    StrategyArtifactMissingError,
    UnsupportedStrategyError,
)
from backend.app.inference.feature_pipeline import (
    FileFeatures,
    PipelineResult,
    _map_file_status,
    extract_features,
)
from backend.app.inference.registry import (
    DEFAULT_STRATEGY,
    get_production_strategy,
    get_strategy,
    list_production_strategies,
)
from backend.app.inference.schemas import (
    AnalyzeRiskRequest,
    AnalyzeRiskResponse,
    FileRiskResult,
)
from backend.app.inference.scoring import score_files
from backend.app.inference.service import InferenceService
from backend.app.schemas.diff import CommitInfo, FileDiff, FileStatus, Hunk
from backend.app.services.git_service import GitServiceError

# ── Errors ─────────────────────────────────────────────────────────────────


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(InferenceError, Exception)
        assert issubclass(RepositoryAccessError, InferenceError)
        assert issubclass(CommitNotFoundError, InferenceError)
        assert issubclass(FeatureExtractionError, InferenceError)
        assert issubclass(UnsupportedStrategyError, InferenceError)
        assert issubclass(StrategyArtifactMissingError, InferenceError)

    def test_instantiate_with_message(self):
        err = InferenceError("test message")
        assert str(err) == "test message"

    def test_subclass_with_message(self):
        err = RepositoryAccessError("cannot clone")
        assert str(err) == "cannot clone"
        assert isinstance(err, InferenceError)


# ── Schemas ────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_request_valid(self):
        req = AnalyzeRiskRequest(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123def456",
        )
        assert req.repo_url == "https://github.com/user/repo"
        assert req.commit_sha == "abc123def456"

    def test_request_empty_repo_url_rejected(self):
        with pytest.raises(Exception):
            AnalyzeRiskRequest(repo_url="", commit_sha="abc123")

    def test_request_empty_commit_sha_rejected(self):
        with pytest.raises(Exception):
            AnalyzeRiskRequest(repo_url="https://github.com/u/r", commit_sha="")

    def test_file_risk_result(self):
        r = FileRiskResult(
            path="src/main.py",
            status="modified",
            investigation_priority_score=0.85,
            rank=1,
            total_files_in_commit=3,
            lines_added=10,
            lines_deleted=5,
            is_binary=False,
            language="Python",
            is_test_file=False,
        )
        assert r.path == "src/main.py"
        assert r.investigation_priority_score == 0.85

    def test_file_risk_result_score_bounds(self):
        with pytest.raises(Exception):
            FileRiskResult(
                path="f.py",
                status="modified",
                investigation_priority_score=1.5,
                rank=1,
                total_files_in_commit=1,
            )

    def test_response_defaults(self):
        resp = AnalyzeRiskResponse(
            repo_url="https://github.com/u/r",
            commit_sha="a" * 40,
            short_sha="a" * 8,
        )
        assert resp.strategy == "B1_CHANGE_SIZE"
        assert resp.strategy_version == "1.0.0"
        assert resp.feature_version == "v1"
        assert resp.score_semantics == "investigation_priority_ranking"
        assert len(resp.limitations) > 0
        assert resp.files == []
        assert resp.warnings == []

    def test_response_no_post_outcome_fields(self):
        resp = AnalyzeRiskResponse(
            repo_url="https://github.com/u/r",
            commit_sha="a" * 40,
            short_sha="a" * 8,
        )
        serialized = resp.model_dump()
        forbidden = [
            "corrective_sha",
            "label_source",
            "content_correspondence",
            "content_restoration",
            "region_overlap",
            "function_analysis",
            "evidence_types",
            "file_evidence_level",
        ]
        for field in forbidden:
            assert field not in serialized, f"Post-outcome field {field!r} found in response"


# ── Registry ───────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_b1_strategy(self):
        meta = get_strategy("B1_CHANGE_SIZE")
        assert meta.name == "B1_CHANGE_SIZE"
        assert meta.status == "PRODUCTION_SAFE"
        assert meta.requires_training is False
        assert meta.requires_model_artifact is False

    def test_get_production_strategy(self):
        meta = get_production_strategy("B1_CHANGE_SIZE")
        assert meta.status == "PRODUCTION_SAFE"

    def test_get_production_strategy_rejects_research_only(self):
        with pytest.raises(UnsupportedStrategyError, match="research-only"):
            get_production_strategy("B0_RANDOM")

    def test_get_production_strategy_rejects_unknown(self):
        with pytest.raises(UnsupportedStrategyError, match="Unknown strategy"):
            get_production_strategy("NONEXISTENT")

    def test_list_production_strategies(self):
        prods = list_production_strategies()
        names = [s.name for s in prods]
        assert "B1_CHANGE_SIZE" in names
        for s in prods:
            assert s.status == "PRODUCTION_SAFE"

    def test_research_strategies_exist(self):
        for name in ["B0_RANDOM", "B2_BIASED_PVU", "B3_EVIDENCE_RANKING", "B4_COMMIT_LEVEL"]:
            meta = get_strategy(name)
            assert meta.status == "RESEARCH_ONLY"

    def test_default_strategy_is_b1(self):
        assert DEFAULT_STRATEGY == "B1_CHANGE_SIZE"

    def test_strategy_meta_frozen(self):
        meta = get_strategy("B1_CHANGE_SIZE")
        with pytest.raises(Exception):
            meta.name = "changed"


# ── Feature Pipeline ───────────────────────────────────────────────────────


def _make_commit_info(
    sha: str = "a" * 40,
    files: list[FileDiff] | None = None,
) -> CommitInfo:
    """Helper to build a CommitInfo for testing."""
    if files is None:
        files = [
            FileDiff(
                path="src/main.py",
                status=FileStatus.MODIFIED,
                lines_added=5,
                lines_deleted=2,
                hunks=[
                    Hunk(
                        old_start=1,
                        old_count=10,
                        new_start=1,
                        new_count=13,
                        content=(
                            " import os\n"
                            "-import sys\n"
                            "+import sys\n"
                            "+import json\n"
                            " \n"
                            " def hello():\n"
                            "-    pass\n"
                            "+    print('hello')\n"
                            "+    return True\n"
                            " \n"
                            " class Foo:\n"
                            "     pass\n"
                        ),
                    )
                ],
            )
        ]
    info = CommitInfo(
        sha=sha,
        short_sha=sha[:8],
        author="test",
        author_date=datetime(2026, 1, 1, tzinfo=UTC),
        message="test commit",
        files=files,
    )
    info.compute_stats()
    return info


class TestFeaturePipeline:
    def test_build_file_features_single(self):
        commit = _make_commit_info()
        extractor = MagicMock()
        mock_result = MagicMock(spec=FeatureExtractionResult)
        mock_result.feature_version = "v1"
        file_feat = MagicMock()
        file_feat.total_lines_changed = 7
        file_feat.language = 0.0
        file_feat.is_test_file = False
        mock_result.file_features = [file_feat]
        extractor.extract.return_value = mock_result

        with patch(
            "backend.app.inference.feature_pipeline.FeatureExtractor",
            return_value=extractor,
        ):
            result = extract_features(commit)

        assert len(result.file_features) == 1
        ff = result.file_features[0]
        assert ff.path == "src/main.py"
        assert ff.status == "modified"
        assert ff.total_lines_changed == 7
        assert ff.is_binary is False

    def test_build_file_features_binary(self):
        commit = _make_commit_info(
            files=[
                FileDiff(
                    path="image.png",
                    status=FileStatus.BINARY,
                    is_binary=True,
                )
            ]
        )
        extractor = MagicMock()
        mock_result = MagicMock(spec=FeatureExtractionResult)
        mock_result.feature_version = "v1"
        file_feat = MagicMock()
        file_feat.total_lines_changed = 0
        file_feat.language = 0.0
        file_feat.is_test_file = False
        mock_result.file_features = [file_feat]
        extractor.extract.return_value = mock_result

        with patch(
            "backend.app.inference.feature_pipeline.FeatureExtractor",
            return_value=extractor,
        ):
            result = extract_features(commit)

        ff = result.file_features[0]
        assert ff.is_binary is True
        assert ff.total_lines_changed == 0

    def test_map_file_status(self):
        for status in FileStatus:
            fd = FileDiff(path="f.py", status=status)
            mapped = _map_file_status(fd)
            assert isinstance(mapped, str)

    def test_empty_commit(self):
        commit = _make_commit_info(files=[])
        extractor = MagicMock()
        mock_result = MagicMock(spec=FeatureExtractionResult)
        mock_result.feature_version = "v1"
        mock_result.file_features = []
        extractor.extract.return_value = mock_result

        with patch(
            "backend.app.inference.feature_pipeline.FeatureExtractor",
            return_value=extractor,
        ):
            result = extract_features(commit)

        assert result.file_features == []


# ── Scoring ────────────────────────────────────────────────────────────────


def _make_file_features(
    path: str,
    total_lines_changed: int,
    is_binary: bool = False,
    lines_added: int = 0,
    lines_deleted: int = 0,
    status: str = "modified",
    language: float = 0.0,
    is_test_file: bool = False,
) -> FileFeatures:
    return FileFeatures(
        path=path,
        status=status,
        total_lines_changed=total_lines_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        is_binary=is_binary,
        language=language,
        is_test_file=is_test_file,
    )


def _make_pipeline_result(
    sha: str = "a" * 40,
    file_features: list[FileFeatures] | None = None,
) -> PipelineResult:
    commit = _make_commit_info(sha=sha)
    mock_extraction = MagicMock(spec=FeatureExtractionResult)
    return PipelineResult(
        commit_info=commit,
        extraction_result=mock_extraction,
        file_features=file_features or [],
    )


class TestB1Scoring:
    def test_single_file_score_one(self):
        ff = _make_file_features("a.py", total_lines_changed=10)
        pr = _make_pipeline_result(file_features=[ff])
        scored = score_files(pr)
        assert len(scored) == 1
        assert scored[0][1] == 1.0

    def test_two_files_ranking(self):
        ff1 = _make_file_features("big.py", total_lines_changed=100)
        ff2 = _make_file_features("small.py", total_lines_changed=5)
        pr = _make_pipeline_result(file_features=[ff1, ff2])
        scored = score_files(pr)
        by_path = {s[0].path: s[1] for s in scored}
        assert by_path["big.py"] > by_path["small.py"]
        assert by_path["big.py"] == 1.0
        assert by_path["small.py"] == 0.0

    def test_equal_scores_tiebreak_by_order(self):
        ff1 = _make_file_features("first.py", total_lines_changed=10)
        ff2 = _make_file_features("second.py", total_lines_changed=10)
        pr = _make_pipeline_result(file_features=[ff1, ff2])
        scored = score_files(pr)
        assert scored[0][0].path == "first.py"
        assert scored[1][0].path == "second.py"

    def test_empty_files(self):
        pr = _make_pipeline_result(file_features=[])
        scored = score_files(pr)
        assert scored == []

    def test_binary_file_score_zero(self):
        ff = _make_file_features("image.png", total_lines_changed=0, is_binary=True)
        pr = _make_pipeline_result(file_features=[ff])
        scored = score_files(pr)
        assert scored[0][1] == 0.0

    def test_all_scores_in_0_1(self):
        files = [
            _make_file_features(f"f{i}.py", total_lines_changed=i * 10)
            for i in range(5)
        ]
        pr = _make_pipeline_result(file_features=files)
        scored = score_files(pr)
        for _, score in scored:
            assert 0.0 <= score <= 1.0

    def test_deterministic_same_input(self):
        files = [
            _make_file_features("a.py", total_lines_changed=50),
            _make_file_features("b.py", total_lines_changed=25),
        ]
        pr1 = _make_pipeline_result(sha="a" * 40, file_features=files)
        pr2 = _make_pipeline_result(sha="a" * 40, file_features=files)
        s1 = score_files(pr1)
        s2 = score_files(pr2)
        assert [(s[0].path, s[1]) for s in s1] == [(s[0].path, s[1]) for s in s2]

    def test_three_file_descending_order(self):
        ff1 = _make_file_features("a.py", total_lines_changed=30)
        ff2 = _make_file_features("b.py", total_lines_changed=10)
        ff3 = _make_file_features("c.py", total_lines_changed=20)
        pr = _make_pipeline_result(file_features=[ff1, ff2, ff3])
        scored = score_files(pr)
        scores = [s[1] for s in scored]
        assert scores == sorted(scores, reverse=True)


# ── Service ────────────────────────────────────────────────────────────────


class TestInferenceService:
    def _fake_commit(self, sha="a" * 40) -> CommitInfo:
        return _make_commit_info(
            sha=sha,
            files=[
                FileDiff(
                    path="src/main.py",
                    status=FileStatus.MODIFIED,
                    lines_added=5,
                    lines_deleted=2,
                    hunks=[
                        Hunk(
                            old_start=1,
                            old_count=10,
                            new_start=1,
                            new_count=13,
                            content=(
                                " import os\n"
                                "-import sys\n"
                                "+import sys\n"
                                "+import json\n"
                                " \n"
                                " def hello():\n"
                                "-    pass\n"
                                "+    print('hello')\n"
                                "+    return True\n"
                                " \n"
                                " class Foo:\n"
                                "     pass\n"
                            ),
                        )
                    ],
                )
            ],
        )

    def test_analyze_local_commit(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)

        assert isinstance(resp, AnalyzeRiskResponse)
        assert resp.commit_sha == "a" * 40
        assert resp.strategy == "B1_CHANGE_SIZE"
        assert resp.total_files == 1
        assert resp.files_analyzed == 1
        assert len(resp.files) == 1
        assert resp.files[0].path == "src/main.py"
        assert resp.files[0].investigation_priority_score == 1.0

    def test_analyze_local_commit_git_error(self):
        git_svc = MagicMock()
        git_svc.get_commit_diff.side_effect = GitServiceError("unknown commit")

        svc = InferenceService(git_service=git_svc)
        with pytest.raises(CommitNotFoundError):
            svc.analyze_local_commit("/tmp/repo", "bad")

    def test_analyze_local_commit_binary_only(self):
        commit = _make_commit_info(
            files=[
                FileDiff(
                    path="image.png",
                    status=FileStatus.BINARY,
                    is_binary=True,
                )
            ]
        )
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)

        assert resp.files_skipped == 1
        assert resp.files_analyzed == 0
        assert any("Binary" in w for w in resp.warnings)

    def test_analyze_local_commit_multiple_files(self):
        commit = _make_commit_info(
            files=[
                FileDiff(
                    path="big.py",
                    status=FileStatus.MODIFIED,
                    lines_added=50,
                    lines_deleted=10,
                    hunks=[
                        Hunk(
                            old_start=1, old_count=20, new_start=1, new_count=60,
                            content=" line\n" * 60,
                        )
                    ],
                ),
                FileDiff(
                    path="small.py",
                    status=FileStatus.MODIFIED,
                    lines_added=2,
                    lines_deleted=1,
                    hunks=[
                        Hunk(
                            old_start=1, old_count=3, new_start=1, new_count=4,
                            content=" line\n" * 4,
                        )
                    ],
                ),
            ]
        )
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)

        assert resp.total_files == 2
        assert resp.files_analyzed == 2
        by_path = {f.path: f.investigation_priority_score for f in resp.files}
        assert by_path["big.py"] > by_path["small.py"]

    def test_analyze_local_commit_rejects_research_strategy(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        with pytest.raises(UnsupportedStrategyError, match="research-only"):
            svc.analyze_local_commit("/tmp/repo", "a" * 40, strategy="B0_RANDOM")

    def test_analyze_local_commit_empty_commit(self):
        commit = _make_commit_info(files=[])
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)

        assert resp.total_files == 0
        assert resp.files_analyzed == 0
        assert resp.files == []

    def test_analyze_local_commit_elapsed_ms_positive(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)
        assert resp.elapsed_ms >= 0.0

    def test_analyze_local_commit_analyzed_at_populated(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)
        assert resp.analyzed_at != ""

    def test_analyze_local_commit_short_sha(self):
        commit = self._fake_commit(sha="abc123" + "0" * 34)
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "abc123" + "0" * 34)
        assert resp.short_sha == "abc12300"

    def test_response_feature_version(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)
        assert resp.feature_version == "v1"

    def test_response_score_semantics_identifier(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)
        assert resp.score_semantics == "investigation_priority_ranking"

    def test_response_limitations_not_empty(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        resp = svc.analyze_local_commit("/tmp/repo", "a" * 40)
        assert len(resp.limitations) > 0
        assert any("defect probabilities" in lim for lim in resp.limitations)

    def test_service_rejects_feature_extraction_error(self):
        commit = self._fake_commit()
        git_svc = MagicMock()
        git_svc.get_commit_diff.return_value = commit

        svc = InferenceService(git_service=git_svc)
        with patch(
            "backend.app.inference.service.extract_features",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(FeatureExtractionError):
                svc.analyze_local_commit("/tmp/repo", "a" * 40)

    def test_service_rejects_clone_error(self):
        git_svc = MagicMock()
        git_svc.clone_repo.side_effect = GitServiceError("network error")

        svc = InferenceService(git_service=git_svc)
        with pytest.raises(RepositoryAccessError):
            svc.analyze_commit("https://github.com/u/r", "a" * 40)
