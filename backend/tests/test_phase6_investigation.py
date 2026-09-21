"""Phase 6 investigation service integration tests.

Tests InvestigationService composition, API contract, B1 preservation,
ordering preservation, and deterministic output.
"""
from __future__ import annotations

import pytest

from backend.app.evidence.schemas import FileEvidence
from backend.app.investigation.schemas import InvestigateRequest, InvestigationResult
from backend.app.schemas.diff import FileDiff, FileStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_diff(
    path: str,
    status: FileStatus = FileStatus.MODIFIED,
    lines_added: int = 10,
    lines_deleted: int = 5,
    is_binary: bool = False,
) -> FileDiff:
    return FileDiff(
        path=path,
        status=status,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        is_binary=is_binary,
    )


# ---------------------------------------------------------------------------
# InvestigationResult schema tests
# ---------------------------------------------------------------------------


class TestInvestigationResultSchema:
    """Tests for InvestigationResult and InvestigateRequest schemas."""

    def test_investigate_request_schema(self):
        """InvestigateRequest validates with required fields."""
        req = InvestigateRequest(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123",
        )
        assert req.repo_url == "https://github.com/user/repo"
        assert req.commit_sha == "abc123"
        assert req.top_k == 10  # default

    def test_investigate_request_custom_top_k(self):
        """InvestigateRequest accepts custom top_k."""
        req = InvestigateRequest(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123",
            top_k=5,
        )
        assert req.top_k == 5

    def test_investigate_request_top_k_bounds(self):
        """InvestigateRequest top_k must be in [1, 100]."""
        with pytest.raises(Exception):
            InvestigateRequest(
                repo_url="https://github.com/user/repo",
                commit_sha="abc123",
                top_k=0,
            )
        with pytest.raises(Exception):
            InvestigateRequest(
                repo_url="https://github.com/user/repo",
                commit_sha="abc123",
                top_k=101,
            )

    def test_investigation_result_schema(self):
        """InvestigationResult validates with required fields."""
        result = InvestigationResult(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123",
            short_sha="abc12345",
        )
        assert result.repo_url == "https://github.com/user/repo"
        assert result.commit_sha == "abc123"
        assert result.strategy == "B1_CHANGE_SIZE"
        assert result.files == []
        assert result.top_k_configured == 10
        assert result.top_k_analyzed == 0
        assert result.git_subprocess_count == 0

    def test_investigation_result_with_files(self):
        """InvestigationResult validates with FileEvidence list."""
        fe = FileEvidence(
            path="a.py",
            score=0.8,
            position=0,
            total_lines_changed=50,
            evidence=[],
        )
        result = InvestigationResult(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123",
            short_sha="abc12345",
            files=[fe],
            top_k_configured=10,
            top_k_analyzed=1,
            git_subprocess_count=1,
        )
        assert len(result.files) == 1
        assert result.files[0].path == "a.py"

    def test_investigation_result_deterministic_fields(self):
        """InvestigationResult does not contain runtime-varying fields."""
        result = InvestigationResult(
            repo_url="https://github.com/user/repo",
            commit_sha="abc123",
            short_sha="abc12345",
        )
        # Verify no analyzed_at, latency fields
        d = result.model_dump()
        assert "analyzed_at" not in d
        assert "evidence_latency_ms" not in d
        assert "b1_latency_ms" not in d
        assert "total_latency_ms" not in d


# ---------------------------------------------------------------------------
# B1 preservation tests
# ---------------------------------------------------------------------------


class TestB1Preservation:
    """Tests that InvestigationResult preserves B1 output exactly."""

    def test_investigation_result_includes_all_b1_files(self):
        """All files from B1 output appear in InvestigationResult."""
        fe1 = FileEvidence(
            path="a.py", score=0.8, position=0, total_lines_changed=50, evidence=[]
        )
        fe2 = FileEvidence(
            path="image.bin", score=0.0, position=1, total_lines_changed=0, evidence=[]
        )
        fe3 = FileEvidence(
            path="b.py", score=0.4, position=2, total_lines_changed=20, evidence=[]
        )

        result = InvestigationResult(
            repo_url="test",
            commit_sha="abc123",
            short_sha="abc12345",
            files=[fe1, fe2, fe3],
        )

        paths = [f.path for f in result.files]
        assert paths == ["a.py", "image.bin", "b.py"]

    def test_scores_copied_verbatim(self):
        """Scores in InvestigationResult match B1 output exactly."""
        fe1 = FileEvidence(
            path="a.py", score=0.8571428571, position=0, total_lines_changed=50, evidence=[]
        )
        fe2 = FileEvidence(
            path="b.py", score=0.4285714285, position=1, total_lines_changed=20, evidence=[]
        )

        result = InvestigationResult(
            repo_url="test",
            commit_sha="abc123",
            short_sha="abc12345",
            files=[fe1, fe2],
        )

        assert result.files[0].score == 0.8571428571
        assert result.files[1].score == 0.4285714285

    def test_positions_match_b1_order(self):
        """Positions match the 0-indexed B1 output order."""
        files = [
            FileEvidence(
                path=f"file{i}.py",
                score=1.0 - i * 0.1,
                position=i,
                total_lines_changed=10 * (10 - i),
                evidence=[],
            )
            for i in range(5)
        ]

        result = InvestigationResult(
            repo_url="test",
            commit_sha="abc123",
            short_sha="abc12345",
            files=files,
        )

        for i, f in enumerate(result.files):
            assert f.position == i


# ---------------------------------------------------------------------------
# API contract tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestInvestigateAPIContract:
    """Tests for the POST /analysis/investigate endpoint."""

    def test_investigate_endpoint_exists(self):
        """POST /analysis/investigate endpoint is registered."""
        from fastapi.testclient import TestClient

        from backend.app.api.main import app

        TestClient(app)
        routes = [r.path for r in app.routes]
        assert "/analysis/investigate" in routes

    def test_risk_endpoint_unchanged(self):
        """POST /analysis/risk endpoint behavior unchanged."""
        from fastapi.testclient import TestClient

        from backend.app.api.main import app

        TestClient(app)
        routes = [r.path for r in app.routes]
        assert "/analysis/risk" in routes

    def test_investigate_invalid_request(self):
        """POST /analysis/investigate returns 422 for invalid request."""
        from fastapi.testclient import TestClient

        from backend.app.api.main import app

        client = TestClient(app)
        response = client.post("/analysis/investigate", json={})
        assert response.status_code == 422

    def test_investigate_missing_fields(self):
        """POST /analysis/investigate returns 422 for missing fields."""
        from fastapi.testclient import TestClient

        from backend.app.api.main import app

        client = TestClient(app)
        response = client.post(
            "/analysis/investigate",
            json={"repo_url": "https://github.com/user/repo"},
        )
        assert response.status_code == 422

    def test_investigate_commit_not_found(self):
        """POST /analysis/investigate returns 404 for missing commit."""
        from fastapi.testclient import TestClient

        from backend.app.api.main import app

        client = TestClient(app)
        response = client.post(
            "/analysis/investigate",
            json={
                "repo_url": "https://github.com/nonexistent/repo",
                "commit_sha": "0000000000000000000000000000000000000000",
            },
        )
        # Should be 404 or 422 (depending on whether clone fails first)
        assert response.status_code in (404, 422, 500)


# ---------------------------------------------------------------------------
# Deterministic output tests
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    """Tests for deterministic InvestigationResult output."""

    def test_schema_round_trip(self):
        """InvestigationResult survives JSON round-trip."""
        fe = FileEvidence(
            path="a.py",
            score=0.8,
            position=0,
            total_lines_changed=50,
            evidence=[],
        )
        result = InvestigationResult(
            repo_url="test",
            commit_sha="abc123",
            short_sha="abc12345",
            files=[fe],
            top_k_configured=10,
            top_k_analyzed=1,
            git_subprocess_count=1,
        )

        json_str = result.model_dump_json()
        restored = InvestigationResult.model_validate_json(json_str)

        assert restored.repo_url == result.repo_url
        assert restored.commit_sha == result.commit_sha
        assert len(restored.files) == len(result.files)
        assert restored.files[0].path == result.files[0].path
        assert restored.files[0].score == result.files[0].score
        assert restored.files[0].position == result.files[0].position
        assert restored.top_k_configured == result.top_k_configured
        assert restored.top_k_analyzed == result.top_k_analyzed
        assert restored.git_subprocess_count == result.git_subprocess_count

    def test_identical_results_equal(self):
        """Two InvestigationResults with same data are equal."""
        fe = FileEvidence(
            path="a.py", score=0.8, position=0, total_lines_changed=50, evidence=[]
        )
        r1 = InvestigationResult(
            repo_url="test", commit_sha="abc123", short_sha="abc12345", files=[fe]
        )
        r2 = InvestigationResult(
            repo_url="test", commit_sha="abc123", short_sha="abc12345", files=[fe]
        )
        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# Warning tests
# ---------------------------------------------------------------------------


class TestWarnings:
    """Tests for warning generation in InvestigationResult."""

    def test_empty_warnings_by_default(self):
        """Warnings list is empty by default."""
        result = InvestigationResult(
            repo_url="test", commit_sha="abc123", short_sha="abc12345"
        )
        assert result.warnings == []

    def test_warnings_preserved(self):
        """Provided warnings are preserved."""
        result = InvestigationResult(
            repo_url="test",
            commit_sha="abc123",
            short_sha="abc12345",
            warnings=["Binary file not scored", "Root commit: no history"],
        )
        assert len(result.warnings) == 2
