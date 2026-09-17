"""Tests for Phase 5.1: FastAPI API Integration (POST /analysis/risk).

Tests the HTTP boundary for the production inference endpoint.
Divided into:
  A. API contract tests — mock InferenceService
  B. Integration tests — real temporary Git repositories
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# TestClient
# ---------------------------------------------------------------------------
# Import the app AFTER inference errors so mock patching targets are clear.
from backend.app.api.main import app  # noqa: E402
from backend.app.inference.errors import (
    CommitNotFoundError,
    FeatureExtractionError,
    InferenceError,
    RepositoryAccessError,
)
from backend.app.inference.schemas import AnalyzeRiskResponse, FileRiskResult

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(**overrides) -> AnalyzeRiskResponse:
    """Build a minimal valid AnalyzeRiskResponse for mocking."""
    defaults = dict(
        repo_url="https://github.com/user/repo",
        commit_sha="a" * 40,
        short_sha="a" * 8,
        strategy="B1_CHANGE_SIZE",
        strategy_version="1.0.0",
        feature_version="v1",
        analyzed_at="2026-01-01T00:00:00+00:00",
        elapsed_ms=1.0,
        total_files=1,
        files_analyzed=1,
        files_skipped=0,
        files=[
            FileRiskResult(
                path="src/main.py",
                status="modified",
                investigation_priority_score=1.0,
                rank=1,
                total_files_in_commit=1,
                lines_added=5,
                lines_deleted=2,
                is_binary=False,
                language="Python",
                is_test_file=False,
            )
        ],
        warnings=[],
        score_semantics="investigation_priority_ranking",
    )
    defaults.update(overrides)
    return AnalyzeRiskResponse(**defaults)


def _make_temp_repo() -> tuple[str, str]:
    """Create a temporary Git repository with two commits.

    The second commit modifies a file, producing a non-empty diff.
    Returns (repo_path, second_commit_sha).
    """
    tmpdir = tempfile.mkdtemp(prefix="phase51_test_")
    repo_path = Path(tmpdir)
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path, check=True, capture_output=True,
    )
    (repo_path / "main.py").write_text("def hello():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_path, check=True, capture_output=True,
    )
    (repo_path / "main.py").write_text("def hello():\n    print('hi')\n\ndef world():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "modify main.py"],
        cwd=repo_path, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path, check=True, capture_output=True, text=True,
    )
    sha = result.stdout.strip()
    return str(repo_path), sha


# ===================================================================
# A. API contract tests (mocked InferenceService)
# ===================================================================


class TestRiskEndpointContract:
    def test_valid_request_returns_200(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        assert r.status_code == 200

    def test_response_conforms_to_schema(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
        parsed = AnalyzeRiskResponse(**data)
        assert parsed.commit_sha == "a" * 40

    def test_b1_strategy_metadata_preserved(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
        assert data["strategy"] == "B1_CHANGE_SIZE"
        assert data["strategy_version"] == "1.0.0"
        assert data["feature_version"] == "v1"

    def test_investigation_priority_score_preserved(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
        assert data["files"][0]["investigation_priority_score"] == 1.0
        assert data["score_semantics"] == "investigation_priority_ranking"

    def test_ranked_files_returned(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
        assert len(data["files"]) == 1
        assert data["files"][0]["rank"] == 1
        assert data["files"][0]["path"] == "src/main.py"

    def test_missing_repo_url_returns_422(self):
        r = client.post("/analysis/risk", json={"commit_sha": "a" * 40})
        assert r.status_code == 422

    def test_missing_commit_sha_returns_422(self):
        r = client.post("/analysis/risk", json={"repo_url": "https://github.com/u/r"})
        assert r.status_code == 422

    def test_empty_repo_url_returns_422(self):
        r = client.post(
            "/analysis/risk",
            json={"repo_url": "", "commit_sha": "a" * 40},
        )
        assert r.status_code == 422

    def test_empty_commit_sha_returns_422(self):
        r = client.post(
            "/analysis/risk",
            json={"repo_url": "https://github.com/u/r", "commit_sha": ""},
        )
        assert r.status_code == 422

    def test_commit_not_found_returns_404(self):
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.side_effect = CommitNotFoundError("not found")
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "bad"},
            )
        assert r.status_code == 404

    def test_repository_access_error_returns_422(self):
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.side_effect = RepositoryAccessError("clone failed")
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        assert r.status_code == 422

    def test_feature_extraction_error_returns_500(self):
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.side_effect = FeatureExtractionError("extraction fail")
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        assert r.status_code == 500

    def test_generic_inference_error_returns_500(self):
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.side_effect = InferenceError("generic failure")
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        assert r.status_code == 500

    def test_error_body_is_machine_readable(self):
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.side_effect = CommitNotFoundError("commit xyz not found")
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "xyz"},
            )
        data = r.json()
        assert "detail" in data
        assert "commit xyz not found" in data["detail"]

    def test_no_post_outcome_fields_exposed(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
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
        serialized = str(data)
        for field in forbidden:
            assert field not in serialized, f"Post-outcome field {field!r} found in response"

    def test_deterministic_same_response_twice(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r1 = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
            r2 = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        assert r1.json() == r2.json()

    def test_limitations_present(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            r = client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/r", "commit_sha": "a" * 40},
            )
        data = r.json()
        assert len(data["limitations"]) > 0
        assert any("investigation-priority" in lim for lim in data["limitations"])

    def test_inference_service_called_with_correct_args(self):
        resp = _fake_response()
        with patch("backend.app.api.main.inference_service") as mock_svc:
            mock_svc.analyze_commit.return_value = resp
            client.post(
                "/analysis/risk",
                json={"repo_url": "https://github.com/u/repo", "commit_sha": "abc123"},
            )
        mock_svc.analyze_commit.assert_called_once_with(
            repo_url="https://github.com/u/repo",
            commit_sha="abc123",
        )


# ===================================================================
# B. Integration tests (real temporary Git repository)
# ===================================================================


class TestRiskEndpointIntegration:
    @pytest.fixture(autouse=True)
    def _setup_repo(self):
        self.repo_path, self.sha = _make_temp_repo()

    def _post_risk(self, repo_url: str, commit_sha: str):
        """POST to /analysis/risk with clone_repo patched for local paths."""
        with patch("backend.app.inference.service.GitService.clone_repo", return_value=repo_url):
            return client.post(
                "/analysis/risk",
                json={"repo_url": repo_url, "commit_sha": commit_sha},
            )

    def test_valid_local_repo_returns_200(self):
        r = self._post_risk(self.repo_path, self.sha)
        assert r.status_code == 200

    def test_response_has_correct_commit_sha(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        assert data["commit_sha"] == self.sha

    def test_file_count_matches(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        assert data["total_files"] >= 1
        assert data["files_analyzed"] >= 1

    def test_files_are_ranked(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        for f in data["files"]:
            if not f["is_binary"]:
                assert f["rank"] >= 1
                assert f["investigation_priority_score"] >= 0.0

    def test_scores_are_valid_investigation_priority(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        for f in data["files"]:
            score = f["investigation_priority_score"]
            assert 0.0 <= score <= 1.0

    def test_nonexistent_commit_returns_404(self):
        r = self._post_risk(self.repo_path, "deadbeef00000000")
        assert r.status_code == 404

    def test_strategy_is_b1(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        assert data["strategy"] == "B1_CHANGE_SIZE"

    def test_score_semantics_identifier(self):
        r = self._post_risk(self.repo_path, self.sha)
        data = r.json()
        assert data["score_semantics"] == "investigation_priority_ranking"
