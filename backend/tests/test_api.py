"""Tests for the FastAPI API layer (Phase 1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAnalyzeCommit:
    def test_missing_fields_returns_422(self) -> None:
        resp = client.post("/analysis/commit", json={})
        assert resp.status_code == 422

    def test_empty_repo_url_returns_422(self) -> None:
        resp = client.post("/analysis/commit", json={"repo_url": "", "commit_sha": "abc"})
        assert resp.status_code == 422

    def test_empty_commit_sha_returns_422(self) -> None:
        resp = client.post(
            "/analysis/commit", json={"repo_url": "https://example.com/repo", "commit_sha": ""}
        )
        assert resp.status_code == 422


class TestAnalyzePR:
    def test_missing_fields_returns_422(self) -> None:
        resp = client.post("/analysis/pull-request", json={})
        assert resp.status_code == 422

    def test_empty_branch_returns_422(self) -> None:
        resp = client.post(
            "/analysis/pull-request",
            json={
                "repo_url": "https://example.com/repo",
                "base_branch": "",
                "head_branch": "main",
            },
        )
        assert resp.status_code == 422
