"""Discovery routes accept every search provider for later import."""

from unittest.mock import AsyncMock

from app.routers import discovery


def test_arxiv_selection_creates_import_job(client_with_db, monkeypatch):
    monkeypatch.setattr(discovery, "process_job", AsyncMock())
    response = client_with_db.post(
        "/api/discovery/import",
        json={"candidates": [{"provider": "arxiv", "provider_id": "2401.12345"}]},
    )
    assert response.status_code == 202
    assert response.json()["estimated_api_calls"] == 1
