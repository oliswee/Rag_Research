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


def test_context_endpoint_accepts_doi_and_returns_only_crossref_metadata(
    client_with_db, monkeypatch
):
    monkeypatch.setattr(
        discovery,
        "get_paper_context",
        AsyncMock(
            return_value={
                "metadata": {"source": "Crossref", "venue": "Research Journal"},
                "errors": [],
            }
        ),
    )
    response = client_with_db.get(
        "/api/discovery/context",
        params={"doi": "10.1234/example"},
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["source"] == "Crossref"
    assert "references" not in response.json()
    assert discovery.get_paper_context.await_args.args[1] == "10.1234/example"


def test_context_endpoint_rejects_invalid_doi(client_with_db):
    response = client_with_db.get("/api/discovery/context", params={"doi": "not-a-doi"})
    assert response.status_code == 422
