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


def test_context_endpoint_returns_source_attributed_references(
    client_with_db, monkeypatch
):
    monkeypatch.setattr(
        discovery,
        "get_paper_context",
        AsyncMock(
            return_value={
                "metadata": {"source": "Crossref", "venue": "Research Journal"},
                "citing_papers": [],
                "references": [
                    {
                        "source": "OpenAlex",
                        "title": "A source",
                        "url": "https://openalex.org/W1",
                        "year": 2020,
                    }
                ],
                "errors": [],
            }
        ),
    )
    response = client_with_db.get(
        "/api/discovery/context",
        params={"provider": "arxiv", "provider_id": "2401.12345"},
    )
    assert response.status_code == 200
    assert response.json()["references"][0]["source"] == "OpenAlex"
