"""Citation context is retrieved on demand and attributed to each provider."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.discovery import DiscoveryService
from app.services.metadata_context import get_paper_context


@pytest.mark.asyncio
async def test_context_combines_crossref_s2_citations_and_openalex_references():
    service = DiscoveryService()
    service.get_candidate = AsyncMock(
        return_value={
            "provider": "openalex",
            "provider_id": "W123",
            "title": "Selected",
            "doi": "10.1234/example",
            "_reference_ids": ["W456"],
        }
    )

    async def fake_get(url, params=None):
        if "crossref.org" in url:
            assert params is None
            return {
                "message": {
                    "DOI": "10.1234/example",
                    "container-title": ["Research Journal"],
                    "publisher": "Example Press",
                    "type": "journal-article",
                    "published": {"date-parts": [[2024, 2, 3]]},
                    "references-count": 6,
                }
            }
        if "semanticscholar.org" in url:
            return {
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "abc123",
                            "title": "Citing study",
                            "year": 2025,
                        }
                    }
                ]
            }
        if "openalex.org" in url:
            assert params["filter"] == "openalex:W456"
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W456",
                        "display_name": "Referenced study",
                        "publication_year": 2020,
                    }
                ]
            }
        raise AssertionError(url)

    service._get = AsyncMock(side_effect=fake_get)
    try:
        context = await get_paper_context(service, "openalex", "W123")
    finally:
        await service.close()
    assert context["metadata"]["venue"] == "Research Journal"
    assert context["metadata"]["source"] == "Crossref"
    assert (
        context["citing_papers"][0]["url"]
        == "https://www.semanticscholar.org/paper/abc123"
    )
    assert context["references"][0]["url"] == "https://openalex.org/W456"
    assert context["errors"] == []


@pytest.mark.asyncio
async def test_context_preserves_other_sources_on_semantic_scholar_429():
    service = DiscoveryService()
    service.get_candidate = AsyncMock(
        return_value={
            "provider": "pubmed",
            "provider_id": "123",
            "doi": "10.1234/example",
            "title": "Selected",
        }
    )

    async def fake_get(url, params=None):
        if "semanticscholar.org" in url:
            response = httpx.Response(429, request=httpx.Request("GET", url))
            raise httpx.HTTPStatusError(
                "rate limited", request=response.request, response=response
            )
        if "crossref.org" in url:
            return {"message": {"DOI": "10.1234/example", "publisher": "Example Press"}}
        if "openalex.org" in url:
            return {"referenced_works": []}
        raise AssertionError(url)

    service._get = AsyncMock(side_effect=fake_get)
    try:
        context = await get_paper_context(service, "pubmed", "123")
    finally:
        await service.close()
    assert context["metadata"]["publisher"] == "Example Press"
    assert context["citing_papers"] == []
    assert "429" in context["errors"][0]
