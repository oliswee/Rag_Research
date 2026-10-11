"""Discovery keeps provider failures and paper evidence understandable."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.discovery import DiscoveryService, _abstract_summary


def test_abstract_summary_omits_bibliographic_lead_in():
    abstract = (
        "A. Smith, B. Jones, C. Lee, D. Chen, E. Miller. "
        "Proceedings of a research conference. "
        "We compare retrieval methods and report higher recall."
    )
    assert _abstract_summary(abstract) == (
        "We compare retrieval methods and report higher recall."
    )


def _semantic(paper_id: str, doi: str | None = None) -> dict:
    return {
        "paperId": paper_id,
        "title": f"Semantic paper {paper_id}",
        "year": 2024,
        "abstract": "The study measures retrieval accuracy. It compares two methods.",
        "authors": [{"name": "A. Author"}],
        "citationCount": 12,
        "fieldsOfStudy": ["Computer Science"],
        "externalIds": {"DOI": doi} if doi else {},
    }


def _openalex(paper_id: str, doi: str | None = None) -> dict:
    return {
        "id": f"https://openalex.org/{paper_id}",
        "display_name": f"OpenAlex paper {paper_id}",
        "publication_year": 2025,
        "abstract_inverted_index": {
            "Retrieval": [0],
            "improves": [1],
            "accuracy.": [2],
        },
        "doi": f"https://doi.org/{doi}" if doi else None,
        "authorships": [],
        "cited_by_count": 8,
        "primary_topic": {"display_name": "Information Retrieval"},
        "keywords": [{"display_name": "Search ranking"}],
    }


@pytest.mark.asyncio
async def test_search_balances_providers_and_includes_grounded_insights():
    service = DiscoveryService()
    service._get = AsyncMock(
        side_effect=[
            {"data": [_semantic("s1"), _semantic("s2"), _semantic("s3")]},
            {"results": [_openalex("W1"), _openalex("W2"), _openalex("W3")]},
        ]
    )
    try:
        result = await service.search("retrieval accuracy", limit=4)
    finally:
        await service.close()
    assert {paper["provider"] for paper in result["candidates"]} == {
        "semantic_scholar",
        "openalex",
    }
    assert len(result["candidates"]) == 4
    assert all(paper["relevance_reason"] for paper in result["candidates"])
    assert all(paper["abstract_summary"] for paper in result["candidates"])
    assert any(paper["tags"] for paper in result["candidates"])


@pytest.mark.asyncio
async def test_search_explains_rate_limit_and_keeps_other_source():
    response = httpx.Response(
        429,
        request=httpx.Request(
            "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
        ),
    )
    service = DiscoveryService()
    service._get = AsyncMock(
        side_effect=[
            httpx.HTTPStatusError(
                "rate limited", request=response.request, response=response
            ),
            {"results": [_openalex("W1")]},
        ]
    )
    try:
        result = await service.search("retrieval accuracy")
    finally:
        await service.close()
    assert result["candidates"][0]["provider"] == "openalex"
    assert "429" in result["errors"][0]
    assert "rate limit" in result["errors"][0].lower()
    assert "HTTPStatusError" not in result["errors"][0]


@pytest.mark.asyncio
async def test_duplicate_doi_prefers_openalex_for_reliable_import():
    service = DiscoveryService()
    service._get = AsyncMock(
        side_effect=[
            {"data": [_semantic("s1", "10.1234/shared")]},
            {"results": [_openalex("W1", "10.1234/shared")]},
        ]
    )
    try:
        result = await service.search("retrieval accuracy")
    finally:
        await service.close()
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["provider"] == "openalex"
