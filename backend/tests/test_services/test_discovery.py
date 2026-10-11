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


def _openalex(
    paper_id: str, doi: str | None = None, references: list[str] | None = None
) -> dict:
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
        "referenced_works": [
            f"https://openalex.org/{value}" for value in references or []
        ],
    }


@pytest.mark.asyncio
async def test_search_balances_providers_and_includes_grounded_insights():
    service = DiscoveryService()
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
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
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
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
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
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
    assert result["candidates"][0]["source_url"] == "https://doi.org/10.1234/shared"


@pytest.mark.asyncio
async def test_result_relationships_and_source_based_roles():
    older = _openalex("W1")
    older["publication_year"] = 2016
    older["cited_by_count"] = 1000
    recent = _openalex("W2", references=["W1"])
    recent["publication_year"] = 2026
    service = DiscoveryService()
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    service._get = AsyncMock(side_effect=[{"data": []}, {"results": [older, recent]}])
    try:
        result = await service.search("retrieval accuracy")
    finally:
        await service.close()
    by_id = {paper["provider_id"]: paper for paper in result["candidates"]}
    assert by_id["W1"]["cited_by_results"] == [
        {"provider_id": "W2", "title": recent["display_name"]}
    ]
    assert "Potential foundation" in by_id["W1"]["role_labels"]
    assert "Recent" in by_id["W2"]["role_labels"]
    assert result["themes"] == [{"label": "Information Retrieval", "count": 2}]


@pytest.mark.asyncio
async def test_load_more_cursor_keeps_unconsumed_pubmed_results():
    service = DiscoveryService()
    pubmed = {
        "provider": "pubmed",
        "provider_id": "12345",
        "title": "PubMed retrieval",
        "authors": [],
        "year": 2024,
        "abstract": "A retrieval study.",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "pdf_url": None,
        "open_access": False,
        "doi": None,
        "citation_count": None,
        "tags": ["Information Retrieval"],
    }
    service._get = AsyncMock(
        side_effect=[
            {"data": [_semantic("s1"), _semantic("s2")], "total": 3},
            {"results": [_openalex("W1"), _openalex("W2")], "meta": {"count": 3}},
            {"data": [_semantic("s2")], "total": 3},
            {"results": [_openalex("W1"), _openalex("W2")], "meta": {"count": 3}},
        ]
    )
    service._search_pubmed = AsyncMock(
        return_value={
            "results": [pubmed],
            "total": 1,
            "fetched_count": 1,
        }
    )
    try:
        first = await service.search("retrieval", limit=2)
        second = await service.search(
            "retrieval", limit=2, offsets=first["next_offsets"]
        )
    finally:
        await service.close()
    assert len(first["candidates"]) == 2
    assert first["has_more"] is True
    assert first["next_offsets"]["pubmed"] == 0
    assert any(paper["provider"] == "pubmed" for paper in second["candidates"])
    assert (
        service._get.call_args_list[2].args[1]["offset"]
        == first["next_offsets"]["semantic_scholar"]
    )
