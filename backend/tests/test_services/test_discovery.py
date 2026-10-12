"""Discovery keeps provider failures and paper evidence understandable."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.discovery import DiscoveryService, _abstract_summary, route_sources


def test_query_router_selects_domain_sources_without_user_mode():
    assert route_sources("How does retrieval improve education?") == (
        "semantic_scholar",
        "openalex",
    )
    assert route_sources("transformer neural network language model") == (
        "semantic_scholar",
        "arxiv",
    )
    assert route_sources("clinical cancer immunotherapy trial") == (
        "openalex",
        "pubmed",
    )
    assert route_sources("machine learning for cancer diagnosis") == (
        "semantic_scholar",
        "openalex",
        "pubmed",
        "arxiv",
    )
    assert route_sources("AI research on LLM evaluation") == (
        "semantic_scholar",
        "arxiv",
    )
    assert "pubmed" in route_sources("protein folding in biology")


@pytest.mark.asyncio
async def test_specialist_search_calls_only_its_two_sources():
    service = DiscoveryService()
    service._get = AsyncMock(return_value={"data": [], "total": 0})
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    try:
        result = await service.search("transformer neural network")
    finally:
        await service.close()
    assert result["searched_sources"] == ["semantic_scholar", "arxiv"]
    assert service._get.await_count == 1
    assert "api.semanticscholar.org" in service._get.await_args.args[0]
    service._search_arxiv.assert_awaited_once()
    service._search_pubmed.assert_not_awaited()


@pytest.mark.asyncio
async def test_medical_search_skips_semantic_scholar_and_arxiv():
    service = DiscoveryService()
    service._get = AsyncMock(return_value={"results": [], "meta": {"count": 0}})
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
    try:
        result = await service.search("clinical cancer therapy")
    finally:
        await service.close()
    assert result["searched_sources"] == ["openalex", "pubmed"]
    assert service._get.await_count == 1
    assert service._get.await_args.args[0] == "https://api.openalex.org/works"
    service._search_pubmed.assert_awaited_once()
    service._search_arxiv.assert_not_awaited()


@pytest.mark.asyncio
async def test_visible_results_rerank_strong_title_match_above_provider_order():
    service = DiscoveryService()
    unrelated = _semantic("s1")
    unrelated["title"] = "Unrelated methods"
    matched = _semantic("s2")
    matched["title"] = "Education policy outcomes"
    service._get = AsyncMock(
        side_effect=[
            {"data": [unrelated, matched], "total": 2},
            {"results": [], "meta": {"count": 0}},
        ]
    )
    try:
        result = await service.search("education policy", limit=2)
    finally:
        await service.close()
    assert result["candidates"][0]["provider_id"] == "s2"
    assert result["candidates"][0]["rank_score"] > result["candidates"][1]["rank_score"]


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
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
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
    assert result["searched_sources"] == ["semantic_scholar", "openalex"]
    assert result["failed_sources"] == []
    assert result["search_latency_ms"] >= 0


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
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
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
    assert result["failed_sources"] == ["semantic_scholar"]


@pytest.mark.asyncio
async def test_duplicate_doi_prefers_openalex_for_reliable_import():
    service = DiscoveryService()
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
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
async def test_source_based_roles_without_reference_graph_lookup():
    older = _openalex("W1")
    older["publication_year"] = 2016
    older["cited_by_count"] = 1000
    recent = _openalex("W2", references=["W1"])
    recent["publication_year"] = 2026
    service = DiscoveryService()
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
    service._get = AsyncMock(side_effect=[{"data": []}, {"results": [older, recent]}])
    try:
        result = await service.search("retrieval accuracy")
    finally:
        await service.close()
    by_id = {paper["provider_id"]: paper for paper in result["candidates"]}
    assert "cited_by_results" not in by_id["W1"]
    assert "Highly cited in results" in by_id["W1"]["role_labels"]
    assert "Recent" in by_id["W2"]["role_labels"]
    assert result["themes"] == [{"label": "Information Retrieval", "count": 2}]
    assert "referenced_works" not in service._get.call_args_list[1].args[1]["select"]


@pytest.mark.asyncio
async def test_load_more_cursor_keeps_unconsumed_pubmed_results():
    service = DiscoveryService()
    pubmed = {
        "provider": "pubmed",
        "provider_id": "12345",
        "title": "Biomedical paper",
        "authors": [],
        "year": 2024,
        "abstract": "A biomedical study.",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "pdf_url": None,
        "open_access": False,
        "doi": None,
        "citation_count": None,
        "tags": ["Biomedical"],
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
    service._search_arxiv = AsyncMock(return_value={"results": [], "total": 0})
    try:
        first = await service.search("clinical machine learning", limit=2)
        second = await service.search(
            "clinical machine learning", limit=2, offsets=first["next_offsets"]
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


@pytest.mark.asyncio
async def test_arxiv_result_is_importable_and_cursor_advances():
    service = DiscoveryService()
    service._get = AsyncMock(
        side_effect=[{"data": [], "total": 0}, {"results": [], "meta": {"count": 0}}]
    )
    service._search_pubmed = AsyncMock(return_value={"results": [], "total": 0})
    arxiv = {
        "provider": "arxiv",
        "provider_id": "2401.12345",
        "arxiv_id": "2401.12345",
        "title": "Retrieval paper",
        "authors": ["A. Author"],
        "year": 2024,
        "abstract": "The study evaluates retrieval.",
        "source_url": "https://arxiv.org/abs/2401.12345",
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "open_access": True,
        "doi": None,
        "citation_count": None,
        "tags": ["cs.IR"],
    }
    service._search_arxiv = AsyncMock(
        return_value={"results": [arxiv], "total": 2, "fetched_count": 1}
    )
    try:
        result = await service.search("machine learning retrieval", limit=1)
    finally:
        await service.close()
    assert result["candidates"][0]["provider"] == "arxiv"
    assert result["candidates"][0]["abstract_summary"]
    assert result["next_offsets"]["arxiv"] == 1
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_arxiv_candidate_lookup_strips_version():
    from app.services.arxiv import parse_arxiv_feed
    from tests.test_services.test_arxiv import FEED

    service = DiscoveryService()
    service._get_xml = AsyncMock(return_value=FEED)
    try:
        candidate = await service.get_candidate("arxiv", "2401.12345")
    finally:
        await service.close()
    assert candidate == parse_arxiv_feed(FEED)["results"][0]
    assert service._get_xml.call_args.args[1]["id_list"] == "2401.12345"


@pytest.mark.asyncio
async def test_arxiv_question_uses_topic_terms_without_question_words():
    service = DiscoveryService()
    service._arxiv_feed = AsyncMock(return_value={"results": [], "total": 0})
    try:
        await service._search_arxiv(
            "How does retrieval improve medical question answering?", 20, 0
        )
    finally:
        await service.close()
    query = service._arxiv_feed.call_args.args[0]["search_query"]
    assert query.startswith("all:retrieval AND all:improve")
    assert "all:How" not in query
    assert "all:does" not in query
