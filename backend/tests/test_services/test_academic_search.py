import json

import httpx
import pytest

from app.schemas.query import QueryRequest
from app.services.academic_search import (
    AcademicSearch,
    PaperIdentity,
    licensed_location,
    match_identity,
)
from app.services.evidence_routing import parse_policy


def policy(**kwargs):
    return parse_policy(QueryRequest(query="memory papers", **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"local_only": True}, {"only_uploaded": True}, {"retrieval_mode": "bm25"}],
)
async def test_prohibited_means_zero_requests(fields):
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: calls.append(r))
    ) as client:
        result = await AcademicSearch(client).search(policy(**fields))
    assert not calls and not result.candidates


def oa_record(doi="10.1234/a", year=2025):
    return {
        "id": "https://openalex.org/W1",
        "title": "Memory",
        "doi": f"https://doi.org/{doi}",
        "publication_year": year,
        "open_access": {"is_oa": True},
        "best_oa_location": {
            "pdf_url": "https://example.org/a.pdf",
            "license": "cc-by",
            "version": "publishedVersion",
        },
        "abstract_inverted_index": {"Memory": [0], "works": [1]},
    }


@pytest.mark.asyncio
async def test_partial_failure_retries_are_bounded_and_no_secrets_in_events():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(
                429, text="secret provider message", headers={"retry-after": "9999"}
            )
        return httpx.Response(200, json={"results": [oa_record()]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AcademicSearch(client, retry_delay=0).search(policy())
    assert len(calls) == 3
    assert len(result.candidates) == 1
    assert licensed_location(result.candidates[0]) == "https://example.org/a.pdf"
    serialized = json.dumps(result.events)
    assert "secret provider message" not in serialized and "api_key" not in serialized


@pytest.mark.asyncio
async def test_year_identity_dedup_and_latest():
    def handler(request):
        assert request.url.params["sort"] == "publication_date:desc"
        return httpx.Response(
            200,
            json={
                "results": [
                    oa_record(),
                    oa_record(),
                    oa_record("10.1234/wrong"),
                    oa_record("10.1234/old", 2010),
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AcademicSearch(client).search(
            policy(
                latest=True, provider="openalex", year_from=2020, paper_ids=["local"]
            ),
            [PaperIdentity("local", doi="10.1234/a")],
        )
    assert [r["doi"] for r in result.candidates] == ["10.1234/a"]
    assert result.candidates[0]["local_paper_id"] == "local"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "oversize", "invalid", "redirect"])
async def test_faults_are_sanitized_and_bounded(fault):
    calls = []

    def handler(request):
        calls.append(request)
        if fault == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        if fault == "oversize":
            return httpx.Response(200, content=b"x" * 524289)
        if fault == "redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AcademicSearch(client, retry_delay=0).search(
            policy(provider="openalex")
        )
    assert not result.candidates and 1 <= len(calls) <= 2
    assert "secret" not in json.dumps(result.events)


@pytest.mark.parametrize(
    "license", [None, "unknown", "publisher-specific-oa", "cc-by-nc"]
)
def test_unknown_or_restricted_license_never_auto_fetches(license):
    assert (
        licensed_location(
            {
                "open_access": True,
                "license": license,
                "pdf_url": "https://example.org/a.pdf",
            }
        )
        is None
    )


@pytest.mark.asyncio
async def test_arxiv_versions_are_not_collapsed_or_matched_to_wrong_version():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>https://arxiv.org/abs/2401.00001v2</id><title>Memory</title>
    <summary>Memory works.</summary><published>2024-01-01</published>
    </entry></feed>"""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=xml))
    ) as client:
        result = await AcademicSearch(client).search(
            policy(provider="arxiv", paper_ids=["a"]),
            [PaperIdentity("a", arxiv_id="2401.00001v1")],
        )
    assert not result.candidates


def test_explicit_arxiv_version_cannot_be_bypassed_by_doi_or_provider_match():
    identity = PaperIdentity(
        "a",
        doi="10.1234/a",
        arxiv_id="2401.00001v1",
        provider="openalex",
        provider_id="W1",
    )
    assert not match_identity(
        {"doi": "10.1234/a", "provider": "openalex", "provider_id": "W1"}, identity
    )
    assert match_identity({"doi": "10.1234/a", "arxiv_id": "2401.00001v1"}, identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,year",
    [
        ({"year_from": 2020, "year_to": 2025}, "2020-2025"),
        ({"year_from": 2020}, "2020-"),
        ({"year_to": 2025}, "-2025"),
    ],
)
async def test_semantic_year_uses_documented_hyphen_range(fields, year):
    def handler(request):
        assert request.url.params["year"] == year
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AcademicSearch(client).search(
            policy(provider="semantic_scholar", **fields)
        )
    # Adapter isolates provider failures, so verify the event as well as the assertion.
    # A caught assertion would produce an unavailable event instead of ok.
    assert result.events[0]["status"] == "ok"
