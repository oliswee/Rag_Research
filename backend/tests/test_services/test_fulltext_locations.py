"""Unpaywall is the only optional open-copy lookup."""

import httpx
import pytest

from app.config import settings
from app.services.fulltext_locations import FullTextResolver, parse_unpaywall_locations


def test_unpaywall_requires_matching_open_doi_and_pdf_url():
    record = {
        "doi": "10.1234/example",
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://repository.example.edu/paper.pdf",
        },
    }
    locations = parse_unpaywall_locations(record, "10.1234/example")
    assert len(locations) == 1
    assert locations[0].source == "Unpaywall"
    assert locations[0].url == "https://repository.example.edu/paper.pdf"
    assert parse_unpaywall_locations(record, "10.1234/different") == []
    assert (
        parse_unpaywall_locations({**record, "is_oa": False}, "10.1234/example") == []
    )


@pytest.mark.asyncio
async def test_resolver_uses_only_unpaywall(monkeypatch):
    monkeypatch.setattr(settings, "unpaywall_email", "research@example.org")
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "api.unpaywall.org"
        assert request.url.params["email"] == "research@example.org"
        return httpx.Response(
            200,
            json={
                "doi": "10.1234/example",
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://repo.example.org/open.pdf"
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        locations, warnings = await FullTextResolver(client).find(
            {"provider": "openalex", "provider_id": "W1", "doi": "10.1234/example"}
        )
    assert [location.source for location in locations] == ["Unpaywall"]
    assert warnings == []
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_resolver_skips_lookup_without_email_or_doi(monkeypatch):
    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("No external lookup expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        resolver = FullTextResolver(client)
        monkeypatch.setattr(settings, "unpaywall_email", "")
        assert await resolver.find(
            {"provider": "pubmed", "provider_id": "123", "doi": "10.1234/example"}
        ) == ([], [])
        monkeypatch.setattr(settings, "unpaywall_email", "research@example.org")
        assert await resolver.find(
            {"provider": "pubmed", "provider_id": "123", "doi": None}
        ) == ([], [])


@pytest.mark.asyncio
async def test_unpaywall_rate_limit_is_sanitized(monkeypatch):
    monkeypatch.setattr(settings, "unpaywall_email", "private@example.org")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(429))
    ) as client:
        locations, warnings = await FullTextResolver(client).find(
            {"provider": "openalex", "provider_id": "W1", "doi": "10.1234/example"}
        )
    assert locations == []
    assert "429" in warnings[0]
    assert "private@example.org" not in warnings[0]
