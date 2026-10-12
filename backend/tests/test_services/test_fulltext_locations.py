"""Open full-text resolvers verify identifiers and availability."""

import httpx
import pytest

from app.config import settings
from app.services.fulltext_locations import (
    FullTextLocation,
    FullTextResolver,
    parse_core_locations,
    parse_europe_pmc_locations,
    parse_jats_body,
    parse_unpaywall_locations,
)


def test_unpaywall_requires_matching_open_doi_and_pdf_url():
    record = {
        "doi": "10.1234/example",
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://repository.example.edu/paper.pdf",
            "license": "cc-by",
        },
    }
    locations = parse_unpaywall_locations(record, "10.1234/example")
    assert len(locations) == 1
    assert locations[0].source == "Unpaywall"
    assert locations[0].kind == "pdf"
    assert locations[0].url == "https://repository.example.edu/paper.pdf"
    assert parse_unpaywall_locations(record, "10.1234/different") == []
    assert (
        parse_unpaywall_locations({**record, "is_oa": False}, "10.1234/example") == []
    )


def test_core_requires_exact_doi_and_download_url():
    payload = {
        "results": [
            {
                "doi": "10.1234/other",
                "downloadUrl": "https://core.ac.uk/download/1.pdf",
            },
            {
                "doi": "10.1234/example",
                "downloadUrl": "https://core.ac.uk/download/2.pdf",
            },
        ]
    }
    locations = parse_core_locations(payload, "10.1234/example")
    assert [location.url for location in locations] == [
        "https://core.ac.uk/download/2.pdf"
    ]


def test_europe_pmc_uses_only_matching_open_access_record():
    payload = {
        "resultList": {
            "result": [
                {"doi": "10.1234/example", "pmcid": "PMC12345", "isOpenAccess": "N"},
                {"doi": "10.1234/example", "pmcid": "PMC67890", "isOpenAccess": "Y"},
            ]
        }
    }
    locations = parse_europe_pmc_locations(payload, "10.1234/example", None)
    assert len(locations) == 1
    assert locations[0].source == "Europe PMC"
    assert locations[0].kind == "xml"
    assert locations[0].url.endswith("/PMC67890/fullTextXML")


def test_jats_body_requires_open_license_and_body():
    payload = b"""<article xmlns:xlink='http://www.w3.org/1999/xlink'>
    <front><article-meta><permissions><license xlink:href='https://creativecommons.org/licenses/by/4.0/'/></permissions></article-meta></front>
    <body><sec><title>Methods</title><p>We tested the intervention.</p></sec></body>
    </article>"""
    assert "We tested the intervention." in parse_jats_body(payload)
    assert "Methods" in parse_jats_body(payload)
    with pytest.raises(ValueError, match="license"):
        parse_jats_body(b"<article><body><p>Private text</p></body></article>")


def test_jats_body_accepts_creative_commons_link_inside_license():
    payload = b"""<article xmlns:xlink='http://www.w3.org/1999/xlink'>
    <front><article-meta><permissions><license><license-p>Distributed under
    <ext-link xlink:href='http://creativecommons.org/licenses/by/4.0/'>CC BY</ext-link>
    </license-p></license></permissions></article-meta></front>
    <body><p>Full study result.</p></body></article>"""
    assert "Full study result." in parse_jats_body(payload)


@pytest.mark.asyncio
async def test_resolver_keeps_available_locations_when_core_is_rate_limited(
    monkeypatch,
):
    monkeypatch.setattr(settings, "unpaywall_email", "research@example.org")
    monkeypatch.setattr(settings, "core_api_key", "secret-core-key")

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.unpaywall.org":
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
        if request.url.host == "api.core.ac.uk":
            assert request.headers["Authorization"] == "Bearer secret-core-key"
            return httpx.Response(429)
        if request.url.host == "www.ebi.ac.uk":
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [
                            {
                                "doi": "10.1234/example",
                                "isOpenAccess": "Y",
                                "pmcid": "PMC12345",
                            }
                        ]
                    }
                },
            )
        raise AssertionError(str(request.url))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        resolver = FullTextResolver(client)
        locations, warnings = await resolver.find(
            {"provider": "openalex", "provider_id": "W1", "doi": "10.1234/example"}
        )
    assert [location.source for location in locations] == ["Unpaywall", "Europe PMC"]
    assert "CORE: 429" in warnings[0]
    assert "secret-core-key" not in warnings[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/private",
        "https://www.ebi.ac.uk:8443/europepmc/webservices/rest/PMC123/fullTextXML",
    ],
)
async def test_europe_pmc_xml_fetch_rejects_other_hosts(url):
    resolver = FullTextResolver()
    try:
        with pytest.raises(ValueError, match="Unsupported XML location"):
            await resolver.fetch_xml(
                FullTextLocation(
                    "Europe PMC",
                    "xml",
                    url,
                    "https://europepmc.org/articles/PMC123",
                )
            )
    finally:
        await resolver.close()
