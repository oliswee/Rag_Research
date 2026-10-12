"""Find verified open copies and read Europe PMC's licensed OA XML."""

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree as ET

import httpx

from app.config import settings
from app.services.discovery import _provider_error, normalize_doi


@dataclass(frozen=True)
class FullTextLocation:
    source: str
    kind: str
    url: str
    source_url: str


def _http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"https", "http"}
        and bool(parsed.hostname)
        and not (parsed.username or parsed.password)
    )


def parse_unpaywall_locations(payload: dict, doi: str) -> list[FullTextLocation]:
    if normalize_doi(payload.get("doi")) != doi or payload.get("is_oa") is not True:
        return []
    locations = [payload.get("best_oa_location"), *(payload.get("oa_locations") or [])]
    seen = set()
    result = []
    for location in locations:
        url = (location or {}).get("url_for_pdf")
        if _http_url(url) and url not in seen:
            seen.add(url)
            result.append(FullTextLocation("Unpaywall", "pdf", url, url))
    return result[:3]


def parse_core_locations(payload: dict, doi: str) -> list[FullTextLocation]:
    locations = []
    for work in payload.get("results") or []:
        if normalize_doi(work.get("doi")) != doi:
            continue
        url = work.get("downloadUrl") or work.get("download_url")
        if _http_url(url):
            locations.append(FullTextLocation("CORE", "pdf", url, url))
    return locations[:3]


def parse_europe_pmc_locations(
    payload: dict, doi: str | None, pmid: str | None
) -> list[FullTextLocation]:
    for record in (payload.get("resultList") or {}).get("result") or []:
        same_paper = (doi is not None and normalize_doi(record.get("doi")) == doi) or (
            doi is None
            and pmid is not None
            and record.get("id") == pmid
            and record.get("source") == "MED"
        )
        pmcid = record.get("pmcid") or ""
        if (
            same_paper
            and record.get("isOpenAccess") == "Y"
            and re.fullmatch(r"PMC\d{1,12}", pmcid)
        ):
            return [
                FullTextLocation(
                    "Europe PMC",
                    "xml",
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                    f"https://europepmc.org/articles/{pmcid}",
                )
            ]
    return []


def parse_jats_body(payload: bytes) -> str:
    """Extract only the article body after checking an explicit OA license."""
    root = ET.fromstring(payload)
    if root.tag != "article":
        raise ValueError("Not an article XML document")
    licensed = False
    for license_node in root.findall("./front/article-meta/permissions/license"):
        for node in license_node.iter():
            href = node.get("{http://www.w3.org/1999/xlink}href") or ""
            parsed = urlsplit(href)
            if parsed.hostname in {
                "creativecommons.org",
                "www.creativecommons.org",
            } and parsed.path.startswith(("/licenses/", "/publicdomain/")):
                licensed = True
                break
    if not licensed:
        raise ValueError("Open reuse license was not found")
    body = root.find("./body")
    if body is None:
        raise ValueError("Article body was not found")
    text = " ".join(" ".join(body.itertext()).split())
    if not text:
        raise ValueError("Article body was empty")
    return text


class FullTextResolver:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False)
        self.owns_client = client is None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def _get_json(
        self, url: str, params: dict, headers: dict | None = None
    ) -> dict:
        payload = bytearray()
        async with self.client.stream(
            "GET", url, params=params, headers=headers
        ) as response:
            response.raise_for_status()
            async for part in response.aiter_bytes():
                payload.extend(part)
                if len(payload) > 3_000_000:
                    raise ValueError("Provider metadata response is too large")
        return json.loads(payload)

    async def find(self, candidate: dict) -> tuple[list[FullTextLocation], list[str]]:
        doi = normalize_doi(candidate.get("doi"))
        pmid = candidate["provider_id"] if candidate["provider"] == "pubmed" else None
        jobs = {}
        if doi and settings.unpaywall_email:
            jobs["Unpaywall"] = self._get_json(
                "https://api.unpaywall.org/v2/" + quote(doi, safe="/"),
                {"email": settings.unpaywall_email},
            )
        if doi and settings.core_api_key:
            jobs["CORE"] = self._get_json(
                "https://api.core.ac.uk/v3/search/works/",
                {"q": f'doi:"{doi}"', "limit": 3},
                {"Authorization": f"Bearer {settings.core_api_key}"},
            )
        if doi or pmid:
            term = f"DOI:{doi}" if doi else f"EXT_ID:{pmid} AND SRC:MED"
            jobs["Europe PMC"] = self._get_json(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                {"query": term, "format": "json", "resultType": "lite", "pageSize": 5},
            )
        outcomes = await asyncio.gather(*jobs.values(), return_exceptions=True)
        locations, warnings = [], []
        for source, outcome in zip(jobs, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not (
                    isinstance(outcome, httpx.HTTPStatusError)
                    and outcome.response.status_code == 404
                ):
                    warnings.append(_provider_error(source, outcome))
                continue
            if source == "Unpaywall":
                locations.extend(parse_unpaywall_locations(outcome, doi))
            elif source == "CORE":
                locations.extend(parse_core_locations(outcome, doi))
            else:
                locations.extend(parse_europe_pmc_locations(outcome, doi, pmid))
        return locations, warnings

    async def fetch_xml(self, location: FullTextLocation) -> str:
        parsed = urlsplit(location.url)
        if (
            location.source != "Europe PMC"
            or location.kind != "xml"
            or parsed.scheme != "https"
            or parsed.hostname != "www.ebi.ac.uk"
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or not re.fullmatch(
                r"/europepmc/webservices/rest/PMC\d{1,12}/fullTextXML",
                parsed.path,
            )
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Unsupported XML location")
        payload = bytearray()
        limit = settings.max_file_size_mb * 1024 * 1024
        async with self.client.stream("GET", location.url) as response:
            response.raise_for_status()
            async for part in response.aiter_bytes():
                payload.extend(part)
                if len(payload) > limit:
                    raise ValueError("Article XML exceeds configured size limit")
        return parse_jats_body(bytes(payload))
