"""Find open PDF copies through one optional Unpaywall DOI lookup."""

import json
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import httpx

from app.config import settings
from app.services.discovery import _provider_error, normalize_doi


@dataclass(frozen=True)
class FullTextLocation:
    source: str
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
            result.append(FullTextLocation("Unpaywall", url, url))
    return result[:3]


class FullTextResolver:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False)
        self.owns_client = client is None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def _get_json(self, url: str, params: dict) -> dict:
        payload = bytearray()
        async with self.client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            async for part in response.aiter_bytes():
                payload.extend(part)
                if len(payload) > 3_000_000:
                    raise ValueError("Provider metadata response is too large")
        return json.loads(payload)

    async def find(self, candidate: dict) -> tuple[list[FullTextLocation], list[str]]:
        doi = normalize_doi(candidate.get("doi"))
        if not doi or not settings.unpaywall_email:
            return [], []
        try:
            payload = await self._get_json(
                "https://api.unpaywall.org/v2/" + quote(doi, safe="/"),
                {"email": settings.unpaywall_email},
            )
            return parse_unpaywall_locations(payload, doi), []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return [], []
            return [], [_provider_error("Unpaywall", exc)]
        except Exception as exc:
            return [], [_provider_error("Unpaywall", exc)]
