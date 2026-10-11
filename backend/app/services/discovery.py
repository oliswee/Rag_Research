"""Normalized paper discovery from Semantic Scholar and OpenAlex."""

import asyncio
import re
import time
from urllib.parse import quote

import httpx

from app.config import settings

_CACHE: dict[tuple[str, tuple], tuple[float, dict]] = {}
_CACHE_TTL = 600


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(
        r"^(https?://(dx\.)?doi\.org/|doi:)", "", value.strip(), flags=re.I
    ).lower()


def normalize_title(value: str) -> str:
    """Loose title key for human duplicate suggestions only."""
    return " ".join(re.findall(r"\w+", value.casefold()))


def _openalex_abstract(index: dict | None) -> str | None:
    if not index:
        return None
    words = [""] * (
        max((p for positions in index.values() for p in positions), default=-1) + 1
    )
    for word, positions in index.items():
        for position in positions:
            words[position] = word
    return " ".join(words) or None


def semantic_candidate(record: dict) -> dict:
    ids = record.get("externalIds") or {}
    return {
        "provider": "semantic_scholar",
        "provider_id": record["paperId"],
        "title": record.get("title") or "Untitled",
        "authors": [a["name"] for a in record.get("authors") or [] if a.get("name")],
        "year": record.get("year"),
        "abstract": record.get("abstract"),
        "source_url": record.get("url"),
        "pdf_url": (record.get("openAccessPdf") or {}).get("url"),
        "open_access": bool(record.get("isOpenAccess") or record.get("openAccessPdf")),
        "doi": normalize_doi(ids.get("DOI")),
        "arxiv_id": ids.get("ArXiv"),
    }


def openalex_candidate(record: dict) -> dict:
    location = record.get("best_oa_location") or {}
    return {
        "provider": "openalex",
        "provider_id": record["id"].rsplit("/", 1)[-1],
        "title": record.get("display_name") or "Untitled",
        "authors": [
            a["author"]["display_name"]
            for a in record.get("authorships") or []
            if a.get("author", {}).get("display_name")
        ],
        "year": record.get("publication_year"),
        "abstract": _openalex_abstract(record.get("abstract_inverted_index")),
        "source_url": record.get("id"),
        "pdf_url": location.get("pdf_url"),
        "open_access": bool((record.get("open_access") or {}).get("is_oa")),
        "doi": normalize_doi(record.get("doi")),
        "arxiv_id": None,
    }


class DiscoveryService:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(
            timeout=15,
            follow_redirects=False,
            headers={"User-Agent": "FolioTrace/0.1 (local research library)"},
        )
        self.owns_client = client is None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def _get(self, url: str, params: dict | None = None) -> dict:
        cache_key = (url, tuple(sorted((params or {}).items())))
        cached = _CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL:
            return cached[1]
        for attempt in range(3):
            headers = (
                {"x-api-key": settings.semantic_scholar_api_key}
                if "api.semanticscholar.org" in url
                and settings.semantic_scholar_api_key
                else None
            )
            response = await self.client.get(url, params=params, headers=headers)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if len(_CACHE) >= 128:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = (time.monotonic(), data)
            return data
        raise RuntimeError("Academic API unavailable")

    async def search(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access: bool = False,
        limit: int = 20,
    ) -> dict:
        limit = max(1, min(limit, 20))
        semantic_params = {
            "query": query,
            "limit": limit,
            "fields": (
                "paperId,title,authors,year,abstract,url,"
                "openAccessPdf,isOpenAccess,externalIds"
            ),
        }
        if year_from or year_to:
            semantic_params["year"] = f"{year_from or ''}-{year_to or ''}"
        filters = []
        if year_from or year_to:
            filters.append(f"from_publication_date:{year_from or 1800}-01-01")
            if year_to:
                filters.append(f"to_publication_date:{year_to}-12-31")
        if open_access:
            filters.append("is_oa:true")
        openalex_params = {
            "search": query,
            "per_page": limit,
            "select": (
                "id,display_name,authorships,publication_year,"
                "abstract_inverted_index,doi,ids,open_access,best_oa_location"
            ),
        }
        if filters:
            openalex_params["filter"] = ",".join(filters)
        calls = await asyncio.gather(
            self._get(
                "https://api.semanticscholar.org/graph/v1/paper/search", semantic_params
            ),
            self._get("https://api.openalex.org/works", openalex_params),
            return_exceptions=True,
        )
        candidates, errors, seen = [], [], set()
        for name, outcome, key, normalizer in (
            ("Semantic Scholar", calls[0], "data", semantic_candidate),
            ("OpenAlex", calls[1], "results", openalex_candidate),
        ):
            if isinstance(outcome, BaseException):
                errors.append(f"{name}: {type(outcome).__name__}")
                continue
            for raw in outcome.get(key, []):
                candidate = normalizer(raw)
                if open_access and not candidate["open_access"]:
                    continue
                identity = candidate["doi"] or (
                    candidate["provider"],
                    candidate["provider_id"],
                )
                if identity not in seen:
                    seen.add(identity)
                    candidates.append(candidate)
        return {"candidates": candidates[:limit], "errors": errors}

    async def get_candidate(self, provider: str, provider_id: str) -> dict:
        if provider == "semantic_scholar":
            if not re.fullmatch(r"[A-Za-z0-9:._-]{1,100}", provider_id):
                raise ValueError("Invalid Semantic Scholar ID")
            raw = await self._get(
                "https://api.semanticscholar.org/graph/v1/paper/"
                + quote(provider_id, safe=""),
                {
                    "fields": (
                        "paperId,title,authors,year,abstract,url,"
                        "openAccessPdf,isOpenAccess,externalIds"
                    )
                },
            )
            return semantic_candidate(raw)
        if provider == "openalex":
            if not re.fullmatch(r"W\d{1,20}", provider_id):
                raise ValueError("Invalid OpenAlex ID")
            raw = await self._get("https://api.openalex.org/works/" + provider_id)
            return openalex_candidate(raw)
        raise ValueError("Unsupported provider")
