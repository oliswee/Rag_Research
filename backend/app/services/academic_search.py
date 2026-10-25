"""Read-only academic search with a request-local budget and no query cache."""

import asyncio
import ipaddress
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import httpx

from app.config import settings
from app.services.arxiv import ATOM, parse_arxiv_feed
from app.services.discovery import (
    normalize_doi,
    openalex_candidate,
    route_sources,
    semantic_candidate,
)
from app.services.pubmed import search_pubmed
from app.services.retrieval_trace import _safe_query

_HOSTS = {
    "api.semanticscholar.org",
    "api.openalex.org",
    "export.arxiv.org",
    "eutils.ncbi.nlm.nih.gov",
    "api.crossref.org",
    "api.unpaywall.org",
}
_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST = 0.0


def safe_url(value):
    """Public citation URL, without credentials/query/fragment; DNS only on fetch."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 80, 443}
        ):
            return None
        if host == "localhost" or host.endswith((".local", ".localhost")):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return None


def public_query(question):
    text = _safe_query(question)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    return re.sub(r"https?://\S+", "[redacted-url]", text)[:500]


@dataclass(frozen=True)
class PaperIdentity:
    paper_id: str
    doi: str | None = None
    arxiv_id: str | None = None
    provider: str | None = None
    provider_id: str | None = None
    document_version: str | None = None


def match_identity(candidate, identity):
    arxiv = candidate.get("arxiv_id")
    if identity.arxiv_id and re.search(r"v\d+$", identity.arxiv_id):
        # DOI/provider equivalence does not establish an explicit arXiv version.
        return arxiv == identity.arxiv_id
    if identity.arxiv_id and arxiv:
        # Explicit versions must match. Unversioned IDs do not prove a version.
        if re.search(r"v\d+$", identity.arxiv_id) and arxiv != identity.arxiv_id:
            return False
        if re.sub(r"v\d+$", "", arxiv) == re.sub(r"v\d+$", "", identity.arxiv_id):
            return True
    if identity.doi and normalize_doi(identity.doi) == normalize_doi(
        candidate.get("doi")
    ):
        return True
    return bool(
        identity.provider
        and identity.provider == candidate.get("provider")
        and identity.provider_id == candidate.get("provider_id")
    )


def licensed_location(candidate):
    license = (candidate.get("license") or "").lower().strip().rstrip("/")
    allowed = {
        "cc0",
        "cc-by",
        "cc-by-sa",
        "https://creativecommons.org/publicdomain/zero/1.0",
    }
    if license not in allowed and not re.fullmatch(
        r"https?://creativecommons\.org/licenses/(?:by|by-sa)/[1-4]\.0", license
    ):
        return None
    return (
        safe_url(candidate.get("pdf_url"))
        if candidate.get("open_access") is True
        else None
    )


@dataclass
class WebBudget:
    started: float = field(default_factory=time.monotonic)
    attempts: int = 0
    events: list[dict] = field(default_factory=list)
    retried: set[str] = field(default_factory=set)

    @property
    def remaining(self):
        return max(0, settings.web_deadline_seconds - (time.monotonic() - self.started))

    def consume(self):
        if self.attempts >= settings.web_max_attempts or self.remaining <= 0:
            raise TimeoutError("Web budget exhausted")
        self.attempts += 1


@dataclass
class SearchResult:
    candidates: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    budget: WebBudget | None = None


class AcademicSearch:
    def __init__(self, client=None, *, retry_delay=1):
        self.client = client
        self.retry_delay = retry_delay

    async def _fetch(self, client, budget, provider, url, params, *, xml=False):
        global _ARXIV_LAST
        if urlsplit(url).hostname not in _HOSTS:
            raise ValueError("Unlisted academic endpoint")
        while True:
            budget.consume()
            event = {
                "provider": provider,
                "attempt": budget.attempts,
                "status": "started",
            }
            budget.events.append(event)
            try:
                if provider == "arxiv":
                    async with _ARXIV_LOCK:
                        await asyncio.sleep(
                            max(0, 3 - (time.monotonic() - _ARXIV_LAST))
                        )
                        _ARXIV_LAST = time.monotonic()
                headers = (
                    {"x-api-key": settings.semantic_scholar_api_key}
                    if urlsplit(url).hostname == "api.semanticscholar.org"
                    and settings.semantic_scholar_api_key
                    else {}
                )
                async with asyncio.timeout(
                    min(settings.web_timeout_seconds, budget.remaining)
                ):
                    async with client.stream(
                        "GET",
                        url,
                        params=params,
                        headers=headers,
                        follow_redirects=False,
                    ) as response:
                        event["http_status"] = response.status_code
                        if (
                            response.status_code in {429, 500, 502, 503, 504}
                            and provider not in budget.retried
                        ):
                            budget.retried.add(provider)
                            event["status"] = "retryable"
                            await asyncio.sleep(min(self.retry_delay, budget.remaining))
                            continue
                        response.raise_for_status()
                        data = bytearray()
                        async for part in response.aiter_bytes():
                            data.extend(part)
                            if len(data) > settings.web_metadata_bytes:
                                raise ValueError("Metadata too large")
                        result = bytes(data) if xml else json.loads(data)
                        event["status"] = "ok"
                        return result
            except Exception as exc:
                event["status"] = "unavailable"
                event["error_type"] = type(exc).__name__
                raise

    async def _provider(self, client, budget, provider, query, policy):
        limit = settings.web_candidates_per_provider

        async def fetch(url, params):
            return await self._fetch(client, budget, provider, url, params)

        if provider == "semantic_scholar":
            params = {
                "query": query,
                "limit": limit,
                "fields": (
                    "paperId,title,authors,year,abstract,url,externalIds,"
                    "openAccessPdf,isOpenAccess,publicationDate"
                ),
            }
            if policy.year_from or policy.year_to:
                params["year"] = f"{policy.year_from or ''}-{policy.year_to or ''}"
            raw = await fetch(
                "https://api.semanticscholar.org/graph/v1/paper/search", params
            )
            rows = []
            for record in raw.get("data", [])[:limit]:
                candidate = semantic_candidate(record)
                location = record.get("openAccessPdf") or {}
                candidate.update(
                    license=location.get("license"),
                    version=location.get("version"),
                    published_at=record.get("publicationDate"),
                )
                rows.append(candidate)
            return rows
        if provider == "openalex":
            params = {"search": query, "per-page": limit}
            filters = []
            if policy.year_from:
                filters.append(f"from_publication_date:{policy.year_from}-01-01")
            if policy.year_to:
                filters.append(f"to_publication_date:{policy.year_to}-12-31")
            if filters:
                params["filter"] = ",".join(filters)
            if policy.intent == "latest":
                params["sort"] = "publication_date:desc"
            raw = await fetch("https://api.openalex.org/works", params)
            rows = []
            for record in raw.get("results", [])[:limit]:
                index = record.get("abstract_inverted_index") or {}
                if any(
                    not isinstance(p, int) or p < 0 or p > 16000
                    for positions in index.values()
                    for p in positions
                ):
                    record = {**record, "abstract_inverted_index": None}
                candidate = openalex_candidate(record)
                location = record.get("best_oa_location") or {}
                candidate.update(
                    license=location.get("license"),
                    version=location.get("version"),
                    published_at=record.get("publication_date"),
                )
                rows.append(candidate)
            return rows
        if provider == "arxiv":
            params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
            if policy.intent == "latest":
                params.update(sortBy="submittedDate", sortOrder="descending")
            xml = await self._fetch(
                client,
                budget,
                provider,
                "https://export.arxiv.org/api/query",
                params,
                xml=True,
            )
            candidates = parse_arxiv_feed(xml)["results"]
            entries = ET.fromstring(xml).findall(f"{ATOM}entry")
            for c, entry in zip(candidates, entries, strict=True):
                if c:
                    versioned = (entry.findtext(f"{ATOM}id") or "").split("/abs/")[-1]
                    c.update(
                        arxiv_id=versioned,
                        version=versioned,
                        source_url=f"https://arxiv.org/abs/{versioned}",
                        published_at=entry.findtext(f"{ATOM}published"),
                    )
            return [c for c in candidates if c]
        if provider == "pubmed":

            async def pub_json(url, params):
                if url.endswith("esearch.fcgi") and policy.intent == "latest":
                    params = {**params, "sort": "pub_date"}
                return await fetch(url, params)

            result = await search_pubmed(
                pub_json,
                lambda url, params: self._fetch(
                    client, budget, provider, url, params, xml=True
                ),
                query,
                limit,
                year_from=policy.year_from,
                year_to=policy.year_to,
                api_key=settings.ncbi_api_key,
                email=settings.ncbi_email,
            )
            return [c for c in result["results"] if c]
        if provider == "crossref":
            params = {"query": query, "rows": limit}
            if policy.intent == "latest":
                params.update(sort="published", order="desc")
            raw = await fetch("https://api.crossref.org/works", params)
            return [
                {
                    "provider": provider,
                    "provider_id": r.get("DOI"),
                    "doi": r.get("DOI"),
                    "title": (r.get("title") or ["Untitled"])[0],
                    "source_url": r.get("URL"),
                    "year": ((r.get("published") or {}).get("date-parts") or [[None]])[
                        0
                    ][0],
                }
                for r in (raw.get("message") or {}).get("items", [])[:limit]
            ]
        return []

    async def search(self, policy, identities=()):
        if policy.web_prohibited:
            return SearchResult(events=[{"status": "prohibited"}])
        budget = WebBudget()
        candidates = []
        providers = (
            (policy.provider,)
            if policy.provider
            else route_sources(policy.question)[:2]
        )
        # Latest needs date ordering, which S2 relevance search cannot guarantee.
        if policy.intent == "latest" and policy.provider is None:
            providers = ("openalex", "crossref")
        identities = tuple(
            i for i in identities if i.paper_id in (policy.paper_ids or ())
        )
        if policy.paper_ids is not None and not identities:
            return SearchResult(
                events=[{"status": "unresolved_scoped_identity"}], budget=budget
            )
        queries = (
            [public_query(policy.question)]
            if policy.paper_ids is None
            else list(
                dict.fromkeys(
                    normalize_doi(i.doi) or i.arxiv_id or i.provider_id
                    for i in identities
                    if i.doi or i.arxiv_id or i.provider_id
                )
            )[:2]
        )
        owns = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=settings.web_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        try:
            async with asyncio.timeout(settings.web_deadline_seconds):
                for provider in providers:
                    for query in queries:
                        try:
                            rows = await self._provider(
                                client, budget, provider, query, policy
                            )
                            for c in rows:
                                year = c.get("year")
                                if (policy.year_from or policy.year_to) and (
                                    not isinstance(year, int)
                                    or year < (policy.year_from or 1800)
                                    or year > (policy.year_to or 2100)
                                ):
                                    continue
                                identity = next(
                                    (i for i in identities if match_identity(c, i)),
                                    None,
                                )
                                if policy.paper_ids is not None and identity is None:
                                    continue
                                c["local_paper_id"] = (
                                    identity.paper_id if identity else None
                                )
                                c["doi"] = normalize_doi(c.get("doi"))
                                c["source_url"] = safe_url(c.get("source_url"))
                                if not c["source_url"]:
                                    continue
                                c["title"] = str(c.get("title") or "Untitled")[:500]
                                c["abstract"] = (
                                    str(c["abstract"])
                                    if c.get("abstract")
                                    and len(str(c["abstract"])) <= 16000
                                    else None
                                )
                                c["retrieved_at"] = datetime.now(UTC).isoformat()
                                candidates.append(c)
                        except Exception:
                            # Events contain no provider error text.
                            continue
        except TimeoutError:
            budget.events.append({"status": "deadline"})
        finally:
            if owns:
                await client.aclose()
        seen, result = set(), []
        for c in candidates:
            key = (
                c.get("doi")
                or c.get("arxiv_id")
                or (c["provider"], c.get("provider_id")),
                c.get("version"),
            )
            if key not in seen:
                seen.add(key)
                result.append(c)
        if policy.intent == "latest":
            result.sort(
                key=lambda c: (c.get("year") or 0, c.get("published_at") or ""),
                reverse=True,
            )
        return SearchResult(result[:16], budget.events, budget)
