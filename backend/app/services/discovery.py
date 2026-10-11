"""Normalized paper discovery from public academic providers."""

import asyncio
import math
import re
import time
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from app.config import settings
from app.services.arxiv import base_arxiv_id, parse_arxiv_feed
from app.services.pubmed import parse_pubmed_xml, search_pubmed

_CACHE: dict[tuple[str, tuple], tuple[float, dict]] = {}
_CACHE_TTL = 600
_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST_REQUEST = 0.0
_QUESTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


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


def _abstract_summary(abstract: str | None) -> str | None:
    """Use only sentences present in the provider abstract; never invent findings."""
    if not abstract:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    informative = [
        sentence
        for sentence in sentences
        if len(sentence.split()) >= 3
        and not re.match(
            r"^(proceedings|journal|conference|published in)\b", sentence, re.I
        )
        and not (
            sentence.count(",") >= 4
            and not re.search(
                r"\b(the|this|we|study|paper|research|results)\b", sentence, re.I
            )
        )
    ]
    summary = " ".join(informative[:2]).strip()
    if len(summary) > 480:
        summary = summary[:480].rsplit(" ", 1)[0] + "…"
    return summary or None


def _tags(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))[:5]


def _provider_error(name: str, error: BaseException) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 429:
            return f"{name}: 429 rate limit reached; retry later or add an API key"
        return f"{name}: HTTP {status} from the provider"
    if isinstance(error, httpx.TimeoutException):
        return f"{name}: request timed out"
    if isinstance(error, httpx.RequestError):
        return f"{name}: network request failed"
    return f"{name}: provider response could not be processed"


def _explain_relevance(candidate: dict, query: str) -> str:
    words = set(re.findall(r"[\w-]{3,}", query.casefold()))
    evidence = " ".join(
        [candidate["title"], candidate.get("abstract") or "", *candidate["tags"]]
    ).casefold()
    matched = [word for word in words if word in evidence]
    if matched:
        return "Matches topic terms: " + ", ".join(sorted(matched)[:4]) + "."
    return "Returned by the academic source; inspect the abstract for relevance."


def semantic_candidate(record: dict) -> dict:
    ids = record.get("externalIds") or {}
    doi = normalize_doi(ids.get("DOI"))
    return {
        "provider": "semantic_scholar",
        "provider_id": record["paperId"],
        "title": record.get("title") or "Untitled",
        "authors": [a["name"] for a in record.get("authors") or [] if a.get("name")],
        "year": record.get("year"),
        "abstract": record.get("abstract"),
        "source_url": f"https://doi.org/{doi}" if doi else record.get("url"),
        "pdf_url": (record.get("openAccessPdf") or {}).get("url"),
        "open_access": bool(record.get("isOpenAccess") or record.get("openAccessPdf")),
        "doi": doi,
        "arxiv_id": ids.get("ArXiv"),
        "citation_count": record.get("citationCount"),
        "tags": _tags(record.get("fieldsOfStudy") or []),
        "abstract_summary": _abstract_summary(record.get("abstract")),
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
        "source_url": record.get("doi")
        or location.get("landing_page_url")
        or record.get("id"),
        "pdf_url": location.get("pdf_url"),
        "open_access": bool((record.get("open_access") or {}).get("is_oa")),
        "doi": normalize_doi(record.get("doi")),
        "arxiv_id": None,
        "citation_count": record.get("cited_by_count"),
        "tags": _tags(
            [
                (record.get("primary_topic") or {}).get("display_name") or "",
                *[
                    keyword.get("display_name") or ""
                    for keyword in record.get("keywords") or []
                ],
            ]
        ),
        "abstract_summary": _abstract_summary(
            _openalex_abstract(record.get("abstract_inverted_index"))
        ),
        "_reference_ids": [
            reference.rsplit("/", 1)[-1]
            for reference in record.get("referenced_works") or []
        ],
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

    async def _get_xml(self, url: str, params: dict) -> bytes:
        for attempt in range(3):
            response = await self.client.get(url, params=params)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()
            if len(response.content) > 3_000_000:
                raise ValueError("Academic XML response is too large")
            return response.content
        raise RuntimeError("Academic API unavailable")

    async def _search_pubmed(
        self,
        query: str,
        limit: int,
        offset: int,
        year_from: int | None,
        year_to: int | None,
    ) -> dict:
        return await search_pubmed(
            self._get,
            self._get_xml,
            query,
            limit,
            offset,
            year_from,
            year_to,
            False,  # Filter after fetching so skipped PubMed IDs advance the cursor.
            settings.ncbi_api_key,
            settings.ncbi_email,
        )

    async def _arxiv_feed(self, params: dict) -> dict:
        global _ARXIV_LAST_REQUEST
        # arXiv asks clients to leave three seconds between consecutive requests.
        async with _ARXIV_LOCK:
            delay = 3 - (time.monotonic() - _ARXIV_LAST_REQUEST)
            if delay > 0:
                await asyncio.sleep(delay)
            _ARXIV_LAST_REQUEST = time.monotonic()
            payload = await self._get_xml("https://export.arxiv.org/api/query", params)
        return parse_arxiv_feed(payload)

    async def _search_arxiv(self, query: str, limit: int, offset: int) -> dict:
        terms = [
            word.casefold()
            for word in re.findall(r"[\w-]{3,}", query)
            if word.casefold() not in _QUESTION_WORDS
        ][:4]
        search_query = " AND ".join(f"all:{term}" for term in terms) or f'all:"{query}"'
        return await self._arxiv_feed(
            {
                "search_query": search_query,
                "start": offset,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )

    async def search(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access: bool = False,
        limit: int = 20,
        offsets: dict[str, int] | None = None,
    ) -> dict:
        limit = max(1, min(limit, 20))
        offsets = offsets or {}
        start = {
            name: max(0, offsets.get(name, 0))
            for name in ("semantic_scholar", "openalex", "pubmed", "arxiv")
        }
        semantic_params = {
            "query": query,
            "limit": limit,
            "offset": start["semantic_scholar"],
            "fields": (
                "paperId,title,authors,year,abstract,url,"
                "openAccessPdf,isOpenAccess,externalIds,citationCount,fieldsOfStudy"
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
            "page": start["openalex"] // limit + 1,
            "select": (
                "id,display_name,authorships,publication_year,"
                "abstract_inverted_index,doi,ids,open_access,best_oa_location,"
                "cited_by_count,primary_topic,keywords,referenced_works"
            ),
        }
        if filters:
            openalex_params["filter"] = ",".join(filters)
        calls = await asyncio.gather(
            self._get(
                "https://api.semanticscholar.org/graph/v1/paper/search", semantic_params
            ),
            self._get("https://api.openalex.org/works", openalex_params),
            self._search_pubmed(query, limit, start["pubmed"], year_from, year_to),
            self._search_arxiv(query, limit, start["arxiv"]),
            return_exceptions=True,
        )
        ranked, errors = [], []
        next_offsets = start.copy()
        more_by_source = {}
        fetched_by_source = {}
        for name, source, outcome, key, normalizer in (
            (
                "Semantic Scholar",
                "semantic_scholar",
                calls[0],
                "data",
                semantic_candidate,
            ),
            ("OpenAlex", "openalex", calls[1], "results", openalex_candidate),
            ("PubMed", "pubmed", calls[2], "results", lambda record: record),
            ("arXiv", "arxiv", calls[3], "results", lambda record: record),
        ):
            if isinstance(outcome, BaseException):
                errors.append(_provider_error(name, outcome))
                continue
            records = outcome.get(key, [])
            if source == "openalex":
                records = records[start[source] % limit :]
            fetched = outcome.get("fetched_count", len(records))
            fetched_by_source[source] = fetched
            total = (
                outcome.get("total")
                if source != "openalex"
                else (outcome.get("meta") or {}).get("count")
            )
            more_by_source[source] = (
                start[source] + fetched < total
                if isinstance(total, int)
                else fetched >= limit
            )
            for rank, raw in enumerate(records):
                if raw is None:
                    ranked.append((rank, source, None))
                    continue
                candidate = normalizer(raw)
                if source == "arxiv" and candidate.get("year"):
                    if year_from and candidate["year"] < year_from:
                        ranked.append((rank, source, None))
                        continue
                    if year_to and candidate["year"] > year_to:
                        ranked.append((rank, source, None))
                        continue
                if open_access and not candidate["open_access"]:
                    ranked.append((rank, source, None))
                    continue
                candidate["abstract_summary"] = candidate.get(
                    "abstract_summary"
                ) or _abstract_summary(candidate.get("abstract"))
                candidate["relevance_reason"] = _explain_relevance(candidate, query)
                candidate["_provider_rank"] = rank
                citations = max(0, candidate.get("citation_count") or 0)
                age = max(0, datetime.now(UTC).year - (candidate.get("year") or 1800))
                # Provider rank drives relevance; other signals break ties.
                candidate["rank_score"] = round(
                    1 / (rank + 1)
                    + 0.01 * min(math.log1p(citations), 8)
                    + 0.02 / (age + 1),
                    4,
                )
                ranked.append((rank, source, candidate))
        # Interleave provider rankings so all available sources remain visible.
        ranked.sort(
            key=lambda entry: (
                entry[0],
                start[entry[1]],
                -(entry[2] or {}).get("rank_score", 0),
            )
        )
        candidates = []
        by_identifier = {}
        for _, source, candidate in ranked:
            next_offsets[source] += 1
            if candidate is None:
                continue
            identifiers = [(candidate["provider"], candidate["provider_id"])]
            if candidate.get("doi"):
                identifiers.append(("doi", candidate["doi"].lower()))
            if candidate.get("arxiv_id"):
                identifiers.append(("arxiv", candidate["arxiv_id"].lower()))
            existing = next(
                (by_identifier[key] for key in identifiers if key in by_identifier),
                None,
            )
            if existing is None:
                existing = len(candidates)
                candidates.append(candidate)
            elif candidate["provider"] == "openalex":
                candidates[existing] = candidate
            for key in identifiers:
                by_identifier[key] = existing
            if len(candidates) >= limit:
                break
        openalex = {
            candidate["provider_id"]: candidate
            for candidate in candidates
            if candidate["provider"] == "openalex"
        }
        for candidate in candidates:
            candidate["cited_by_results"] = []
        for candidate in openalex.values():
            for reference_id in candidate.pop("_reference_ids", []):
                target = openalex.get(reference_id)
                if target and target is not candidate:
                    target["cited_by_results"].append(
                        {
                            "provider_id": candidate["provider_id"],
                            "title": candidate["title"],
                        }
                    )
        citation_counts = sorted(
            candidate["citation_count"]
            for candidate in candidates
            if isinstance(candidate.get("citation_count"), int)
            and candidate["citation_count"] > 0
        )
        upper_quartile = (
            citation_counts[math.ceil(0.75 * len(citation_counts)) - 1]
            if citation_counts
            else 0
        )
        for candidate in candidates:
            labels = []
            year = candidate.get("year")
            if year and year >= datetime.now(UTC).year - 2:
                labels.append("Recent")
            highly_cited = (candidate.get("citation_count") or 0) >= max(
                10, upper_quartile
            )
            if highly_cited:
                labels.append("Highly cited in results")
            if candidate["cited_by_results"]:
                labels.append("Referenced by related results")
                candidate["rank_score"] += (
                    min(len(candidate["cited_by_results"]), 3) * 0.03
                )
            if (
                highly_cited
                and candidate["cited_by_results"]
                and year
                and year <= datetime.now(UTC).year - 5
            ):
                labels.append("Potential foundation")
            candidate["role_labels"] = labels
        candidates.sort(
            key=lambda candidate: (
                candidate["_provider_rank"],
                -candidate["rank_score"],
            )
        )
        for candidate in candidates:
            candidate.pop("_provider_rank", None)
        themes = Counter(
            candidate["tags"][0] for candidate in candidates if candidate["tags"]
        )
        return {
            "candidates": candidates,
            "errors": errors,
            "next_offsets": next_offsets,
            "has_more": any(more_by_source.values())
            or any(
                next_offsets[source] < start[source] + fetched_by_source[source]
                for source in fetched_by_source
            ),
            "themes": [
                {"label": label, "count": count}
                for label, count in themes.most_common(8)
            ],
        }

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
                        "openAccessPdf,isOpenAccess,externalIds,citationCount,fieldsOfStudy"
                    )
                },
            )
            return semantic_candidate(raw)
        if provider == "openalex":
            if not re.fullmatch(r"W\d{1,20}", provider_id):
                raise ValueError("Invalid OpenAlex ID")
            raw = await self._get("https://api.openalex.org/works/" + provider_id)
            return openalex_candidate(raw)
        if provider == "pubmed":
            if not re.fullmatch(r"\d{1,12}", provider_id):
                raise ValueError("Invalid PubMed ID")
            params = {
                "db": "pubmed",
                "id": provider_id,
                "retmode": "xml",
                "tool": "FolioTrace",
            }
            if settings.ncbi_api_key:
                params["api_key"] = settings.ncbi_api_key
            if settings.ncbi_email:
                params["email"] = settings.ncbi_email
            payload = await self._get_xml(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params,
            )
            papers = parse_pubmed_xml(payload)
            if not papers or papers[0]["provider_id"] != provider_id:
                raise ValueError("PubMed record was not found")
            return papers[0]
        if provider == "arxiv":
            paper_id = base_arxiv_id(provider_id)
            if paper_id is None or paper_id != provider_id:
                raise ValueError("Invalid arXiv ID")
            result = await self._arxiv_feed({"id_list": paper_id})
            papers = [
                paper
                for paper in result["results"]
                if paper and paper["provider_id"] == paper_id
            ]
            if not papers:
                raise ValueError("arXiv record was not found")
            return papers[0]
        raise ValueError("Unsupported provider")
