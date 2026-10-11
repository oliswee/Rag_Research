"""Normalized paper discovery from Semantic Scholar and OpenAlex."""

import asyncio
import re
import xml.etree.ElementTree as ET
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



def pubmed_candidate(article: ET.Element) -> dict:
    citation = article.find("MedlineCitation")
    if citation is None or citation.findtext("PMID") is None:
        raise ValueError("Missing PMID")
    pmid = citation.findtext("PMID")
    metadata = citation.find("Article")
    if metadata is None:
        raise ValueError("Missing article metadata")
    node = metadata.find("ArticleTitle")
    title = " ".join("".join(node.itertext()).split()) if node is not None else "Untitled"
    abstract_node = metadata.find("Abstract")
    abstract = " ".join("".join(n.itertext()) for n in abstract_node.findall("AbstractText")) if abstract_node is not None else None
    authors = []
    for author in metadata.findall("./AuthorList/Author"):
        name = " ".join(filter(None, (author.findtext("ForeName"), author.findtext("LastName"))))
        if name:
            authors.append(name)
    year_text = (metadata.findtext("./Journal/JournalIssue/PubDate/Year")
                 or metadata.findtext("./Journal/JournalIssue/PubDate/MedlineDate")
                 or citation.findtext("./DateCompleted/Year") or "")
    match = re.search(r"(?:18|19|20)\d{2}", year_text)
    ids = {n.get("IdType"): n.text for n in article.findall("./PubmedData/ArticleIdList/ArticleId")}
    return {
        "provider": "pubmed", "provider_id": pmid, "title": title,
        "authors": authors, "year": int(match.group()) if match else None,
        "abstract": " ".join(abstract.split()) if abstract else None,
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "pdf_url": None, "open_access": bool(ids.get("pmc")),
        "doi": normalize_doi(ids.get("doi")), "arxiv_id": None,
        "tags": [],
        "tags": [n.text for n in metadata.findall("./PublicationTypeList/PublicationType") if n.text],
    }


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
        "tags": [],
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

    async def _pubmed(self, query: str, offset: int, limit: int) -> dict:
        found = await self._get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {"db": "pubmed", "term": query, "retmode": "json",
             "retstart": offset, "retmax": limit, "tool": "FolioTrace"},
        )
        result = found.get("esearchresult", {})
        ids = result.get("idlist", [])
        if not ids:
            return {"candidates": [], "total": int(result.get("count", 0))}
        response = await self.client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "FolioTrace"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return {"candidates": [pubmed_candidate(node) for node in root.findall("PubmedArticle")],
                "total": int(result.get("count", 0))}

    async def search(
        self, query: str, year_from: int | None = None,
        year_to: int | None = None, open_access: bool = False,
        limit: int = 20, offset: int = 0, provider: str | None = None,
    ) -> dict:
        limit = max(1, min(limit, 50))
        offset = max(0, offset)
        sources = [provider] if provider else ["semantic_scholar", "openalex", "pubmed"]
        if any(name not in {"semantic_scholar", "openalex", "pubmed"} for name in sources):
            raise ValueError("Unsupported provider")
        sem_params = {"query": query, "limit": limit, "offset": offset,
                      "fields": "paperId,title,authors,year,abstract,url,openAccessPdf,isOpenAccess,externalIds"}
        if year_from or year_to:
            sem_params["year"] = f"{year_from or ''}-{year_to or ''}"
        filters = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if open_access:
            filters.append("is_oa:true")
        oa_params = {"search": query, "per_page": limit, "page": offset // limit + 1,
                     "select": "id,display_name,authorships,publication_year,abstract_inverted_index,doi,ids,open_access,best_oa_location"}
        if filters:
            oa_params["filter"] = ",".join(filters)
        pubmed_query = query
        if year_from or year_to:
            pubmed_query += f" AND {year_from or 1800}:{year_to or 2100}[dp]"
        if open_access:
            pubmed_query += " AND free full text[sb]"
        calls = {
            "semantic_scholar": lambda: self._get("https://api.semanticscholar.org/graph/v1/paper/search", sem_params),
            "openalex": lambda: self._get("https://api.openalex.org/works", oa_params),
            "pubmed": lambda: self._pubmed(pubmed_query, offset, limit),
        }
        outcomes = await asyncio.gather(*(calls[name]() for name in sources), return_exceptions=True)
        candidates, errors, seen = [], [], set()
        has_more = False
        for name, outcome in zip(sources, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                errors.append(f"{name}: {type(outcome).__name__}")
                continue
            if name == "pubmed":
                records, total = outcome["candidates"], outcome["total"]
            elif name == "semantic_scholar":
                records, total = outcome.get("data", []), outcome.get("total", 0)
            else:
                records, total = outcome.get("results", []), outcome.get("meta", {}).get("count", 0)
            has_more = has_more or offset + limit < total
            normalizer = {"semantic_scholar": semantic_candidate, "openalex": openalex_candidate}.get(name)
            for raw in records:
                candidate = normalizer(raw) if normalizer else raw
                if open_access and not candidate["open_access"]:
                    continue
                identity = candidate.get("doi") or normalize_title(candidate["title"])
                if identity not in seen:
                    seen.add(identity)
                    candidates.append(candidate)
        return {"candidates": candidates, "errors": errors,
                "has_more": has_more, "next_offset": offset + limit if has_more else None}

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
        if provider == "pubmed":
            if not re.fullmatch(r"\d{1,12}", provider_id):
                raise ValueError("Invalid PMID")
            response = await self.client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": provider_id, "retmode": "xml", "tool": "FolioTrace"},
            )
            response.raise_for_status()
            article = ET.fromstring(response.content).find("PubmedArticle")
            if article is None:
                raise ValueError("PubMed record not found")
            return pubmed_candidate(article)
        raise ValueError("Unsupported provider")
