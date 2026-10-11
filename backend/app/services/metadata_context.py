"""On-demand, source-attributed DOI metadata and citation relationships."""

import asyncio
import re
from urllib.parse import quote

import httpx

from app.services.discovery import DiscoveryService, _provider_error, normalize_doi


def _crossref_metadata(payload: dict, doi: str) -> dict | None:
    record = payload.get("message") or {}
    if normalize_doi(record.get("DOI")) != doi:
        return None
    date_parts = (record.get("published") or {}).get("date-parts") or []
    parts = date_parts[0] if date_parts else []
    published = (
        "-".join(
            f"{part:02d}" if index else str(part)
            for index, part in enumerate(parts[:3])
            if isinstance(part, int)
        )
        or None
    )
    venue = record.get("container-title") or []
    return {
        "source": "Crossref",
        "doi": doi,
        "venue": venue[0] if isinstance(venue, list) and venue else None,
        "publisher": record.get("publisher"),
        "type": record.get("type"),
        "published": published,
        "reference_count": record.get("references-count"),
    }


def _semantic_id(candidate: dict) -> str | None:
    if candidate["provider"] == "semantic_scholar":
        return candidate["provider_id"]
    if candidate.get("doi"):
        return f"DOI:{candidate['doi']}"
    if candidate.get("arxiv_id"):
        return f"ARXIV:{candidate['arxiv_id']}"
    if candidate["provider"] == "pubmed":
        return f"PMID:{candidate['provider_id']}"
    return None


def _citing_papers(payload: dict) -> list[dict]:
    papers = []
    for entry in payload.get("data") or []:
        record = entry.get("citingPaper") or {}
        paper_id = record.get("paperId") or ""
        if not re.fullmatch(r"[A-Za-z0-9]{1,100}", paper_id):
            continue
        papers.append(
            {
                "source": "Semantic Scholar",
                "title": record.get("title") or "Untitled",
                "year": record.get("year"),
                "url": f"https://www.semanticscholar.org/paper/{paper_id}",
            }
        )
    return papers


def _references(payload: dict, identifiers: list[str]) -> list[dict]:
    by_id = {
        record["id"].rsplit("/", 1)[-1]: record
        for record in payload.get("results") or []
        if record.get("id")
    }
    return [
        {
            "source": "OpenAlex",
            "title": by_id[identifier].get("display_name") or "Untitled",
            "year": by_id[identifier].get("publication_year"),
            "url": f"https://openalex.org/{identifier}",
        }
        for identifier in identifiers
        if identifier in by_id
    ]


def _record_error(errors: list[str], source: str, outcome: BaseException) -> None:
    if (
        isinstance(outcome, httpx.HTTPStatusError)
        and outcome.response.status_code == 404
    ):
        return
    errors.append(_provider_error(source, outcome))


async def get_paper_context(
    service: DiscoveryService, provider: str, provider_id: str
) -> dict:
    candidate = await service.get_candidate(provider, provider_id)
    doi = normalize_doi(candidate.get("doi"))
    semantic_id = _semantic_id(candidate)
    jobs = {}
    if doi:
        jobs["Crossref"] = service._get(
            "https://api.crossref.org/works/" + quote(doi, safe="/"),
        )
    if semantic_id:
        jobs["Semantic Scholar"] = service._get(
            "https://api.semanticscholar.org/graph/v1/paper/"
            + quote(semantic_id, safe=":/")
            + "/citations",
            {"limit": 5, "offset": 0, "fields": "title,url,year"},
        )
    if candidate["provider"] != "openalex" and doi:
        jobs["OpenAlex"] = service._get(
            "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="/"),
        )
    results = await asyncio.gather(*jobs.values(), return_exceptions=True)
    outcomes = dict(zip(jobs, results, strict=True))
    errors = []
    for source, outcome in outcomes.items():
        if isinstance(outcome, BaseException):
            _record_error(errors, source, outcome)
    crossref = outcomes.get("Crossref")
    metadata = (
        _crossref_metadata(crossref, doi)
        if doi and isinstance(crossref, dict)
        else None
    )
    semantic = outcomes.get("Semantic Scholar")
    citations = _citing_papers(semantic) if isinstance(semantic, dict) else []
    openalex = (
        candidate if candidate["provider"] == "openalex" else outcomes.get("OpenAlex")
    )
    raw_ids = (
        (openalex or {}).get("_reference_ids")
        or ((openalex or {}).get("referenced_works") or [])
        if isinstance(openalex, dict)
        else []
    )
    reference_ids = [
        value.rsplit("/", 1)[-1]
        for value in raw_ids
        if re.fullmatch(r"(?:https://openalex.org/)?W\d{1,20}", value)
    ][:5]
    references = []
    if reference_ids:
        try:
            payload = await service._get(
                "https://api.openalex.org/works",
                {
                    "filter": "openalex:" + "|".join(reference_ids),
                    "per_page": len(reference_ids),
                    "select": "id,display_name,publication_year",
                },
            )
            references = _references(payload, reference_ids)
        except Exception as exc:
            _record_error(errors, "OpenAlex", exc)
    return {
        "metadata": metadata,
        "citing_papers": citations,
        "references": references,
        "errors": errors,
    }
