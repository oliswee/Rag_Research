"""Fetch optional Crossref metadata for a selected DOI."""

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


async def get_paper_context(service: DiscoveryService, doi: str) -> dict:
    """Use one Crossref request; absent metadata does not block the paper card."""
    normalized = normalize_doi(doi)
    try:
        payload = await service._get(
            "https://api.crossref.org/works/" + quote(normalized, safe="/")
        )
        return {"metadata": _crossref_metadata(payload, normalized), "errors": []}
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return {"metadata": None, "errors": []}
        return {"metadata": None, "errors": [_provider_error("Crossref", exc)]}
    except Exception as exc:
        return {"metadata": None, "errors": [_provider_error("Crossref", exc)]}
