"""Explicit Crossref update direction and public DOI relations, with safe unknowns."""

import asyncio
import json
from datetime import UTC, date, datetime
from urllib.parse import quote

import httpx
from sqlalchemy import select

from app.models import (
    Paper,
    PaperIdentifier,
    PublicationState,
    SourceDocument,
    SourceIndexGeneration,
)
from app.services.academic_search import normalize_doi
from app.services.trace_versions import digest

MAX_BYTES = 262144
RELATIONS = {"is-preprint-of", "has-preprint", "is-version-of", "has-version"}
STATUSES = {
    "retraction": "retracted",
    "withdrawal": "withdrawn",
    "expression_of_concern": "expression_of_concern",
}


def event_date(value):
    parts = (value or {}).get("date-parts") if isinstance(value, dict) else None
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return None
    values = parts[0]
    if not 1 <= len(values) <= 3 or any(type(x) is not int for x in values):
        return None
    try:
        known = date(
            values[0],
            values[1] if len(values) > 1 else 1,
            values[2] if len(values) > 2 else 1,
        )
    except ValueError:
        return None
    if known > datetime.now(UTC).date():
        return None
    return "-".join(str(x) if i == 0 else f"{x:02d}" for i, x in enumerate(values))


def status_from_records(doi, record, updates):
    doi = normalize_doi(doi)
    result = {
        "doi": doi,
        "status": "unknown",
        "relations": [],
        "updates": [],
        "notices": [],
    }
    if (
        not doi
        or not isinstance(record, dict)
        or normalize_doi(record.get("DOI")) != doi
    ):
        result["notices"].append("record_identity_unverified")
        return result
    relations = record.get("relation") or {}
    if isinstance(relations, dict):
        for kind in sorted(RELATIONS):
            rows = relations.get(kind, [])
            for relation in rows[:10] if isinstance(rows, list) else []:
                related = (
                    normalize_doi(relation.get("id"))
                    if isinstance(relation, dict)
                    else None
                )
                if related and related != doi and relation.get("id-type") == "doi":
                    result["relations"].append(
                        {
                            "kind": kind,
                            "doi": related,
                            "url": "https://doi.org/" + quote(related, safe="/"),
                        }
                    )
    for notice in updates[:20] if isinstance(updates, list) else []:
        if not isinstance(notice, dict):
            continue
        notice_doi = normalize_doi(notice.get("DOI"))
        if not notice_doi or notice_doi == doi:
            continue
        events = notice.get("update-to", [])
        for event in events[:10] if isinstance(events, list) else []:
            if not isinstance(event, dict) or normalize_doi(event.get("DOI")) != doi:
                continue
            when = event_date(event.get("updated"))
            kind = event.get("type")
            if when and kind in {*STATUSES, "correction", "reinstatement"}:
                result["updates"].append(
                    {
                        "kind": kind,
                        "date": when,
                        "notice_doi": notice_doi,
                        "url": "https://doi.org/" + quote(notice_doi, safe="/"),
                    }
                )
    states = {STATUSES[e["kind"]] for e in result["updates"] if e["kind"] in STATUSES}
    if any(e["kind"] == "reinstatement" for e in result["updates"]) or len(states) > 1:
        result["notices"].append("conflicting_status_events")
    elif states:
        result["status"] = states.pop()
    result["source_url"] = "https://api.crossref.org/works/" + quote(doi, safe="/")
    result["event_digest"] = digest([result["relations"], result["updates"]])
    return result


async def get_status(db, paper_id):
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise LookupError("Paper not found")
    identifiers = list(
        (
            await db.scalars(
                select(PaperIdentifier).where(PaperIdentifier.paper_id == paper_id)
            )
        ).all()
    )
    doi = next(
        (normalize_doi(item.value) for item in identifiers if item.kind == "doi"), None
    )
    stored = await db.get(PublicationState, paper_id)
    document = await db.scalar(
        select(SourceDocument)
        .join(
            SourceIndexGeneration,
            SourceIndexGeneration.document_id == SourceDocument.id,
        )
        .where(
            SourceIndexGeneration.paper_id == paper_id,
            SourceIndexGeneration.is_active.is_(True),
        )
    )
    result = {
        "doi": doi,
        "status": "unknown",
        "relations": [],
        "updates": [],
        "notices": [],
        "source_url": None,
    }
    bound = bool(stored and doi and stored.details.get("doi") == doi)
    if bound:
        result.update(stored.details)
    checked = stored.checked_at if bound else None
    error = stored.refresh_error if bound else None
    return {
        **result,
        "paper_id": paper_id,
        "arxiv_ids": [i.value for i in identifiers if i.kind == "arxiv"],
        "source_version": document.document_version if document else None,
        "source_date": paper.published_date.isoformat()
        if paper.published_date
        else None,
        "checked_at": checked.replace(tzinfo=UTC).isoformat() if checked else None,
        "refresh_error": error,
        "stale": bool(checked and error),
    }


async def fetch_json(client, url, **kwargs):
    async with client.stream(
        "GET", url, follow_redirects=False, timeout=10, **kwargs
    ) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > MAX_BYTES:
            raise ValueError("Metadata response too large")
        parts, size = [], 0
        async for block in response.aiter_bytes():
            size += len(block)
            if size > MAX_BYTES:
                raise ValueError("Metadata response too large")
            parts.append(block)
        data = json.loads(b"".join(parts))
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise ValueError("Metadata shape invalid")
        return data["message"]


async def refresh_status(db, paper_id, *, client=None, local_only=False):
    current = await get_status(db, paper_id)
    if local_only or not current["doi"]:
        return {
            **current,
            "notices": [
                *current["notices"],
                "local_only" if local_only else "doi_unavailable",
            ],
        }
    owned = client is None
    client = client or httpx.AsyncClient(
        timeout=10, trust_env=False, follow_redirects=False
    )
    state = await db.get(PublicationState, paper_id)
    if state is None:
        state = PublicationState(
            paper_id=paper_id, details={"doi": current["doi"], "status": "unknown"}
        )
        db.add(state)
    elif state.details.get("doi") != current["doi"]:
        state.details = {"doi": current["doi"], "status": "unknown"}
        state.checked_at = None
    state.last_attempt_at = datetime.now(UTC)
    try:
        async with asyncio.timeout(20):
            record = await fetch_json(
                client,
                "https://api.crossref.org/works/" + quote(current["doi"], safe="/"),
            )
            if normalize_doi(record.get("DOI")) != current["doi"]:
                raise ValueError("Metadata DOI mismatch")
            updates = await fetch_json(
                client,
                "https://api.crossref.org/works",
                params={"filter": "updates:" + current["doi"], "rows": 10},
            )
            state.details = status_from_records(
                current["doi"], record, updates.get("items", [])
            )
            state.checked_at, state.refresh_error = datetime.now(UTC), None
    except Exception as exc:
        state.refresh_error = type(exc).__name__
    finally:
        if owned:
            await client.aclose()
    await db.commit()
    return await get_status(db, paper_id)


async def citation_status(db, paper_id, doi):
    result = {
        "publication_status": "unknown",
        "publication_source_url": None,
        "publication_checked_at": None,
        "publication_stale": False,
    }
    state = await db.get(PublicationState, paper_id) if doi else None
    if (
        state
        and state.checked_at
        and isinstance(state.details, dict)
        and state.details.get("doi") == doi
    ):
        status = state.details.get("status")
        if status in {"unknown", *STATUSES.values()}:
            result.update(
                publication_status=status,
                publication_source_url="https://api.crossref.org/works/"
                + quote(doi, safe="/"),
                publication_checked_at=state.checked_at.replace(tzinfo=UTC).isoformat(),
                publication_stale=bool(state.refresh_error),
            )
    return result
