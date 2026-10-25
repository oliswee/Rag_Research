"""Request-local evidence. No database, import service or vector-store dependency."""

import asyncio
import hashlib
import re
import uuid
from dataclasses import dataclass, field

from app.config import settings
from app.services.academic_search import (
    WebBudget,
    licensed_location,
    match_identity,
    safe_url,
)
from app.services.pdf_download import download_pdf
from app.services.source_cleaning import clean_document
from app.services.temporary_pdf import parse_temporary_pdf


@dataclass
class WebEvidencePool:
    sources: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def _ranges(text, size=1000):
    """Keep whole sentences; skip an overlong indivisible sentence, never cut a number."""
    start = 0
    for match in re.finditer(r"(?<=[.!?。！？])\s+|\n\n+", text):
        end = match.end()
        if end - start >= size:
            if end - start <= 4000:
                yield start, end
            start = end
    if start < len(text) and len(text) - start <= 4000:
        yield start, len(text)


def _source(
    record, excerpt, version, *, page=None, block_id=None, start=0, origin="web"
):
    end = start + len(excerpt)
    span = {
        "paper_id": record["paper_id"],
        "document_id": record["paper_id"],
        "document_version": version,
        "page": page,
        "block_id": block_id or "abstract",
        "raw_start": start,
        "raw_end": end,
        "source_origin": origin,
        "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
    }
    return {
        **record,
        "chunk_id": "web:" + str(uuid.uuid4()),
        "excerpt": excerpt,
        "document_id": record["paper_id"],
        "document_version": version,
        "parent_id": None,
        "source_spans": [span],
        "content_scope": "full_text" if page is not None else "abstract",
        "page_start": page,
        "page_end": page,
        "raw_start": start,
        "raw_end": end,
        "evidence_role": "body",
        "citation_url": record["source_url"],
        "score": None,
    }


async def build_web_evidence(
    candidates,
    policy,
    identities=(),
    *,
    fetcher=None,
    parser=None,
    trace=None,
    budget=None,
):
    pool = WebEvidencePool()
    if policy.web_prohibited:
        return pool
    budget = budget or WebBudget()
    fetcher, parser = fetcher or download_pdf, parser or parse_temporary_pdf
    pdf_attempted = False
    seen = set()
    for candidate in candidates[:16]:
        url = safe_url(candidate.get("source_url"))
        if not url:
            continue
        identity = next(
            (
                i
                for i in identities
                if i.paper_id in (policy.paper_ids or ())
                and match_identity(candidate, i)
            ),
            None,
        )
        if policy.paper_ids is not None and identity is None:
            continue
        key = (
            candidate.get("doi")
            or candidate.get("arxiv_id")
            or (candidate.get("provider"), candidate.get("provider_id")),
            candidate.get("version"),
        )
        if key in seen:
            continue
        seen.add(key)
        record = {
            k: candidate.get(k)
            for k in (
                "doi",
                "arxiv_id",
                "provider",
                "provider_id",
                "title",
                "year",
                "published_at",
                "retrieved_at",
                "license",
                "version",
                "open_access",
            )
        }
        record.update(
            paper_id=identity.paper_id
            if identity
            else "web-paper:" + str(uuid.uuid4()),
            source_url=url,
            source_origin="web",
            evidence_level="metadata_only",
            page_start=None,
            page_end=None,
            access_status="licensed_oa"
            if licensed_location(candidate)
            else "permission_unknown",
            version_status="unknown",
        )
        abstract = candidate.get("abstract")
        # A native PDF version cannot be established by a provider's DOI alone.
        scoped_native = bool(identity and identity.document_version)
        sources = []
        pdf_url = licensed_location(candidate)
        if pdf_url and not pdf_attempted and budget.remaining > 0:
            pdf_attempted = True
            try:
                async with asyncio.timeout(budget.remaining):
                    payload = await fetcher(
                        pdf_url,
                        max_bytes=settings.web_pdf_bytes,
                        max_redirects=2,
                        same_host=True,
                        on_attempt=budget.consume,
                    )
                    parsed = await parser(payload)
                    if (
                        scoped_native
                        and parsed.document_version != identity.document_version
                    ):
                        record["version_status"] = "mismatch"
                    else:
                        record.update(
                            evidence_level="full_text",
                            version_status="matched"
                            if scoped_native
                            else "retrieved_version",
                        )
                        cleaned = {b.block_id: b for b in clean_document(parsed).blocks}
                        for page in parsed.pages:
                            for block in page.blocks:
                                flags = cleaned[block.id].flags
                                if {
                                    "reference_section",
                                    "page_number_candidate",
                                    "repeated_header_candidate",
                                    "repeated_footer_candidate",
                                } & set(flags):
                                    continue
                                if (
                                    page.text[block.raw_start : block.raw_end]
                                    != block.text
                                ):
                                    continue
                                for a, b in _ranges(block.text):
                                    sources.append(
                                        _source(
                                            record,
                                            block.text[a:b],
                                            parsed.document_version,
                                            page=page.number,
                                            block_id=block.id,
                                            start=block.raw_start + a,
                                        )
                                    )
                                    if len(sources) >= 30:
                                        break
                                if len(sources) >= 30:
                                    break
                            if len(sources) >= 30:
                                break
                        if not sources:
                            record["evidence_level"] = "metadata_only"
            except Exception as exc:
                pool.events.append(
                    {
                        "stage": "temporary_pdf",
                        "status": "unavailable",
                        "error_type": type(exc).__name__,
                    }
                )
        if not sources and abstract and not scoped_native:
            record["evidence_level"] = "abstract_only"
            version = "sha256:" + hashlib.sha256(abstract.encode()).hexdigest()
            for a, b in _ranges(abstract):
                sources.append(_source(record, abstract[a:b], version, start=a))
        pool.sources.extend(sources)
        pool.records.append(record)
        pool.events.append(
            {
                "stage": "web_evidence",
                "status": record["evidence_level"],
                "version_status": record["version_status"],
            }
        )
    if trace is not None:
        trace.record["web_events"].extend(pool.events)
    return pool
