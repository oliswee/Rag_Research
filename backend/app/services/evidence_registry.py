"""Revalidate retrieved evidence against SQLite or sealed current-request Web data."""

import hashlib
from dataclasses import dataclass
from types import MappingProxyType

from sqlalchemy import select

from app.models import (
    Paper,
    PaperChunk,
    PaperIdentifier,
    PaperStatus,
    SourceIndexGeneration,
)
from app.schemas.evidence import Citation, EvidenceSnapshot, EvidenceSpan
from app.services.academic_search import normalize_doi, safe_url
from app.services.source_anchors import resolve_child


def _hash(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _snapshot(source, *, origin, title=None, doi=None):
    text = source.get("excerpt")
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError("Invalid original excerpt")
    level = source.get("evidence_level") or (
        "abstract_only" if source.get("content_scope") == "abstract" else "full_text"
    )
    native = bool(source.get("document_version") and source.get("source_spans"))
    version = source.get("document_version") if native else _hash(text)
    spans = tuple(
        EvidenceSpan.model_validate(s) for s in source.get("source_spans", [])
    )
    if not spans:
        if origin == "web":
            raise ValueError("Web original span missing")
        spans = (
            EvidenceSpan(
                paper_id=source["paper_id"],
                document_id="legacy:" + source["chunk_id"],
                document_version=version,
                block_id=source["chunk_id"],
                raw_start=0,
                raw_end=len(text),
            ),
        )
    if any(
        s.paper_id != source["paper_id"] or s.document_version != version for s in spans
    ):
        raise ValueError("Original span identity mismatch")
    if sum(s.raw_end - s.raw_start for s in spans) + 2 * (len(spans) - 1) != len(text):
        raise ValueError("Original span length mismatch")
    pages = tuple(sorted({s.page for s in spans if s.page is not None}))
    if origin == "web":
        if level != "full_text" and (
            pages
            or source.get("page_start") is not None
            or source.get("page_end") is not None
        ):
            raise ValueError("Non-PDF evidence cannot claim PDF pages")
        if level == "full_text" and (not pages or not source.get("license")):
            raise ValueError("Full-text Web provenance incomplete")
        for span in source.get("source_spans", []):
            if span.get("excerpt_sha256") != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("Web excerpt digest mismatch")
        url = safe_url(source.get("source_url"))
        if not url:
            raise ValueError("Unsafe Web citation URL")
    else:
        url = f"/api/sources/chunks/{source['chunk_id']}"
    return EvidenceSnapshot(
        source_id=source["chunk_id"],
        paper_id=source["paper_id"],
        title=title or source.get("title") or "Unknown paper",
        doi=normalize_doi(doi or source.get("doi")),
        source_origin=origin,
        parser_origin=source.get("source_origin"),
        evidence_type=level,
        source_url=url,
        raw_text=text,
        spans=spans,
        version=version,
        pdf_pages=pages if native and level == "full_text" else (),
        retrieved_at=source.get("retrieved_at"),
        license=source.get("license"),
        year=source.get("year"),
        parent_id=source.get("parent_id"),
        parent_text=(source.get("parent") or {}).get("raw_text"),
        parent_spans=tuple(
            EvidenceSpan.model_validate(s)
            for s in (source.get("parent") or {}).get("source_spans", [])
        ),
    )


def seal_web_sources(sources):
    """Called by temporary Web pool only, before model/user data can mutate sources."""
    if len(sources) > 50:
        raise ValueError("Too many Web snapshots")
    return tuple(_snapshot(s, origin="web") for s in sources)


@dataclass(frozen=True)
class EvidenceRegistry:
    sources: MappingProxyType
    rejections: tuple[dict, ...] = ()

    def cite(self, source_id, quote):
        source = self.sources.get(source_id)
        if (
            source is None
            or source.evidence_type == "metadata_only"
            or not isinstance(quote, str)
            or len(quote.strip()) < 12
        ):
            raise ValueError("Unusable factual citation")
        left = source.raw_text.find(quote)
        if left < 0:
            raise ValueError("Quote is not in original child")
        right, offset, anchors = left + len(quote), 0, []
        for span in source.spans:
            size = span.raw_end - span.raw_start
            a, b = max(left, offset), min(right, offset + size)
            if a < b:
                anchors.append(
                    span.model_copy(
                        update={
                            "raw_start": span.raw_start + a - offset,
                            "raw_end": span.raw_start + b - offset,
                        }
                    )
                )
            offset += size + 2
        if not anchors:
            raise ValueError("Quote has no original anchor")
        pages = {s.page for s in anchors if s.page is not None}
        return Citation(
            citation_id="cite:"
            + hashlib.sha256((source_id + "\0" + quote).encode()).hexdigest()[:24],
            source_id=source_id,
            paper_id=source.paper_id,
            doi=source.doi,
            source_origin=source.source_origin,
            evidence_type=source.evidence_type,
            source_url=source.source_url,
            raw_quote=quote,
            span=tuple(anchors),
            pdf_page=next(iter(pages))
            if len(pages) == 1 and source.pdf_pages
            else None,
            version=source.version,
            retrieved_at=source.retrieved_at,
            license=source.license,
        )

    def source_dicts(self):
        return [
            {
                "chunk_id": s.source_id,
                "paper_id": s.paper_id,
                "title": s.title,
                "doi": s.doi,
                "source_origin": s.source_origin,
                "evidence_level": s.evidence_type,
                "content_scope": "abstract"
                if s.evidence_type == "abstract_only"
                else "full_text",
                "source_url": s.source_url,
                "citation_url": s.source_url,
                "excerpt": s.raw_text,
                "document_version": s.version,
                "document_id": s.spans[0].document_id,
                "source_spans": [span.model_dump() for span in s.spans],
                "parent_id": s.parent_id,
                "parent": {
                    "id": s.parent_id,
                    "raw_text": s.parent_text,
                    "source_spans": [span.model_dump() for span in s.parent_spans],
                }
                if s.parent_text
                else None,
                "page_start": min(s.pdf_pages) if s.pdf_pages else None,
                "page_end": max(s.pdf_pages) if s.pdf_pages else None,
                "license": s.license,
                "retrieved_at": s.retrieved_at,
                "year": s.year,
            }
            for s in self.sources.values()
        ]


async def build_registry(
    db,
    sources,
    *,
    web_snapshots=(),
    paper_ids=None,
    local_only=False,
    only_uploaded=False,
):
    if len(sources) > 50:
        raise ValueError("Too many verification sources")
    trusted_web = {s.source_id: s for s in web_snapshots}
    accepted, rejections = {}, []
    for source in sources:
        source_id = source.get("chunk_id")
        try:
            if (
                source.get("evidence_role") in {"bibliography", "page_marker"}
                or source.get("evidence_level") == "metadata_only"
            ):
                raise ValueError("Non-factual evidence")
            if paper_ids is not None and source.get("paper_id") not in paper_ids:
                raise ValueError("Source outside paper scope")
            if source.get("source_origin") == "web":
                if local_only or only_uploaded:
                    raise ValueError("Web prohibited")
                snapshot = trusted_web.get(source_id)
                current = _snapshot(source, origin="web")
                # Display title is not scientific identity; immutable original wins.
                if (
                    snapshot is None
                    or current.model_copy(update={"title": snapshot.title}) != snapshot
                ):
                    raise ValueError("Web snapshot unavailable or changed")
            else:
                child = await db.get(PaperChunk, source_id)
                paper = await db.get(Paper, child.paper_id) if child else None
                if (
                    child is None
                    or paper is None
                    or paper.status == PaperStatus.archived
                ):
                    raise ValueError("Local evidence unavailable")
                if only_uploaded and paper.import_source is not None:
                    raise ValueError("Imported source outside uploaded-only scope")
                original = await resolve_child(db, child, allow_parent_failure=True)
                if original.get("status") == "retired":
                    raise ValueError("Retired generation cannot support new findings")
                if child.index_version == "legacy" and await db.scalar(
                    select(SourceIndexGeneration.index_version)
                    .where(
                        SourceIndexGeneration.paper_id == paper.id,
                        SourceIndexGeneration.is_active.is_(True),
                    )
                    .limit(1)
                ):
                    raise ValueError(
                        "Superseded legacy source cannot support new findings"
                    )
                for key in (
                    "paper_id",
                    "excerpt",
                    "document_version",
                    "source_spans",
                    "page_start",
                    "page_end",
                ):
                    if source.get(key) != original.get(key):
                        raise ValueError("Retrieved original changed")
                doi = await db.scalar(
                    select(PaperIdentifier.value).where(
                        PaperIdentifier.paper_id == paper.id,
                        PaperIdentifier.kind == "doi",
                    )
                )
                snapshot = _snapshot(
                    {
                        **original,
                        "year": paper.published_date.year
                        if paper.published_date
                        else None,
                    },
                    origin="local",
                    title=paper.title,
                    doi=doi,
                )
            if source_id in accepted and snapshot != accepted[source_id]:
                accepted.pop(source_id)
                raise ValueError("Duplicate source identity conflict")
            accepted[source_id] = snapshot
        except (ValueError, TypeError, KeyError) as exc:
            rejections.append(
                {
                    "source_id": source_id,
                    "reason": "source_revalidation_failed",
                    "error_type": type(exc).__name__,
                }
            )
    return EvidenceRegistry(MappingProxyType(accepted), tuple(rejections))
