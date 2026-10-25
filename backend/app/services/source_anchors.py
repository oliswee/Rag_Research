"""Resolve retained child citations using validated original transcript anchors."""

import hashlib

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PaperChunk, SourceDocument, SourceIndexGeneration, SourceParent
from app.services.trace_versions import digest


def validate_document(document: SourceDocument) -> None:
    expected = "sha256:" + digest(
        [
            document.document_version,
            document.parser_version,
            document.raw_pages,
            document.source_url,
        ]
    )
    if expected != document.snapshot_digest:
        raise ValueError("Source snapshot integrity check failed")


def original_text(document: SourceDocument, spans: list[dict]) -> str:
    pages = {page["number"]: page for page in document.raw_pages}
    parts = []
    for span in spans:
        page = pages.get(span.get("page"))
        block = (
            next(
                (
                    block
                    for block in page["blocks"]
                    if block["id"] == span.get("block_id")
                ),
                None,
            )
            if page
            else None
        )
        left, right = span.get("raw_start"), span.get("raw_end")
        if (
            span.get("document_id") != document.id
            or span.get("paper_id") != document.paper_id
            or span.get("document_version") != document.document_version
            or block is None
            or type(left) is not int
            or type(right) is not int
            or not block["raw_start"] <= left < right <= block["raw_end"]
            or (list(span["bbox"]) if span.get("bbox") is not None else None)
            != (list(block["bbox"]) if block["bbox"] is not None else None)
        ):
            raise ValueError("Original source anchor integrity check failed")
        parts.append(page["structured_text"][left:right])
    if not parts:
        raise ValueError("Original source anchors are missing")
    return "\n\n".join(parts)


async def resolve_child(
    db: AsyncSession, child: PaperChunk, *, allow_parent_failure: bool = False
) -> dict:
    result = {
        "chunk_id": child.id,
        "paper_id": child.paper_id,
        "excerpt": child.text,
        "content_scope": child.content_scope,
        "page_start": child.page_start,
        "page_end": child.page_end,
        "index_version": child.index_version,
        "document_id": child.document_id,
        "parent_id": child.parent_id,
        "citation_url": f"/api/sources/chunks/{child.id}",
        "source_spans": [],
        "parent": None,
        "original_pdf_url": None,
        "document_version": None,
        "parser_version": None,
        "status": "legacy",
        "section": child.section,
    }
    if child.index_version == "legacy":
        return result
    document = await db.get(SourceDocument, child.document_id)
    generation = await db.get(SourceIndexGeneration, child.index_version)
    parent = await db.get(SourceParent, child.parent_id)
    if (
        document is None
        or generation is None
        or document.paper_id != child.paper_id
        or generation.document_id != document.id
        or generation.paper_id != child.paper_id
    ):
        raise ValueError("Citation provenance integrity check failed")
    validate_document(document)
    if (
        original_text(document, child.source_spans or []) != child.text
        or hashlib.sha256(child.text.encode()).hexdigest() != child.checksum
    ):
        raise ValueError("Original citation text integrity check failed")
    parent_valid = bool(
        parent is not None
        and parent.document_id == document.id
        and parent.generation_id == generation.index_version
    )
    if parent_valid:
        try:
            parent_valid = original_text(document, parent.raw_spans) == parent.raw_text
        except (KeyError, ValueError, TypeError):
            parent_valid = False
    if not parent_valid and not allow_parent_failure:
        raise ValueError("Parent source integrity check failed")
    result.update(
        {
            "source_spans": child.source_spans,
            "status": "active" if generation.is_active else "retired",
            "document_version": document.document_version,
            "parser_version": document.parser_version,
            "cleaner_version": generation.cleaner_version,
            "chunker_version": generation.chunker_version,
            "embedding_model_id": generation.embedding_model_id,
            "section_id": parent.section_id if parent_valid else None,
            "parent_status": "validated" if parent_valid else "unavailable",
            "source_origin": document.source_origin,
            "original_pdf_url": f"/api/sources/documents/{document.id}/pdf"
            if document.original_pdf is not None
            else None,
            "parent": {
                "id": parent.id,
                "raw_text": parent.raw_text,
                "source_spans": parent.raw_spans,
                "page_start": parent.page_start,
                "page_end": parent.page_end,
            }
            if parent_valid
            else None,
        }
    )
    flags = {
        block["block_id"]: block.get("flags", [])
        for block in generation.cleaning_manifest.get("blocks", [])
    }
    result["quality_flags"] = sorted(
        {
            flag
            for span in child.source_spans
            for flag in flags.get(span["block_id"], [])
        }
    )
    semantic_spans = [
        span
        for span in child.source_spans
        if "page_number_candidate" not in flags.get(span["block_id"], [])
    ]
    result["evidence_role"] = (
        "page_marker"
        if not semantic_spans
        else "bibliography"
        if all(
            "reference_section" in flags.get(span["block_id"], [])
            for span in semantic_spans
        )
        else "body"
    )
    return result
