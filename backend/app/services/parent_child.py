"""Stage deterministic parent/child generations while retaining original sources."""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import (
    Paper,
    PaperChunk,
    SourceDocument,
    SourceIndexGeneration,
    SourceParent,
)
from app.services.pdf_parser import PDFParseResult
from app.services.pdf_structure import PDFBlock
from app.services.source_cleaning import CLEANER_VERSION, CleanedBlock, clean_document

CHUNKER_VERSION = "parent-child-characters-v1"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _id(*values: object) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, _hash(values)))


def configured_model_id() -> str:
    return f"{settings.embedding_provider}:" + (
        settings.fastembed_model
        if settings.embedding_provider == "fastembed"
        else "text-embedding-3-small"
    )


@dataclass
class GenerationBuild:
    document: SourceDocument
    generation: SourceIndexGeneration
    parents: list[SourceParent]
    children: list[PaperChunk]
    review_items: list[dict]


@dataclass
class _Piece:
    source: PDFBlock
    cleaned: CleanedBlock
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.cleaned.cleaned_text[self.start : self.end]


def _ranges(text: str, size: int):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(" ", start + size // 2, end),
                text.rfind("\n", start + size // 2, end),
            )
            if boundary >= 0:
                end = boundary + 1
        yield start, end
        start = end


def _validate(parsed: PDFParseResult) -> None:
    if not parsed.document_version.startswith("sha256:") or not parsed.source_id:
        raise ValueError("Source version and identity are required")
    pages, identifiers = set(), set()
    for page in parsed.pages:
        if page.number in pages:
            raise ValueError("Duplicate source page anchor")
        pages.add(page.number)
        for block in page.blocks:
            if (
                block.id in identifiers
                or block.page != page.number
                or block.document_version != parsed.document_version
                or block.source_id != parsed.source_id
                or not 0 <= block.raw_start < block.raw_end <= len(page.structured_text)
                or page.structured_text[block.raw_start : block.raw_end] != block.text
            ):
                raise ValueError("Invalid original source anchor")
            identifiers.add(block.id)
    if not identifiers:
        raise ValueError("Source has no supported text anchors")


def _span(piece: _Piece, start: int, end: int, document_id: str, paper_id: str) -> dict:
    left, right = piece.cleaned.raw_span(start, end)
    return {
        "paper_id": paper_id,
        "document_id": document_id,
        "document_version": piece.source.document_version,
        "page": piece.source.page,
        "raw_start": left,
        "raw_end": right,
        "block_id": piece.source.id,
        "bbox": piece.source.bbox,
    }


def _page_range(spans: list[dict]) -> tuple[int | None, int | None]:
    pages = [item["page"] for item in spans if item["page"] is not None]
    return (min(pages), max(pages)) if pages else (None, None)


async def stage_generation(
    db: AsyncSession,
    paper: Paper,
    parsed: PDFParseResult,
    *,
    child_size: int = 1200,
    parent_size: int = 4800,
    embedding_model_id: str | None = None,
    source_url: str | None = None,
    source_bytes: bytes | None = None,
) -> GenerationBuild:
    """Stage source rows and activate within the caller's SQLite transaction.

    This function never commits. Publication to Qdrant is a separate recoverable
    operation; pending vectors must not prevent source/keyword availability.
    """
    if not 32 <= child_size <= parent_size:
        raise ValueError("Require 32 <= child_size <= parent_size")
    _validate(parsed)
    from app.services.source_index import ensure_fts

    await ensure_fts(db)
    if (
        source_bytes is not None
        and "sha256:" + hashlib.sha256(source_bytes).hexdigest()
        != parsed.document_version
    ):
        raise ValueError("Original PDF bytes do not match source version")
    cleaned = clean_document(parsed)
    raw_pages = json.loads(json.dumps([asdict(page) for page in parsed.pages]))
    source_url = source_url or paper.source
    snapshot = _hash(
        [parsed.document_version, parsed.parser_version, raw_pages, source_url]
    )
    document_id = _id(paper.id, snapshot)
    manifest = asdict(cleaned)
    model_id = embedding_model_id or configured_model_id()
    index_version = "sha256:" + _hash(
        [
            document_id,
            CLEANER_VERSION,
            CHUNKER_VERSION,
            child_size,
            parent_size,
            model_id,
            manifest,
        ]
    )
    document = await db.get(SourceDocument, document_id)
    if document is None:
        document = SourceDocument(
            id=document_id,
            paper_id=paper.id,
            document_version=parsed.document_version,
            parser_version=parsed.parser_version,
            snapshot_digest="sha256:" + snapshot,
            raw_pages=raw_pages,
            content_scope=parsed.content_scope,
            source_origin=next(
                block.source_origin for page in parsed.pages for block in page.blocks
            ),
            source_url=source_url,
            original_pdf=source_bytes,
        )
        db.add(document)
        await db.flush()
    elif (
        document.raw_pages != raw_pages
        or document.snapshot_digest != "sha256:" + snapshot
        or document.document_version != parsed.document_version
        or document.parser_version != parsed.parser_version
        or document.source_url != source_url
        or document.content_scope != parsed.content_scope
    ):
        raise ValueError("Stored source snapshot failed immutable anchor validation")
    elif source_bytes is not None and document.original_pdf is None:
        document.original_pdf = source_bytes
    generation = await db.get(SourceIndexGeneration, index_version)
    existing = generation is not None
    if existing:
        if (
            generation.document_id != document.id
            or generation.paper_id != paper.id
            or _hash(generation.cleaning_manifest) != _hash(manifest)
            or generation.cleaner_version != CLEANER_VERSION
            or generation.chunker_version != CHUNKER_VERSION
            or generation.embedding_model_id != model_id
            or generation.child_size != child_size
            or generation.parent_size != parent_size
        ):
            raise ValueError("Stored index generation failed immutable validation")
    else:
        generation = SourceIndexGeneration(
            index_version=index_version,
            paper_id=paper.id,
            document_id=document.id,
            cleaner_version=CLEANER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model_id=model_id,
            child_size=child_size,
            parent_size=parent_size,
            is_active=False,
            vector_status="pending",
            cleaning_manifest=manifest,
        )
        db.add(generation)
        await db.flush()
    cleaned_by_id = {block.block_id: block for block in cleaned.blocks}
    groups, current = [], []
    length = 0
    for page in parsed.pages:
        for block in page.blocks:
            cleaned_block = cleaned_by_id[block.id]
            atomic = block.kind in {"table", "formula"}
            ranges = (
                [(0, len(cleaned_block.cleaned_text))]
                if atomic
                else _ranges(cleaned_block.cleaned_text, parent_size)
            )
            for start, end in ranges:
                piece = _Piece(block, cleaned_block, start, end)
                if current and (
                    atomic
                    or current[-1].source.kind in {"table", "formula"}
                    or current[-1].source.section_id != block.section_id
                    or length + 2 + len(piece.text) > parent_size
                ):
                    groups.append(current)
                    current, length = [], 0
                current.append(piece)
                length += len(piece.text) + (2 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    parents, children = [], []
    pages_by_number = {page.number: page for page in parsed.pages}
    for parent_index, pieces in enumerate(groups):
        parent_text, positions, spans = "", [], []
        for piece in pieces:
            if parent_text:
                parent_text += "\n\n"
            start = len(parent_text)
            parent_text += piece.text
            positions.append((start, len(parent_text), piece))
            spans.append(_span(piece, piece.start, piece.end, document_id, paper.id))
        raw_text = "\n\n".join(
            pages_by_number[span["page"]].structured_text[
                span["raw_start"] : span["raw_end"]
            ]
            for span in spans
        )
        first_page, last_page = _page_range(spans)
        parent = SourceParent(
            id=_id(index_version, parent_index, spans),
            generation_id=index_version,
            document_id=document.id,
            section_id=pieces[0].source.section_id,
            parent_index=parent_index,
            raw_text=raw_text,
            cleaned_text=parent_text,
            raw_spans=spans,
            page_start=first_page,
            page_end=last_page,
        )
        if not existing:
            db.add(parent)
            await db.flush()
        parents.append(parent)
        for start, end in _ranges(parent_text, child_size):
            if not parent_text[start:end].strip():
                continue
            child_spans = [
                _span(
                    piece,
                    piece.start + max(start, left) - left,
                    piece.start + min(end, right) - left,
                    document_id,
                    paper.id,
                )
                for left, right, piece in positions
                if max(start, left) < min(end, right)
            ]
            raw = "\n\n".join(
                pages_by_number[span["page"]].structured_text[
                    span["raw_start"] : span["raw_end"]
                ]
                for span in child_spans
            )
            first_page, last_page = _page_range(child_spans)
            child = PaperChunk(
                id=_id(index_version, parent.id, start, end),
                paper_id=paper.id,
                document_id=document.id,
                parent_id=parent.id,
                index_version=index_version,
                chunk_index=len(children),
                text=raw,
                cleaned_text=parent_text[start:end],
                content_scope=parsed.content_scope,
                section=parent.section_id,
                source_spans=child_spans,
                page_start=first_page,
                page_end=last_page,
                source_url=source_url,
                start_char=child_spans[0]["raw_start"]
                if len(child_spans) == 1
                else None,
                end_char=child_spans[0]["raw_end"] if len(child_spans) == 1 else None,
                checksum=hashlib.sha256(raw.encode()).hexdigest(),
            )
            if not existing:
                db.add(child)
            children.append(child)
    if existing:
        stored_parents = list(
            (
                await db.scalars(
                    select(SourceParent)
                    .where(SourceParent.generation_id == index_version)
                    .order_by(SourceParent.parent_index)
                )
            ).all()
        )
        stored_children = list(
            (
                await db.scalars(
                    select(PaperChunk)
                    .where(PaperChunk.index_version == index_version)
                    .order_by(PaperChunk.chunk_index)
                )
            ).all()
        )
        for expected, stored, omitted in (
            (parents, stored_parents, set()),
            (children, stored_children, {"created_at", "search_text"}),
        ):
            fields = (
                [
                    column.name
                    for column in expected[0].__table__.columns
                    if column.name not in omitted
                ]
                if expected
                else []
            )
            if len(expected) != len(stored) or any(
                _hash([getattr(wanted, key) for key in fields])
                != _hash([getattr(actual, key) for key in fields])
                for wanted, actual in zip(expected, stored, strict=True)
            ):
                raise ValueError("Stored source rows failed immutable validation")
        parents, children = stored_parents, stored_children
    else:
        await db.flush()
    if not generation.is_active:
        await db.execute(
            update(SourceIndexGeneration)
            .where(SourceIndexGeneration.paper_id == paper.id)
            .values(is_active=False)
        )
        generation.is_active = True
        generation.vector_status = "pending"
        generation.physical_revision = None
        generation.index_manifest = None
        generation.verified_at = None
        generation.error_type = None
    await db.flush()
    return GenerationBuild(
        document, generation, parents, children, cleaned.review_items
    )
