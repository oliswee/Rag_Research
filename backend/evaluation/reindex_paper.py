"""Rebuild one paper without deleting source history or the shared vector collection.

Run from backend/: uv run python -m evaluation.reindex_paper --paper-id UUID
Use --no-vectors for source/keyword repair while Qdrant is unavailable.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_maker, engine
from app.models import Paper, PaperChunk, SourceDocument, SourceIndexGeneration
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFPage, PDFParseResult, PDFParserService
from app.services.pdf_structure import PDFBlock
from app.services.source_anchors import validate_document
from app.services.versioned_vectors import publish_generation


async def reindex_paper(
    db: AsyncSession,
    paper_id: str,
    *,
    pdf_bytes: bytes | None = None,
    child_size: int | None = None,
    parent_size: int | None = None,
    publish: bool = True,
) -> dict:
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise ValueError("Paper not found")
    active = (
        await db.scalars(
            select(SourceIndexGeneration).where(
                SourceIndexGeneration.paper_id == paper_id,
                SourceIndexGeneration.is_active.is_(True),
            )
        )
    ).one_or_none()
    document = await db.get(SourceDocument, active.document_id) if active else None
    source_url = document.source_url if document else paper.source
    parser = PDFParserService()
    if pdf_bytes is not None:
        parsed = parser.parse(pdf_bytes)
    elif document is not None:
        validate_document(document)
        pdf_bytes = document.original_pdf
        if pdf_bytes is not None:
            parsed = parser.parse(pdf_bytes, source_origin=document.source_origin)
        else:
            pages = [
                PDFPage(
                    **{
                        **page,
                        "blocks": [PDFBlock(**block) for block in page["blocks"]],
                    }
                )
                for page in document.raw_pages
            ]
            parsed = PDFParseResult(
                text="\n\n".join(page.text for page in pages),
                page_count=len([page for page in pages if page.number is not None]),
                metadata={},
                pages=pages,
                document_version=document.document_version,
                source_id=next(
                    block.source_id for page in pages for block in page.blocks
                ),
                parser_version=document.parser_version,
                content_scope=document.content_scope,
            )
    else:
        content = paper.content_text
        if not content or not content.strip():
            chunks = list(
                (
                    await db.scalars(
                        select(PaperChunk)
                        .where(
                            PaperChunk.paper_id == paper_id,
                            PaperChunk.index_version == "legacy",
                        )
                        .order_by(PaperChunk.chunk_index)
                    )
                ).all()
            )
            content = "\n\n".join(chunk.text for chunk in chunks)
        if not content or not content.strip():
            raise ValueError("Paper has no source text to reindex")
        parsed = parser.parse_text(
            content,
            content_scope=paper.content_scope or "full_text",
            source_origin="legacy_extraction",
        )
    build = await stage_generation(
        db,
        paper,
        parsed,
        child_size=child_size
        if child_size is not None
        else (active.child_size if active else 1200),
        parent_size=parent_size
        if parent_size is not None
        else (active.parent_size if active else 4800),
        source_url=source_url,
        source_bytes=pdf_bytes,
    )
    await db.commit()
    error_type = None
    if publish:
        try:
            await publish_generation(db, build)
        except Exception as exc:
            error_type = type(exc).__name__
        await db.commit()
    return {
        "paper_id": paper.id,
        "document_id": build.document.id,
        "document_version": build.document.document_version,
        "index_version": build.generation.index_version,
        "source_origin": build.document.source_origin,
        "parent_count": len(build.parents),
        "child_count": len(build.children),
        "pdf_pages_available": any(page.number is not None for page in parsed.pages),
        "original_pdf_available": build.document.original_pdf is not None,
        "vector_status": build.generation.vector_status,
        "physical_revision": build.generation.physical_revision,
        "error_type": error_type,
    }


async def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--paper-id", required=True)
    arguments.add_argument("--pdf", type=Path)
    arguments.add_argument("--child-size", type=int)
    arguments.add_argument("--parent-size", type=int)
    arguments.add_argument("--no-vectors", action="store_true")
    args = arguments.parse_args()
    try:
        async with async_session_maker() as db:
            result = await reindex_paper(
                db,
                args.paper_id,
                pdf_bytes=args.pdf.read_bytes() if args.pdf else None,
                child_size=args.child_size,
                parent_size=args.parent_size,
                publish=not args.no_vectors,
            )
        print(json.dumps(result))
        if result["error_type"]:
            raise SystemExit(2)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
