"""Persisted, resumable-by-retry paper imports from public providers."""

import logging
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import async_session_maker
from app.models import (
    ImportItem,
    ImportJob,
    Paper,
    PaperChunk,
    PaperIdentifier,
    PaperStatus,
    PaperType,
)
from app.services.discovery import DiscoveryService
from app.services.fulltext_locations import FullTextResolver
from app.services.pdf_download import download_pdf
from app.services.pdf_parser import PDFParserService
from app.services.source_index import SourceVectorIndex, build_chunks

logger = logging.getLogger(__name__)


def identifiers(candidate: dict) -> list[tuple[str, str]]:
    values = [(candidate["provider"], candidate["provider_id"])]
    if candidate.get("doi"):
        values.append(("doi", candidate["doi"]))
    if candidate.get("arxiv_id"):
        values.append(("arxiv", candidate["arxiv_id"].lower()))
    return values


async def existing_paper(db: AsyncSession, candidate: dict) -> str | None:
    for kind, value in identifiers(candidate):
        found = (
            await db.execute(
                select(PaperIdentifier.paper_id).where(
                    PaperIdentifier.kind == kind, PaperIdentifier.value == value
                )
            )
        ).scalar_one_or_none()
        if found:
            return found
    return None


async def import_one(
    db: AsyncSession, item: ImportItem, discovery: DiscoveryService
) -> None:
    """Resolve source again on the server, then save original text before indexing."""
    item.status = "running"
    await db.commit()
    try:
        candidate = await discovery.get_candidate(item.provider, item.provider_id)
        duplicate = await existing_paper(db, candidate)
        if duplicate:
            item.paper_id, item.status = duplicate, "duplicate"
            await db.commit()
            return
        warnings = []
        pages: list[tuple[str, int | None]] = []
        scope = "metadata"
        content_url = candidate.get("source_url")
        if candidate.get("pdf_url") and candidate.get("open_access"):
            try:
                payload = await download_pdf(candidate["pdf_url"])
                parsed = PDFParserService(
                    max_pages=min(settings.max_pdf_pages, 200)
                ).parse(payload)
                pages = [(page.text, page.number) for page in parsed.pages]
                scope = "full_text"
                content_url = candidate["pdf_url"]
            except Exception as exc:
                warnings.append(f"Provider PDF unavailable: {type(exc).__name__}")
                logger.warning(
                    "Open PDF import failed for %s: %s",
                    item.provider_id,
                    type(exc).__name__,
                )
        if not pages:
            resolver = FullTextResolver()
            try:
                locations, lookup_warnings = await resolver.find(candidate)
                warnings.extend(lookup_warnings)
                for location in locations:
                    try:
                        if location.kind == "pdf":
                            payload = await download_pdf(location.url)
                            parsed = PDFParserService(
                                max_pages=min(settings.max_pdf_pages, 200)
                            ).parse(payload)
                            pages = [(page.text, page.number) for page in parsed.pages]
                        else:
                            pages = [(await resolver.fetch_xml(location), None)]
                        if not any(text.strip() for text, _ in pages):
                            raise ValueError("Open copy contains no extractable text")
                        scope = "full_text"
                        content_url = location.source_url
                        warnings.append(f"Full text retrieved via {location.source}")
                        break
                    except Exception as exc:
                        pages = []
                        warnings.append(
                            f"{location.source} copy unavailable: {type(exc).__name__}"
                        )
            except Exception as exc:
                warnings.append(f"Open-copy lookup unavailable: {type(exc).__name__}")
            finally:
                await resolver.close()
        if not pages and candidate.get("abstract"):
            pages = [(candidate["abstract"], None)]
            scope = "abstract"
        if not pages:
            warnings.append("No open full text or abstract available; metadata only")
        warning = "; ".join(warnings) or None
        paper = Paper(
            title=candidate["title"],
            authors=candidate["authors"],
            published_date=date(candidate["year"], 1, 1)
            if candidate.get("year")
            else None,
            source=candidate.get("source_url"),
            type=PaperType.research_paper,
            content_text="\n\n".join(content for content, _ in pages) or None,
            import_source=item.provider,
            imported_at=datetime.now(UTC),
            open_access=candidate.get("open_access"),
            content_scope=scope,
            status=PaperStatus.ready_for_review,
        )
        db.add(paper)
        await db.flush()
        for kind, value in identifiers(candidate):
            db.add(PaperIdentifier(paper_id=paper.id, kind=kind, value=value))
        chunks = []
        for data in build_chunks(
            pages,
            scope,
            content_url if scope == "full_text" else candidate.get("source_url"),
        ):
            chunk = PaperChunk(paper_id=paper.id, **data)
            db.add(chunk)
            chunks.append(chunk)
        item.paper_id, item.content_scope, item.warning, item.status = (
            paper.id,
            scope,
            warning,
            "done",
        )
        await db.commit()
        # Index is rebuildable: a failed external service never erases source content.
        if chunks and settings.openai_api_key:
            try:
                from app.services.embedding_service import EmbeddingService

                embeddings = EmbeddingService()
                index = SourceVectorIndex()
                vectors = await embeddings.generate_text_batch(
                    [chunk.text for chunk in chunks]
                )
                index.upsert_many(
                    [
                        (
                            chunk.id,
                            paper.id,
                            vector,
                            candidate.get("year"),
                            item.provider,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                )
            except Exception as exc:
                item.warning = (
                    (item.warning + "; ") if item.warning else ""
                ) + f"Vector indexing pending: {type(exc).__name__}"
                await db.commit()
        elif chunks:
            item.warning = (
                (item.warning + "; ") if item.warning else ""
            ) + "Vector indexing pending: OPENAI_API_KEY is not configured"
            await db.commit()
    except Exception as exc:
        await db.rollback()
        item = await db.get(ImportItem, item.id)
        item.status = "failed"
        item.error = f"Import failed: {type(exc).__name__}"
        await db.commit()
        logger.error("Import failed for %s (%s)", item.provider_id, type(exc).__name__)


async def process_job(job_id: str) -> None:
    async with async_session_maker() as db:
        job = await db.get(ImportJob, job_id)
        if job is None:
            return
        job.status = "running"
        await db.commit()
        discovery = DiscoveryService()
        try:
            items = (
                (
                    await db.execute(
                        select(ImportItem).where(ImportItem.job_id == job_id)
                    )
                )
                .scalars()
                .all()
            )
            for item in items:
                if item.status not in {"done", "duplicate"}:
                    await import_one(db, item, discovery)
        finally:
            await discovery.close()
        job.status = (
            "done"
            if all(item.status in {"done", "duplicate"} for item in items)
            else "partial"
        )
        job.completed_at = datetime.now(UTC)
        await db.commit()
