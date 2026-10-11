"""Paper management API routes."""

import logging
from datetime import date

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models import Paper, PaperChunk, PaperStatus, PaperType, Pattern
from app.schemas.paper import PaperMetadataInput, PaperResponse
from app.services import VectorStoreService
from app.services.source_index import SourceVectorIndex
from app.workflows import IngestionInput, IngestionWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/papers", tags=["papers"])


# --- Response Models ---


class UploadResponse:
    """Response model for paper upload."""

    def __init__(
        self,
        paper_id: str,
        metadata: dict,
        patterns: list[dict],
        extraction_warnings: list[str],
    ):
        self.paper_id = paper_id
        self.metadata = metadata
        self.patterns = patterns
        self.extraction_warnings = extraction_warnings


# --- Upload Route ---


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_paper(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    title: str | None = Form(None),
    authors: str | None = Form(None),
    source: str | None = Form(None),
    published_date: date | None = Form(None),
    type: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload a paper (PDF or text) for pattern extraction.

    Either `file` (PDF) or `text` must be provided.

    Returns the paper ID, extracted metadata, extracted patterns,
    and any warnings from the extraction process.
    """
    # Validate input
    if file is None and (text is None or not text.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": "No file or text provided"},
        )

    # Build user metadata if provided
    user_metadata = None
    if any([title, authors, source, published_date, type]):
        user_metadata = PaperMetadataInput(
            title=title,
            authors=authors.split(",") if authors else None,
            source=source,
            published_date=published_date,
            type=PaperType(type) if type else PaperType.research_paper,
        )

    # Build ingestion input
    try:
        if file is not None:
            # Check file size before reading into memory
            max_size_mb = settings.max_file_size_mb
            max_size_bytes = max_size_mb * 1024 * 1024
            if file.size and file.size > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "FILE_TOO_LARGE",
                        "message": f"File exceeds maximum size of {max_size_mb}MB",
                    },
                )
            pdf_bytes = await file.read()
            # Double-check size after read (file.size may be None for some uploads)
            if len(pdf_bytes) > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "FILE_TOO_LARGE",
                        "message": f"File exceeds maximum size of {max_size_mb}MB",
                    },
                )
            ingestion_input = IngestionInput(
                pdf_bytes=pdf_bytes,
                user_metadata=user_metadata,
            )
        else:
            ingestion_input = IngestionInput(
                text=text,
                user_metadata=user_metadata,
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    # Run ingestion workflow
    workflow = IngestionWorkflow(db=db)
    result = await workflow.run(ingestion_input)

    if not result.success:
        error_msg = result.error or "Unknown error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "EXTRACTION_FAILED", "message": error_msg},
        )

    # Get paper metadata for response
    paper = await db.get(Paper, result.paper_id)
    pub_date = paper.published_date if paper else None
    metadata = {
        "title": paper.title if paper else "Unknown",
        "authors": paper.authors if paper else [],
        "source": paper.source if paper else None,
        "published_date": str(pub_date) if pub_date else None,
        "type": paper.type.value if paper and paper.type else "research_paper",
    }

    return {
        "paper_id": result.paper_id,
        "metadata": metadata,
        "patterns": [p.model_dump() for p in result.patterns],
        "extraction_warnings": result.warnings,
    }


# --- List Route ---


@router.get("")
async def list_papers(
    status: PaperStatus | None = Query(None, alias="status"),
    paper_type: PaperType | None = Query(None, alias="type"),
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List papers with optional filters and pagination.

    Query parameters:
    - status: Filter by paper status
    - type: Filter by paper type
    - search: Search in title
    - limit: Max results (1-100, default 50)
    - offset: Pagination offset
    """
    # Clamp limit
    limit = max(1, min(limit, 100))

    # Build query
    query = select(Paper)

    if status:
        query = query.where(Paper.status == status)
    if paper_type:
        query = query.where(Paper.type == paper_type)
    if search:
        query = query.where(Paper.title.ilike(f"%{search}%"))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Apply pagination and get results
    query = query.order_by(Paper.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    papers = result.scalars().all()

    # Get pattern counts for each paper
    paper_ids = [p.id for p in papers]
    pattern_counts: dict[str, int] = {}

    if paper_ids:
        count_query = (
            select(Pattern.paper_id, func.count(Pattern.id))
            .where(Pattern.paper_id.in_(paper_ids))
            .group_by(Pattern.paper_id)
        )
        count_result = await db.execute(count_query)
        pattern_counts = dict(count_result.all())

    # Build response
    paper_responses = []
    for paper in papers:
        paper_responses.append(
            PaperResponse(
                id=paper.id,
                title=paper.title,
                authors=paper.authors or [],
                date=paper.date,
                source=paper.source,
                published_date=paper.published_date,
                type=paper.type or PaperType.research_paper,
                status=paper.status,
                pattern_count=pattern_counts.get(paper.id, 0),
                content_scope=paper.content_scope,
                import_source=paper.import_source,
                created_at=paper.created_at.isoformat() if paper.created_at else "",
                updated_at=paper.updated_at.isoformat() if paper.updated_at else "",
            ).model_dump()
        )

    return {
        "papers": paper_responses,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# --- Get Single Paper Route ---


@router.get("/{paper_id}")
async def get_paper(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a single paper by ID with full details."""
    paper = await db.get(Paper, paper_id)

    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Paper not found"},
        )

    # Get pattern count
    count_query = select(func.count(Pattern.id)).where(Pattern.paper_id == paper_id)
    pattern_count = (await db.execute(count_query)).scalar() or 0

    return PaperResponse(
        id=paper.id,
        title=paper.title,
        authors=paper.authors or [],
        date=paper.date,
        source=paper.source,
        published_date=paper.published_date,
        type=paper.type or PaperType.research_paper,
        status=paper.status,
        pattern_count=pattern_count,
        content_scope=paper.content_scope,
        import_source=paper.import_source,
        created_at=paper.created_at.isoformat() if paper.created_at else "",
        updated_at=paper.updated_at.isoformat() if paper.updated_at else "",
    ).model_dump()


# --- Delete Route ---


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a paper and all associated patterns.

    Also removes embeddings from vector store.
    """
    paper = await db.get(Paper, paper_id)

    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Paper not found"},
        )

    # Delete patterns from database (cascade should handle this, but be explicit)
    chunk_ids = (
        (await db.execute(select(PaperChunk.id).where(PaperChunk.paper_id == paper_id)))
        .scalars()
        .all()
    )
    pattern_query = select(Pattern).where(Pattern.paper_id == paper_id)
    patterns = (await db.execute(pattern_query)).scalars().all()

    for pattern in patterns:
        await db.delete(pattern)

    # Delete paper
    await db.delete(paper)
    await db.commit()

    # Delete from vector store
    try:
        vector_store = VectorStoreService()
        await vector_store.delete_by_paper_id(paper_id)
    except Exception as e:
        # Log but don't fail - paper is already deleted
        logger.warning(f"Failed to delete vectors for paper {paper_id}: {e}")
    try:
        SourceVectorIndex().delete(list(chunk_ids))
    except Exception as e:
        logger.warning("Failed to delete source vectors for paper %s: %s", paper_id, e)
