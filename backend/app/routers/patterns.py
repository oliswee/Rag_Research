"""Pattern management API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Paper, Pattern, PatternStatus
from app.schemas.pattern import (
    PatternResponse,
    PatternReviewRequest,
    PatternReviewResponse,
    PatternUpdateRequest,
)
from app.services import EmbeddingService, VectorStoreService
from app.workflows import ReviewWorkflow

logger = logging.getLogger(__name__)

# Router for paper-scoped pattern operations
paper_patterns_router = APIRouter(prefix="/api/papers", tags=["papers"])

# Router for direct pattern operations
patterns_router = APIRouter(prefix="/api/patterns", tags=["patterns"])


# --- Review Patterns Route ---


@paper_patterns_router.post("/{paper_id}/patterns")
async def review_patterns(
    paper_id: str,
    request: PatternReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Submit pattern review decisions (approve/reject).

    Approved patterns are embedded and stored in the vector store.
    Rejected patterns have their status updated only.
    """
    workflow = ReviewWorkflow(db=db)
    result = await workflow.run(paper_id, request.decisions)

    if not result.success:
        # Check if it's a "not found" error
        if any("not found" in err.lower() for err in result.errors):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": result.errors[0]},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "REVIEW_FAILED", "message": "; ".join(result.errors)},
        )

    # Get approved patterns for response
    approved_patterns = await _get_approved_patterns(db, paper_id)

    return PatternReviewResponse(
        saved_count=result.approved_count,
        rejected_count=result.rejected_count,
        patterns=approved_patterns,
    ).model_dump()


async def _get_approved_patterns(
    db: AsyncSession, paper_id: str
) -> list[PatternResponse]:
    """Get all approved patterns for a paper."""
    result = await db.execute(
        select(Pattern).where(
            Pattern.paper_id == paper_id,
            Pattern.status == PatternStatus.approved,
        )
    )
    patterns = result.scalars().all()

    return [
        PatternResponse(
            id=p.id,
            paper_id=p.paper_id,
            name=p.name,
            claim=p.claim,
            evidence=p.evidence,
            context=p.context,
            tags=p.tags or [],
            custom_fields=p.custom_fields,
            status=p.status,
            created_at=p.created_at.isoformat() if p.created_at else "",
            updated_at=p.updated_at.isoformat() if p.updated_at else "",
        )
        for p in patterns
    ]


# --- List Patterns Route ---


@paper_patterns_router.get("/{paper_id}/patterns")
async def list_patterns(
    paper_id: str,
    pattern_status: PatternStatus | None = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get all patterns for a paper with optional status filter."""
    # Verify paper exists
    paper = await db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Paper not found"},
        )

    # Build query
    query = select(Pattern).where(Pattern.paper_id == paper_id)

    if pattern_status:
        query = query.where(Pattern.status == pattern_status)

    query = query.order_by(Pattern.created_at.desc())

    result = await db.execute(query)
    patterns = result.scalars().all()

    pattern_responses = [
        PatternResponse(
            id=p.id,
            paper_id=p.paper_id,
            name=p.name,
            claim=p.claim,
            evidence=p.evidence,
            context=p.context,
            tags=p.tags or [],
            custom_fields=p.custom_fields,
            status=p.status,
            created_at=p.created_at.isoformat() if p.created_at else "",
            updated_at=p.updated_at.isoformat() if p.updated_at else "",
        ).model_dump()
        for p in patterns
    ]

    return {"patterns": pattern_responses}


# --- Get Pattern Route ---


@patterns_router.get("/{pattern_id}")
async def get_pattern(
    pattern_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a single pattern by ID with full details."""
    pattern = await db.get(Pattern, pattern_id)

    if not pattern:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Pattern not found"},
        )

    # Get the paper title
    paper = await db.get(Paper, pattern.paper_id)
    paper_title = paper.title if paper else "Unknown"

    return {
        "id": pattern.id,
        "paper_id": pattern.paper_id,
        "paper_title": paper_title,
        "name": pattern.name,
        "claim": pattern.claim,
        "evidence": pattern.evidence,
        "context": pattern.context,
        "tags": pattern.tags or [],
        "custom_fields": pattern.custom_fields,
        "status": pattern.status,
        "created_at": pattern.created_at.isoformat() if pattern.created_at else "",
        "updated_at": pattern.updated_at.isoformat() if pattern.updated_at else "",
    }


# --- Update Pattern Route ---


@patterns_router.put("/{pattern_id}")
async def update_pattern(
    pattern_id: str,
    request: PatternUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update an existing pattern.

    If content fields (claim, evidence, context) are changed,
    the pattern is re-embedded in the vector store.
    """
    pattern = await db.get(Pattern, pattern_id)

    if not pattern:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Pattern not found"},
        )

    # Track if content changed (for re-embedding)
    content_changed = False

    # Update fields if provided
    if request.name is not None:
        pattern.name = request.name
        content_changed = True
    if request.claim is not None:
        pattern.claim = request.claim
        content_changed = True
    if request.evidence is not None:
        pattern.evidence = request.evidence
        content_changed = True
    if request.context is not None:
        pattern.context = request.context
        content_changed = True
    if request.tags is not None:
        pattern.tags = request.tags

    # Re-embed if content changed and pattern is approved
    if content_changed and pattern.status == PatternStatus.approved:
        try:
            embedding_service = EmbeddingService()
            vector_store = VectorStoreService()

            embedding_text = embedding_service.prepare_embedding_text(
                {
                    "name": pattern.name,
                    "claim": pattern.claim,
                    "evidence": pattern.evidence,
                    "context": pattern.context,
                }
            )
            embedding_result = await embedding_service.generate_embedding(
                embedding_text
            )

            await vector_store.upsert(
                pattern_id=pattern.id,
                embedding=embedding_result.embedding,
                payload={
                    "paper_id": pattern.paper_id,
                    "name": pattern.name,
                    "claim": pattern.claim,
                    "evidence": pattern.evidence,
                    "context": pattern.context,
                    "tags": pattern.tags or [],
                },
            )
        except Exception as e:
            logger.warning(f"Failed to re-embed pattern {pattern_id}: {e}")
            # Don't fail the update - just log the warning

    await db.commit()
    await db.refresh(pattern)

    return PatternResponse(
        id=pattern.id,
        paper_id=pattern.paper_id,
        name=pattern.name,
        claim=pattern.claim,
        evidence=pattern.evidence,
        context=pattern.context,
        tags=pattern.tags or [],
        custom_fields=pattern.custom_fields,
        status=pattern.status,
        created_at=pattern.created_at.isoformat() if pattern.created_at else "",
        updated_at=pattern.updated_at.isoformat() if pattern.updated_at else "",
    ).model_dump()


# --- Delete Pattern Route ---


@patterns_router.delete("/{pattern_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pattern(
    pattern_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a pattern and remove it from the vector store."""
    pattern = await db.get(Pattern, pattern_id)

    if not pattern:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Pattern not found"},
        )

    # Delete from database
    await db.delete(pattern)
    await db.commit()

    # Delete from vector store
    try:
        vector_store = VectorStoreService()
        vector_store.delete_pattern(pattern_id)
    except Exception as e:
        logger.warning(f"Failed to delete pattern {pattern_id} from vector store: {e}")
