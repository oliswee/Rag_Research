"""Query API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.query import FollowUpRequest, QueryRequest, QueryResponse
from app.workflows import QueryInput, QueryWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("")
async def submit_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Submit a natural language query.

    Creates a new session if session_id is not provided.
    Returns synthesized answer with pattern citations.
    """
    workflow = QueryWorkflow(db=db)
    result = await workflow.run(
        QueryInput(
            query=request.query,
            session_id=request.session_id,
        )
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "QUERY_FAILED",
                "message": result.error or "Query processing failed",
            },
        )

    return QueryResponse(
        session_id=result.session_id,
        response=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        follow_up_suggestions=result.follow_up_suggestions,
    ).model_dump()


@router.post("/followup")
async def follow_up_query(
    request: FollowUpRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Submit a follow-up query in an existing session.

    Requires a valid session_id from a previous query.
    Uses conversation history for context.
    """
    workflow = QueryWorkflow(db=db)
    result = await workflow.run(
        QueryInput(
            query=request.query,
            session_id=request.session_id,
        )
    )

    if not result.success:
        # Check if it's a session error
        if result.error and "session" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": "Session expired or not found",
                },
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "QUERY_FAILED",
                "message": result.error or "Follow-up query failed",
            },
        )

    return QueryResponse(
        session_id=result.session_id,
        response=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        follow_up_suggestions=result.follow_up_suggestions,
    ).model_dump()
