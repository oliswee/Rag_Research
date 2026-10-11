"""Query API routes."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Paper
from app.schemas.query import FollowUpRequest, QueryRequest, QueryResponse
from app.schemas.session import ConversationTurn, SourceReference
from app.services.session_manager import session_manager
from app.workflows import QueryInput, QueryWorkflow
from app.workflows.source_query import answer_from_evidence, has_source_chunks, retrieve

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
    has_online_imports = (
        await db.execute(
            select(Paper.id).where(Paper.import_source.is_not(None)).limit(1)
        )
    ).scalar_one_or_none() is not None
    if (
        "retrieval_mode" in request.model_fields_set
        or request.paper_ids is not None
        or (has_online_imports and await has_source_chunks(db))
    ):
        session = (
            session_manager.get_session(request.session_id)
            if request.session_id
            else None
        )
        if request.session_id and session is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": "Session expired or not found",
                },
            )
        session = session or session_manager.create_session()
        previous = next(
            (turn.content for turn in reversed(session.history) if turn.role == "user"),
            "",
        )
        retrieval_query = f"{previous} {request.query}" if previous else request.query
        evidence, mode = await retrieve(
            db,
            retrieval_query,
            request.paper_ids,
            request.retrieval_mode,
            year_from=request.year_from,
            year_to=request.year_to,
            provider=request.provider,
        )
        answer, cited = await answer_from_evidence(request.query, evidence)
        now = datetime.now(UTC)
        session_manager.update_session(
            session.session_id,
            ConversationTurn(role="user", content=request.query, timestamp=now),
        )
        session_manager.update_session(
            session.session_id,
            ConversationTurn(
                role="assistant",
                content=answer,
                timestamp=now,
                sources=[
                    SourceReference(
                        pattern_id=source["chunk_id"], summary=source["title"]
                    )
                    for source in cited
                ],
            ),
        )
        return QueryResponse(
            session_id=session.session_id,
            response=answer,
            sources=[],
            confidence="medium" if cited else "low",
            evidence=cited,
            retrieval_mode=mode,
        ).model_dump()
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
    has_online_imports = (
        await db.execute(
            select(Paper.id).where(Paper.import_source.is_not(None)).limit(1)
        )
    ).scalar_one_or_none() is not None
    if request.retrieval_mode or (has_online_imports and await has_source_chunks(db)):
        return await submit_query(
            QueryRequest(
                query=request.query,
                session_id=request.session_id,
                paper_ids=request.paper_ids,
                retrieval_mode=request.retrieval_mode or "dense",
                year_from=request.year_from,
                year_to=request.year_to,
                provider=request.provider,
            ),
            db,
        )
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
