"""Query API routes."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models import Paper
from app.schemas.query import FollowUpRequest, QueryRequest, QueryResponse
from app.schemas.session import ConversationTurn, SourceReference
from app.services.llm_client import llm_available
from app.services.local_execution import local_only_request, loopback_host, loopback_url
from app.services.retrieval_trace import RetrievalTrace
from app.services.session_manager import session_manager
from app.services.trace_versions import corpus_version
from app.workflows import QueryInput, QueryWorkflow
from app.workflows.query_rewrite import rewrite_for_retrieval
from app.workflows.source_query import answer_from_evidence, has_source_chunks, retrieve

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


def _enforce_local_only(local_only: bool) -> None:
    if local_only and (
        settings.embedding_provider != "fastembed"
        or settings.llm_provider != "ollama"
        or not loopback_url(settings.ollama_base_url)
        or not loopback_host(settings.qdrant_host)
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "LOCAL_ONLY_REMOTE_PROVIDER",
                "message": (
                    "local_only requires FastEmbed and loopback "
                    "Ollama/Qdrant endpoints."
                ),
            },
        )


def _write_trace(trace: RetrievalTrace) -> None:
    try:
        trace.append(Path(settings.retrieval_trace_path))
    except OSError as exc:
        logger.warning("Retrieval trace unavailable: %s", type(exc).__name__)


def _record_terminal_failure(trace: RetrievalTrace, stage: str, exc: Exception) -> None:
    previous = trace.record["failures"][-1:]
    already_recorded = (
        previous
        and previous[0]["stage"]
        in {
            "query",
            "legacy_workflow",
        }
        and previous[0]["error_type"] == type(exc).__name__
    )
    if not already_recorded:
        trace.record_failure(stage, exc)


async def _capture_corpus_version(db: AsyncSession, trace: RetrievalTrace) -> None:
    if trace.record["corpus_version"] != "unversioned":
        return
    started = time.perf_counter()
    try:
        trace.record["corpus_version"] = await corpus_version(db)
    except Exception as exc:
        trace.record["corpus_version"] = "unavailable"
        trace.record_failure("corpus_snapshot", exc)
        raise
    finally:
        trace.record["stage_ms"]["corpus_snapshot"] = round(
            (time.perf_counter() - started) * 1000, 3
        )


@router.post("")
async def submit_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Submit a natural language query.

    Creates a new session if session_id is not provided.
    Returns synthesized answer with pattern citations.
    """
    trace = RetrievalTrace(
        request.query,
        {
            "paper_ids": request.paper_ids,
            "year_from": request.year_from,
            "year_to": request.year_to,
            "provider": request.provider,
            "requested_mode": request.retrieval_mode,
            "local_only": request.local_only,
            "latest": request.latest,
        },
        corpus_version="unversioned",
    )
    local_token = local_only_request.set(request.local_only)
    try:
        return await _submit_query_impl(request, db, trace)
    except Exception as exc:
        _record_terminal_failure(trace, "query", exc)
        trace.finish("error", trace.record["prompt_evidence_ids"])
        raise
    finally:
        local_only_request.reset(local_token)
        _write_trace(trace)


async def _submit_query_impl(
    request: QueryRequest, db: AsyncSession, trace: RetrievalTrace
) -> dict:
    _enforce_local_only(request.local_only)
    has_online_imports = (
        await db.execute(
            select(Paper.id).where(Paper.import_source.is_not(None)).limit(1)
        )
    ).scalar_one_or_none() is not None
    await _capture_corpus_version(db, trace)
    if (
        "retrieval_mode" in request.model_fields_set
        or request.paper_ids is not None
        or (request.rewrite_query and await has_source_chunks(db))
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
        prompt_ids: list[str] = []
        try:
            rewrite_start = time.perf_counter()
            retrieval_query, rewrite_status = await rewrite_for_retrieval(
                request.query, session.history, request.rewrite_query, trace=trace
            )
            trace.record_query(retrieval_query, rewrite_status)
            trace.record["stage_ms"]["rewrite"] = round(
                (time.perf_counter() - rewrite_start) * 1000, 3
            )
            evidence, mode = await retrieve(
                db,
                retrieval_query,
                request.paper_ids,
                request.retrieval_mode,
                year_from=request.year_from,
                year_to=request.year_to,
                provider=request.provider,
                trace=trace,
            )
            prompt_ids = (
                [item["chunk_id"] for item in evidence] if llm_available() else []
            )
            answer_start = time.perf_counter()
            answer, cited = await answer_from_evidence(request.query, evidence, trace)
            trace.record["stage_ms"]["answer"] = round(
                (time.perf_counter() - answer_start) * 1000, 3
            )
            support_status = (
                "supported"
                if cited and llm_available()
                else "evidence_only"
                if cited
                else "abstain"
            )
            trace.finish(
                support_status,
                prompt_ids,
                [item["chunk_id"] for item in cited],
            )
        except Exception as exc:
            trace.record_failure("query", exc)
            trace.finish("error", prompt_ids)
            raise
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
            run_id=trace.run_id,
            response=answer,
            sources=[],
            confidence="medium" if cited else "low",
            evidence=cited,
            retrieval_mode=mode,
            retrieval_query=retrieval_query,
            rewrite_status=rewrite_status,
        ).model_dump()
    trace.record_query(request.query, "legacy")
    trace.record_index("legacy_pattern_unavailable")
    workflow_start = time.perf_counter()
    try:
        workflow = QueryWorkflow(db=db)
        result = await workflow.run(
            QueryInput(
                query=request.query,
                session_id=request.session_id,
                trace=trace,
            )
        )
        trace.record["stage_ms"]["legacy_workflow"] = round(
            (time.perf_counter() - workflow_start) * 1000, 3
        )
        support_status = (
            "error"
            if not result.success
            else ("cited_unverified" if result.sources else "abstain")
        )
        trace.finish(
            support_status,
            trace.record["prompt_evidence_ids"],
            [source.pattern_id for source in result.sources],
        )
    except Exception as exc:
        trace.record_failure("legacy_workflow", exc)
        trace.finish("error", [])
        raise
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
        run_id=trace.run_id,
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
    trace = RetrievalTrace(
        request.query,
        {
            "paper_ids": request.paper_ids,
            "year_from": request.year_from,
            "year_to": request.year_to,
            "provider": request.provider,
            "requested_mode": request.retrieval_mode,
            "local_only": request.local_only,
            "latest": request.latest,
        },
        corpus_version="unversioned",
    )
    local_token = local_only_request.set(request.local_only)
    try:
        return await _follow_up_query_impl(request, db, trace)
    except Exception as exc:
        _record_terminal_failure(trace, "followup", exc)
        trace.finish("error", trace.record["prompt_evidence_ids"])
        raise
    finally:
        local_only_request.reset(local_token)
        _write_trace(trace)


async def _follow_up_query_impl(
    request: FollowUpRequest, db: AsyncSession, trace: RetrievalTrace
) -> dict:
    _enforce_local_only(request.local_only)
    has_online_imports = (
        await db.execute(
            select(Paper.id).where(Paper.import_source.is_not(None)).limit(1)
        )
    ).scalar_one_or_none() is not None
    await _capture_corpus_version(db, trace)
    if (
        request.retrieval_mode
        or (request.rewrite_query and await has_source_chunks(db))
        or (has_online_imports and await has_source_chunks(db))
    ):
        return await _submit_query_impl(
            QueryRequest(
                query=request.query,
                session_id=request.session_id,
                paper_ids=request.paper_ids,
                retrieval_mode=request.retrieval_mode or "dense",
                rewrite_query=request.rewrite_query,
                local_only=request.local_only,
                latest=request.latest,
                year_from=request.year_from,
                year_to=request.year_to,
                provider=request.provider,
            ),
            db,
            trace,
        )
    trace.record_query(request.query, "legacy")
    trace.record_index("legacy_pattern_unavailable")
    workflow_start = time.perf_counter()
    try:
        workflow = QueryWorkflow(db=db)
        result = await workflow.run(
            QueryInput(
                query=request.query,
                session_id=request.session_id,
                trace=trace,
            )
        )
        trace.record["stage_ms"]["legacy_workflow"] = round(
            (time.perf_counter() - workflow_start) * 1000, 3
        )
        support_status = (
            "error"
            if not result.success
            else ("cited_unverified" if result.sources else "abstain")
        )
        trace.finish(
            support_status,
            trace.record["prompt_evidence_ids"],
            [source.pattern_id for source in result.sources],
        )
    except Exception as exc:
        trace.record_failure("legacy_workflow", exc)
        trace.finish("error", [])
        raise
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
        run_id=trace.run_id,
        response=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        follow_up_suggestions=result.follow_up_suggestions,
    ).model_dump()
