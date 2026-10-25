"""Local-first evidence query routes with immutable inherited scope."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models import Paper
from app.schemas.query import FollowUpRequest, QueryRequest, QueryResponse
from app.schemas.session import ConversationTurn, SourceReference
from app.services.evidence_routing import parse_policy
from app.services.llm_client import llm_available
from app.services.local_execution import local_only_request, loopback_host, loopback_url
from app.services.retrieval_trace import RetrievalTrace
from app.services.session_manager import session_manager
from app.services.trace_versions import corpus_version
from app.workflows.local_web import local_retrieval, run_local_web
from app.workflows.source_query import answer_from_evidence, retrieve

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/query", tags=["query"])
_SCOPE_FIELDS = (
    "paper_ids",
    "retrieval_mode",
    "rewrite_query",
    "local_only",
    "only_uploaded",
    "latest",
    "intent",
    "year_from",
    "year_to",
    "provider",
)


def _enforce_local_only(local_only):
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


def _write_trace(trace):
    try:
        trace.append(Path(settings.retrieval_trace_path))
    except OSError as exc:
        logger.warning("Retrieval trace unavailable: %s", type(exc).__name__)


def _effective_request(request):
    """Omitted constraints inherit; explicit null/false resets are intentional."""
    data = request.model_dump(exclude_unset=True)
    if data.get("latest") is False and "intent" not in data:
        data["intent"] = None
    session = (
        session_manager.get_session(request.session_id) if request.session_id else None
    )
    if session is not None:
        for key, value in session.query_scope.items():
            data.setdefault(key, value)
    if data.get("retrieval_mode") is None:
        data["retrieval_mode"] = "hybrid"
    effective = QueryRequest(**data)
    policy = parse_policy(effective)
    return effective.model_copy(
        update={
            "local_only": policy.local_only,
            "only_uploaded": policy.only_uploaded,
            "intent": policy.intent,
        }
    )


async def _capture_corpus_version(db, trace):
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
    request: QueryRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    request = _effective_request(request)
    trace = RetrievalTrace(
        request.query,
        {**request.model_dump(), "requested_mode": request.retrieval_mode},
        corpus_version="unversioned",
    )
    token = local_only_request.set(request.local_only or request.only_uploaded)
    try:
        return await _submit_query_impl(request, db, trace)
    except Exception as exc:
        if not trace.record["failures"] or trace.record["failures"][-1] != {
            "stage": "query",
            "error_type": type(exc).__name__,
        }:
            trace.record_failure("query", exc)
        trace.finish("error", trace.record["prompt_evidence_ids"])
        raise
    finally:
        local_only_request.reset(token)
        _write_trace(trace)


async def _submit_query_impl(request, db, trace):
    _enforce_local_only(request.local_only or request.only_uploaded)
    await _capture_corpus_version(db, trace)
    session = (
        session_manager.get_session(request.session_id) if request.session_id else None
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
    policy = parse_policy(request)

    async def local_runner(p, history, t, *, correction=False):
        return await local_retrieval(
            db, p, history, t, correction=correction, retriever=retrieve
        )

    result = await run_local_web(
        db, policy, session.history, trace, local_runner=local_runner
    )
    packed = result.packed
    started = time.perf_counter()
    answer, cited = await answer_from_evidence(
        request.query,
        packed.sources,
        trace,
        **({"packed_context": packed} if request.retrieval_mode == "hybrid" else {}),
    )
    if request.paper_ids:
        missing = [
            p
            for p in request.paper_ids
            if packed.coverage.get(p, {}).get("context_status") != "included"
        ]
        if missing:
            titles = dict(
                (
                    await db.execute(
                        select(Paper.id, Paper.title).where(Paper.id.in_(missing))
                    )
                ).all()
            )
            answer += (
                "\n\nNo usable context for: "
                + "; ".join(
                    titles.get(p, "Unknown selected paper")[:200] for p in missing
                )
                + ". Results for these papers remain unknown."
            )
    trace.record["stage_ms"]["answer"] = round(
        (time.perf_counter() - started) * 1000, 3
    )
    trace.finish(
        "supported"
        if cited and llm_available()
        else "evidence_only"
        if cited
        else "abstain",
        trace.record["prompt_evidence_ids"],
        [s["chunk_id"] for s in cited],
    )
    now = datetime.now(UTC)
    session_manager.set_query_scope(
        session.session_id, {k: getattr(request, k) for k in _SCOPE_FIELDS}
    )
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
                SourceReference(pattern_id=s["chunk_id"], summary=s["title"])
                for s in cited
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
        context_coverage=packed.coverage,
        retrieval_mode=result.local.mode,
        retrieval_query=request.query,
        rewrite_status=result.local.rewrite_status,
        query_count=result.local.query_count,
        retrieval_status=result.local.retrieval_status,
        routing={
            "intent": policy.intent,
            "action": result.decision.action,
            "sufficiency": result.assessment.state,
            "reasons": list(result.decision.reasons),
            "local_retry_count": trace.record["local_retry_count"],
            "local_only": policy.local_only,
            "only_uploaded": policy.only_uploaded,
        },
        web_records=result.web_records,
        notices=result.notices,
    ).model_dump()


@router.post("/followup")
async def follow_up_query(
    request: FollowUpRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    return await submit_query(_effective_request(request), db)
