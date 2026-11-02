"""Verified matrix API with the same privacy and trace boundary as ordinary QA."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.routers.query import _capture_corpus_version, _enforce_local_only, _write_trace
from app.schemas.query import QueryRequest
from app.schemas.research import CompareRequest
from app.services.evidence_routing import parse_policy
from app.services.local_execution import local_only_request
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.comparison import DEFAULT_QUESTION, run_comparison

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/compare")
async def compare(request: CompareRequest, db: AsyncSession = Depends(get_db)) -> dict:
    policy = parse_policy(
        QueryRequest(
            query=request.question or DEFAULT_QUESTION,
            intent="compare",
            **request.model_dump(exclude={"question"}),
        )
    )
    trace = RetrievalTrace(
        policy.question,
        {
            **request.model_dump(),
            "local_only": policy.local_only,
            "only_uploaded": policy.only_uploaded,
            "requested_mode": "hybrid",
            "intent": policy.intent,
        },
        corpus_version="unversioned",
    )
    token = local_only_request.set(policy.local_only or policy.only_uploaded)
    try:
        _enforce_local_only(policy.local_only or policy.only_uploaded)
        await _capture_corpus_version(db, trace)
        return (await run_comparison(db, request, trace)).model_dump()
    except Exception as exc:
        trace.record_failure("comparison", exc)
        trace.finish("error", trace.record["prompt_evidence_ids"])
        raise
    finally:
        local_only_request.reset(token)
        _write_trace(trace)
