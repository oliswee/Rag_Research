from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.models import Paper, PaperType
from app.routers.query import follow_up_query, submit_query
from app.schemas.query import FollowUpRequest, QueryRequest
from app.services.context_packing import pack_context
from app.services.session_manager import session_manager
from app.workflows.local_web import LocalRun


@pytest.fixture
def local_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))


@pytest.mark.asyncio
async def test_followup_inherits_privacy_and_explicit_false_resets(
    async_db_session, monkeypatch, local_endpoints
):
    seen = []

    async def local(db, policy, history, trace, **kwargs):
        seen.append(policy)
        return LocalRun(
            pack_context([], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    monkeypatch.setattr("app.routers.query.local_retrieval", local)
    first = await submit_query(
        QueryRequest(query="memory", local_only=True, year_from=2020, paper_ids=["a"]),
        async_db_session,
    )
    second = await follow_up_query(
        FollowUpRequest(query="latest memory", session_id=first["session_id"]),
        async_db_session,
    )
    assert (
        seen[-1].local_only
        and seen[-1].paper_ids == ("a",)
        and seen[-1].year_from == 2020
    )
    assert second["routing"]["action"] == "answer_local"
    await follow_up_query(
        FollowUpRequest(
            query="memory",
            session_id=first["session_id"],
            local_only=False,
            paper_ids=None,
            year_from=None,
            latest=False,
            intent=None,
        ),
        async_db_session,
    )
    assert (
        not seen[-1].local_only
        and seen[-1].paper_ids is None
        and seen[-1].year_from is None
    )
    scope = session_manager.get_session(first["session_id"]).query_scope
    assert "web_records" not in scope and "excerpt" not in str(scope)


@pytest.mark.asyncio
async def test_uploaded_scope_intersects_selected_papers(
    async_db_session, monkeypatch, local_endpoints
):
    uploaded = Paper(title="Uploaded", type=PaperType.research_paper)
    imported = Paper(
        title="Imported", type=PaperType.research_paper, import_source="openalex"
    )
    async_db_session.add_all([uploaded, imported])
    await async_db_session.flush()
    seen = []

    async def local(db, policy, history, trace, **kwargs):
        seen.append(policy)
        return LocalRun(
            pack_context([], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    monkeypatch.setattr("app.routers.query.local_retrieval", local)
    response = await submit_query(
        QueryRequest(
            query="latest papers", only_uploaded=True, paper_ids=[imported.id]
        ),
        async_db_session,
    )
    assert seen[0].paper_ids == ()
    assert response["routing"]["action"] == "answer_local"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question", ["只用上传的论文，查最新结果", "Only use local papers; latest results"]
)
async def test_literal_privacy_enforced_before_database(
    question, monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    db = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await submit_query(QueryRequest(query=question), db)
    assert exc.value.status_code == 400
    db.execute.assert_not_called()


def test_explicit_latest_false_resets_inherited_latest_intent():
    from app.routers.query import _effective_request

    session = session_manager.create_session()
    session_manager.set_query_scope(
        session.session_id, {"latest": True, "intent": "latest"}
    )
    effective = _effective_request(
        FollowUpRequest(
            query="memory accuracy", session_id=session.session_id, latest=False
        )
    )
    assert effective.latest is False and effective.intent == "factual"
