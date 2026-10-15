import json

import pytest
from fastapi import HTTPException

from app.config import settings
from app.models import Paper, PaperChunk, PaperStatus, PaperType
from app.routers.query import follow_up_query, submit_query
from app.schemas.query import FollowUpRequest, QueryRequest


@pytest.mark.asyncio
async def test_source_query_and_followup_use_local_evidence(
    async_db_session, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(trace_path))
    paper = Paper(
        title="Source",
        authors=[],
        type=PaperType.research_paper,
        status=PaperStatus.ready_for_review,
        import_source="openalex",
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    async_db_session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text="Active retrieval improves answers.",
            content_scope="abstract",
            checksum="x",
        )
    )
    await async_db_session.commit()
    first = await submit_query(
        QueryRequest(query="active retrieval", retrieval_mode="dense"), async_db_session
    )
    assert first["retrieval_mode"] == "bm25_fallback"
    assert first["evidence"][0]["paper_id"] == paper.id
    assert first["evidence"][0]["page_start"] is None
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert trace["run_id"] == first["run_id"]
    assert trace["retrieval_mode"] == "bm25_fallback"
    assert trace["rankings"]["bm25"][0]["child_id"] == first["evidence"][0]["chunk_id"]
    assert trace["prompt_evidence_ids"] == []
    assert trace["returned_evidence_ids"] == [first["evidence"][0]["chunk_id"]]
    assert trace["web_decision"] == "not_implemented"
    second = await follow_up_query(
        FollowUpRequest(query="What improves?", session_id=first["session_id"]),
        async_db_session,
    )
    assert second["session_id"] == first["session_id"]
    assert second["evidence"]


@pytest.mark.asyncio
async def test_query_uses_rewritten_search_but_answers_original_question(
    async_db_session, monkeypatch
):
    async def fake_rewrite(question, history, enabled):
        assert enabled
        assert question == "What about its latency?"
        return "active retrieval latency", "applied"

    async def fake_retrieve(db, query, *args, **kwargs):
        if query != "active retrieval latency":
            return [], "bm25"
        return [
            {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "title": "Latency study",
                "source_url": None,
                "content_scope": "abstract",
                "excerpt": "Active retrieval reduced latency.",
                "page_start": None,
                "page_end": None,
            }
        ], "bm25"

    async def fake_answer(question, evidence):
        return f"Answer to {question}: {evidence[0]['excerpt']}", evidence

    monkeypatch.setattr("app.routers.query.rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr("app.routers.query.retrieve", fake_retrieve)
    monkeypatch.setattr("app.routers.query.answer_from_evidence", fake_answer)

    response = await submit_query(
        QueryRequest(
            query="What about its latency?",
            retrieval_mode="bm25",
            rewrite_query=True,
        ),
        async_db_session,
    )

    assert response["response"] == (
        "Answer to What about its latency?: Active retrieval reduced latency."
    )
    assert response["retrieval_query"] == "active retrieval latency"
    assert response["rewrite_status"] == "applied"


@pytest.mark.asyncio
async def test_source_query_records_missing_session_failure(
    async_db_session, monkeypatch, tmp_path
):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(trace_path))
    with pytest.raises(HTTPException) as error:
        await submit_query(
            QueryRequest(
                query="session question", retrieval_mode="bm25", session_id="missing"
            ),
            async_db_session,
        )
    assert error.value.status_code == 404
    saved = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(saved) == 1
    assert saved[0]["support_status"] == "error"
    assert saved[0]["failures"][0]["error_type"] == "HTTPException"


@pytest.mark.asyncio
async def test_query_records_database_probe_failure(monkeypatch, tmp_path):
    class BrokenDB:
        async def execute(self, _statement):
            raise RuntimeError("private database detail")

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(trace_path))
    with pytest.raises(RuntimeError):
        await submit_query(QueryRequest(query="question"), BrokenDB())
    saved = json.loads(trace_path.read_text().strip())
    assert saved["support_status"] == "error"
    assert saved["failures"][0]["error_type"] == "RuntimeError"
    assert "private database detail" not in str(saved)
