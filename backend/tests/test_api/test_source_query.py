import json

import pytest
from fastapi import HTTPException

from app.config import settings
from app.models import Paper, PaperChunk, PaperStatus, PaperType
from app.routers.query import follow_up_query, submit_query
from app.schemas.query import FollowUpRequest, QueryRequest


@pytest.mark.asyncio
async def test_m4_coverage_names_missing_paper_and_distinguishes_unsent_context(
    async_db_session, monkeypatch, tmp_path
):
    from app.services.retrieval_branches import BranchResult

    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    papers = [
        Paper(title=t, type=PaperType.research_paper)
        for t in ["Available study", "Missing study"]
    ]
    async_db_session.add_all(papers)
    await async_db_session.flush()
    child = PaperChunk(
        paper_id=papers[0].id,
        chunk_index=0,
        text="Latency is 1.5 ms.",
        content_scope="abstract",
        checksum="legacy",
    )
    async_db_session.add(child)
    await async_db_session.commit()

    async def branches(*args, **kwargs):
        return [BranchResult("Q0", "bm25", [(child.id, 1)], "ok", 0)]

    monkeypatch.setattr("app.workflows.source_query.collect_rankings", branches)
    response = await submit_query(
        QueryRequest(
            query="Compare latency",
            rewrite_query=False,
            paper_ids=[p.id for p in papers],
        ),
        async_db_session,
    )
    assert "Missing study" in response["response"] and "unknown" in response["response"]
    assert response["context_coverage"][papers[0].id]["context_status"] == "included"
    assert response["context_coverage"][papers[0].id]["prompt_status"] == "not_sent"
    record = json.loads((tmp_path / "trace.jsonl").read_text())
    assert record["retrieval_budget"]["candidate_k"] == 30
    assert record["prompt_evidence_ids"] == []
    assert record["context_packing"]["actual_prompt_evidence_coverage"] == 0


@pytest.mark.asyncio
async def test_default_uploaded_source_request_and_followup_execute_hybrid(
    async_db_session, monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    paper = Paper(title="Uploaded only", authors=[], type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        text="Retrieval latency improves.",
        content_scope="full_text",
        checksum="x",
    )
    async_db_session.add(chunk)
    await async_db_session.commit()
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService.generate_text_batch",
        AsyncMock(return_value=[[1.0]]),
    )

    class Index:
        def __init__(self):
            self.client = SimpleNamespace(close=lambda: None)

        def search(self, *args, **kwargs):
            return [(chunk.id, 0.9)]

    monkeypatch.setattr("app.workflows.source_query.SourceVectorIndex", Index)
    response = await submit_query(
        QueryRequest(query="retrieval latency"), async_db_session
    )
    assert QueryRequest(query="q").retrieval_mode == "hybrid"
    assert response["retrieval_mode"] == "hybrid" and response["query_count"] <= 3
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert all(stage in record["rankings"] for stage in ["bm25", "dense", "rrf"])
    assert len(record["retrieval_branches"]) == 2
    response = await follow_up_query(
        FollowUpRequest(query="retrieval latency", session_id=response["session_id"]),
        async_db_session,
    )
    assert response["retrieval_mode"] == "hybrid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("ollama_base_url", "https://remote.example/v1"),
        ("ollama_base_url", "http://localhost.evil.example/v1"),
        ("qdrant_host", "remote.example"),
    ],
)
async def test_local_only_rejects_remote_endpoints_before_database(
    field, value, monkeypatch, tmp_path
):
    from unittest.mock import AsyncMock

    from app.services.local_execution import local_only_request

    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, field, value)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    db = AsyncMock()
    with pytest.raises(HTTPException):
        await submit_query(QueryRequest(query="private", local_only=True), db)
    db.execute.assert_not_called()
    assert local_only_request.get() is False


@pytest.mark.asyncio
async def test_local_only_rejects_configured_remote_models_before_query(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    with pytest.raises(HTTPException) as error:
        await submit_query(
            QueryRequest(query="local question", local_only=True), async_db_session
        )
    assert error.value.status_code == 400
    trace = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert trace["scope"]["local_only"] is True
    assert trace["support_status"] == "error"
    assert trace["llm_calls"] == 0


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
    assert trace["index_locator"] == "sqlite_fts5"
    assert any("source_chunks" in item for item in trace["attempted_index_locators"])
    assert trace["rankings"]["bm25"][0]["child_id"] == first["evidence"][0]["chunk_id"]
    assert trace["prompt_evidence_ids"] == []
    assert trace["returned_evidence_ids"] == [first["evidence"][0]["chunk_id"]]
    assert trace["web_decision"] == "answer_local"
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
    async def fake_rewrite(*args, **kwargs):
        pytest.fail("Explicit baseline must retain literal Q0")

    async def fake_retrieve(db, query, *args, **kwargs):
        if query != "What about its latency?":
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

    async def fake_answer(question, evidence, trace=None):
        assert trace is not None
        return f"Answer to {question}: {evidence[0]['excerpt']}", evidence

    monkeypatch.setattr(
        "app.workflows.query_rewrite.rewrite_for_retrieval", fake_rewrite
    )
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
    assert response["retrieval_query"] == "What about its latency?"
    assert response["rewrite_status"] == "disabled"


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
        async def stream(self, _statement):
            raise RuntimeError("private database detail")

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(trace_path))
    with pytest.raises(RuntimeError):
        await submit_query(QueryRequest(query="question"), BrokenDB())
    saved = json.loads(trace_path.read_text().strip())
    assert saved["support_status"] == "error"
    assert saved["failures"][0]["error_type"] == "RuntimeError"
    assert "private database detail" not in str(saved)


@pytest.mark.asyncio
async def test_query_records_terminal_failure_after_recoverable_failure(
    async_db_session, monkeypatch, tmp_path
):
    async def fake_retrieve(*_args, **kwargs):
        kwargs["trace"].record_failure("dense", TimeoutError("dense unavailable"))
        return [], "bm25_fallback"

    async def fake_answer(*_args):
        return "Insufficient evidence.", []

    def broken_update(*_args):
        raise RuntimeError("later private failure")

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(trace_path))
    monkeypatch.setattr("app.routers.query.retrieve", fake_retrieve)
    monkeypatch.setattr("app.routers.query.answer_from_evidence", fake_answer)
    monkeypatch.setattr(
        "app.routers.query.session_manager.update_session", broken_update
    )
    with pytest.raises(RuntimeError):
        await submit_query(
            QueryRequest(query="question", retrieval_mode="dense"), async_db_session
        )
    saved = json.loads(trace_path.read_text().strip())
    assert saved["support_status"] == "error"
    assert saved["failures"] == [
        {"stage": "dense", "error_type": "TimeoutError"},
        {"stage": "query", "error_type": "RuntimeError"},
    ]
