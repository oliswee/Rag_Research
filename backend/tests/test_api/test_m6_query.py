import json

import pytest

from app.config import settings
from app.routers.query import submit_query
from app.schemas.query import QueryRequest
from app.services.context_packing import pack_context
from app.workflows.local_web import LocalRun
from tests.test_services.test_evidence_registry import local_source


@pytest.fixture
def generation_model(monkeypatch):
    class Model:
        async def complete_json(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "evidence_blocks" in payload:
                block = json.loads(payload["evidence_blocks"].split("\n\n")[0])
                source_id, paper_id, quote = (
                    block["citations"][0]["id"],
                    block["paper_ids"][0],
                    block["text"],
                )
                return {
                    "claims": [
                        {
                            "paper_id": paper_id,
                            "text": quote.strip(),
                            "source_ids": [source_id],
                            "supporting_quotes": {source_id: quote},
                        }
                    ]
                }
            return {
                "verdicts": [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "supported",
                        "characteristics": {},
                    }
                    for claim in payload["claims"]
                ]
            }

        async def close(self):
            pass

    monkeypatch.setattr("app.services.llm_client.LLMClient", lambda **kwargs: Model())
    monkeypatch.setattr(settings, "anthropic_api_key", "test")


@pytest.mark.asyncio
async def test_qa_returns_same_verified_schema_as_matrix(
    async_db_session, monkeypatch, tmp_path, generation_model
):
    paper, source = await local_source(async_db_session)

    async def retrieve(*args, **kwargs):
        return [source], "bm25"

    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(
            query="memory accuracy",
            retrieval_mode="bm25",
            rewrite_query=False,
            paper_ids=[paper.id],
        ),
        async_db_session,
    )
    assert (
        response["support_status"] == "supported"
        and response["claims"][0]["support_status"] == "supported"
    )
    assert response["citations"][0]["raw_quote"] == source["excerpt"]
    assert response["citations"][0]["pdf_page"] is None
    assert response["citations"][0]["source_origin"] == "local"
    assert response["evidence"][0]["chunk_id"] == source["chunk_id"]


@pytest.mark.asyncio
async def test_qa_rejects_same_id_corruption_before_any_factual_generation(
    async_db_session, monkeypatch, tmp_path
):
    paper, source = await local_source(async_db_session)

    async def retrieve(*args, **kwargs):
        return [{**source, "excerpt": "The invented accuracy is 99 percent."}], "bm25"

    def forbidden(**kwargs):
        pytest.fail("Corrupt original reached factual generation")

    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr("app.services.llm_client.LLMClient", forbidden)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    response = await submit_query(
        QueryRequest(
            query="memory accuracy",
            retrieval_mode="bm25",
            rewrite_query=False,
            paper_ids=[paper.id],
        ),
        async_db_session,
    )
    assert response["claims"] == [] and response["citations"] == []
    assert response["support_status"] == "abstain" and "99" not in response["response"]


@pytest.mark.asyncio
async def test_unbound_pronoun_requests_one_necessary_clarification(
    async_db_session, monkeypatch, tmp_path
):
    await local_source(async_db_session)
    await local_source(async_db_session)

    async def forbidden(*args, **kwargs):
        pytest.fail("Unresolved study should be clarified before retrieval")

    monkeypatch.setattr("app.routers.query.retrieve", forbidden)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(query="What about its accuracy?", rewrite_query=False),
        async_db_session,
    )
    assert response["support_status"] == "clarification"
    assert response["clarification"]["question"].count("?") == 1
    assert response["claims"] == []


@pytest.mark.asyncio
async def test_multi_paper_qa_decomposes_by_identity_and_preserves_coverage(
    async_db_session, monkeypatch, tmp_path, generation_model
):
    samples = [await local_source(async_db_session) for _ in range(2)]
    sources = {p.id: s for p, s in samples}
    calls = []

    async def local(db, policy, history, trace, **kwargs):
        calls.append(policy.paper_ids)
        return LocalRun(
            pack_context([sources[policy.paper_ids[0]]], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    async def forbidden(*args, **kwargs):
        pytest.fail("Selected papers must not use one shared candidate search")

    monkeypatch.setattr("app.workflows.comparison.local_retrieval", local)
    monkeypatch.setattr("app.routers.query.retrieve", forbidden)
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(
            query="memory accuracy",
            paper_ids=list(sources),
            local_only=True,
            rewrite_query=False,
        ),
        async_db_session,
    )
    assert calls == [(p.id,) for p, _ in samples]
    assert set(response["context_coverage"]) == set(sources)


@pytest.mark.asyncio
async def test_pronoun_is_bound_to_authoritative_previous_paper(
    async_db_session, monkeypatch, tmp_path, generation_model
):
    from datetime import UTC, datetime

    from app.schemas.session import ConversationTurn, SourceReference
    from app.services.session_manager import session_manager

    paper, source = await local_source(async_db_session)
    await local_source(async_db_session)
    session = session_manager.create_session()
    session_manager.update_session(
        session.session_id,
        ConversationTurn(
            role="assistant",
            content="Original evidence",
            timestamp=datetime.now(UTC),
            sources=[SourceReference(pattern_id=source["chunk_id"])],
        ),
    )
    seen = []

    async def retrieve(db, question, paper_ids=None, mode="bm25", **kwargs):
        seen.append(paper_ids)
        return [source], "bm25"

    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(
            query="What is its accuracy?",
            session_id=session.session_id,
            retrieval_mode="bm25",
            rewrite_query=False,
        ),
        async_db_session,
    )
    assert seen and all(scope == [paper.id] or scope == (paper.id,) for scope in seen)
    assert response["citations"][0]["paper_id"] == paper.id
