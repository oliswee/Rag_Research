"""Injected HTTP failures through the actual provider client and QA boundary."""

import asyncio
import json

import httpx
import openai
import pytest
from fastapi import HTTPException

from app.config import settings
from app.routers.query import submit_query
from app.schemas.query import QueryRequest
from app.services.llm_client import LLMClient
from app.services.request_budget import current_budget
from tests.test_services.test_evidence_registry import local_source


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["429", "timeout", "unavailable", "budget"])
async def test_provider_fault_retains_originals_without_unverified_claim(
    async_db_session,
    monkeypatch,
    tmp_path,
    fault,
):
    paper, source = await local_source(async_db_session)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    attempts = []

    async def retrieve(*args, **kwargs):
        return [source], "bm25"

    async def transport(request):
        attempts.append(request)
        if fault == "429":
            return httpx.Response(
                429, json={"error": {"message": "synthetic rate limit"}}
            )
        if fault == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if fault == "unavailable":
            raise httpx.ConnectError("synthetic unavailable", request=request)
        proposal = {
            "claims": [
                {
                    "text": source["excerpt"],
                    "paper_id": paper.id,
                    "source_ids": [source["chunk_id"]],
                    "supporting_quotes": {source["chunk_id"]: source["excerpt"]},
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "model": "test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(proposal),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    model = LLMClient(provider="ollama", max_retries=1)
    await model.client.close()
    model.client = openai.AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1/",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    monkeypatch.setattr("app.services.llm_client.LLMClient", lambda **kwargs: model)
    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr(settings, "request_max_model_attempts", 1)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(
            query="What memory accuracy was reported?",
            paper_ids=[paper.id],
            retrieval_mode="bm25",
            rewrite_query=False,
        ),
        async_db_session,
    )
    assert response["support_status"] == "evidence_only"
    assert response["citations"] == []
    assert response["evidence"][0]["excerpt"] == source["excerpt"]
    assert len(attempts) == 1 and current_budget.get() is None
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert trace["request_budget"]["model_attempts"] == 1
    assert trace["failures"]


@pytest.mark.asyncio
async def test_query_deadline_records_interrupted_trace_and_resets_context(
    async_db_session,
    monkeypatch,
    tmp_path,
):
    from app.services.local_execution import local_only_request

    paper, _ = await local_source(async_db_session)

    async def retrieve(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.02)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    with pytest.raises(HTTPException) as caught:
        await submit_query(
            QueryRequest(
                query="What memory accuracy was reported?",
                paper_ids=[paper.id],
                retrieval_mode="bm25",
                rewrite_query=False,
            ),
            async_db_session,
        )
    assert caught.value.status_code == 504
    assert current_budget.get() is None and local_only_request.get() is False
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert trace["support_status"] == "interrupted" and trace["total_ms"] > 0
