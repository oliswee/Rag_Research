"""Optional rewrite keeps retrieval context without changing the user's question."""

from datetime import UTC, datetime

import pytest

from app.config import settings
from app.schemas.session import ConversationTurn


def user_turn(content: str) -> ConversationTurn:
    return ConversationTurn(role="user", content=content, timestamp=datetime.now(UTC))


@pytest.mark.asyncio
async def test_rewrite_turns_followup_into_standalone_search(monkeypatch):
    from app.workflows.query_rewrite import rewrite_for_retrieval

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            assert "active retrieval" in kwargs["user_prompt"]
            assert "its latency" in kwargs["user_prompt"]
            return {"query": "active retrieval latency"}

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    query, status = await rewrite_for_retrieval(
        "What about its latency?", [user_turn("How does active retrieval work?")], True
    )

    assert query == "active retrieval latency"
    assert status == "applied"


@pytest.mark.asyncio
async def test_rewrite_without_key_keeps_context_and_reports_unavailable(monkeypatch):
    from app.workflows.query_rewrite import rewrite_for_retrieval

    monkeypatch.setattr(settings, "anthropic_api_key", "")
    query, status = await rewrite_for_retrieval(
        "What about its latency?", [user_turn("active retrieval")], True
    )

    assert query == "active retrieval What about its latency?"
    assert status == "unavailable"


@pytest.mark.asyncio
async def test_invalid_rewrite_keeps_context_and_reports_failure(monkeypatch):
    from app.workflows.query_rewrite import rewrite_for_retrieval

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            return {"query": "  "}

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    query, status = await rewrite_for_retrieval(
        "What about its latency?", [user_turn("active retrieval")], True
    )

    assert query == "active retrieval What about its latency?"
    assert status == "failed"


@pytest.mark.asyncio
async def test_first_question_adds_bounded_related_search_terms(monkeypatch):
    from app.workflows.query_rewrite import rewrite_for_retrieval

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            return {
                "query": "active retrieval latency",
                "keywords": [
                    "response time",
                    "latency",
                    "throughput",
                    "four word expansion is invalid",
                ],
            }

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    query, status = await rewrite_for_retrieval(
        "How fast is active retrieval?", [], True
    )

    assert query == "active retrieval latency response time throughput"
    assert status == "applied"


@pytest.mark.asyncio
async def test_expansion_limits_terms_to_three(monkeypatch):
    from app.workflows.query_rewrite import rewrite_for_retrieval

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            return {
                "query": "graph search",
                "keywords": ["network", "nodes", "edges", "traversal"],
            }

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    query, status = await rewrite_for_retrieval("graph search", [], True)

    assert query == "graph search network nodes edges"
    assert status == "applied"
