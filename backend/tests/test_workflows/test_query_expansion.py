"""Query variants must retain Q0 and fail closed on ambiguous constraints."""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from app.config import settings
from app.schemas.session import ConversationTurn
from app.services.retrieval_trace import RetrievalTrace


def install_llm(monkeypatch, result):
    monkeypatch.setattr("app.services.llm_client.llm_available", lambda: True)

    class FakeLLM:
        async def complete_json(self, **kwargs):
            assert '"history"' in kwargs["user_prompt"]
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)


@pytest.mark.asyncio
async def test_three_queries_remain_independent_and_private_trace_is_hashed(
    monkeypatch,
):
    from app.workflows.query_expansion import expand_queries

    question = "How does active retrieval reduce latency?"
    install_llm(
        monkeypatch,
        {
            "query": "active retrieval reduce latency performance",
            "keywords_query": "active retrieval reduce latency response time",
        },
    )
    trace = RetrievalTrace(question, {"local_only": True}, corpus_version="test")
    plan = await expand_queries(question, [], trace=trace)
    assert [q.query_id for q in plan.queries] == ["Q0", "Q1", "Q2"]
    assert [q.query_type for q in plan.queries] == ["original", "rewrite", "keywords"]
    assert plan.queries[0].text == question
    assert all(q.parent_question == question for q in plan.queries)
    assert plan.status == "applied"
    encoded = json.dumps(trace.record)
    assert question not in encoded and "response time" not in encoded
    assert len(trace.record["query_variants"]) == 3
    assert trace.record["query_variants"][0]["query_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Find DOI 10.1038/s41586-020-2649-2",
        "Explain arXiv:2307.03172",
        "Results excluding mice",
        "What did not improve?",
        "Results without external data",
        'Explain "retrieval augmented generation"',
        "Compare only human studies",
        "Find paper_id:abc-123",
    ],
)
async def test_exact_or_negative_constraints_use_q0_without_llm(monkeypatch, question):
    from app.workflows.query_expansion import expand_queries

    monkeypatch.setattr(
        "app.services.llm_client.llm_available",
        lambda: pytest.fail("Unexpected model call"),
    )
    plan = await expand_queries(question, [])
    assert [q.text for q in plan.queries] == [question]
    assert plan.reason in {"exact_identifier", "constraint_guard"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "active retrieval improves accuracy",  # Missing original research outcome.
        "active retrieval latency 2025",  # New numerical constraint.
        "active retrieval latency without caching",  # New exclusion.
        "active retrieval latency DeepSeek",  # New named entity.
        "",
        "x" * 501,
        None,
    ],
)
async def test_invalid_variants_do_not_participate(monkeypatch, candidate):
    from app.workflows.query_expansion import expand_queries

    install_llm(monkeypatch, {"query": candidate, "keywords_query": candidate})
    plan = await expand_queries("active retrieval latency", [])
    assert len(plan.queries) == 1
    assert plan.status == "failed"


@pytest.mark.asyncio
async def test_duplicate_case_spacing_and_punctuation_variants_do_not_add_votes(
    monkeypatch,
):
    from app.workflows.query_expansion import expand_queries

    install_llm(
        monkeypatch,
        {
            "query": " LATENCY active retrieval ",
            "keywords_query": "active retrieval latency!",
        },
    )
    plan = await expand_queries("active retrieval latency", [])
    assert len(plan.queries) == 1
    assert plan.reason == "no_valid_variants"


@pytest.mark.asyncio
async def test_history_resolves_followup_without_replacing_literal_q0(monkeypatch):
    from app.workflows.query_expansion import expand_queries

    install_llm(
        monkeypatch,
        {
            "query": "active retrieval latency",
            "keywords_query": "active retrieval latency performance",
        },
    )
    turn = ConversationTurn(
        role="user",
        content="How does active retrieval work?",
        timestamp=datetime.now(UTC),
    )
    plan = await expand_queries("What about its latency?", [turn])
    assert plan.queries[0].text == "What about its latency?"
    assert plan.queries[1].text == "active retrieval latency"


@pytest.mark.asyncio
async def test_year_is_retained_and_new_year_rejected(monkeypatch):
    from app.workflows.query_expansion import expand_queries

    install_llm(
        monkeypatch,
        {
            "query": "retrieval latency 2024 performance",
            "keywords_query": "retrieval latency 2025",
        },
    )
    plan = await expand_queries("retrieval latency 2024", [])
    assert len(plan.queries) == 2 and plan.queries[-1].query_id == "Q1"


@pytest.mark.asyncio
async def test_missing_provider_and_disabled_plan_do_not_call_model(monkeypatch):
    from app.workflows.query_expansion import expand_queries

    monkeypatch.setattr("app.services.llm_client.llm_available", lambda: False)
    assert (await expand_queries("retrieval latency", [])).status == "unavailable"
    assert (
        await expand_queries("retrieval latency", [], enabled=False)
    ).status == "disabled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [RuntimeError("429 private secret"), [], {"query": 123}]
)
async def test_provider_failure_or_malformed_output_returns_original(
    monkeypatch, result
):
    from app.workflows.query_expansion import expand_queries

    install_llm(monkeypatch, result)
    plan = await expand_queries("retrieval latency", [])
    assert [q.text for q in plan.queries] == ["retrieval latency"]
    assert plan.status == "failed"


@pytest.mark.asyncio
async def test_rewrite_has_bounded_timeout(monkeypatch):
    from app.workflows.query_expansion import expand_queries

    install_llm(monkeypatch, {})
    monkeypatch.setattr(settings, "query_rewrite_timeout_seconds", 0.01)

    async def slow(self, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr("app.services.llm_client.LLMClient.complete_json", slow)
    plan = await expand_queries("retrieval latency", [])
    assert plan.status == "failed" and plan.reason == "timeout"
