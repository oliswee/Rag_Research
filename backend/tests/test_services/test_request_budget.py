import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.request_budget import (
    BudgetExceeded,
    RequestBudget,
    current_budget,
    request_limits,
)


def test_reservations_are_atomic_and_unknown_prices_fail_closed():
    budget = RequestBudget(max_calls=1, max_input_bytes=10, max_output_tokens=4)
    budget.reserve_model("abcd", 4, provider="ollama")
    with pytest.raises(BudgetExceeded):
        budget.reserve_model("a", 1, provider="ollama")
    assert budget.calls == 1
    assert budget.snapshot()["currency_cost"] == 0
    unknown = RequestBudget(max_currency=0.1)
    with pytest.raises(BudgetExceeded):
        unknown.reserve_model("a", 1, provider="deepseek")
    assert unknown.calls == 0


def test_retry_and_external_attempts_are_charged_before_io():
    budget = RequestBudget(max_external_attempts=1, max_calls=2)
    budget.reserve_model("a", 2, provider="deepseek")
    with pytest.raises(BudgetExceeded):
        budget.reserve_external()
    assert budget.external_attempts == 1


@pytest.mark.parametrize(
    "limits,text,tokens",
    [
        ({"max_input_bytes": 2}, "four", 1),
        ({"max_output_tokens": 2}, "a", 3),
    ],
)
def test_rejected_reservation_has_no_partial_counter_mutation(limits, text, tokens):
    budget = RequestBudget(**limits)
    with pytest.raises(BudgetExceeded):
        budget.reserve_model(text, tokens, provider="ollama")
    assert budget.calls == budget.output_tokens == budget.input_bytes == 0


def test_explicit_currency_prices_enforce_ceiling(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "request_input_price_per_million", 1)
    monkeypatch.setattr(settings, "request_output_price_per_million", 2)
    budget = RequestBudget(max_currency=0.000005)
    budget.reserve_model("a", 2, provider="deepseek")
    with pytest.raises(BudgetExceeded):
        budget.reserve_model("a", 1, provider="deepseek")
    assert budget.calls == budget.external_attempts == 1


def test_local_embeddings_use_persistent_cache_without_request_download(monkeypatch):
    import fastembed

    from app.config import settings
    from app.services.embedding_service import EmbeddingService
    from app.services.local_execution import local_only_request

    observed = {}

    class Encoder:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def embed(self, texts):
            return []

    monkeypatch.setattr(fastembed, "TextEmbedding", Encoder)
    token = local_only_request.set(True)
    try:
        EmbeddingService(provider="fastembed")._embed_local(["test"])
    finally:
        local_only_request.reset(token)
    assert observed["cache_dir"] == settings.fastembed_cache_dir
    assert observed["local_files_only"] is True


@pytest.mark.asyncio
async def test_deadline_and_context_reset(monkeypatch):
    from fastapi import HTTPException

    from app.config import settings

    monkeypatch.setattr(settings, "request_deadline_seconds", 0.02)

    @request_limits
    async def slow():
        assert current_budget.get() is not None
        await asyncio.sleep(0.2)

    with pytest.raises(HTTPException) as caught:
        await slow()
    assert caught.value.status_code == 504
    assert current_budget.get() is None


@pytest.mark.asyncio
async def test_concurrent_budgets_are_isolated():
    @request_limits
    async def run():
        budget = current_budget.get()
        budget.reserve_model("a", 1, provider="ollama")
        await asyncio.sleep(0)
        return budget

    left, right = await asyncio.gather(run(), run())
    assert left is not right
    assert left.calls == right.calls == 1
    assert current_budget.get() is None


@pytest.mark.asyncio
async def test_actual_llm_retry_cannot_escape_call_cap():
    import httpx
    import openai

    from app.services.llm_client import LLMClient

    client = LLMClient(provider="ollama", max_retries=2)
    await client.client.close()
    create = AsyncMock(
        side_effect=openai.RateLimitError(
            "synthetic 429",
            response=httpx.Response(
                429, request=httpx.Request("POST", "http://localhost")
            ),
            body=None,
        )
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    budget = RequestBudget(max_calls=1)
    token = current_budget.set(budget)
    try:
        with pytest.raises(BudgetExceeded):
            await client.complete("system", "user", max_tokens=3)
    finally:
        current_budget.reset(token)
    assert create.await_count == 1
