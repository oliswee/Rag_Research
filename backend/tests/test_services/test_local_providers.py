"""Local defaults must run without paid API keys or mixing vector dimensions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from app.config import ModelID, settings
from app.services.embedding_service import EmbeddingService, embedding_collection
from app.services.llm_client import LLMClient
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_index import SourceVectorIndex
from evaluation.deepeval_adapter import JudgeConfig, evaluate_trace


def test_local_only_embedding_uses_cached_files(monkeypatch):
    from app.services.local_execution import local_only_request

    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    token = local_only_request.set(True)
    try:
        with patch("fastembed.TextEmbedding") as model:
            model.return_value.embed.return_value = []
            EmbeddingService()._embed_local(["private text"])
            assert model.call_args.kwargs["local_files_only"] is True
    finally:
        local_only_request.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 409, 429, 503])
async def test_local_llm_retries_are_bounded_and_counted(status_code, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = openai.APIStatusError(
        "temporary", response=httpx.Response(status_code, request=request), body=None
    )
    if status_code == 429:
        error = openai.RateLimitError(
            "temporary",
            response=httpx.Response(status_code, request=request),
            body=None,
        )
    with (
        patch("app.services.llm_client.openai.AsyncOpenAI") as sdk,
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        sdk.return_value.chat.completions.create = AsyncMock(side_effect=error)
        client = LLMClient(max_retries=2)
        client.trace = RetrievalTrace("q", {}, corpus_version="test")
        from app.services.llm_client import LLMError

        with pytest.raises(LLMError):
            await client.complete("system", "question")
        assert sdk.call_args.kwargs["max_retries"] == 0
        assert sdk.return_value.chat.completions.create.await_count == 2
        assert client.trace.record["llm_retry_count"] == 1
        assert sleep.await_count == 1


@pytest.mark.asyncio
async def test_fastembed_needs_no_key_and_preserves_batch_order(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    monkeypatch.setattr(settings, "openai_api_key", "")
    service = EmbeddingService()
    monkeypatch.setattr(
        service,
        "_embed_local",
        lambda texts: [[float(index)] * 384 for index, _ in enumerate(texts)],
    )

    vectors = await service.generate_text_batch(["first", "second"])
    single = await service.generate_embedding("single")

    assert service.client is None
    assert service.dimensions == 384
    assert vectors[0] != vectors[1]
    assert len(single.embedding) == 384
    assert single.token_count == 0


@pytest.mark.asyncio
async def test_ollama_needs_no_key_and_uses_instruct_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    response = SimpleNamespace(
        model="qwen3:4b-instruct",
        choices=[SimpleNamespace(message=SimpleNamespace(content="你好"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    with patch("app.services.llm_client.openai.AsyncOpenAI") as client_class:
        client_class.return_value.chat.completions.create = AsyncMock(
            return_value=response
        )
        client = LLMClient(model=ModelID.SONNET_4)
        trace = RetrievalTrace("local question", {}, corpus_version="test")
        client.trace = trace
        result = await client.complete("Respond briefly", "Say hello", max_tokens=20)

    assert client.model == "qwen3:4b-instruct"
    assert result.content == "你好"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert trace.as_dict()["token_usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert trace.as_dict()["llm_calls"] == 1
    assert client_class.call_args.kwargs["base_url"] == "http://localhost:11434/v1/"


def test_deepseek_is_explicit_and_requires_its_own_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        LLMClient()
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    with patch("app.services.llm_client.openai.AsyncOpenAI") as client_class:
        client = LLMClient()
    assert client.model == "deepseek-flash"
    assert client_class.call_args.kwargs["base_url"] == "https://api.deepseek.com"


def test_local_vectors_use_distinct_collections_and_reject_wrong_size(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    original = embedding_collection("source_chunks")
    monkeypatch.setattr(settings, "fastembed_model", "BAAI/bge-small-en-v1.5-other")
    assert embedding_collection("source_chunks") != original

    class ExistingCollection:
        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=1536))
                )
            )

    index = SourceVectorIndex.__new__(SourceVectorIndex)
    index.client = ExistingCollection()
    index.dimensions = 384
    with pytest.raises(ValueError, match="wrong dimensions"):
        index.ensure_collection()


def test_source_search_scopes_results_to_current_library():
    seen = {}

    class Client:
        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=384))
                )
            )

        def query_points(self, _name, **kwargs):
            seen["filter"] = kwargs["query_filter"]
            return SimpleNamespace(points=[])

    index = SourceVectorIndex.__new__(SourceVectorIndex)
    index.client = Client()
    index.dimensions = 384
    index.dataset_id = "current-library"
    assert index.search([0.0] * 384) == []
    assert seen["filter"].must[0].match.value == "current-library"


def test_local_judge_accepts_private_context_without_remote_upload():
    trace = {
        "run_id": "local-1",
        "original_query": "[private-query]",
        "prompt_evidence_ids": ["chunk-1"],
        "corpus_public": False,
    }
    result = evaluate_trace(
        trace,
        question="Private question",
        actual_output="Answer",
        evidence_by_id={"chunk-1": "Private evidence"},
        expected_output=None,
        judge=JudgeConfig(provider="ollama", model="qwen3:4b-instruct", api_key=""),
        measure=lambda *_: {"score": 0.5, "cost": 0.0},
    )
    assert result["status"] == "evaluated"
    assert "Private evidence" not in str(result)
