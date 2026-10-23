import json

import pytest

from app.config import settings
from app.services.retrieval_trace import RetrievalTrace


def test_trace_persists_ranked_ids_without_secrets_or_source_text(tmp_path):
    trace = RetrievalTrace(
        original_query="What is the result? ANTHROPIC_API_KEY=sk-secretvalue123456",
        scope={"paper_ids": ["paper-1"], "provider": None},
        corpus_version="qasper-v0.3",
    )
    trace.record_query("result", "applied")
    trace.add_ranked("bm25", [("chunk-1", -2.5)], duration_ms=3.0)
    trace.add_ranked("rrf", [("chunk-1", 0.02)], duration_ms=0.2)
    trace.record_failure("dense", TimeoutError("sk-othersecret123456 source text"))
    trace.record_llm_usage("local-judge", 12, 4)
    trace.record_llm_retry("TimeoutError")
    trace.finish("supported", ["chunk-1"])
    path = tmp_path / "retrieval_trace.jsonl"
    trace.append(path)
    record = json.loads(path.read_text(encoding="utf-8").strip())

    assert record["run_id"]
    assert record["index_version"] == "unversioned"
    assert record["retrieval_schema"] == "legacy_flat_v1"
    assert record["rankings"]["bm25"] == [
        {"child_id": "chunk-1", "rank": 1, "score": -2.5}
    ]
    assert record["failures"] == [
        {"stage": "dense", "error_type": "TimeoutError"},
        {"stage": "llm_retry", "error_type": "TimeoutError"},
    ]
    assert record["token_usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert record["llm_retry_count"] == 1
    assert record["prompt_evidence_ids"] == ["chunk-1"]
    assert record["support_status"] == "supported"
    assert "sk-secretvalue" not in str(record)
    assert "source text" not in str(record)


def test_trace_rejects_non_numeric_scores_and_duplicate_prompt_ids():
    trace = RetrievalTrace("question", {}, corpus_version="unversioned")
    trace.add_ranked("bm25", [("a", float("nan")), ("b", 1.0)], duration_ms=1)
    trace.finish("abstain", ["a", "a", "b"])
    record = trace.as_dict()
    assert record["rankings"]["bm25"][0]["score"] is None
    assert record["prompt_evidence_ids"] == ["a", "b"]


def test_trace_redacts_configured_and_bearer_credentials(monkeypatch):
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "fixture-private-token")
    trace = RetrievalTrace(
        "Authorization: Bearer bearer-private-token fixture-private-token",
        {},
        corpus_version="unversioned",
    )
    saved = str(trace.as_dict())
    assert "bearer-private-token" not in saved
    assert "fixture-private-token" not in saved


def test_private_query_text_is_not_persisted_even_for_unknown_secret_format():
    trace = RetrievalTrace(
        "SEMANTIC_SCHOLAR_API_KEY=another-valid-key",
        {},
        corpus_version="private",
    )
    trace.record_query("private rewrite with another-valid-key", "applied")
    saved = str(trace.as_dict())
    assert "another-valid-key" not in saved
    assert trace.as_dict()["original_query"] == "[private-query]"
    assert trace.as_dict()["original_query_sha256"]
    assert trace.as_dict()["retrieval_query"] == "[private-query]"


def test_public_query_redacts_named_api_keys():
    trace = RetrievalTrace(
        "Find SEMANTIC_SCHOLAR_API_KEY=another-valid-key paper",
        {},
        corpus_version="public",
        corpus_public=True,
    )
    assert "another-valid-key" not in trace.as_dict()["original_query"]


def test_trace_scope_never_persists_untrusted_paper_ids_or_provider(tmp_path):
    trace = RetrievalTrace(
        "public question",
        {
            "paper_ids": ["paper-1", "Authorization: Bearer private-token"],
            "provider": "private-provider-text",
            "requested_mode": "bm25",
            "year_from": 2020,
            "local_only": True,
            "latest": False,
        },
        corpus_version="fixture",
    )
    output = tmp_path / "trace.jsonl"
    trace.append(output)
    saved = output.read_text(encoding="utf-8")
    assert "private-token" not in saved
    assert "private-provider-text" not in saved
    assert "paper-1" not in saved
    assert trace.as_dict()["scope"]["paper_id_count"] == 2
    assert trace.as_dict()["scope"]["requested_mode"] == "bm25"
    assert trace.as_dict()["scope"]["local_only"] is True
    assert trace.as_dict()["scope"]["latest"] is False


@pytest.mark.parametrize("unknown_first", [True, False])
def test_partial_token_usage_is_unknown(unknown_first):
    trace = RetrievalTrace("query", {}, corpus_version="test")
    values = [(None, 2), (5, 3)] if unknown_first else [(5, 3), (None, 2)]
    for input_tokens, output_tokens in values:
        trace.record_llm_usage("local", input_tokens, output_tokens)
    assert trace.record["token_usage"] == {"input_tokens": None, "output_tokens": 5}
