import json

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
    assert record["failures"] == [{"stage": "dense", "error_type": "TimeoutError"}]
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
