import pytest


@pytest.mark.asyncio
async def test_timed_out_benchmark_reads_use_worker_owned_clients(monkeypatch):
    import asyncio
    import threading
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from app.config import settings
    from evaluation.isolated_corpus import evaluate_bm25
    from tests.test_evaluation_isolated import sample_corpus

    parent = MagicMock()
    parent.dataset_id = "benchmark:test"
    ended = threading.Event()
    closed = threading.Event()
    readers = []

    class Reader:
        def __init__(self, *, dataset_id_override):
            assert dataset_id_override == parent.dataset_id
            self.active = False
            self.client = SimpleNamespace(close=self.close)
            readers.append(self)

        def search(self, *args, **kwargs):
            self.active = True
            time.sleep(0.08)
            self.active = False
            ended.set()
            return [("chunk-a", 1.0)]

        def close(self):
            assert not self.active
            closed.set()

    embedder = SimpleNamespace(
        provider="fastembed",
        model="test",
        generate_text_batch=AsyncMock(return_value=[[1.0]]),
    )
    monkeypatch.setattr("evaluation.isolated_corpus.EmbeddingService", lambda: embedder)
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService", lambda: embedder
    )
    monkeypatch.setattr(
        "evaluation.isolated_corpus.prepare_vector_index",
        AsyncMock(return_value=parent),
    )
    monkeypatch.setattr("evaluation.isolated_corpus.SourceVectorIndex", Reader)
    monkeypatch.setattr(settings, "retrieval_branch_timeout_seconds", 0.02)
    with pytest.raises(RuntimeError, match="Expected hybrid"):
        await evaluate_bm25(*sample_corpus(), mode="hybrid")
    assert parent.search.call_count == 0
    parent.client.close.assert_called_once()
    assert await asyncio.to_thread(ended.wait, 1)
    assert await asyncio.to_thread(closed.wait, 1)
    assert readers


def test_cached_plans_reject_question_and_configuration_mismatch():
    from evaluation.multiquery_benchmark import restore_plan

    row = {
        "question": "retrieval latency",
        "plan_version": "v1",
        "queries": [
            {
                "query_id": "Q0",
                "query_type": "original",
                "text": "retrieval latency",
                "parent_question": "retrieval latency",
            }
        ],
        "status": "disabled",
        "reason": "disabled",
        "rejected_count": 0,
    }
    assert restore_plan(row, "retrieval latency", "v1").queries[0].query_id == "Q0"
    with pytest.raises(ValueError):
        restore_plan(row, "other question", "v1")
    with pytest.raises(ValueError):
        restore_plan(row, "retrieval latency", "v2")
    row["queries"][0]["text"] = "rewritten original"
    with pytest.raises(ValueError):
        restore_plan(row, "retrieval latency", "v1")


def test_comparison_keeps_unanswerable_metrics_out_of_mean():
    from evaluation.multiquery_benchmark import paired_comparison

    single = {
        "results": [
            {"query": "a", "recall_at_10": 0.5, "mrr_at_10": 0.25},
            {"query": "b", "recall_at_10": None, "mrr_at_10": None},
        ]
    }
    multi = {
        "results": [
            {"query": "a", "recall_at_10": 1.0, "mrr_at_10": 0.5},
            {"query": "b", "recall_at_10": None, "mrr_at_10": None},
        ]
    }
    result = paired_comparison(single, multi)
    assert result["answerable_count"] == 1
    assert result["mean_recall_delta"] == 0.5
    multi["results"][0]["query"] = "mismatched"
    with pytest.raises(ValueError):
        paired_comparison(single, multi)


def test_parameter_replay_uses_only_independent_lists_and_no_fake_latency():
    from evaluation.multiquery_benchmark import replay_fusion

    labels = [{"query": "q", "relevant_chunk_ids": ["b"]}]
    trace = {
        "original_query": "q",
        "rankings": {
            "Q0.bm25": [{"child_id": "a"}, {"child_id": "b"}],
            "Q0.dense": [{"child_id": "b"}, {"child_id": "a"}],
            "bm25": [{"child_id": "a"}],
        },
        "retrieval_branches": [
            {"query_id": "Q0", "retriever": r} for r in ["bm25", "dense"]
        ],
    }
    report = replay_fusion(labels, [trace], top_k=1, rrf_k=20)
    assert report["results"][0]["ranked_chunk_ids"] == ["a", "b"]
    assert report["metrics"]["mrr_at_10"] == 0.5
    assert report["latency_p95_ms"] is None
    assert report["evaluation_kind"] == "saved_rank_list_replay"
    with pytest.raises(ValueError):
        replay_fusion(labels, [], top_k=1, rrf_k=20)
