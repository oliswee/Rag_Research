"""Reranking must preserve candidates, scope and bounded worker lifetime."""

import asyncio
import threading

import pytest

from app.services.retrieval_trace import RetrievalTrace


def sources(count=3):
    return [
        {"chunk_id": f"c{i}", "paper_id": f"p{i}", "excerpt": f"Result {i}."}
        for i in range(count)
    ]


def trace():
    return RetrievalTrace("private original question", {}, corpus_version="test")


@pytest.mark.asyncio
async def test_rerank_uses_original_question_and_preserves_all_candidates():
    from app.services.child_reranker import rerank_children

    class NumericScorer:
        model_id = "fixture"
        revision = "fixture-v1"

        def score(self, question, texts):
            assert question == "Was latency 1.5 ms or 5.1 ms?"
            assert texts == ["Latency was 5.1 ms.", "Latency was 1.5 ms.", "Other."]
            return [0.2, 0.9, 0.2]

    items = sources()
    for source, text in zip(
        items, ["Latency was 5.1 ms.", "Latency was 1.5 ms.", "Other."], strict=True
    ):
        source["excerpt"] = text
    recorder = trace()
    ranked = await rerank_children(
        "Was latency 1.5 ms or 5.1 ms?", items, scorer=NumericScorer(), trace=recorder
    )
    assert [s["chunk_id"] for s in ranked] == ["c1", "c0", "c2"]
    assert ranked[0]["excerpt"] == "Latency was 1.5 ms."
    assert items[0]["chunk_id"] == "c0"
    assert recorder.record["reranking"]["status"] == "ok"
    assert "Latency" not in str(recorder.record)


@pytest.mark.asyncio
@pytest.mark.parametrize("scores", [[1], [1, float("nan"), 0], [1, float("inf"), 0]])
async def test_partial_or_nonfinite_model_output_falls_back_without_drops(scores):
    from app.services.child_reranker import rerank_children

    class InvalidScorer:
        def score(self, question, texts):
            return scores

    recorder = trace()
    ranked = await rerank_children(
        "q", sources(), scorer=InvalidScorer(), trace=recorder
    )
    assert [s["chunk_id"] for s in ranked] == ["c0", "c1", "c2"]
    assert recorder.record["reranking"]["status"] == "failed"


@pytest.mark.asyncio
async def test_model_failure_trace_contains_only_error_type():
    from app.services.child_reranker import rerank_children

    class BrokenScorer:
        def score(self, question, texts):
            raise RuntimeError("private text and token-secret")

    recorder = trace()
    ranked = await rerank_children(
        "q", sources(), scorer=BrokenScorer(), trace=recorder
    )
    assert [s["chunk_id"] for s in ranked] == ["c0", "c1", "c2"]
    assert "token-secret" not in str(recorder.record)
    assert recorder.record["reranking"]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_timeout_keeps_worker_slot_until_cpu_job_finishes():
    from app.services.child_reranker import rerank_children

    started, release = threading.Event(), threading.Event()

    class BlockingScorer:
        def score(self, question, texts):
            started.set()
            assert release.wait(3)
            return [0, 2, 1]

    first_trace, busy_trace = trace(), trace()
    try:
        ranked = await rerank_children(
            "q", sources(), scorer=BlockingScorer(), timeout=0.03, trace=first_trace
        )
        assert started.is_set()
        assert [s["chunk_id"] for s in ranked] == ["c0", "c1", "c2"]
        assert first_trace.record["reranking"]["status"] == "timeout"
        await rerank_children("q", sources(), scorer=BlockingScorer(), trace=busy_trace)
        assert busy_trace.record["reranking"]["status"] == "busy"
    finally:
        release.set()
        await asyncio.sleep(0.05)
    assert first_trace.record["reranking"]["status"] == "timeout"


@pytest.mark.parametrize("paper_count", [2, 3, 4, 5])
def test_selection_retains_each_explicit_paper_before_filling_slots(paper_count):
    from app.services.child_reranker import select_children

    items = [dict(s, paper_id="p0") for s in sources(8)] + [
        {"chunk_id": f"extra{i}", "paper_id": f"p{i}", "excerpt": "Evidence."}
        for i in range(1, paper_count)
    ]
    chosen = select_children(items, 5, [f"p{i}" for i in range(paper_count)])
    assert len(chosen) == 5
    assert {s["paper_id"] for s in chosen} == {f"p{i}" for i in range(paper_count)}
    assert len({s["chunk_id"] for s in chosen}) == 5


def test_runtime_loader_is_cache_only_and_cpu(monkeypatch, tmp_path):
    from app.services.child_reranker import LocalChildScorer

    class FakeEncoder:
        def __init__(self, **kwargs):
            assert kwargs["local_files_only"] is True
            assert kwargs["providers"] == ["CPUExecutionProvider"]
            assert kwargs["threads"] == 2

        def rerank(self, query, documents, **kwargs):
            return [0.1, 0.8]

    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", FakeEncoder)
    scorer = LocalChildScorer(cache_dir=str(tmp_path))
    assert scorer.score("q", ["wrong", "right"]) == [0.1, 0.8]
