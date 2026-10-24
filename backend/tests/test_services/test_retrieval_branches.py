"""Event barriers verify overlap and session safety rather than timing speedups."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock

import pytest

from app.services.retrieval_trace import RetrievalTrace


def plans():
    from app.workflows.query_expansion import RetrievalQuery

    return [
        RetrievalQuery(f"Q{i}", "original" if i == 0 else "rewrite", text, "original")
        for i, text in enumerate(["original", "rewrite", "terms"])
    ]


@pytest.mark.asyncio
async def test_six_branches_overlap_without_concurrent_sqlite_use(monkeypatch):
    from app.services.retrieval_branches import collect_rankings

    dense_entered = threading.Event()
    sqlite_active = 0
    scope = {
        "paper_ids": ["p"],
        "year_from": 2020,
        "year_to": 2024,
        "provider": "arxiv",
    }

    async def lexical(db, text, **kwargs):
        nonlocal sqlite_active
        assert sqlite_active == 0
        sqlite_active += 1
        try:
            assert all(kwargs[k] == value for k, value in scope.items())
            assert await asyncio.to_thread(dense_entered.wait, 1)
            return [("common", -1), (text, -0.5)]
        finally:
            sqlite_active -= 1

    class Index:
        def search(self, vector, **kwargs):
            assert kwargs["active_versions"] == {"p": "v2"}
            assert all(kwargs[k] == value for k, value in scope.items())
            dense_entered.set()
            return [("common", 0.9), ("dense", 0.8)]

    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService.generate_text_batch",
        AsyncMock(return_value=[[1.0]] * 3),
    )
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    results = await collect_rankings(
        None,
        plans(),
        active_versions={"p": "v2"},
        scope=scope,
        mode="hybrid",
        lexical_search=lexical,
        vector_index=Index(),
        timeout=2,
        concurrency=3,
        per_query_k=30,
    )
    assert len(results) == 6 and all(r.status == "ok" for r in results)
    assert {(r.query_id, r.retriever) for r in results} == {
        (f"Q{i}", retriever) for i in range(3) for retriever in ["bm25", "dense"]
    }


@pytest.mark.asyncio
async def test_partial_failure_and_timeout_do_not_mutate_trace_later(monkeypatch):
    from app.services.retrieval_branches import collect_rankings

    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService.generate_text_batch",
        AsyncMock(return_value=[[1.0]] * 3),
    )
    ended = threading.Event()

    class Index:
        def search(self, vector, **kwargs):
            time.sleep(0.08)
            ended.set()
            return [("late-private-id", 0.9)]

    async def lexical(db, text, **kwargs):
        if text == "rewrite":
            raise RuntimeError("private raw text")
        return [("safe", -1)]

    trace = RetrievalTrace("private", {}, corpus_version="test")
    results = await collect_rankings(
        None,
        plans(),
        active_versions={},
        scope={},
        mode="hybrid",
        lexical_search=lexical,
        vector_index=Index(),
        timeout=0.02,
        concurrency=6,
        per_query_k=30,
    )
    assert sum(r.status == "timeout" for r in results) == 3
    assert sum(r.status == "error" for r in results) == 1
    assert sum(r.status == "ok" for r in results) == 2
    snapshot = trace.as_dict()
    assert await asyncio.to_thread(ended.wait, 1)
    assert trace.as_dict() == snapshot
    assert all(not r.ranked for r in results if r.status != "ok")


@pytest.mark.asyncio
async def test_no_embedding_work_for_keyword_only_mode(monkeypatch):
    from app.services.retrieval_branches import collect_rankings

    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService",
        lambda: pytest.fail("keyword mode cannot embed"),
    )
    results = await collect_rankings(
        None,
        plans()[:1],
        active_versions={},
        scope={},
        mode="bm25",
        lexical_search=AsyncMock(return_value=[]),
        per_query_k=30,
        timeout=1,
        concurrency=3,
    )
    assert len(results) == 1 and results[0].retriever == "bm25"


@pytest.mark.asyncio
async def test_query_dedup_and_scope_validation_before_branch_work(monkeypatch):
    from app.services.retrieval_branches import collect_rankings
    from app.workflows.query_expansion import RetrievalQuery

    with pytest.raises(ValueError):
        await collect_rankings(
            None,
            [RetrievalQuery("Q1", "rewrite", "no original", "q")],
            active_versions={},
            scope={},
            mode="hybrid",
            lexical_search=AsyncMock(),
            per_query_k=30,
            timeout=1,
            concurrency=3,
        )
