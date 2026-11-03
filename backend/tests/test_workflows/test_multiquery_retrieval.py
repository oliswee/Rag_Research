from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.models import Paper, PaperChunk, PaperType
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.query_expansion import RetrievalQuery
from app.workflows.source_query import retrieve


@pytest.mark.asyncio
async def test_six_list_rrf_revalidates_scope_before_fusion(
    async_db_session, monkeypatch
):
    good = Paper(
        title="Allowed",
        type=PaperType.research_paper,
        published_date=date(2022, 1, 1),
        import_source="arxiv",
    )
    wrong = Paper(
        title="Foreign",
        type=PaperType.research_paper,
        published_date=date(2025, 1, 1),
        import_source="openalex",
    )
    async_db_session.add_all([good, wrong])
    await async_db_session.flush()
    children = [
        PaperChunk(
            paper_id=p.id,
            chunk_index=0,
            text="Retrieval evidence.",
            content_scope="abstract",
            checksum="x",
        )
        for p in [good, wrong]
    ]
    async_db_session.add_all(children)
    await async_db_session.commit()
    target, foreign = [c.id for c in children]
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService.generate_text_batch",
        AsyncMock(return_value=[[1.0]] * 3),
    )
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    monkeypatch.setattr(
        "app.workflows.source_query.keyword_search",
        AsyncMock(return_value=[(target, -1), (target, -1)]),
    )

    class Index:
        dataset_id = "test"

        def search(self, *args, **kwargs):
            assert kwargs["paper_ids"] == [good.id]
            return [(foreign, 1), ("missing", 0.95), (target, 0.9), (target, 0.8)]

    variants = [
        RetrievalQuery(f"Q{i}", "original" if i == 0 else "rewrite", q, "retrieval")
        for i, q in enumerate(["retrieval", "retrieval latency", "retrieval accuracy"])
    ]
    trace = RetrievalTrace("retrieval", {}, corpus_version="test")
    rows, actual = await retrieve(
        async_db_session,
        "retrieval",
        paper_ids=[good.id],
        year_from=2020,
        year_to=2024,
        provider="arxiv",
        queries=variants,
        vector_index=Index(),
        trace=trace,
    )
    assert actual == "hybrid" and [r["chunk_id"] for r in rows] == [target]
    assert len(trace.record["retrieval_branches"]) == 6
    assert trace.record["rankings"]["rrf"][0]["score"] == pytest.approx(6 / 61)
    assert foreign not in str(trace.record["rankings"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing,expected",
    [
        ("bm25", "dense_fallback"),
        ("dense", "bm25_fallback"),
        ("both", "retrieval_failed"),
    ],
)
async def test_branch_failure_is_explicit_even_with_empty_corpus(
    async_db_session, monkeypatch, failing, expected
):
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService.generate_text_batch",
        AsyncMock(return_value=[[1.0]]),
    )

    async def lexical(*args, **kwargs):
        if failing in {"bm25", "both"}:
            raise RuntimeError("private")
        return []

    class Index:
        dataset_id = "test"

        def search(self, *args, **kwargs):
            if failing in {"dense", "both"}:
                raise RuntimeError("private")
            return []

    monkeypatch.setattr("app.workflows.source_query.keyword_search", lexical)
    trace = RetrievalTrace("retrieval", {}, corpus_version="test")
    rows, actual = await retrieve(
        async_db_session, "retrieval", trace=trace, vector_index=Index()
    )
    assert rows == [] and actual == expected
    assert len(trace.record["failures"]) >= 1
    assert "private" not in str(trace.record["failures"])
