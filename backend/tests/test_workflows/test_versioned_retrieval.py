"""SQLite is authoritative even when a vector service returns stale IDs."""

import pytest

from app.models import Paper, PaperType
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.retrieval_trace import RetrievalTrace
from app.services.trace_versions import corpus_version
from app.workflows.source_query import retrieve


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_status", ["retired", "pending"])
async def test_stale_dense_ids_fall_back_to_active_sources_and_record_anchors(
    async_db_session,
    monkeypatch,
    vector_status,
):
    paper = Paper(title="Retrieval", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    old = await stage_generation(
        async_db_session,
        paper,
        PDFParserService().parse_abstract("Obsolete retrieval evidence."),
    )
    active = await stage_generation(
        async_db_session,
        paper,
        PDFParserService().parse_abstract("Current retrieval evidence."),
    )
    await async_db_session.commit()

    class Embedder:
        async def generate_text_batch(self, queries):
            return [[1.0] for _ in queries]

    class StaleIndex:
        dataset_id = "stale-test"

        def search(self, *args, **kwargs):
            self.versions = kwargs.get("active_versions")
            child = (
                old.children[0] if vector_status == "retired" else active.children[0]
            )
            return [(child.id, 1.0)]

    monkeypatch.setattr("app.services.embedding_service.EmbeddingService", Embedder)
    index = StaleIndex()
    trace = RetrievalTrace("retrieval", {}, corpus_version="test")
    rows, mode = await retrieve(
        async_db_session,
        "retrieval",
        mode="dense",
        trace=trace,
        vector_index=index,
    )
    assert mode == "bm25_fallback"
    assert index.versions == {paper.id: None}  # Active vectors are still pending.
    assert [row["chunk_id"] for row in rows] == [active.children[0].id]
    assert rows[0]["parent_id"] == active.parents[0].id
    assert rows[0]["source_spans"] == active.children[0].source_spans
    assert trace.record["retrieval_schema"] == "parent_child_v1"
    assert trace.record["resolved_anchors"][0]["parent_id"] == active.parents[0].id
    assert "Current retrieval evidence" not in str(trace.record)


@pytest.mark.asyncio
async def test_corpus_fingerprint_changes_on_structured_provenance_edit(
    async_db_session,
):
    paper = Paper(title="Snapshot", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    build = await stage_generation(
        async_db_session,
        paper,
        PDFParserService().parse_abstract("Evidence."),
    )
    await async_db_session.commit()
    before = await corpus_version(async_db_session)
    build.children[0].source_spans = [
        {**build.children[0].source_spans[0], "raw_end": 3}
    ]
    await async_db_session.commit()
    assert await corpus_version(async_db_session) != before
