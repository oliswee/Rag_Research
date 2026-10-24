"""Paper repair preserves old citations and never fabricates legacy PDF geometry."""

import pytest

from app.models import Paper, PaperChunk, PaperType


@pytest.mark.asyncio
async def test_reindex_legacy_text_and_repair_reuses_source_snapshot(async_db_session):
    from evaluation.reindex_paper import reindex_paper

    paper = Paper(
        title="Legacy text",
        type=PaperType.research_paper,
        content_text="Original evidence for retrieval. " * 12,
        content_scope="full_text",
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    legacy = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        text="Original evidence for retrieval.",
        content_scope="full_text",
        checksum="legacy",
    )
    async_db_session.add(legacy)
    await async_db_session.commit()
    first = await reindex_paper(
        async_db_session, paper.id, publish=False, child_size=80
    )
    assert first["source_origin"] == "legacy_extraction"
    assert first["vector_status"] == "pending"
    assert first["child_count"] > 1
    assert first["pdf_pages_available"] is False
    second = await reindex_paper(
        async_db_session, paper.id, publish=False, child_size=100
    )
    assert second["document_id"] == first["document_id"]
    assert second["index_version"] != first["index_version"]
    assert (await async_db_session.get(PaperChunk, legacy.id)).text == legacy.text
    same = await reindex_paper(
        async_db_session, paper.id, publish=False, child_size=100
    )
    assert same["index_version"] == second["index_version"]
