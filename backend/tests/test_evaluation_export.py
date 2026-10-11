import json

import pytest

from app.models import Paper, PaperChunk, PaperStatus, PaperType


@pytest.mark.asyncio
async def test_export_writes_traceable_local_excerpts(async_db_session, tmp_path):
    from evaluation.export_for_labeling import export_chunks

    paper = Paper(
        title="Retrieval study",
        authors=["A. Researcher"],
        type=PaperType.research_paper,
        status=PaperStatus.active,
        source="https://example.org/paper",
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        text="Active retrieval reduced latency in the experiment.",
        content_scope="abstract",
        page_start=None,
        checksum="example-checksum",
    )
    async_db_session.add(chunk)
    await async_db_session.commit()

    output = tmp_path / "chunks.jsonl"
    count = await export_chunks(async_db_session, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert count == 1
    assert record["chunk_id"] == chunk.id
    assert record["paper_id"] == paper.id
    assert record["title"] == "Retrieval study"
    assert record["text"] == "Active retrieval reduced latency in the experiment."
    assert record["content_scope"] == "abstract"
    assert record["page_start"] is None
