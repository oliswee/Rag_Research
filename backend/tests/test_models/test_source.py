"""Source records keep imported content traceable and imports idempotent."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Paper, PaperType
from app.models.source import PaperChunk, PaperIdentifier


@pytest.mark.asyncio
async def test_identifier_is_unique_across_papers(async_db_session):
    first = Paper(title="First", type=PaperType.research_paper)
    second = Paper(title="Second", type=PaperType.research_paper)
    async_db_session.add_all([first, second])
    await async_db_session.flush()
    async_db_session.add(
        PaperIdentifier(paper_id=first.id, kind="doi", value="10.1234/example")
    )
    await async_db_session.commit()

    async_db_session.add(
        PaperIdentifier(paper_id=second.id, kind="doi", value="10.1234/example")
    )
    with pytest.raises(IntegrityError):
        await async_db_session.commit()


@pytest.mark.asyncio
async def test_chunk_page_and_scope_survive_round_trip(async_db_session):
    paper = Paper(title="Paged", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        text="Evidence on page three",
        content_scope="full_text",
        page_start=3,
        page_end=3,
        start_char=0,
        end_char=22,
        source_url="https://example.org/paper.pdf",
        checksum="sha256:example",
    )
    async_db_session.add(chunk)
    await async_db_session.commit()
    loaded = (
        await async_db_session.execute(
            select(PaperChunk).where(PaperChunk.id == chunk.id)
        )
    ).scalar_one()
    assert (loaded.page_start, loaded.page_end, loaded.content_scope) == (
        3,
        3,
        "full_text",
    )


@pytest.mark.asyncio
async def test_deleting_paper_cascades_to_chunks_and_identifiers(async_db_session):
    paper = Paper(title="Disposable", type=PaperType.research_paper)
    paper.identifiers.append(PaperIdentifier(kind="doi", value="10.1234/disposable"))
    paper.chunks.append(
        PaperChunk(
            text="Abstract text", content_scope="abstract", checksum="sha256:abstract"
        )
    )
    async_db_session.add(paper)
    await async_db_session.commit()
    await async_db_session.delete(paper)
    await async_db_session.commit()
    assert (await async_db_session.execute(select(PaperChunk))).scalars().all() == []
    assert (
        await async_db_session.execute(select(PaperIdentifier))
    ).scalars().all() == []
