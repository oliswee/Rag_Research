"""Structured provenance must be consistent while legacy rows remain usable."""

import pytest
from sqlalchemy.exc import IntegrityError

from app import models
from app.models import Paper, PaperChunk, PaperType


def source_models():
    names = ("SourceDocument", "SourceIndexGeneration", "SourceParent")
    assert all(hasattr(models, name) for name in names), (
        "Versioned source models are missing"
    )
    return tuple(getattr(models, name) for name in names)


@pytest.mark.asyncio
async def test_source_generations_keep_old_children_and_computed_search_text(
    async_db_session,
):
    Document, Generation, Parent = source_models()
    paper = Paper(id="paper-a", title="Paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    doc = Document(
        id="doc-a",
        paper_id=paper.id,
        document_version="sha256:raw",
        parser_version="parser",
        snapshot_digest="sha256:snapshot",
        raw_pages=[],
        content_scope="abstract",
    )
    async_db_session.add(doc)
    await async_db_session.flush()
    for i in range(2):
        generation = Generation(
            index_version=f"index-{i}",
            paper_id=paper.id,
            document_id=doc.id,
            cleaner_version="cleaner",
            chunker_version="chunker",
            embedding_model_id="local",
            is_active=i == 1,
        )
        async_db_session.add(generation)
        await async_db_session.flush()
        parent = Parent(
            id=f"parent-{i}",
            generation_id=generation.index_version,
            document_id=doc.id,
            raw_text="Original raw source.",
            cleaned_text="Search text.",
            raw_spans=[],
        )
        async_db_session.add(parent)
        await async_db_session.flush()
        child = PaperChunk(
            id=f"child-{i}",
            paper_id=paper.id,
            parent_id=parent.id,
            document_id=doc.id,
            index_version=generation.index_version,
            chunk_index=0,
            text="Original raw source.",
            cleaned_text="Search text.",
            source_spans=[],
            checksum="checksum",
            content_scope="abstract",
        )
        async_db_session.add(child)
    await async_db_session.commit()
    await async_db_session.refresh(child)
    assert child.search_text == "Search text."
    assert (
        await async_db_session.get(PaperChunk, "child-0")
    ).text == "Original raw source."
    assert child.page_start is None


@pytest.mark.asyncio
async def test_only_one_generation_can_be_active_for_each_paper(async_db_session):
    Document, Generation, _ = source_models()
    paper = Paper(id="paper-a", title="Paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    doc = Document(
        id="doc-a",
        paper_id=paper.id,
        document_version="raw",
        parser_version="parser",
        snapshot_digest="snapshot",
        raw_pages=[],
        content_scope="abstract",
    )
    async_db_session.add(doc)
    await async_db_session.flush()
    async_db_session.add_all(
        [
            Generation(
                index_version=f"index-{i}",
                paper_id=paper.id,
                document_id=doc.id,
                cleaner_version="cleaner",
                chunker_version="chunker",
                embedding_model_id="local",
                is_active=True,
            )
            for i in range(2)
        ]
    )
    with pytest.raises(IntegrityError):
        await async_db_session.commit()


@pytest.mark.asyncio
async def test_legacy_chunk_still_uses_raw_search_text(async_db_session):
    source_models()
    paper = Paper(title="Legacy paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        text="Legacy evidence.",
        checksum="legacy",
        content_scope="full_text",
    )
    async_db_session.add(chunk)
    await async_db_session.commit()
    await async_db_session.refresh(chunk)
    assert chunk.index_version == "legacy"
    assert chunk.search_text == "Legacy evidence."
    assert chunk.document_id is None and chunk.parent_id is None
