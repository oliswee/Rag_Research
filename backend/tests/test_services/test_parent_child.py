"""Stable source anchors must survive rechunking and repeated text/pages."""

import importlib.util

import fitz
import pytest
from sqlalchemy import select

from app.models import Paper, PaperChunk, PaperType, SourceIndexGeneration
from app.services.pdf_parser import PDFParserService


def builder():
    assert importlib.util.find_spec("app.services.parent_child"), (
        "Parent-child builder is missing"
    )
    from app.services.parent_child import stage_generation

    return stage_generation


@pytest.mark.asyncio
async def test_rechunking_preserves_old_children_and_original_spans(async_db_session):
    stage = builder()
    paper = Paper(title="Paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    parsed = PDFParserService().parse_abstract(
        "The retrieval method improves recall. " * 40
    )
    first = await stage(
        async_db_session,
        paper,
        parsed,
        child_size=100,
        parent_size=800,
        embedding_model_id="local:test",
    )
    await async_db_session.commit()
    assert len(first.parents) > 1
    assert any(
        sum(child.parent_id == parent.id for child in first.children) > 1
        for parent in first.parents
    )
    original = {child.id: child.text for child in first.children}
    again = await stage(
        async_db_session,
        paper,
        parsed,
        child_size=100,
        parent_size=800,
        embedding_model_id="local:test",
    )
    assert [child.id for child in again.children] == [
        child.id for child in first.children
    ]
    changed = await stage(
        async_db_session,
        paper,
        parsed,
        child_size=150,
        parent_size=800,
        embedding_model_id="local:test",
    )
    await async_db_session.commit()
    assert changed.document.id == first.document.id
    assert changed.generation.index_version != first.generation.index_version
    assert not set(original) & {child.id for child in changed.children}
    await async_db_session.refresh(first.generation)
    assert first.generation.is_active is False
    assert changed.generation.is_active is True
    for identifier, text in original.items():
        assert (await async_db_session.get(PaperChunk, identifier)).text == text
    for child in changed.children:
        assert child.page_start is None and child.page_end is None
        pieces = []
        for span in child.source_spans:
            page = next(
                page
                for page in changed.document.raw_pages
                if page["number"] == span["page"]
            )
            pieces.append(page["structured_text"][span["raw_start"] : span["raw_end"]])
        assert child.text == "\n\n".join(pieces)
    active = (
        (
            await async_db_session.execute(
                select(SourceIndexGeneration).where(
                    SourceIndexGeneration.is_active.is_(True)
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(active) == 1


@pytest.mark.asyncio
async def test_cross_page_parent_and_repeated_text_keep_distinct_original_spans(
    async_db_session,
):
    stage = builder()
    document = fitz.open()
    for _ in range(2):
        document.new_page().insert_text(
            (50, 80), "Repeated evidence from separate pages."
        )
    data = document.tobytes()
    document.close()
    parsed = PDFParserService().parse(data)
    paper = Paper(title="Paged", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    result = await stage(
        async_db_session,
        paper,
        parsed,
        child_size=500,
        parent_size=1000,
        embedding_model_id="local:test",
    )
    await async_db_session.commit()
    assert len(result.parents) == len(result.children) == 1
    parent, child = result.parents[0], result.children[0]
    assert (parent.page_start, parent.page_end) == (1, 2)
    assert [span["page"] for span in child.source_spans] == [1, 2]
    assert child.source_spans[0]["block_id"] != child.source_spans[1]["block_id"]
    assert child.text.count("Repeated evidence from separate pages.") == 2


@pytest.mark.asyncio
async def test_invalid_source_anchor_is_rejected_before_any_generation(
    async_db_session,
):
    stage = builder()
    paper = Paper(title="Paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    parsed = PDFParserService().parse_abstract("Actual evidence.")
    parsed.pages[0].blocks[0].raw_start = 3
    with pytest.raises(ValueError, match="anchor"):
        await stage(async_db_session, paper, parsed, embedding_model_id="local:test")
    assert (
        await async_db_session.execute(select(SourceIndexGeneration))
    ).scalars().all() == []


@pytest.mark.asyncio
async def test_keyword_index_uses_cleaned_active_generation_and_excludes_history(
    async_db_session,
):
    from app.services.source_index import keyword_search

    stage = builder()
    paper = Paper(title="Versioned", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    async_db_session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text="Obsolete legacy evidence.",
            checksum="legacy",
            content_scope="full_text",
        )
    )
    await async_db_session.commit()
    assert await keyword_search(async_db_session, "obsolete")
    first = await stage(
        async_db_session,
        paper,
        PDFParserService().parse_abstract(
            "The re\u00ad\ntrieval method gives alpha evidence."
        ),
        embedding_model_id="local:test",
    )
    await async_db_session.commit()
    assert await keyword_search(async_db_session, "retrieval") == [first.children[0].id]
    assert await keyword_search(async_db_session, "obsolete") == []
    second = await stage(
        async_db_session,
        paper,
        PDFParserService().parse_abstract("Beta evidence replaces the source."),
        embedding_model_id="local:test",
    )
    await async_db_session.commit()
    assert await keyword_search(async_db_session, "alpha") == []
    assert await keyword_search(async_db_session, "beta") == [second.children[0].id]
    assert (await async_db_session.get(PaperChunk, first.children[0].id)).text.endswith(
        "alpha evidence."
    )
    await async_db_session.delete(paper)
    await async_db_session.commit()
    assert await keyword_search(async_db_session, "beta") == []


@pytest.mark.asyncio
async def test_parent_from_another_paper_is_rejected_by_database(async_db_session):
    from sqlalchemy.exc import IntegrityError

    first = Paper(title="First", type=PaperType.research_paper)
    second = Paper(title="Second", type=PaperType.research_paper)
    async_db_session.add_all([first, second])
    await async_db_session.flush()
    builds = [
        await builder()(
            async_db_session,
            paper,
            PDFParserService().parse_abstract("Same evidence."),
        )
        for paper in (first, second)
    ]
    await async_db_session.commit()
    builds[0].children[0].parent_id = builds[1].parents[0].id
    with pytest.raises(IntegrityError):
        await async_db_session.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["text", "cleaned_text"])
async def test_idempotent_rebuild_rejects_changed_text_under_an_existing_id(
    async_db_session,
    field,
):
    paper = Paper(title="Integrity", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    parsed = PDFParserService().parse_abstract("Original retrieval evidence.")
    build = await builder()(async_db_session, paper, parsed)
    await async_db_session.commit()
    setattr(build.children[0], field, "Changed text under the same child ID.")
    await async_db_session.commit()
    with pytest.raises(ValueError, match="immutable"):
        await builder()(async_db_session, paper, parsed)


@pytest.mark.asyncio
async def test_table_parent_keeps_cells_and_rebuilds_after_session_reload(
    async_db_session,
):
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    for x in (50, 200, 350):
        page.draw_line((x, 100), (x, 190))
    for y in (100, 130, 160, 190):
        page.draw_line((50, y), (350, y))
    for row, values in enumerate(
        (
            ("Method used for evidence", "Measured recall"),
            ("Alpha retrieval method", "0.7"),
            ("Beta retrieval method", "0.8"),
        )
    ):
        for column, value in enumerate(values):
            page.insert_text((60 + column * 150, 120 + row * 30), value, fontsize=9)
    parsed = PDFParserService().parse(document.tobytes())
    document.close()
    paper = Paper(title="Table", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    first = await builder()(
        async_db_session,
        paper,
        parsed,
        child_size=32,
        parent_size=64,
    )
    await async_db_session.commit()
    assert len(first.parents) == 1 and len(first.children) > 1
    assert len(first.parents[0].raw_text) > 64
    assert (
        first.document.raw_pages[0]["blocks"][0]["cells"][2][0]
        == "Beta retrieval method"
    )
    version = first.generation.index_version
    async_db_session.expunge_all()
    again = await builder()(
        async_db_session,
        paper,
        parsed,
        child_size=32,
        parent_size=64,
    )
    assert again.generation.index_version == version
