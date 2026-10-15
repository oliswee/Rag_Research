"""Source indexing preserves genuine page boundaries and searchable evidence."""

from datetime import date

import pytest
from sqlalchemy import select

from app.models import Paper, PaperChunk, PaperStatus, PaperType
from app.services.source_index import (
    SourceVectorIndex,
    build_chunks,
    keyword_search,
    rrf_merge,
)


def test_chunks_never_invent_page_numbers():
    chunks = build_chunks(
        [("first page text", 1), ("second page text", 2)],
        "full_text",
        "https://example.org/a.pdf",
    )
    assert [c["page_start"] for c in chunks] == [1, 2]
    abstract = build_chunks(
        [("abstract text", None)], "abstract", "https://example.org/a"
    )
    assert abstract[0]["page_start"] is None
    assert abstract[0]["content_scope"] == "abstract"


def test_rrf_uses_rank_and_deduplicates():
    assert rrf_merge([["a", "b"], ["b", "c"]], k=60)[0] == "b"


def test_vector_search_can_return_ranked_scores_without_changing_default_ids():
    class Client:
        def collection_exists(self, _name):
            return True

        def query_points(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(points=[SimpleNamespace(id="chunk-1", score=0.8)])

    index = SourceVectorIndex.__new__(SourceVectorIndex)
    index.client = Client()
    assert index.search([0.1], with_scores=True) == [("chunk-1", 0.8)]
    assert index.search([0.1]) == ["chunk-1"]


@pytest.mark.asyncio
async def test_keyword_search_and_delete(async_db_session):
    db = async_db_session
    paper = Paper(
        title="Alpha",
        authors=[],
        type=PaperType.research_paper,
        status=PaperStatus.active,
        published_date=date(2024, 1, 1),
        import_source="openalex",
    )
    db.add(paper)
    await db.flush()
    db.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text="mitochondrial metabolism",
            content_scope="abstract",
            checksum="x",
        )
    )
    await db.commit()
    hits = await keyword_search(db, "mitochondrial metabolism")
    assert len(hits) == 1
    scored_hits = await keyword_search(db, "mitochondrial metabolism", with_scores=True)
    assert scored_hits[0][0] == hits[0]
    assert isinstance(scored_hits[0][1], float)
    assert await keyword_search(db, "mitochondrial", year_from=2025) == []
    assert (
        await keyword_search(db, "mitochondrial", year_to=2024, provider="openalex")
        == hits
    )
    assert await keyword_search(db, "mitochondrial", provider="semantic_scholar") == []
    assert (
        await db.execute(select(PaperChunk).where(PaperChunk.id == hits[0]))
    ).scalar_one().paper_id == paper.id
    await db.delete(paper)
    await db.commit()
    assert await keyword_search(db, "mitochondrial metabolism") == []
