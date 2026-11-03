import importlib.util

import pytest
from sqlalchemy import select

from app.models import Paper, PaperType
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.source_index import CHUNK_COLLECTION
from tests.test_services.test_versioned_vectors import (
    vector_index as vector_index_fixture,
)

vector_index = vector_index_fixture


class CountingEmbedder:
    provider = "local"
    model = "test"

    def __init__(self):
        self.texts = []

    async def generate_text_batch(self, texts):
        self.texts.extend(texts)
        return [[1.0, 0.2, 0.3, 0.4] for _ in texts]


def api():
    assert importlib.util.find_spec("app.services.incremental_index"), (
        "Incremental publication and memo validation are missing"
    )
    from app.services.incremental_index import sync_generation

    return sync_generation


async def build(db, content="Memory accuracy is 95 percent.", paper=None):
    paper = paper or Paper(title="Lifecycle", type=PaperType.research_paper)
    db.add(paper)
    await db.flush()
    result = await stage_generation(
        db,
        paper,
        PDFParserService().parse_abstract(content),
        child_size=32,
        parent_size=64,
        embedding_model_id="local:test",
    )
    await db.commit()
    return paper, result


@pytest.mark.asyncio
async def test_new_then_identical_content_does_not_reembed(
    async_db_session, vector_index
):
    sync = api()
    embedder = CountingEmbedder()
    paper, first = await build(async_db_session)
    await sync(async_db_session, first, embedder=embedder, vector_index=vector_index)
    await async_db_session.commit()
    assert first.generation.vector_status == "ready" and embedder.texts
    before = list(embedder.texts)
    _, repeated = await build(async_db_session, paper=paper)
    result = await sync(
        async_db_session, repeated, embedder=embedder, vector_index=vector_index
    )
    assert result["unchanged"] and embedder.texts == before
    assert first.generation.index_version == repeated.generation.index_version


@pytest.mark.asyncio
async def test_changed_source_reuses_untouched_embedding_input(
    async_db_session, vector_index
):
    sync = api()
    embedder = CountingEmbedder()
    text = "Memory accuracy is 95 percent.\n\nThe sample included 200 adults."
    paper, first = await build(async_db_session, text)
    await sync(async_db_session, first, embedder=embedder, vector_index=vector_index)
    await async_db_session.commit()
    before = len(embedder.texts)
    _, changed = await build(async_db_session, text.replace("200", "300"), paper)
    await sync(async_db_session, changed, embedder=embedder, vector_index=vector_index)
    assert changed.generation.index_version != first.generation.index_version
    assert not first.generation.is_active
    assert len(embedder.texts) - before < len(changed.children)
    assert changed.generation.vector_status == "ready"


@pytest.mark.asyncio
async def test_failed_publication_can_retry_cached_work(
    async_db_session, vector_index, monkeypatch
):
    sync = api()
    _, source = await build(async_db_session)
    embedder = CountingEmbedder()
    original = vector_index.upsert_many

    def failed(*args, **kwargs):
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(vector_index, "upsert_many", failed)
    with pytest.raises(RuntimeError):
        await sync(
            async_db_session, source, embedder=embedder, vector_index=vector_index
        )
    await async_db_session.commit()
    assert source.generation.vector_status == "error"
    before = list(embedder.texts)
    monkeypatch.setattr(vector_index, "upsert_many", original)
    await sync(async_db_session, source, embedder=embedder, vector_index=vector_index)
    assert source.generation.vector_status == "ready" and embedder.texts == before


@pytest.mark.asyncio
async def test_ready_but_missing_vector_is_repaired_without_reembedding(
    async_db_session, vector_index
):
    sync = api()
    _, source = await build(async_db_session)
    embedder = CountingEmbedder()
    await sync(async_db_session, source, embedder=embedder, vector_index=vector_index)
    await async_db_session.commit()
    before = list(embedder.texts)
    vector_index.delete([source.children[0].id])
    result = await sync(
        async_db_session, source, embedder=embedder, vector_index=vector_index
    )
    assert not result["unchanged"] and source.generation.vector_status == "ready"
    assert embedder.texts == before
    assert vector_index.client.retrieve(CHUNK_COLLECTION, [source.children[0].id])


@pytest.mark.asyncio
async def test_corrupt_embedding_memo_is_never_reused(async_db_session, vector_index):
    sync = api()
    from app.models.maintenance import EmbeddingMemo

    _, source = await build(async_db_session)
    embedder = CountingEmbedder()
    await sync(async_db_session, source, embedder=embedder, vector_index=vector_index)
    await async_db_session.commit()
    memo = (await async_db_session.scalars(select(EmbeddingMemo))).first()
    memo.vector = [0.0, 1.0, 0.0, 0.0]
    await async_db_session.commit()
    before = len(embedder.texts)
    await sync(
        async_db_session,
        source,
        embedder=embedder,
        vector_index=vector_index,
        force=True,
    )
    assert len(embedder.texts) > before


@pytest.mark.asyncio
async def test_metadata_change_refreshes_payload_without_reembedding(
    async_db_session, vector_index
):
    from datetime import date

    sync = api()
    paper, source = await build(async_db_session)
    embedder = CountingEmbedder()
    await sync(async_db_session, source, embedder=embedder, vector_index=vector_index)
    await async_db_session.commit()
    before = list(embedder.texts)
    paper.published_date = date(2024, 1, 2)
    await async_db_session.commit()
    result = await sync(
        async_db_session, source, embedder=embedder, vector_index=vector_index
    )
    assert not result["unchanged"] and embedder.texts == before
    saved = vector_index.client.retrieve(CHUNK_COLLECTION, [source.children[0].id])[0]
    assert saved.payload["year"] == 2024


@pytest.mark.asyncio
async def test_embedding_model_change_requires_new_embedding(
    async_db_session, vector_index
):
    sync = api()
    paper, source = await build(async_db_session)
    await sync(
        async_db_session, source, embedder=CountingEmbedder(), vector_index=vector_index
    )
    await async_db_session.commit()
    changed = await stage_generation(
        async_db_session,
        paper,
        PDFParserService().parse_abstract("Memory accuracy is 95 percent."),
        child_size=32,
        parent_size=64,
        embedding_model_id="local:other-model",
    )
    await async_db_session.commit()
    embedder = CountingEmbedder()
    embedder.model = "other-model"
    await sync(async_db_session, changed, embedder=embedder, vector_index=vector_index)
    assert embedder.texts and changed.generation.vector_status == "ready"
