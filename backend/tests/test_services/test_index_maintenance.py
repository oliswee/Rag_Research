import importlib.util

import pytest
from sqlalchemy import select, text

from app.services.source_index import CHUNK_COLLECTION
from tests.test_services.test_incremental_index import CountingEmbedder, build
from tests.test_services.test_versioned_vectors import identifier
from tests.test_services.test_versioned_vectors import (
    vector_index as vector_index_fixture,
)

vector_index = vector_index_fixture


def api():
    assert importlib.util.find_spec("app.services.index_maintenance"), (
        "Scoped index check/repair is missing"
    )
    from app.services import index_maintenance

    return index_maintenance


@pytest.mark.asyncio
async def test_missing_fts_postings_and_vector_detected_and_repaired(
    async_db_session, vector_index
):
    maintenance = api()
    from app.services.incremental_index import sync_generation

    _, source = await build(async_db_session)
    await sync_generation(
        async_db_session, source, embedder=CountingEmbedder(), vector_index=vector_index
    )
    await async_db_session.commit()
    # External-content SELECT still returns the row after this posting deletion.
    rowid = await async_db_session.scalar(
        text("SELECT rowid FROM paper_chunks WHERE id=:id"),
        {"id": source.children[0].id},
    )
    await async_db_session.execute(
        text(
            "INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
            "VALUES('delete',:id,:text)"
        ),
        {"id": rowid, "text": source.children[0].cleaned_text},
    )
    await async_db_session.commit()
    vector_index.delete([source.children[0].id])
    before = await maintenance.check_index(async_db_session, vector_index)
    assert before["fts"]["missing_count"] == 1
    assert before["vectors"]["missing_count"] == 1
    assert not before["consistent"]
    repaired = await maintenance.repair_index(
        async_db_session, vector_index, embedder=CountingEmbedder()
    )
    assert repaired["after"]["consistent"]
    assert (
        await maintenance.repair_index(
            async_db_session, vector_index, embedder=CountingEmbedder()
        )
    )["after"]["consistent"]


@pytest.mark.asyncio
async def test_orphan_own_vector_removed_foreign_dataset_untouched(
    async_db_session, vector_index
):
    maintenance = api()
    vector_index.upsert(identifier("orphan"), "deleted-paper", [1, 0, 0, 0])
    vector_index.upsert(identifier("foreign"), "other-paper", [1, 0, 0, 0])
    vector_index.client.set_payload(
        CHUNK_COLLECTION,
        {"dataset_id": "other-library"},
        points=[identifier("foreign")],
    )
    before = await maintenance.check_index(async_db_session, vector_index)
    assert before["vectors"]["orphan_count"] == 1
    result = await maintenance.repair_index(
        async_db_session, vector_index, embedder=CountingEmbedder()
    )
    assert result["after"]["consistent"]
    assert not vector_index.client.retrieve(CHUNK_COLLECTION, [identifier("orphan")])
    assert vector_index.client.retrieve(CHUNK_COLLECTION, [identifier("foreign")])


@pytest.mark.asyncio
async def test_deletion_retry_record_survives_paper_removal(
    async_db_session, vector_index, monkeypatch
):
    maintenance = api()
    from app.models.maintenance import SourceDeletionJob
    from app.services.incremental_index import sync_generation

    paper, source = await build(async_db_session)
    await sync_generation(
        async_db_session, source, embedder=CountingEmbedder(), vector_index=vector_index
    )
    await maintenance.queue_source_deletion(
        async_db_session, vector_index.dataset_id, [c.id for c in source.children]
    )
    await async_db_session.delete(paper)
    await async_db_session.commit()
    original = vector_index.delete

    def failed(ids):
        raise TimeoutError("private endpoint detail")

    monkeypatch.setattr(vector_index, "delete", failed)
    await maintenance.retry_source_deletions(async_db_session, vector_index)
    await async_db_session.commit()
    job = (await async_db_session.scalars(select(SourceDeletionJob))).one()
    assert job.status == "pending" and job.error_type == "TimeoutError"
    monkeypatch.setattr(vector_index, "delete", original)
    await maintenance.retry_source_deletions(async_db_session, vector_index)
    assert job.status == "done" and job.attempts == 2


@pytest.mark.asyncio
async def test_queued_id_that_changes_scope_is_not_deleted(
    async_db_session, vector_index
):
    maintenance = api()
    point = identifier("changed-ownership")
    vector_index.upsert(point, "paper", [1, 0, 0, 0])
    await maintenance.queue_source_deletion(
        async_db_session, vector_index.dataset_id, [point]
    )
    await async_db_session.commit()
    vector_index.client.set_payload(
        CHUNK_COLLECTION, {"dataset_id": "other-library"}, points=[point]
    )
    await maintenance.retry_source_deletions(async_db_session, vector_index)
    saved = vector_index.client.retrieve(CHUNK_COLLECTION, [point])[0]
    assert saved.payload["dataset_id"] == "other-library"


def test_source_delete_is_atomically_scoped_to_its_dataset(vector_index):
    point = identifier("foreign-delete-boundary")
    vector_index.upsert(point, "paper", [1, 0, 0, 0])
    vector_index.client.set_payload(
        CHUNK_COLLECTION, {"dataset_id": "other-library"}, points=[point]
    )
    vector_index.delete([point])
    assert vector_index.client.retrieve(CHUNK_COLLECTION, [point])


@pytest.mark.asyncio
async def test_legacy_missing_vector_can_be_rebuilt_without_losing_old_citation(
    async_db_session, vector_index, monkeypatch
):
    import hashlib

    from app.config import settings
    from app.models import Paper, PaperChunk, PaperType
    from app.services.source_anchors import resolve_child

    maintenance = api()
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    paper = Paper(
        title="Legacy repair",
        type=PaperType.research_paper,
        content_scope="abstract",
        content_text="Memory accuracy is 95 percent.",
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    old = PaperChunk(
        id=identifier("legacy-original"),
        paper_id=paper.id,
        chunk_index=0,
        text=paper.content_text,
        content_scope="abstract",
        checksum=hashlib.sha256(paper.content_text.encode()).hexdigest(),
    )
    async_db_session.add(old)
    await async_db_session.commit()
    embedder = CountingEmbedder()
    embedder.provider, embedder.model = "fastembed", settings.fastembed_model
    result = await maintenance.repair_index(
        async_db_session, vector_index, embedder=embedder
    )
    assert result["after"]["consistent"]
    original = await resolve_child(async_db_session, old)
    assert original["excerpt"] == paper.content_text
    from app.services.evidence_registry import build_registry

    # Historical access survives; superseded flat fragments cannot become new facts.
    assert not (await build_registry(async_db_session, [original])).sources
