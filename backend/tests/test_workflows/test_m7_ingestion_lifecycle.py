from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from app.config import settings
from app.models import SourceDeletionJob
from app.services.source_index import SourceVectorIndex
from app.workflows.ingestion import IngestionInput, IngestionWorkflow
from tests.test_services.test_incremental_index import CountingEmbedder


@pytest.mark.asyncio
async def test_duplicate_upload_reuses_embeddings_through_normal_workflow(
    async_db_session, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    embedder = CountingEmbedder()
    embedder.provider, embedder.model = "fastembed", settings.fastembed_model

    def index_factory():
        index = SourceVectorIndex.__new__(SourceVectorIndex)
        index.client = QdrantClient(path=str(tmp_path / "vectors"))
        index.dimensions, index.dataset_id = 4, "ingestion-lifecycle"
        return index

    monkeypatch.setattr(
        "app.services.versioned_vectors.SourceVectorIndex", index_factory
    )
    monkeypatch.setattr(
        "app.services.versioned_vectors.EmbeddingService", lambda: embedder
    )
    monkeypatch.setattr(
        "app.services.incremental_index.SourceVectorIndex", index_factory
    )
    monkeypatch.setattr(
        "app.services.embedding_service.EmbeddingService", lambda: embedder
    )
    metadata = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("offline")))
    patterns = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("offline")))
    for _ in range(2):
        response = await IngestionWorkflow(
            async_db_session, metadata_extractor=metadata, pattern_extractor=patterns
        ).run(IngestionInput(text="Memory accuracy is 95 percent."))
        assert response.success
    assert len(embedder.texts) == 1


@pytest.mark.asyncio
async def test_api_delete_keeps_durable_source_cleanup_when_index_offline(
    async_db_session, client_with_db, monkeypatch
):
    from tests.test_services.test_evidence_registry import local_source

    paper, source = await local_source(async_db_session)
    await async_db_session.commit()

    def offline():
        raise ConnectionError("private provider body")

    monkeypatch.setattr("app.routers.papers.SourceVectorIndex", offline)
    monkeypatch.setattr(
        "app.routers.papers.VectorStoreService",
        lambda: SimpleNamespace(delete_by_paper_id=AsyncMock()),
    )
    response = client_with_db.delete(f"/api/papers/{paper.id}")
    assert response.status_code == 204
    job = (await async_db_session.scalars(select(SourceDeletionJob))).one()
    assert job.status == "pending" and source["chunk_id"] in job.point_ids
