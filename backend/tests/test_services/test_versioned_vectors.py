"""Exercise generation filters and publication against a real local Qdrant client."""

import hashlib
import uuid

import pytest
from qdrant_client import QdrantClient

from app.models import Paper, PaperType
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.source_index import SourceVectorIndex, keyword_search


@pytest.fixture
def vector_index():
    index = SourceVectorIndex.__new__(SourceVectorIndex)
    index.client = QdrantClient(":memory:")
    index.dimensions = 4
    index.dataset_id = "test-versioned-library"
    yield index
    index.client.close()


def identifier(label):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, label))


def test_search_excludes_retired_pending_and_legacy_for_active_papers(vector_index):
    rows = [
        ("old", "p1", "old-version"),
        ("active", "p1", "new-version"),
        ("pending", "p2", "pending-version"),
    ]
    for label, paper, version in rows:
        vector_index.upsert_many(
            [(identifier(label), paper, [1.0, 0.0, 0.0, 0.0], None, None)],
            provenance={identifier(label): {"index_version": version}},
        )
    for label, paper in (("legacy-active", "p1"), ("legacy-valid", "p3")):
        vector_index.upsert(identifier(label), paper, [1.0, 0.0, 0.0, 0.0])
    hits = vector_index.search(
        [1.0, 0.0, 0.0, 0.0],
        active_versions={"p1": "new-version", "p2": None},
    )
    assert set(hits) == {identifier("active"), identifier("legacy-valid")}
    assert len(vector_index.search([1.0, 0.0, 0.0, 0.0])) == 5


class Embedder:
    provider = "local"
    model = "test"

    async def generate_text_batch(self, texts):
        self.texts = texts
        return [[1.0, 0.2, 0.3, 0.4] for _ in texts]


async def build_source(db):
    paper = Paper(title="Publication", type=PaperType.research_paper)
    db.add(paper)
    await db.flush()
    build = await stage_generation(
        db,
        paper,
        PDFParserService().parse_abstract(
            "The re\u00ad\ntrieval system finds evidence."
        ),
        embedding_model_id="local:test",
    )
    await db.commit()
    return build


@pytest.mark.asyncio
async def test_publication_verifies_payload_vectors_and_records_physical_manifest(
    async_db_session,
    vector_index,
):
    from app.services.versioned_vectors import publish_generation

    build = await build_source(async_db_session)
    embedder = Embedder()
    await publish_generation(
        async_db_session, build, embedder=embedder, vector_index=vector_index
    )
    generation = build.generation
    assert generation.vector_status == "ready"
    assert generation.physical_revision.startswith("sha256:")
    assert generation.verified_at is not None
    assert generation.index_manifest["point_count"] == len(build.children)
    assert "retrieval" in embedder.texts[0]
    assert "\u00ad" in build.children[0].text
    point = generation.index_manifest["points"][0]
    assert (
        point["source_checksum"]
        == hashlib.sha256(build.children[0].text.encode()).hexdigest()
    )
    assert point["vector_sha256"]
    from app.services.trace_versions import digest

    logical = generation.index_version
    await publish_generation(
        async_db_session, build, embedder=embedder, vector_index=vector_index
    )
    assert generation.index_version == logical
    assert generation.physical_revision == "sha256:" + digest(generation.index_manifest)


@pytest.mark.asyncio
async def test_failed_publication_keeps_source_and_keyword_search(
    async_db_session,
    vector_index,
):
    from app.services.versioned_vectors import publish_generation

    class Broken(Embedder):
        async def generate_text_batch(self, texts):
            raise RuntimeError("private credential must not be stored")

    build = await build_source(async_db_session)
    with pytest.raises(RuntimeError):
        await publish_generation(
            async_db_session, build, embedder=Broken(), vector_index=vector_index
        )
    await async_db_session.commit()
    assert build.generation.vector_status == "error"
    assert build.generation.physical_revision is None
    assert build.generation.error_type == "RuntimeError"
    assert await keyword_search(async_db_session, "retrieval") == [build.children[0].id]


@pytest.mark.asyncio
async def test_foreign_point_collision_rejected_before_any_publication_write(
    async_db_session,
    vector_index,
):
    from app.services.source_index import CHUNK_COLLECTION
    from app.services.versioned_vectors import publish_generation

    build = await build_source(async_db_session)
    foreign = SourceVectorIndex.__new__(SourceVectorIndex)
    foreign.client, foreign.dimensions = vector_index.client, vector_index.dimensions
    foreign.dataset_id = "another-library"
    foreign.upsert(build.children[0].id, "foreign-paper", [0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="collision"):
        await publish_generation(
            async_db_session, build, embedder=Embedder(), vector_index=vector_index
        )
    saved = vector_index.client.retrieve(CHUNK_COLLECTION, [build.children[0].id])[0]
    assert saved.payload["dataset_id"] == "another-library"
    assert build.generation.vector_status == "error"


@pytest.mark.asyncio
async def test_interrupted_republication_never_leaves_a_ready_old_certificate(
    tmp_path,
    vector_index,
):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.models import SourceIndexGeneration
    from app.models.base import Base
    from app.services.source_index import CHUNK_COLLECTION
    from app.services.versioned_vectors import publish_generation

    engine = create_async_engine(
        "sqlite+aiosqlite:///" + (tmp_path / "shared.db").as_posix()
    )
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as db:
            paper = Paper(title="Republish", type=PaperType.research_paper)
            db.add(paper)
            await db.flush()
            build = await stage_generation(
                db,
                paper,
                PDFParserService().parse_abstract("Original evidence. " * 80),
                child_size=32,
                embedding_model_id="local:test",
            )
            await db.commit()
            await publish_generation(
                db, build, embedder=Embedder(), vector_index=vector_index
            )
            await db.commit()
            version = build.generation.index_version
            first_child_id = build.children[0].id
            assert len(build.children) > 20
            observed = []

            class Interrupted(Embedder):
                calls = 0

                async def generate_text_batch(self, texts):
                    self.calls += 1
                    if self.calls == 2:
                        async with sessions() as reader:
                            observed.append(
                                (
                                    await reader.execute(
                                        select(
                                            SourceIndexGeneration.vector_status,
                                            SourceIndexGeneration.physical_revision,
                                        ).where(
                                            SourceIndexGeneration.index_version
                                            == version
                                        )
                                    )
                                ).one()
                            )
                        raise RuntimeError("interrupted after the first vector batch")
                    return [[0.0, 1.0, 0.0, 0.0] for _ in texts]

            with pytest.raises(RuntimeError):
                await publish_generation(
                    db,
                    build,
                    embedder=Interrupted(),
                    vector_index=vector_index,
                )
            await db.rollback()  # Same persisted result as an interrupted process.
            point = vector_index.client.retrieve(
                CHUNK_COLLECTION,
                [first_child_id],
                with_vectors=True,
            )[0]
            assert point.vector == [0.0, 1.0, 0.0, 0.0]
            assert observed == [("pending", None)]
            async with sessions() as reader:
                persisted = await reader.get(SourceIndexGeneration, version)
                assert persisted.vector_status == "pending"
                assert persisted.physical_revision is None
    finally:
        await engine.dispose()
