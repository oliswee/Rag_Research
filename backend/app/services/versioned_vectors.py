"""Recoverable Qdrant publication, certified only at the recorded verification time."""

import hashlib
import math
import struct
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper
from app.services.embedding_service import EmbeddingService
from app.services.parent_child import GenerationBuild
from app.services.source_index import CHUNK_COLLECTION, SourceVectorIndex
from app.services.trace_versions import digest


def _vector_hash(vector: list[float]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(vector)}f", *vector)).hexdigest()


async def publish_generation(
    db: AsyncSession,
    build: GenerationBuild,
    *,
    embedder: EmbeddingService | None = None,
    vector_index: SourceVectorIndex | None = None,
) -> None:
    """Commit pending before external writes; caller commits final ready/error.

    Call after committing source rows, using expire_on_commit=False. The early
    commit is essential when replacing an already-ready generation: readers and
    interrupted processes must never retain its old ready certificate.

    The all-ID preflight precedes writes but cannot lock out external Qdrant
    writers. Publication/reindex jobs must be serialized for each library.
    Partial writes remain excluded by pending/error generation status.
    """
    generation = build.generation
    generation.vector_status = "pending"
    generation.physical_revision = None
    generation.index_manifest = None
    generation.verified_at = None
    generation.error_type = None
    index = vector_index
    try:
        if not generation.is_active:
            raise ValueError("Cannot publish an inactive generation")
        await db.commit()
        embedder = embedder or EmbeddingService()
        if f"{embedder.provider}:{embedder.model}" != generation.embedding_model_id:
            raise ValueError("Generation embedding model does not match publisher")
        index = index or SourceVectorIndex()
        index.ensure_collection()
        identifiers = [child.id for child in build.children]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Generation has missing or duplicate children")
        provenance = {
            child.id: {
                "index_version": generation.index_version,
                "document_id": build.document.id,
                "parent_id": child.parent_id,
                "embedding_model_id": generation.embedding_model_id,
                "source_checksum": hashlib.sha256(child.text.encode()).hexdigest(),
                "search_checksum": hashlib.sha256(
                    child.cleaned_text.encode()
                ).hexdigest(),
            }
            for child in build.children
        }
        # Preflight the whole generation before writing any batch.
        for start in range(0, len(identifiers), 100):
            for point in index.client.retrieve(
                CHUNK_COLLECTION, identifiers[start : start + 100], with_payload=True
            ):
                if (
                    point.payload.get("dataset_id") != index.dataset_id
                    or point.payload.get("paper_id") != generation.paper_id
                    or point.payload.get("index_version") != generation.index_version
                ):
                    raise ValueError("Source vector ID collision")
        paper = await db.get(Paper, generation.paper_id)
        manifests = []
        for start in range(0, len(build.children), 20):
            children = build.children[start : start + 20]
            vectors = await embedder.generate_text_batch(
                [child.cleaned_text for child in children]
            )
            if len(vectors) != len(children):
                raise ValueError("Embedding count does not match children")
            for vector in vectors:
                if (
                    len(vector) != index.dimensions
                    or not all(math.isfinite(value) for value in vector)
                    or sum(value * value for value in vector) <= 0
                ):
                    raise ValueError("Invalid source embedding vector")
            index.upsert_many(
                [
                    (
                        child.id,
                        generation.paper_id,
                        vector,
                        paper.published_date.year if paper.published_date else None,
                        paper.import_source,
                    )
                    for child, vector in zip(children, vectors, strict=True)
                ],
                provenance=provenance,
            )
            actual = {
                str(point.id): point
                for point in index.client.retrieve(
                    CHUNK_COLLECTION,
                    [child.id for child in children],
                    with_payload=True,
                    with_vectors=True,
                )
            }
            for child, expected in zip(children, vectors, strict=True):
                point = actual.get(child.id)
                payload = {
                    **provenance[child.id],
                    "dataset_id": index.dataset_id,
                    "paper_id": generation.paper_id,
                    "year": paper.published_date.year if paper.published_date else None,
                    "provider": paper.import_source,
                }
                if point is None or point.payload != payload:
                    raise ValueError("Published point provenance verification failed")
                length = math.sqrt(sum(value * value for value in expected))
                normalized = [value / length for value in expected]
                if (
                    not isinstance(point.vector, list)
                    or len(point.vector) != index.dimensions
                    or any(
                        not math.isfinite(got) or abs(got - wanted) > 1e-5
                        for got, wanted in zip(point.vector, normalized, strict=True)
                    )
                ):
                    raise ValueError("Published point vector verification failed")
                manifests.append(
                    {
                        "child_id": child.id,
                        **payload,
                        "vector_sha256": _vector_hash(point.vector),
                    }
                )
        manifest = {
            "schema": "source_physical_v1",
            "collection": CHUNK_COLLECTION,
            "dataset_id": index.dataset_id,
            "index_version": generation.index_version,
            "dimensions": index.dimensions,
            "point_count": len(manifests),
            "points": sorted(manifests, key=lambda point: point["child_id"]),
        }
        generation.index_manifest = manifest
        generation.physical_revision = "sha256:" + digest(manifest)
        generation.verified_at = datetime.now(UTC)
        generation.vector_status = "ready"
    except Exception as exc:
        generation.vector_status = "error"
        generation.error_type = type(exc).__name__
        raise
    finally:
        if vector_index is None and index is not None:
            index.client.close()
        await db.flush()
