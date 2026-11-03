"""Reuse exact embedding inputs while keeping full physical publication checks."""

import hashlib
import math
from datetime import UTC, datetime

from sqlalchemy.dialects.sqlite import insert

from app.models import EmbeddingMemo, Paper, SourceSyncState
from app.services.source_index import CHUNK_COLLECTION, SourceVectorIndex
from app.services.trace_versions import digest
from app.services.versioned_vectors import _vector_hash, publish_generation


def valid_vector(vector, dimensions):
    return (
        isinstance(vector, list)
        and len(vector) == dimensions
        and all(type(x) in {int, float} and math.isfinite(x) for x in vector)
        and sum(x * x for x in vector) > 0
    )


def generation_points_valid(build, index, *, paper=None):
    """A previous certificate alone cannot establish current physical integrity."""
    generation = build.generation
    manifest = generation.index_manifest
    if (
        not generation.is_active
        or generation.vector_status != "ready"
        or not manifest
        or generation.physical_revision != "sha256:" + digest(manifest)
        or manifest.get("dataset_id") != index.dataset_id
        or manifest.get("collection") != CHUNK_COLLECTION
        or manifest.get("dimensions") != index.dimensions
        or manifest.get("index_version") != generation.index_version
    ):
        return False
    expected = {c.id: c for c in build.children}
    points = {p["child_id"]: p for p in manifest.get("points", [])}
    if not expected or expected.keys() != points.keys():
        return False
    for start in range(0, len(expected), 100):
        ids = list(expected)[start : start + 100]
        actual = {
            str(p.id): p
            for p in index.client.retrieve(
                CHUNK_COLLECTION, ids, with_payload=True, with_vectors=True
            )
        }
        if actual.keys() != set(ids):
            return False
        for child_id, point in actual.items():
            wanted = points[child_id]
            payload = {
                k: v
                for k, v in wanted.items()
                if k not in {"child_id", "vector_sha256"}
            }
            child = expected[child_id]
            if (
                point.payload != payload
                or payload.get("paper_id") != generation.paper_id
                or payload.get("index_version") != generation.index_version
                or payload.get("document_id") != build.document.id
                or payload.get("parent_id") != child.parent_id
                or payload.get("embedding_model_id") != generation.embedding_model_id
                or (
                    paper is not None
                    and (
                        payload.get("year")
                        != (paper.published_date.year if paper.published_date else None)
                        or payload.get("provider") != paper.import_source
                    )
                )
                or payload.get("source_checksum")
                != hashlib.sha256(child.text.encode()).hexdigest()
                or payload.get("search_checksum")
                != hashlib.sha256(child.cleaned_text.encode()).hexdigest()
                or not valid_vector(point.vector, index.dimensions)
                or _vector_hash(point.vector) != wanted["vector_sha256"]
            ):
                return False
    return True


class MemoizedEmbedder:
    def __init__(self, db, embedder, dimensions):
        self.db, self.embedder, self.dimensions = db, embedder, dimensions
        self.provider, self.model = embedder.provider, embedder.model
        self.model_id = f"{self.provider}:{self.model}"
        self.generated_count = 0
        self.reused_count = 0

    async def generate_text_batch(self, texts):
        values, missing = {}, {}
        for source in dict.fromkeys(texts):
            input_hash = hashlib.sha256(source.encode()).hexdigest()
            key = digest([self.model_id, self.dimensions, input_hash])
            memo = await self.db.get(EmbeddingMemo, key)
            if (
                memo is not None
                and memo.model_id == self.model_id
                and memo.dimensions == self.dimensions
                and memo.input_hash == input_hash
                and valid_vector(memo.vector, self.dimensions)
                and memo.vector_hash == digest(memo.vector)
            ):
                values[source] = list(memo.vector)
                self.reused_count += 1
            else:
                missing[source] = (key, input_hash)
        if missing:
            vectors = await self.embedder.generate_text_batch(list(missing))
            if len(vectors) != len(missing) or any(
                not valid_vector(v, self.dimensions) for v in vectors
            ):
                raise ValueError("Invalid embedding memo input")
            for source, vector in zip(missing, vectors, strict=True):
                vector = [float(x) for x in vector]
                key, input_hash = missing[source]
                data = dict(
                    id=key,
                    model_id=self.model_id,
                    dimensions=self.dimensions,
                    input_hash=input_hash,
                    vector=vector,
                    vector_hash=digest(vector),
                    created_at=datetime.now(UTC),
                )
                await self.db.execute(
                    insert(EmbeddingMemo)
                    .values(**data)
                    .on_conflict_do_update(index_elements=[EmbeddingMemo.id], set_=data)
                )
                # Expire a previously loaded corrupt row without rolling back work.
                memo = await self.db.get(EmbeddingMemo, key)
                if memo is not None:
                    await self.db.refresh(memo)
                values[source] = vector
                self.generated_count += 1
        return [list(values[source]) for source in texts]


async def sync_generation(db, build, *, embedder=None, vector_index=None, force=False):
    from app.services.embedding_service import EmbeddingService

    index, owned = vector_index or SourceVectorIndex(), vector_index is None
    outcome, failure, memo = "error", None, None
    try:
        if (
            embedder is not None
            and f"{embedder.provider}:{embedder.model}"
            != build.generation.embedding_model_id
        ):
            raise ValueError("Embedding model differs from generation")
        current = False
        if not force:
            try:
                current = generation_points_valid(
                    build, index, paper=await db.get(Paper, build.generation.paper_id)
                )
            except Exception:
                current = False
        if current:
            outcome = "unchanged"
            return {"unchanged": True, "generated_count": 0, "reused_count": 0}
        memo = MemoizedEmbedder(db, embedder or EmbeddingService(), index.dimensions)
        await publish_generation(db, build, embedder=memo, vector_index=index)
        outcome = "published"
        return {
            "unchanged": False,
            "generated_count": memo.generated_count,
            "reused_count": memo.reused_count,
        }
    except Exception as exc:
        failure = type(exc).__name__
        raise
    finally:
        if owned:
            index.client.close()
        state = await db.get(SourceSyncState, build.generation.paper_id)
        if state is None:
            state = SourceSyncState(paper_id=build.generation.paper_id)
            db.add(state)
        state.content_hash = build.document.document_version
        state.index_version = build.generation.index_version
        state.parser_version = build.document.parser_version
        state.chunker_version = build.generation.chunker_version
        state.embedding_model_id = build.generation.embedding_model_id
        state.checked_at = datetime.now(UTC)
        state.outcome, state.error_type = outcome, failure
        await db.flush()
