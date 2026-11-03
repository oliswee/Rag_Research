"""Serial, scoped reconciliation; SQLite originals are never rebuilt or deleted."""

from datetime import UTC, datetime

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import select, text

from app.db.source_fts_v2 import active_chunk_sql
from app.models import (
    Paper,
    PaperChunk,
    SourceDeletionJob,
    SourceDocument,
    SourceIndexGeneration,
    SourceParent,
)
from app.services.incremental_index import generation_points_valid, sync_generation
from app.services.parent_child import GenerationBuild
from app.services.source_anchors import resolve_child
from app.services.source_index import CHUNK_COLLECTION, ensure_fts
from app.services.trace_versions import digest

MAX_POINTS = 100000


async def load_build(db, generation):
    document = await db.get(SourceDocument, generation.document_id)
    children = list(
        (
            await db.scalars(
                select(PaperChunk)
                .where(PaperChunk.index_version == generation.index_version)
                .order_by(PaperChunk.chunk_index)
            )
        ).all()
    )
    parents = list(
        (
            await db.scalars(
                select(SourceParent)
                .where(SourceParent.generation_id == generation.index_version)
                .order_by(SourceParent.parent_index)
            )
        ).all()
    )
    if document is None or not children:
        raise ValueError("Generation originals unavailable")
    for child in children:
        await resolve_child(db, child, allow_parent_failure=True)
    return GenerationBuild(document, generation, parents, children, [])


async def fts_report(db):
    exists = await db.scalar(
        text("SELECT count(*) FROM sqlite_master WHERE name='paper_chunks_fts'")
    )
    if not exists:
        count = await db.scalar(
            text("SELECT count(*) FROM paper_chunks c WHERE " + active_chunk_sql())
        )
        return {
            "missing_count": count,
            "orphan_count": 0,
            "mismatch_count": 0,
            "schema_missing": True,
        }
    # FTS external-content SELECT mirrors content rows even if postings vanished.
    # Compare actual tokenizer instances against an identically tokenized temp index.
    for name in ("m7_expected_vocab", "m7_actual_vocab", "m7_expected"):
        await db.execute(text(f"DROP TABLE IF EXISTS temp.{name}"))
    await db.execute(text("CREATE VIRTUAL TABLE temp.m7_expected USING fts5(text)"))
    await db.execute(
        text(
            "INSERT INTO temp.m7_expected(rowid,text) "
            "SELECT rowid,text FROM source_search_content"
        )
    )
    await db.execute(
        text(
            "CREATE VIRTUAL TABLE temp.m7_expected_vocab "
            "USING fts5vocab(temp,m7_expected,instance)"
        )
    )
    await db.execute(
        text(
            "CREATE VIRTUAL TABLE temp.m7_actual_vocab "
            "USING fts5vocab(main,paper_chunks_fts,instance)"
        )
    )
    tables = []
    for name in ("m7_expected_vocab", "m7_actual_vocab"):
        values = {}
        for term, doc, column, offset in (
            await db.execute(text(f"SELECT term,doc,col,offset FROM temp.{name}"))
        ).all():
            values.setdefault(doc, set()).add((term, column, offset))
        tables.append(values)
    expected, actual = tables
    return {
        "missing_count": len(expected.keys() - actual.keys()),
        "orphan_count": len(actual.keys() - expected.keys()),
        "mismatch_count": sum(
            expected[k] != actual[k] for k in expected.keys() & actual.keys()
        ),
        "schema_missing": False,
    }


def own_points(index):
    if not index.client.collection_exists(CHUNK_COLLECTION):
        return {}
    result, offset = {}, None
    while True:
        rows, offset = index.client.scroll(
            CHUNK_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="dataset_id", match=MatchValue(value=index.dataset_id)
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        result.update((str(point.id), point) for point in rows)
        if len(result) > MAX_POINTS:
            raise ValueError("Maintenance scope exceeds the bounded local limit")
        if offset is None:
            return result


async def check_index(db, index):
    fts = await fts_report(db)
    active = list(
        (
            await db.scalars(
                select(SourceIndexGeneration).where(
                    SourceIndexGeneration.is_active.is_(True)
                )
            )
        ).all()
    )
    expected = set(
        (
            await db.execute(
                text("SELECT id FROM paper_chunks c WHERE " + active_chunk_sql())
            )
        ).scalars()
    )
    invalid = []
    unavailable = None
    try:
        points = own_points(index)
        for generation in active:
            try:
                build = await load_build(db, generation)
                if not generation_points_valid(
                    build, index, paper=await db.get(Paper, generation.paper_id)
                ):
                    invalid.append(generation.index_version)
            except (ValueError, TypeError, KeyError):
                invalid.append(generation.index_version)
        vector_report = {
            "missing_count": len(expected - points.keys()),
            "orphan_count": len(points.keys() - expected),
            "mismatch_count": len(invalid),
            "invalid_generations": invalid,
            "unavailable": None,
        }
    except Exception as exc:
        unavailable = type(exc).__name__
        vector_report = {
            "missing_count": None,
            "orphan_count": None,
            "mismatch_count": None,
            "invalid_generations": [],
            "unavailable": unavailable,
        }
    consistent = (
        not unavailable
        and not any(fts.values())
        and not any(
            vector_report[k]
            for k in ("missing_count", "orphan_count", "mismatch_count")
        )
    )
    return {
        "consistent": consistent,
        "fts": fts,
        "vectors": vector_report,
        "dataset_id": index.dataset_id,
        "collection": CHUNK_COLLECTION,
        "checked_at": datetime.now(UTC).isoformat(),
    }


async def queue_source_deletion(db, dataset_id, point_ids):
    point_ids = sorted(set(point_ids))
    if not point_ids:
        return None
    if len(point_ids) > MAX_POINTS:
        raise ValueError("Deletion work exceeds bounded local limit")
    key = digest([dataset_id, point_ids])
    job = await db.get(SourceDeletionJob, key)
    if job is None:
        job = SourceDeletionJob(
            id=key,
            dataset_id=dataset_id,
            point_ids=point_ids,
            status="pending",
            attempts=0,
        )
        db.add(job)
    elif job.status == "done":
        job.status = "pending"
    await db.flush()
    return job


async def retry_source_deletions(db, index):
    jobs = list(
        (
            await db.scalars(
                select(SourceDeletionJob)
                .where(
                    SourceDeletionJob.dataset_id == index.dataset_id,
                    SourceDeletionJob.status == "pending",
                )
                .order_by(SourceDeletionJob.created_at)
                .limit(100)
            )
        ).all()
    )
    for job in jobs:
        job.attempts += 1
        job.updated_at = datetime.now(UTC)
        try:
            for start in range(0, len(job.point_ids), 100):
                ids = job.point_ids[start : start + 100]
                if not index.client.collection_exists(CHUNK_COLLECTION):
                    continue
                points = index.client.retrieve(CHUNK_COLLECTION, ids, with_payload=True)
                owned = [
                    str(p.id)
                    for p in points
                    if p.payload.get("dataset_id") == index.dataset_id
                ]
                # index.delete also applies an atomic ownership filter server-side.
                index.delete(owned)
                if any(
                    p.payload.get("dataset_id") == index.dataset_id
                    for p in index.client.retrieve(
                        CHUNK_COLLECTION, owned, with_payload=True
                    )
                ):
                    raise ValueError("Deletion verification failed")
            job.status, job.error_type = "done", None
        except Exception as exc:
            job.error_type = type(exc).__name__
    await db.flush()
    return {
        "attempted_jobs": len(jobs),
        "pending_jobs": sum(j.status == "pending" for j in jobs),
    }


async def repair_index(db, index, *, embedder=None):
    before = await check_index(db, index)
    errors = []
    if any(before["fts"].values()):
        await ensure_fts(db)
        await db.execute(
            text("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild')")
        )
        await db.commit()
    if not before["vectors"]["unavailable"]:
        expected = set(
            (
                await db.execute(
                    text("SELECT id FROM paper_chunks c WHERE " + active_chunk_sql())
                )
            ).scalars()
        )
        missing = expected - own_points(index).keys()
        legacy_papers = list(
            (
                await db.scalars(
                    select(PaperChunk.paper_id)
                    .where(
                        PaperChunk.id.in_(missing), PaperChunk.index_version == "legacy"
                    )
                    .distinct()
                )
            ).all()
        )
        await queue_source_deletion(
            db, index.dataset_id, list(own_points(index).keys() - expected)
        )
        await db.commit()
        await retry_source_deletions(db, index)
        await db.commit()
        for version in before["vectors"]["invalid_generations"]:
            generation = await db.get(SourceIndexGeneration, version)
            try:
                build = await load_build(db, generation)
                await sync_generation(
                    db, build, embedder=embedder, vector_index=index, force=True
                )
            except Exception as exc:
                errors.append(
                    {"index_version": version, "error_type": type(exc).__name__}
                )
            await db.commit()
        if legacy_papers:
            from evaluation.reindex_paper import reindex_paper

            for paper_id in legacy_papers:
                try:
                    rebuilt = await reindex_paper(db, paper_id, publish=False)
                    generation = await db.get(
                        SourceIndexGeneration, rebuilt["index_version"]
                    )
                    await sync_generation(
                        db,
                        await load_build(db, generation),
                        embedder=embedder,
                        vector_index=index,
                    )
                except Exception as exc:
                    errors.append(
                        {"paper_id": paper_id, "error_type": type(exc).__name__}
                    )
                await db.commit()
            expected = set(
                (
                    await db.execute(
                        text(
                            "SELECT id FROM paper_chunks c WHERE " + active_chunk_sql()
                        )
                    )
                ).scalars()
            )
            await queue_source_deletion(
                db, index.dataset_id, list(own_points(index).keys() - expected)
            )
            await db.commit()
            await retry_source_deletions(db, index)
            await db.commit()
    else:
        errors.append({"error_type": before["vectors"]["unavailable"]})
    return {"before": before, "after": await check_index(db, index), "errors": errors}
