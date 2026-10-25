"""Bounded retrieval branches; workers return data and never mutate a Trace."""

import asyncio
import time
from dataclasses import dataclass

from app.services.source_index import SourceVectorIndex
from app.services.sqlite_deadline import sqlite_read_deadline
from app.workflows.query_expansion import RetrievalQuery, content_tokens


@dataclass
class BranchResult:
    query_id: str
    retriever: str
    ranked: list[tuple[str, float | None]]
    status: str
    duration_ms: float
    error_type: str | None = None


async def collect_rankings(
    db,
    queries: list[RetrievalQuery],
    *,
    active_versions: dict[str, str | None],
    scope: dict,
    mode: str,
    lexical_search,
    vector_index=None,
    index_factory=SourceVectorIndex,
    per_query_k: int = 30,
    timeout: float = 15,
    concurrency: int = 3,
) -> list[BranchResult]:
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Unknown retrieval mode")
    if (
        not queries
        or queries[0].query_id != "Q0"
        or queries[0].text != queries[0].parent_question
    ):
        raise ValueError("Original query Q0 is required")
    if not 1 <= per_query_k <= 30 or not 1 <= concurrency <= 3 or not 0 < timeout <= 15:
        raise ValueError("Invalid retrieval budget")
    unique, seen, identifiers = [], set(), set()
    for query in queries:
        if query.parent_question != queries[0].text or query.query_id in identifiers:
            raise ValueError("Query provenance mismatch")
        key = frozenset(content_tokens(query.text))
        identifiers.add(query.query_id)
        if key not in seen:
            unique.append(query)
            seen.add(key)
        if len(unique) == 3:
            break
    scope = {
        key: scope.get(key) for key in ("paper_ids", "year_from", "year_to", "provider")
    }
    session_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    embeddings = None
    if mode != "bm25":
        from app.services.embedding_service import EmbeddingService

        async def embed():
            return await EmbeddingService().generate_text_batch(
                [q.text for q in unique]
            )

        embeddings = asyncio.create_task(embed())

    def vector_read(vector):
        # Keep owned-client teardown in the thread, even after request timeout.
        index = vector_index if vector_index is not None else index_factory()
        try:
            return index.search(
                vector,
                **scope,
                limit=per_query_k,
                with_scores=True,
                active_versions=active_versions,
            )
        finally:
            if vector_index is None:
                index.client.close()

    async def run(query, position, retriever):
        started = time.perf_counter()
        admitted = locked = False
        try:
            # Async cancellation is safe while queued, never during SQLite work.
            async with asyncio.timeout(timeout):
                await semaphore.acquire()
                admitted = True
                if retriever == "bm25":
                    await session_lock.acquire()
                    locked = True
            remaining = timeout - (time.perf_counter() - started)
            if retriever == "bm25":
                ranked = await sqlite_read_deadline(
                    db,
                    lambda: lexical_search(
                        db, query.text, **scope, limit=per_query_k, with_scores=True
                    ),
                    remaining,
                )
            else:
                async with asyncio.timeout(remaining):
                    vectors = await asyncio.shield(embeddings)
                    if len(vectors) != len(unique):
                        raise ValueError("Embedding count mismatch")
                    ranked = await asyncio.to_thread(vector_read, vectors[position])
            result = BranchResult(query.query_id, retriever, ranked, "ok", 0)
        except Exception as exc:
            result = BranchResult(
                query.query_id,
                retriever,
                [],
                "timeout" if isinstance(exc, TimeoutError) else "error",
                0,
                type(exc).__name__,
            )
        finally:
            if locked:
                session_lock.release()
            if admitted:
                semaphore.release()
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    retrievers = ["bm25", "dense"] if mode == "hybrid" else [mode]
    try:
        return await asyncio.gather(
            *(
                run(q, i, retriever)
                for i, q in enumerate(unique)
                for retriever in retrievers
            )
        )
    finally:
        if embeddings is not None:
            if not embeddings.done():
                embeddings.cancel()
            await asyncio.gather(embeddings, return_exceptions=True)
