"""Rebuild the source Qdrant collection from authoritative SQLite chunks.

Run after setting OPENAI_API_KEY: uv run python -m evaluation.reindex
"""

import asyncio

from sqlalchemy import select

from app.db import async_session_maker
from app.models import Paper, PaperChunk
from app.services.embedding_service import EmbeddingService
from app.services.source_index import CHUNK_COLLECTION, SourceVectorIndex


async def main() -> None:
    async with async_session_maker() as db:
        rows = (
            await db.execute(
                select(PaperChunk, Paper).join(Paper, Paper.id == PaperChunk.paper_id)
            )
        ).all()
    embeddings = EmbeddingService()
    index = SourceVectorIndex()
    if index.client.collection_exists(CHUNK_COLLECTION):
        index.client.delete_collection(CHUNK_COLLECTION)
    index.ensure_collection()
    count = 0
    for offset in range(0, len(rows), 20):
        batch = rows[offset : offset + 20]
        vectors = await embeddings.generate_text_batch(
            [chunk.text for chunk, _ in batch]
        )
        index.upsert_many(
            [
                (
                    chunk.id,
                    paper.id,
                    vector,
                    paper.published_date.year if paper.published_date else None,
                    paper.import_source,
                )
                for (chunk, paper), vector in zip(batch, vectors, strict=True)
            ]
        )
        count += len(batch)
        print(f"Indexed {count}/{len(rows)} excerpts")


if __name__ == "__main__":
    asyncio.run(main())
