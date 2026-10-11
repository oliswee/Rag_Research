"""Export the current SQLite excerpt corpus for local relevance annotation."""

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_maker, engine
from app.models import Paper, PaperChunk


async def export_chunks(db: AsyncSession, output: Path) -> int:
    rows = (
        await db.execute(
            select(PaperChunk, Paper)
            .join(Paper, Paper.id == PaperChunk.paper_id)
            .order_by(Paper.title, PaperChunk.chunk_index)
        )
    ).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk, paper in rows:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.id,
                        "paper_id": paper.id,
                        "title": paper.title,
                        "text": chunk.text,
                        "content_scope": chunk.content_scope,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "source_url": chunk.source_url or paper.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/local/chunks.jsonl")
    )
    args = parser.parse_args()
    try:
        async with async_session_maker() as db:
            count = await export_chunks(db, args.output)
    finally:
        await engine.dispose()
    print(f"Exported {count} local excerpts to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
