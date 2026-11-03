"""Serially repair all paper generations without deleting history or collections.

Run: uv run python -m evaluation.reindex
For one paper or offline source repair, use evaluation.reindex_paper instead.
"""

import asyncio
import json

from sqlalchemy import exists, func, or_, select

from app.db import async_session_maker, engine
from app.models import Paper, PaperChunk
from evaluation.reindex_paper import reindex_paper


async def main() -> None:
    failed = False
    try:
        async with async_session_maker() as db:
            identifiers = list(
                (
                    await db.scalars(
                        select(Paper.id)
                        .where(
                            or_(
                                func.length(func.trim(Paper.content_text)) > 0,
                                exists(
                                    select(PaperChunk.id).where(
                                        PaperChunk.paper_id == Paper.id
                                    )
                                ),
                            )
                        )
                        .order_by(Paper.id)
                    )
                ).all()
            )
            for identifier in identifiers:
                result = await reindex_paper(db, identifier)
                print(json.dumps(result))
                failed |= bool(result["error_type"])
    finally:
        await engine.dispose()
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
