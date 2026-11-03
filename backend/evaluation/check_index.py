"""Inspect this library's SQLite/FTS/Qdrant indexes; --repair explicitly repairs.

Run serially from backend: uv run python -m evaluation.check_index [--repair]
Retains immutable source history and all foreign-dataset vector points.
"""

import argparse
import asyncio
import json

from app.db import async_session_maker, engine
from app.services.index_maintenance import check_index, repair_index
from app.services.source_index import SourceVectorIndex


async def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--repair", action="store_true")
    args = arguments.parse_args()
    index = None
    try:
        index = SourceVectorIndex()
        async with async_session_maker() as db:
            result = await (
                repair_index(db, index) if args.repair else check_index(db, index)
            )
            await db.commit()
            print(json.dumps(result, ensure_ascii=False))
            good = (result["after"] if args.repair else result)["consistent"]
        if not good:
            raise SystemExit(2)
    except Exception as exc:
        print(json.dumps({"consistent": False, "error_type": type(exc).__name__}))
        raise SystemExit(2) from None
    finally:
        if index is not None:
            index.client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
