"""Stop SQLite work without cancelling and invalidating its async connection."""

import asyncio
import time

from sqlalchemy.ext.asyncio import AsyncSession


async def sqlite_read_deadline(db, operation, seconds: float):
    if seconds <= 0:
        raise TimeoutError
    if not isinstance(db, AsyncSession):
        async with asyncio.timeout(seconds):
            return await operation()
    if db.get_bind().dialect.name != "sqlite":
        raise ValueError("Native retrieval deadlines require SQLite")
    deadline = time.perf_counter() + seconds
    connection = await db.connection()
    raw = await connection.get_raw_connection()
    driver = raw.driver_connection
    cursor = await driver.execute("PRAGMA busy_timeout")
    original_busy = (await cursor.fetchone())[0]
    await cursor.close()
    cursor = await driver.execute(f"PRAGMA busy_timeout={max(1, int(seconds * 1000))}")
    await cursor.close()
    await driver.set_progress_handler(
        lambda: int(time.perf_counter() >= deadline), 1000
    )

    # Interrupt only from SQLite's own execution callback. An asynchronous
    # interrupt racing a sequence of statements can also hit finalization or
    # rollback; the progress callback is scoped to the executing operation.
    try:
        try:
            result = await operation()
        except Exception:
            if time.perf_counter() >= deadline:
                raise TimeoutError from None
            raise
        if time.perf_counter() >= deadline:
            raise TimeoutError
        return result
    finally:
        await driver.set_progress_handler(None, 0)
        cursor = await driver.execute(f"PRAGMA busy_timeout={int(original_busy)}")
        await cursor.close()
