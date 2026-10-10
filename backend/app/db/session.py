"""Async database session configuration."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Convert sqlite:/// to sqlite+aiosqlite:///
_database_url = settings.database_url
if _database_url.startswith("sqlite:///"):
    _database_url = _database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(
    _database_url,
    echo=False,
    future=True,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI routes to get a database session.

    Yields:
        AsyncSession: Database session that auto-closes after use.
    """
    async with async_session_maker() as session:
        yield session
