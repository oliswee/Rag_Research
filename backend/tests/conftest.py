"""Pytest fixtures for backend tests."""

from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models.base import Base

# --- Database Fixtures ---


@pytest.fixture(scope="function")
async def async_db_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture(scope="function")
async def async_db_session(async_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = sessionmaker(
        async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def override_get_db(async_db_session: AsyncSession):
    """Override the get_db dependency for testing."""

    async def _override_get_db():
        yield async_db_session

    return _override_get_db


# --- Client Fixtures ---


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def client_with_db(override_get_db) -> Generator[TestClient, None, None]:
    """Create a test client with database override."""
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- LLM Mock Fixtures ---


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""

    def _create_response(
        content: str, input_tokens: int = 100, output_tokens: int = 50
    ):
        return MagicMock(
            content=[MagicMock(text=content)],
            model="claude-sonnet-4-20250514",
            usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    return _create_response


@pytest.fixture
def mock_anthropic_client(mock_llm_response):
    """Mock the Anthropic client for LLM tests."""
    with patch("app.services.llm_client.anthropic.AsyncAnthropic") as mock_class:
        mock_instance = AsyncMock()
        mock_class.return_value = mock_instance
        yield mock_instance, mock_llm_response
