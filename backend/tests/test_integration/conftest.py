"""Fixtures for integration tests.

These fixtures provide mocked external services (LLM, embeddings, Qdrant)
while allowing the full workflow logic to execute.
"""

import re
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models.base import Base
from app.schemas.session import ConversationSession

# --- Database Fixtures ---


@pytest.fixture(scope="function")
async def integration_db_engine():
    """Create an in-memory SQLite engine for integration testing."""
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
async def integration_db_session(
    integration_db_engine,
) -> AsyncSession:
    """Create a test database session for integration tests."""
    async_session_maker = sessionmaker(
        integration_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


# --- Mock Service Fixtures ---


@pytest.fixture
def mock_metadata_extractor() -> MagicMock:
    """Mock metadata extractor returning deterministic metadata."""
    from app.models import PaperType

    extractor = MagicMock()

    async def mock_extract(text, pdf_hints=None, user_metadata=None):
        result = MagicMock()
        # Use user metadata if provided, otherwise use defaults
        if user_metadata:
            result.title = user_metadata.title or "Extracted Title"
            result.authors = user_metadata.authors or ["Author One"]
            result.source = user_metadata.source
            result.published_date = user_metadata.published_date
            # Always return a valid type (never None)
            result.type = user_metadata.type or PaperType.research_paper
        else:
            result.title = "Test Paper: Agent Memory Patterns"
            result.authors = ["Test Author", "Another Author"]
            result.source = "https://arxiv.org/abs/1234"
            result.published_date = None
            result.type = PaperType.research_paper  # Always return valid type
        return result

    extractor.extract = mock_extract
    return extractor


@pytest.fixture
def mock_pattern_extractor() -> MagicMock:
    """Mock pattern extractor returning deterministic patterns."""
    extractor = MagicMock()

    async def mock_extract(text, metadata_context=None):
        # Create mock patterns
        pattern1 = MagicMock()
        pattern1.name = "Memory Consolidation Pattern"
        pattern1.claim = "Agents benefit from periodic memory consolidation"
        pattern1.evidence = "Studies show 40% improvement in recall after consolidation"
        pattern1.context = "This applies to long-running agent tasks"
        pattern1.tags = ["memory", "agents"]
        pattern1.confidence = MagicMock(value="high")

        pattern2 = MagicMock()
        pattern2.name = "Context Window Management"
        pattern2.claim = "Efficient context management reduces hallucinations"
        pattern2.evidence = "Experiments demonstrate 25% reduction in false claims"
        pattern2.context = "Particularly important for multi-turn conversations"
        pattern2.tags = ["context", "hallucination"]
        pattern2.confidence = MagicMock(value="medium")

        result = MagicMock()
        result.patterns = MagicMock(patterns=[pattern1, pattern2])
        result.warnings = []
        result.is_successful = True
        result.pattern_count = 2

        return result

    extractor.extract = mock_extract
    return extractor


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock embedding service returning deterministic embeddings."""
    service = MagicMock()

    # Return different but consistent embeddings based on input
    def prepare_text(pattern_dict):
        parts = []
        for key in ["name", "claim", "evidence", "context"]:
            if pattern_dict.get(key):
                parts.append(str(pattern_dict[key]))
        return " | ".join(parts)

    async def generate_embedding(text):
        # Generate a deterministic embedding based on text hash
        import hashlib

        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        # Create a 1536-dim embedding with some variation
        base = (text_hash % 100) / 100.0
        embedding = [base + (i * 0.0001) for i in range(1536)]

        result = MagicMock()
        result.embedding = embedding
        return result

    service.prepare_embedding_text = prepare_text
    service.generate_embedding = generate_embedding
    return service


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store with in-memory storage."""
    store = MagicMock()

    # In-memory storage for vectors
    vectors: dict = {}

    async def upsert(pattern_id, embedding, payload):
        vectors[pattern_id] = {"embedding": embedding, "payload": payload}

    async def search(query_vector, limit=10):
        # Return all stored vectors as results (matching real VectorStoreService.search)
        results = []
        for pattern_id, data in list(vectors.items())[:limit]:
            results.append(
                {
                    "id": pattern_id,
                    "score": 0.95 - len(results) * 0.05,  # Decreasing scores
                    "payload": data["payload"],
                }
            )
        return results

    async def delete_by_paper_id(paper_id):
        to_delete = [
            pid
            for pid, data in vectors.items()
            if data["payload"].get("paper_id") == paper_id
        ]
        for pid in to_delete:
            del vectors[pid]

    def delete_pattern(pattern_id):
        if pattern_id in vectors:
            del vectors[pattern_id]

    store.upsert = upsert
    store.search = search
    store.delete_by_paper_id = delete_by_paper_id
    store.delete_pattern = delete_pattern
    store._vectors = vectors  # Expose for inspection in tests
    return store


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client returning deterministic synthesis results."""
    client = MagicMock()

    async def complete_json(system_prompt, user_prompt):
        pattern_ids = re.findall(r"(?m)^\[P:([^\]]+)\]", user_prompt)
        assert len(pattern_ids) >= 2
        return {
            "answer": (
                "Based on the research patterns in your library, agents benefit "
                f"significantly from memory consolidation [P:{pattern_ids[0]}]. "
                "Additionally, efficient context window management helps "
                f"reduce hallucinations [P:{pattern_ids[1]}]."
            ),
            "sources": [
                {
                    "pattern_id": pattern_ids[0],
                    "paper_title": "Test Paper: Agent Memory Patterns",
                    "relevance": "supporting",
                },
                {
                    "pattern_id": pattern_ids[1],
                    "paper_title": "Test Paper: Agent Memory Patterns",
                    "relevance": "supporting",
                },
            ],
            "confidence": "high",
            "follow_up_suggestions": [
                "What specific consolidation techniques are most effective?",
                "How does context window size affect hallucination rates?",
            ],
        }

    client.complete_json = complete_json
    return client


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Mock session manager with in-memory session storage."""
    manager = MagicMock()

    sessions: dict[str, ConversationSession] = {}
    session_counter = {"count": 0}

    def create_session():
        session_counter["count"] += 1
        session_id = f"session-{session_counter['count']}"
        now = datetime.now(UTC)
        session = ConversationSession(
            session_id=session_id,
            history=[],
            created_at=now,
            last_accessed=now,
        )
        sessions[session_id] = session
        return session

    def get_session(session_id):
        return sessions.get(session_id)

    def update_session(session_id, turn):
        if session_id in sessions:
            sessions[session_id].history.append(turn)
            sessions[session_id].last_accessed = datetime.now(UTC)
            return True
        return False

    manager.create_session = create_session
    manager.get_session = get_session
    manager.update_session = update_session
    manager._sessions = sessions  # Expose for inspection
    return manager


@pytest.fixture
def mock_pdf_parser() -> MagicMock:
    """Mock PDF parser returning deterministic results."""
    parser = MagicMock()

    def parse(pdf_bytes):
        result = MagicMock()
        result.text = "This is the extracted text content from the PDF."
        result.page_count = 10
        result.metadata = {"title": "PDF Title from Metadata"}
        return result

    parser.parse = parse
    return parser


# --- Integration Client Fixture ---


@pytest.fixture
def integration_client(
    integration_db_session,
    mock_metadata_extractor,
    mock_pattern_extractor,
    mock_embedding_service,
    mock_vector_store,
    mock_llm_client,
    mock_session_manager,
    mock_pdf_parser,
) -> Generator[TestClient, None, None]:
    """Create a test client with all services mocked for integration testing.

    This client allows the full workflow logic to execute while
    mocking external dependencies (LLM, embeddings, Qdrant).
    """

    async def override_get_db():
        yield integration_db_session

    app.dependency_overrides[get_db] = override_get_db

    # Patch service constructors to return our mocks
    with (
        patch(
            "app.workflows.ingestion.MetadataExtractor",
            return_value=mock_metadata_extractor,
        ),
        patch(
            "app.workflows.ingestion.PatternExtractor",
            return_value=mock_pattern_extractor,
        ),
        patch("app.workflows.ingestion.PDFParserService", return_value=mock_pdf_parser),
        patch(
            "app.workflows.review.EmbeddingService",
            return_value=mock_embedding_service,
        ),
        patch(
            "app.workflows.review.VectorStoreService", return_value=mock_vector_store
        ),
        patch(
            "app.workflows.query.EmbeddingService", return_value=mock_embedding_service
        ),
        patch("app.workflows.query.VectorStoreService", return_value=mock_vector_store),
        patch("app.workflows.query.LLMClient", return_value=mock_llm_client),
        patch("app.workflows.query.SessionManager", return_value=mock_session_manager),
        patch("app.routers.papers.VectorStoreService", return_value=mock_vector_store),
        patch(
            "app.routers.patterns.EmbeddingService", return_value=mock_embedding_service
        ),
        patch(
            "app.routers.patterns.VectorStoreService", return_value=mock_vector_store
        ),
    ):
        yield TestClient(app)

    app.dependency_overrides.clear()
