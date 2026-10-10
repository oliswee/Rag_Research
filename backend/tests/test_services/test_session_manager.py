"""Tests for SessionManager service."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from app.schemas.session import ConversationSession, ConversationTurn, SourceReference
from app.services.session_manager import (
    MAX_CONTENT_LENGTH,
    MAX_HISTORY_TURNS,
    SessionManager,
)

# --- Fixtures ---


@pytest.fixture
def session_manager():
    """Create a fresh SessionManager for each test."""
    return SessionManager(ttl_seconds=60, cleanup_interval_seconds=1)


@pytest.fixture
def sample_turn():
    """Create a sample conversation turn."""
    return ConversationTurn(
        role="user",
        content="What patterns exist for agent memory?",
        sources=None,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def assistant_turn_with_sources():
    """Create an assistant turn with source references."""
    return ConversationTurn(
        role="assistant",
        content="Based on the patterns in your library...",
        sources=[
            SourceReference(
                pattern_id="pattern-123",
                summary="Memory pattern",
                relevance_score=0.95,
            )
        ],
        timestamp=datetime.now(UTC),
    )


# --- SessionManager Initialization Tests ---


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_default_ttl(self):
        """Uses default TTL of 30 minutes."""
        manager = SessionManager()
        assert manager.ttl_seconds == 1800

    def test_init_default_cleanup_interval(self):
        """Uses default cleanup interval of 5 minutes."""
        manager = SessionManager()
        assert manager.cleanup_interval_seconds == 300

    def test_init_custom_ttl(self):
        """Accepts custom TTL."""
        manager = SessionManager(ttl_seconds=120)
        assert manager.ttl_seconds == 120

    def test_init_custom_cleanup_interval(self):
        """Accepts custom cleanup interval."""
        manager = SessionManager(cleanup_interval_seconds=60)
        assert manager.cleanup_interval_seconds == 60

    def test_init_empty_sessions(self):
        """Initializes with empty sessions dict."""
        manager = SessionManager()
        assert manager.session_count == 0

    def test_init_no_cleanup_task(self):
        """No cleanup task running initially."""
        manager = SessionManager()
        assert manager._cleanup_task is None


# --- SessionManager.create_session() Tests ---


class TestSessionManagerCreateSession:
    """Tests for SessionManager.create_session() method."""

    def test_create_session_returns_session(self, session_manager):
        """Returns a ConversationSession instance."""
        session = session_manager.create_session()
        assert isinstance(session, ConversationSession)

    def test_create_session_generates_uuid(self, session_manager):
        """Generates a valid UUID for session_id."""
        session = session_manager.create_session()
        assert len(session.session_id) == 36  # UUID format

    def test_create_session_empty_history(self, session_manager):
        """Session starts with empty history."""
        session = session_manager.create_session()
        assert session.history == []

    def test_create_session_sets_timestamps(self, session_manager):
        """Sets created_at and last_accessed timestamps."""
        before = datetime.now(UTC)
        session = session_manager.create_session()
        after = datetime.now(UTC)

        assert before <= session.created_at <= after
        assert before <= session.last_accessed <= after

    def test_create_session_increments_count(self, session_manager):
        """Increments session count."""
        assert session_manager.session_count == 0
        session_manager.create_session()
        assert session_manager.session_count == 1
        session_manager.create_session()
        assert session_manager.session_count == 2

    def test_create_session_unique_ids(self, session_manager):
        """Generates unique session IDs."""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()
        assert session1.session_id != session2.session_id


# --- SessionManager.get_session() Tests ---


class TestSessionManagerGetSession:
    """Tests for SessionManager.get_session() method."""

    def test_get_session_returns_session(self, session_manager):
        """Returns existing session by ID."""
        created = session_manager.create_session()
        retrieved = session_manager.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_session_returns_none_for_unknown(self, session_manager):
        """Returns None for unknown session ID."""
        result = session_manager.get_session("nonexistent-id")
        assert result is None

    def test_get_session_updates_last_accessed(self, session_manager):
        """Updates last_accessed on retrieval."""
        session = session_manager.create_session()
        original_time = session.last_accessed

        # Wait a tiny bit and retrieve
        import time

        time.sleep(0.01)

        retrieved = session_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.last_accessed > original_time

    def test_get_session_removes_expired(self, session_manager):
        """Removes and returns None for expired sessions."""
        session = session_manager.create_session()

        # Manually expire the session
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=120)

        result = session_manager.get_session(session.session_id)
        assert result is None
        assert session_manager.session_count == 0


# --- SessionManager.update_session() Tests ---


class TestSessionManagerUpdateSession:
    """Tests for SessionManager.update_session() method."""

    def test_update_session_adds_turn(self, session_manager, sample_turn):
        """Adds turn to session history."""
        session = session_manager.create_session()
        result = session_manager.update_session(session.session_id, sample_turn)

        assert result is True
        updated = session_manager.get_session(session.session_id)
        assert len(updated.history) == 1
        assert updated.history[0].content == sample_turn.content

    def test_update_session_returns_false_for_unknown(
        self, session_manager, sample_turn
    ):
        """Returns False for unknown session ID."""
        result = session_manager.update_session("nonexistent-id", sample_turn)
        assert result is False

    def test_update_session_returns_false_for_expired(
        self, session_manager, sample_turn
    ):
        """Returns False for expired session."""
        session = session_manager.create_session()
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=120)

        result = session_manager.update_session(session.session_id, sample_turn)
        assert result is False

    def test_update_session_updates_last_accessed(self, session_manager, sample_turn):
        """Updates last_accessed on update."""
        session = session_manager.create_session()
        original_time = session.last_accessed

        import time

        time.sleep(0.01)

        session_manager.update_session(session.session_id, sample_turn)
        updated = session_manager.get_session(session.session_id)
        assert updated.last_accessed > original_time

    def test_update_session_preserves_turn_data(
        self, session_manager, assistant_turn_with_sources
    ):
        """Preserves all turn data including sources."""
        session = session_manager.create_session()
        session_manager.update_session(session.session_id, assistant_turn_with_sources)

        updated = session_manager.get_session(session.session_id)
        turn = updated.history[0]
        assert turn.role == "assistant"
        assert turn.sources is not None
        assert len(turn.sources) == 1
        assert turn.sources[0].pattern_id == "pattern-123"


# --- SessionManager Content Truncation Tests ---


class TestSessionManagerContentTruncation:
    """Tests for content truncation (B-006)."""

    def test_truncates_oversized_content(self, session_manager):
        """Truncates content exceeding MAX_CONTENT_LENGTH."""
        oversized_content = "x" * (MAX_CONTENT_LENGTH + 1000)
        turn = ConversationTurn(
            role="user",
            content=oversized_content,
            sources=None,
            timestamp=datetime.now(UTC),
        )

        session = session_manager.create_session()
        session_manager.update_session(session.session_id, turn)

        updated = session_manager.get_session(session.session_id)
        assert len(updated.history[0].content) == MAX_CONTENT_LENGTH

    def test_preserves_normal_content(self, session_manager, sample_turn):
        """Preserves content within limit."""
        session = session_manager.create_session()
        session_manager.update_session(session.session_id, sample_turn)

        updated = session_manager.get_session(session.session_id)
        assert updated.history[0].content == sample_turn.content


# --- SessionManager History Limit Tests ---


class TestSessionManagerHistoryLimit:
    """Tests for history length limit (sliding window)."""

    def test_enforces_max_history_turns(self, session_manager):
        """Enforces MAX_HISTORY_TURNS limit with sliding window."""
        session = session_manager.create_session()

        # Add MAX_HISTORY_TURNS + 1 turns
        for i in range(MAX_HISTORY_TURNS + 5):
            turn = ConversationTurn(
                role="user",
                content=f"Message {i}",
                sources=None,
                timestamp=datetime.now(UTC),
            )
            session_manager.update_session(session.session_id, turn)

        updated = session_manager.get_session(session.session_id)
        assert len(updated.history) == MAX_HISTORY_TURNS

    def test_sliding_window_removes_oldest(self, session_manager):
        """Sliding window removes oldest turns first."""
        session = session_manager.create_session()

        # Fill to max + 2
        for i in range(MAX_HISTORY_TURNS + 2):
            turn = ConversationTurn(
                role="user",
                content=f"Message {i}",
                sources=None,
                timestamp=datetime.now(UTC),
            )
            session_manager.update_session(session.session_id, turn)

        updated = session_manager.get_session(session.session_id)
        # First message should be Message 2 (0 and 1 were removed)
        assert updated.history[0].content == "Message 2"


# --- SessionManager.delete_session() Tests ---


class TestSessionManagerDeleteSession:
    """Tests for SessionManager.delete_session() method."""

    def test_delete_session_removes_session(self, session_manager):
        """Deletes session by ID."""
        session = session_manager.create_session()
        result = session_manager.delete_session(session.session_id)

        assert result is True
        assert session_manager.get_session(session.session_id) is None

    def test_delete_session_returns_false_for_unknown(self, session_manager):
        """Returns False for unknown session ID."""
        result = session_manager.delete_session("nonexistent-id")
        assert result is False

    def test_delete_session_decrements_count(self, session_manager):
        """Decrements session count."""
        session = session_manager.create_session()
        assert session_manager.session_count == 1

        session_manager.delete_session(session.session_id)
        assert session_manager.session_count == 0


# --- SessionManager.cleanup_expired() Tests ---


class TestSessionManagerCleanupExpired:
    """Tests for SessionManager.cleanup_expired() method."""

    def test_cleanup_removes_expired_sessions(self, session_manager):
        """Removes expired sessions."""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()

        # Expire session1
        session1.last_accessed = datetime.now(UTC) - timedelta(seconds=120)

        removed = session_manager.cleanup_expired()

        assert removed == 1
        assert session_manager.get_session(session1.session_id) is None
        assert session_manager.get_session(session2.session_id) is not None

    def test_cleanup_returns_count(self, session_manager):
        """Returns count of removed sessions."""
        for _ in range(3):
            session = session_manager.create_session()
            session.last_accessed = datetime.now(UTC) - timedelta(seconds=120)

        removed = session_manager.cleanup_expired()
        assert removed == 3

    def test_cleanup_preserves_active_sessions(self, session_manager):
        """Preserves sessions that are not expired."""
        session_manager.create_session()  # Create session to test preservation

        removed = session_manager.cleanup_expired()

        assert removed == 0
        assert session_manager.session_count == 1


# --- SessionManager Session Limit Tests ---


class TestSessionManagerSessionLimit:
    """Tests for maximum session limit."""

    def test_raises_on_max_sessions(self):
        """Raises ValueError when MAX_SESSIONS reached."""
        manager = SessionManager()

        # Patch MAX_SESSIONS for testing
        with patch("app.services.session_manager.MAX_SESSIONS", 3):
            manager.create_session()
            manager.create_session()
            manager.create_session()

            with pytest.raises(ValueError) as exc_info:
                manager.create_session()

            assert "Maximum session limit" in str(exc_info.value)


# --- SessionManager.session_count Tests ---


class TestSessionManagerSessionCount:
    """Tests for SessionManager.session_count property."""

    def test_session_count_starts_at_zero(self, session_manager):
        """Starts at zero."""
        assert session_manager.session_count == 0

    def test_session_count_reflects_sessions(self, session_manager):
        """Reflects actual session count."""
        session_manager.create_session()
        session_manager.create_session()
        assert session_manager.session_count == 2


# --- SessionManager Cleanup Task Tests ---


class TestSessionManagerCleanupTask:
    """Tests for background cleanup task."""

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self, session_manager):
        """Starts background cleanup task."""
        await session_manager.start_cleanup_task()

        assert session_manager._cleanup_task is not None
        assert not session_manager._cleanup_task.done()

        await session_manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self, session_manager):
        """Stops background cleanup task."""
        await session_manager.start_cleanup_task()
        await session_manager.stop_cleanup_task()

        assert session_manager._cleanup_task is None

    @pytest.mark.asyncio
    async def test_start_cleanup_task_idempotent(self, session_manager):
        """Starting multiple times doesn't create multiple tasks."""
        await session_manager.start_cleanup_task()
        task1 = session_manager._cleanup_task

        await session_manager.start_cleanup_task()
        task2 = session_manager._cleanup_task

        assert task1 is task2

        await session_manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_cleanup_task_runs_periodically(self):
        """Cleanup task runs at configured interval."""
        manager = SessionManager(ttl_seconds=1, cleanup_interval_seconds=0.1)

        # Create and expire a session
        session = manager.create_session()
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=5)

        await manager.start_cleanup_task()

        # Wait for cleanup to run
        await asyncio.sleep(0.2)

        assert manager.session_count == 0

        await manager.stop_cleanup_task()


# --- Schema Tests ---


class TestConversationTurnSchema:
    """Tests for ConversationTurn schema."""

    def test_turn_requires_role(self):
        """Role is required."""
        with pytest.raises(ValueError):
            ConversationTurn(
                content="test",
                sources=None,
                timestamp=datetime.now(UTC),
            )

    def test_turn_requires_content(self):
        """Content is required."""
        with pytest.raises(ValueError):
            ConversationTurn(
                role="user",
                sources=None,
                timestamp=datetime.now(UTC),
            )

    def test_turn_requires_timestamp(self):
        """Timestamp is required."""
        with pytest.raises(ValueError):
            ConversationTurn(
                role="user",
                content="test",
                sources=None,
            )

    def test_turn_role_validation(self):
        """Role must be 'user' or 'assistant'."""
        with pytest.raises(ValueError):
            ConversationTurn(
                role="system",
                content="test",
                sources=None,
                timestamp=datetime.now(UTC),
            )

    def test_turn_accepts_valid_roles(self):
        """Accepts 'user' and 'assistant' roles."""
        user_turn = ConversationTurn(
            role="user",
            content="test",
            sources=None,
            timestamp=datetime.now(UTC),
        )
        assert user_turn.role == "user"

        assistant_turn = ConversationTurn(
            role="assistant",
            content="test",
            sources=None,
            timestamp=datetime.now(UTC),
        )
        assert assistant_turn.role == "assistant"


class TestSourceReferenceSchema:
    """Tests for SourceReference schema."""

    def test_source_requires_pattern_id(self):
        """Pattern ID is required."""
        with pytest.raises(ValueError):
            SourceReference(summary="test", relevance_score=0.5)

    def test_source_optional_summary(self):
        """Summary is optional."""
        source = SourceReference(pattern_id="test-id")
        assert source.summary is None

    def test_source_optional_relevance_score(self):
        """Relevance score is optional."""
        source = SourceReference(pattern_id="test-id")
        assert source.relevance_score is None

    def test_source_relevance_score_bounds(self):
        """Relevance score must be between 0 and 1."""
        with pytest.raises(ValueError):
            SourceReference(pattern_id="test-id", relevance_score=1.5)

        with pytest.raises(ValueError):
            SourceReference(pattern_id="test-id", relevance_score=-0.1)

        # Valid scores
        source = SourceReference(pattern_id="test-id", relevance_score=0.0)
        assert source.relevance_score == 0.0

        source = SourceReference(pattern_id="test-id", relevance_score=1.0)
        assert source.relevance_score == 1.0


class TestConversationSessionSchema:
    """Tests for ConversationSession schema."""

    def test_session_requires_session_id(self):
        """Session ID is required."""
        with pytest.raises(ValueError):
            ConversationSession(
                history=[],
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
            )

    def test_session_default_empty_history(self):
        """History defaults to empty list."""
        session = ConversationSession(
            session_id="test-id",
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
        )
        assert session.history == []
