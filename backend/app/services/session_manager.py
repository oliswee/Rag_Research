"""In-memory session manager for conversation history."""

import asyncio
import contextlib
import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta

from app.schemas.session import ConversationSession, ConversationTurn

logger = logging.getLogger(__name__)

# Content truncation limit (100KB) - addresses B-006 from backlog
MAX_CONTENT_LENGTH = 100_000

# Maximum number of active sessions to prevent memory exhaustion
MAX_SESSIONS = 10_000

# Maximum conversation turns per session (sliding window)
MAX_HISTORY_TURNS = 100


class SessionManager:
    """Thread-safe in-memory session storage with TTL-based expiration.

    Sessions expire after 30 minutes of inactivity. A background cleanup
    task removes expired sessions every 5 minutes.

    Usage:
        manager = SessionManager()
        await manager.start_cleanup_task()

        session = manager.create_session()
        manager.update_session(session.session_id, turn)
        session = manager.get_session(session.session_id)

        await manager.stop_cleanup_task()
    """

    TTL_SECONDS = 1800  # 30 minutes
    CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        ttl_seconds: int | None = None,
        cleanup_interval_seconds: int | None = None,
    ):
        """Initialize session manager.

        Args:
            ttl_seconds: Session TTL in seconds. Defaults to 1800 (30 min).
            cleanup_interval_seconds: Cleanup interval. Defaults to 300 (5 min).
        """
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None

        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else self.TTL_SECONDS
        self.cleanup_interval_seconds = (
            cleanup_interval_seconds
            if cleanup_interval_seconds is not None
            else self.CLEANUP_INTERVAL_SECONDS
        )

    def create_session(self) -> ConversationSession:
        """Create a new conversation session.

        Returns:
            New ConversationSession with unique UUID.

        Raises:
            ValueError: If maximum session limit is reached.
        """
        now = datetime.now(UTC)
        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            history=[],
            created_at=now,
            last_accessed=now,
        )

        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise ValueError(
                    f"Maximum session limit ({MAX_SESSIONS}) reached. "
                    "Please try again later."
                )
            self._sessions[session.session_id] = session

        logger.debug("Created session %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Get a session by ID if it exists and is not expired.

        Args:
            session_id: The session ID to look up.

        Returns:
            The session if found and not expired, None otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Check if expired
            if self._is_expired(session):
                del self._sessions[session_id]
                logger.debug("Session %s expired on access", session_id)
                return None

            # Update last_accessed on read
            session.last_accessed = datetime.now(UTC)
            return session

    def update_session(
        self,
        session_id: str,
        turn: ConversationTurn,
    ) -> bool:
        """Add a conversation turn to a session.

        Args:
            session_id: The session ID to update.
            turn: The conversation turn to append.

        Returns:
            True if session was updated, False if session not found/expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            # Check if expired
            if self._is_expired(session):
                del self._sessions[session_id]
                logger.debug("Session %s expired on update", session_id)
                return False

            # Truncate content if too long (addresses B-006)
            if len(turn.content) > MAX_CONTENT_LENGTH:
                logger.warning(
                    "Truncating turn content from %d to %d chars",
                    len(turn.content),
                    MAX_CONTENT_LENGTH,
                )
                turn = ConversationTurn(
                    role=turn.role,
                    content=turn.content[:MAX_CONTENT_LENGTH],
                    sources=turn.sources,
                    timestamp=turn.timestamp,
                )

            # Enforce history length limit with sliding window
            if len(session.history) >= MAX_HISTORY_TURNS:
                session.history.pop(0)
                logger.warning(
                    "Session %s history truncated (max %d turns)",
                    session_id,
                    MAX_HISTORY_TURNS,
                )

            session.history.append(turn)
            session.last_accessed = datetime.now(UTC)

            logger.debug(
                "Updated session %s with %s turn (history: %d)",
                session_id,
                turn.role,
                len(session.history),
            )
            return True

    def set_query_scope(self, session_id: str, scope: dict) -> None:
        """Keep only primitive request constraints, never the ephemeral Web pool."""
        import copy

        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.query_scope = copy.deepcopy(scope)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug("Deleted session %s", session_id)
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed.
        """
        removed = 0
        with self._lock:
            expired_ids = [
                sid
                for sid, session in self._sessions.items()
                if self._is_expired(session)
            ]
            for sid in expired_ids:
                del self._sessions[sid]
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d expired sessions", removed)

        return removed

    def _is_expired(self, session: ConversationSession) -> bool:
        """Check if a session has expired based on last_accessed time."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.ttl_seconds)
        return session.last_accessed < cutoff

    @property
    def session_count(self) -> int:
        """Return current number of active sessions."""
        with self._lock:
            return len(self._sessions)

    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return

        self._shutdown_event = asyncio.Event()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "Started session cleanup task (interval: %ds, TTL: %ds)",
            self.cleanup_interval_seconds,
            self.ttl_seconds,
        )

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task is None:
            return

        if self._shutdown_event:
            self._shutdown_event.set()

        self._cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._cleanup_task

        self._cleanup_task = None
        self._shutdown_event = None
        logger.info("Stopped session cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background loop that periodically cleans up expired sessions."""
        while True:
            try:
                # Wait for interval or shutdown
                if self._shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.cleanup_interval_seconds,
                        )
                        # Shutdown event was set
                        break
                    except TimeoutError:
                        # Timeout expired, do cleanup
                        pass

                self.cleanup_expired()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in cleanup loop: %s", e)
                # Continue running despite errors
                await asyncio.sleep(self.cleanup_interval_seconds)


# Global singleton instance
session_manager = SessionManager()
