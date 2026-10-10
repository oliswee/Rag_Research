"""Tests for query workflow."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.session import ConversationSession
from app.workflows.query import QueryInput, QueryResult, QueryWorkflow


class TestQueryWorkflow:
    """Tests for QueryWorkflow."""

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def mock_session_manager(self) -> MagicMock:
        """Create mock session manager."""
        manager = MagicMock()
        now = datetime.now(UTC)
        manager.create_session.return_value = ConversationSession(
            session_id="session-123",
            history=[],
            created_at=now,
            last_accessed=now,
        )
        manager.get_session.return_value = ConversationSession(
            session_id="session-123",
            history=[],
            created_at=now,
            last_accessed=now,
        )
        manager.update_session.return_value = True
        return manager

    @pytest.fixture
    def mock_embedding_service(self) -> MagicMock:
        """Create mock embedding service."""
        service = MagicMock()
        service.generate_embedding = AsyncMock(
            return_value=MagicMock(embedding=[0.1] * 1536)
        )
        return service

    @pytest.fixture
    def mock_vector_store(self) -> AsyncMock:
        """Create mock vector store service."""
        store = AsyncMock()
        # Return mock search results as dicts (matching real VectorStoreService.search)
        store.search.return_value = [
            {"id": "pattern-123", "score": 0.95, "payload": {}}
        ]
        return store

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        """Create mock LLM client."""
        client = MagicMock()
        client.complete_json = AsyncMock(
            return_value={
                "answer": "Here is the answer [P:pattern-123]",
                "sources": [
                    {
                        "pattern_id": "pattern-123",
                        "paper_title": "Test Paper",
                        "relevance": "supporting",
                    }
                ],
                "confidence": "high",
                "follow_up_suggestions": ["What about X?"],
            }
        )
        return client

    @pytest.fixture
    def mock_pattern(self) -> MagicMock:
        """Create mock pattern."""
        pattern = MagicMock()
        pattern.id = "pattern-123"
        pattern.paper_id = "paper-123"
        pattern.name = "Test Pattern"
        pattern.claim = "Test claim"
        pattern.evidence = "Test evidence"
        pattern.context = "Test context"
        return pattern

    @pytest.fixture
    def mock_paper(self) -> MagicMock:
        """Create mock paper."""
        paper = MagicMock()
        paper.id = "paper-123"
        paper.title = "Test Paper"
        return paper

    @pytest.fixture
    def workflow(
        self,
        mock_db: AsyncMock,
        mock_session_manager: MagicMock,
        mock_embedding_service: MagicMock,
        mock_vector_store: AsyncMock,
        mock_llm_client: MagicMock,
    ) -> QueryWorkflow:
        """Create workflow with mocked dependencies."""
        return QueryWorkflow(
            db=mock_db,
            session_manager=mock_session_manager,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )

    @pytest.mark.asyncio
    async def test_new_session_creates_session(
        self,
        workflow: QueryWorkflow,
        mock_session_manager: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should create new session when no session_id provided."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(QueryInput(query="test query"))

        mock_session_manager.create_session.assert_called_once()
        assert result.session_id == "session-123"

    @pytest.mark.asyncio
    async def test_existing_session_uses_session(
        self,
        workflow: QueryWorkflow,
        mock_session_manager: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should use existing session when session_id provided."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(
            QueryInput(query="test query", session_id="session-123")
        )

        mock_session_manager.get_session.assert_called_with("session-123")
        assert result.session_id == "session-123"

    @pytest.mark.asyncio
    async def test_expired_session_creates_new(
        self,
        workflow: QueryWorkflow,
        mock_session_manager: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should create new session if provided session expired."""
        mock_session_manager.get_session.return_value = None  # Session not found
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(
            QueryInput(query="test query", session_id="expired-session")
        )

        mock_session_manager.create_session.assert_called_once()
        assert result.session_id == "session-123"

    @pytest.mark.asyncio
    async def test_generates_query_embedding(
        self,
        workflow: QueryWorkflow,
        mock_embedding_service: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should generate embedding for query."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        await workflow.run(QueryInput(query="test query"))

        mock_embedding_service.generate_embedding.assert_called_with("test query")

    @pytest.mark.asyncio
    async def test_searches_vector_store(
        self,
        workflow: QueryWorkflow,
        mock_vector_store: AsyncMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should search vector store with query embedding."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        await workflow.run(QueryInput(query="test query", limit=5))

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_synthesizes_answer(
        self,
        workflow: QueryWorkflow,
        mock_llm_client: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should synthesize answer using LLM."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(QueryInput(query="test query"))

        mock_llm_client.complete_json.assert_called_once()
        assert "Here is the answer" in result.answer

    @pytest.mark.asyncio
    async def test_updates_session_with_turns(
        self,
        workflow: QueryWorkflow,
        mock_session_manager: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should update session with user and assistant turns."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        await workflow.run(QueryInput(query="test query"))

        # Should be called twice: user turn and assistant turn
        assert mock_session_manager.update_session.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_sources(
        self,
        workflow: QueryWorkflow,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should return sources in result."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(QueryInput(query="test query"))

        assert len(result.sources) == 1
        assert result.sources[0].pattern_id == "pattern-123"

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_error(
        self,
        workflow: QueryWorkflow,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Should return error if embedding fails."""
        mock_embedding_service.generate_embedding.side_effect = Exception("API error")

        result = await workflow.run(QueryInput(query="test query"))

        assert result.success is False
        assert "embedding" in result.error.lower()

    @pytest.mark.asyncio
    async def test_vector_search_failure_returns_error(
        self,
        workflow: QueryWorkflow,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should return error if vector search fails."""
        mock_vector_store.search.side_effect = Exception("Connection error")

        result = await workflow.run(QueryInput(query="test query"))

        assert result.success is False
        assert "search failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_error(
        self,
        workflow: QueryWorkflow,
        mock_llm_client: MagicMock,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should return error if synthesis fails."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )
        mock_llm_client.complete_json.side_effect = Exception("LLM error")

        result = await workflow.run(QueryInput(query="test query"))

        assert result.success is False
        assert "synthesis failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_results_returns_helpful_message(
        self,
        workflow: QueryWorkflow,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should return helpful message when no patterns found."""
        mock_vector_store.search.return_value = []

        result = await workflow.run(QueryInput(query="test query"))

        assert result.success is True
        assert "couldn't find any patterns" in result.answer.lower()
        assert result.confidence == "low"

    @pytest.mark.asyncio
    async def test_includes_follow_up_suggestions(
        self,
        workflow: QueryWorkflow,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should include follow-up suggestions in result."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(QueryInput(query="test query"))

        assert len(result.follow_up_suggestions) > 0

    @pytest.mark.asyncio
    async def test_includes_confidence(
        self,
        workflow: QueryWorkflow,
        mock_pattern: MagicMock,
        mock_paper: MagicMock,
    ) -> None:
        """Should include confidence in result."""
        workflow._get_patterns_with_papers = AsyncMock(
            return_value=[(mock_pattern, mock_paper)]
        )

        result = await workflow.run(QueryInput(query="test query"))

        assert result.confidence == "high"


class TestQueryInput:
    """Tests for QueryInput dataclass."""

    def test_creates_with_query_only(self) -> None:
        """Should create with just query."""
        input_data = QueryInput(query="test query")
        assert input_data.query == "test query"
        assert input_data.session_id is None
        assert input_data.limit == 10

    def test_creates_with_all_fields(self) -> None:
        """Should accept all optional fields."""
        input_data = QueryInput(
            query="test query",
            session_id="session-123",
            limit=20,
        )
        assert input_data.session_id == "session-123"
        assert input_data.limit == 20


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        result = QueryResult()
        assert result.answer == ""
        assert result.sources == []
        assert result.confidence == "medium"
        assert result.follow_up_suggestions == []
        assert result.session_id == ""
        assert result.success is True
        assert result.error is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        result = QueryResult(
            answer="Test answer",
            confidence="high",
            session_id="session-123",
            success=True,
        )
        assert result.answer == "Test answer"
        assert result.confidence == "high"
        assert result.session_id == "session-123"
