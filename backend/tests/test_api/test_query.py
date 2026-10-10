"""Tests for query API routes."""

from unittest.mock import AsyncMock, patch

from fastapi import status

from app.schemas.session import SourceReference
from app.workflows import QueryResult


class TestSubmitQuery:
    """Tests for POST /api/query."""

    def test_query_success_creates_session(self, client_with_db):
        """New query creates session and returns answer."""
        mock_result = QueryResult(
            answer="Based on the patterns...",
            sources=[
                SourceReference(
                    pattern_id="pattern-1",
                    summary="Paper Title",
                    relevance_score=0.92,
                ),
            ],
            confidence="high",
            follow_up_suggestions=["Tell me more about X"],
            session_id="session-123",
            success=True,
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query",
                json={"query": "What patterns exist about agent memory?"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "session-123"
        assert "Based on the patterns" in data["response"]
        assert len(data["sources"]) == 1
        assert data["confidence"] == "high"

    def test_query_with_session_id_continues_session(self, client_with_db):
        """Query with session_id continues existing session."""
        mock_result = QueryResult(
            answer="Continuing the conversation...",
            sources=[],
            confidence="medium",
            follow_up_suggestions=[],
            session_id="existing-session",
            success=True,
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query",
                json={
                    "query": "Tell me more",
                    "session_id": "existing-session",
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "existing-session"

    def test_query_empty_returns_400(self, client_with_db):
        """Empty query returns validation error."""
        response = client_with_db.post(
            "/api/query",
            json={"query": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_workflow_failure_returns_500(self, client_with_db):
        """Workflow failure returns 500."""
        mock_result = QueryResult(
            success=False,
            error="Embedding generation failed",
            session_id="session-1",
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query",
                json={"query": "What about X?"},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_query_no_results_returns_graceful_response(self, client_with_db):
        """Query with no matching patterns returns helpful message."""
        mock_result = QueryResult(
            answer="I couldn't find any patterns...",
            sources=[],
            confidence="low",
            follow_up_suggestions=[],
            session_id="session-456",
            success=True,
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query",
                json={"query": "Something not in library"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["confidence"] == "low"

    def test_query_includes_follow_up_suggestions(self, client_with_db):
        """Query response includes follow-up suggestions."""
        mock_result = QueryResult(
            answer="Here's what I found...",
            sources=[],
            confidence="high",
            follow_up_suggestions=[
                "What are the limitations?",
                "How does this compare to X?",
            ],
            session_id="session-789",
            success=True,
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query",
                json={"query": "Tell me about agent memory"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["follow_up_suggestions"]) == 2


class TestFollowUpQuery:
    """Tests for POST /api/query/followup."""

    def test_followup_with_session_success(self, client_with_db):
        """Follow-up with valid session continues conversation."""
        mock_result = QueryResult(
            answer="Building on our previous discussion...",
            sources=[],
            confidence="medium",
            follow_up_suggestions=[],
            session_id="session-abc",
            success=True,
        )

        with patch("app.routers.query.QueryWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/query/followup",
                json={
                    "query": "Can you elaborate on that?",
                    "session_id": "session-abc",
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "session-abc"

    def test_followup_without_session_id_returns_400(self, client_with_db):
        """Follow-up without session_id returns validation error."""
        response = client_with_db.post(
            "/api/query/followup",
            json={"query": "Tell me more"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_followup_empty_query_returns_400(self, client_with_db):
        """Follow-up with empty query returns validation error."""
        response = client_with_db.post(
            "/api/query/followup",
            json={"query": "", "session_id": "session-123"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
