"""Tests for health check endpoint."""

from unittest.mock import AsyncMock, patch


def test_health_endpoint_returns_status(client):
    """Health endpoint should return a valid response structure."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "qdrant" in data
    assert "sessions" in data
    assert "tracing" in data


def test_health_endpoint_tracing_disabled_by_default(client):
    """Tracing should be disabled when not configured."""
    response = client.get("/health")
    data = response.json()
    assert data["tracing"] == "disabled"


def test_health_endpoint_qdrant_status(client):
    """Qdrant should show status based on connection check."""
    response = client.get("/health")
    data = response.json()
    # Qdrant status depends on whether Qdrant is running
    assert data["qdrant"] in ["ok", "error"]


def test_health_endpoint_sessions_zero(client):
    """Sessions should be 0 until SessionManager is implemented."""
    response = client.get("/health")
    data = response.json()
    assert data["sessions"] == 0


def test_health_endpoint_database_status(client):
    """Database status should be either 'ok' or 'error'."""
    response = client.get("/health")
    data = response.json()
    # Database status depends on whether connection succeeds
    assert data["database"] in ["ok", "error"]


def test_health_endpoint_status_logic(client):
    """Overall status reflects combined health of database and Qdrant."""
    response = client.get("/health")
    data = response.json()
    # Status should be:
    # - "ok" if both database and Qdrant are healthy
    # - "degraded" if database OK but Qdrant fails
    # - "error" if database fails
    if data["database"] == "error":
        assert data["status"] == "error"
    elif data["database"] == "ok" and data["qdrant"] == "ok":
        assert data["status"] == "ok"
    else:
        assert data["status"] == "degraded"


def test_health_endpoint_database_failure(client):
    """Health endpoint should return error status when database connection fails."""
    # Create a mock session maker that raises an exception
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
    mock_session.__aexit__ = AsyncMock()

    mock_session_maker = AsyncMock(return_value=mock_session)

    with patch("app.main.async_session_maker", mock_session_maker):
        response = client.get("/health")
        data = response.json()

        assert data["database"] == "error"
        assert data["status"] == "error"


def test_health_endpoint_database_query_failure(client):
    """Health endpoint should handle query execution failures gracefully."""
    # Create a mock that succeeds on context entry but fails on execute
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=Exception("Query failed"))

    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_session_maker = AsyncMock(return_value=mock_context)

    with patch("app.main.async_session_maker", mock_session_maker):
        response = client.get("/health")
        data = response.json()

        assert data["database"] == "error"
        assert data["status"] == "error"
