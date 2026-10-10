"""Tests for VectorStoreService."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from app.services.vector_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    VectorStoreService,
)

# --- Fixtures ---


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    with patch("app.services.vector_store.QdrantClient") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def vector_store(mock_qdrant_client):
    """Create a VectorStoreService with mocked client."""
    return VectorStoreService(host="localhost", port=6333)


@pytest.fixture
def sample_vector():
    """Create a sample 1536-dimensional vector."""
    return [0.1] * VECTOR_SIZE


@pytest.fixture
def sample_payload():
    """Create a sample pattern payload."""
    return {
        "pattern_id": "test-pattern-123",
        "paper_id": "test-paper-456",
        "paper_type": "research_paper",
        "summary": "Test pattern summary",
    }


# --- VectorStoreService Initialization Tests ---


class TestVectorStoreServiceInit:
    """Tests for VectorStoreService initialization."""

    def test_init_creates_client_with_host_port(self, mock_qdrant_client):
        """Service creates Qdrant client with provided host and port."""
        with patch("app.services.vector_store.QdrantClient") as mock_class:
            VectorStoreService(host="qdrant.example.com", port=6334)
            mock_class.assert_called_once_with(host="qdrant.example.com", port=6334)

    def test_init_sets_collection_name(self, vector_store):
        """Service initializes with correct collection name."""
        assert vector_store._collection_name == COLLECTION_NAME

    def test_init_stores_client_reference(self, vector_store, mock_qdrant_client):
        """Service stores reference to Qdrant client."""
        assert vector_store.client is mock_qdrant_client


# --- VectorStoreService.ensure_collection() Tests ---


class TestVectorStoreServiceEnsureCollection:
    """Tests for VectorStoreService.ensure_collection() method."""

    def test_ensure_collection_creates_when_not_exists(
        self, vector_store, mock_qdrant_client
    ):
        """Creates collection when it doesn't exist."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        result = vector_store.ensure_collection()

        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

    def test_ensure_collection_skips_when_exists(
        self, vector_store, mock_qdrant_client
    ):
        """Skips creation when collection already exists."""
        existing_collection = MagicMock()
        existing_collection.name = COLLECTION_NAME
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        result = vector_store.ensure_collection()

        assert result is True
        mock_qdrant_client.create_collection.assert_not_called()

    def test_ensure_collection_uses_correct_vector_params(
        self, vector_store, mock_qdrant_client
    ):
        """Creates collection with correct vector parameters."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        vector_store.ensure_collection()

        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION_NAME
        vectors_config = call_kwargs["vectors_config"]
        assert vectors_config.size == VECTOR_SIZE

    def test_ensure_collection_returns_false_on_error(
        self, vector_store, mock_qdrant_client
    ):
        """Returns False when Qdrant operations fail."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

        result = vector_store.ensure_collection()

        assert result is False


# --- VectorStoreService.health_check() Tests ---


class TestVectorStoreServiceHealthCheck:
    """Tests for VectorStoreService.health_check() method."""

    def test_health_check_returns_true_when_healthy(
        self, vector_store, mock_qdrant_client
    ):
        """Returns True when Qdrant is accessible."""
        mock_qdrant_client.get_collections.return_value = MagicMock()

        result = vector_store.health_check()

        assert result is True

    def test_health_check_returns_false_on_connection_error(
        self, vector_store, mock_qdrant_client
    ):
        """Returns False when Qdrant is not accessible."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection refused")

        result = vector_store.health_check()

        assert result is False

    def test_health_check_calls_get_collections(self, vector_store, mock_qdrant_client):
        """Calls get_collections to verify connectivity."""
        vector_store.health_check()

        mock_qdrant_client.get_collections.assert_called_once()


# --- VectorStoreService.upsert_pattern() Tests ---


class TestVectorStoreServiceUpsertPattern:
    """Tests for VectorStoreService.upsert_pattern() method."""

    def test_upsert_pattern_calls_upsert(
        self, vector_store, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Calls Qdrant upsert with correct parameters."""
        vector_store.upsert_pattern(
            pattern_id="pattern-123",
            vector=sample_vector,
            payload=sample_payload,
        )

        mock_qdrant_client.upsert.assert_called_once()

    def test_upsert_pattern_uses_correct_collection(
        self, vector_store, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Upserts to the patterns collection."""
        vector_store.upsert_pattern(
            pattern_id="pattern-123",
            vector=sample_vector,
            payload=sample_payload,
        )

        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    def test_upsert_pattern_creates_point_struct(
        self, vector_store, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Creates PointStruct with pattern_id, vector, and payload."""
        vector_store.upsert_pattern(
            pattern_id="pattern-123",
            vector=sample_vector,
            payload=sample_payload,
        )

        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        points = call_kwargs["points"]
        assert len(points) == 1
        point = points[0]
        assert point.id == "pattern-123"
        assert point.vector == sample_vector
        assert point.payload == sample_payload


# --- VectorStoreService.delete_pattern() Tests ---


class TestVectorStoreServiceDeletePattern:
    """Tests for VectorStoreService.delete_pattern() method."""

    def test_delete_pattern_calls_delete(self, vector_store, mock_qdrant_client):
        """Calls Qdrant delete with correct parameters."""
        vector_store.delete_pattern("pattern-123")

        mock_qdrant_client.delete.assert_called_once()

    def test_delete_pattern_uses_correct_collection(
        self, vector_store, mock_qdrant_client
    ):
        """Deletes from the patterns collection."""
        vector_store.delete_pattern("pattern-123")

        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    def test_delete_pattern_passes_pattern_id(self, vector_store, mock_qdrant_client):
        """Passes pattern_id as points selector."""
        vector_store.delete_pattern("pattern-xyz")

        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["points_selector"] == ["pattern-xyz"]


# --- VectorStoreService.search() Tests ---


class TestVectorStoreServiceSearch:
    """Tests for VectorStoreService.search() method.

    Note: Tests use _search_sync directly since the async search() wrapper
    just delegates to asyncio.to_thread(_search_sync, ...).
    """

    def test_search_calls_query_points(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Calls Qdrant query_points."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        vector_store._search_sync(query_vector=sample_vector)

        mock_qdrant_client.query_points.assert_called_once()

    def test_search_uses_correct_collection(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Searches in the patterns collection."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        vector_store._search_sync(query_vector=sample_vector)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    def test_search_passes_query_vector(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Passes query vector to Qdrant."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        vector_store._search_sync(query_vector=sample_vector)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query"] == sample_vector

    def test_search_uses_default_limit(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Uses default limit of 10."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        vector_store._search_sync(query_vector=sample_vector)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_search_accepts_custom_limit(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Accepts custom limit parameter."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        vector_store._search_sync(query_vector=sample_vector, limit=5)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_search_passes_filter_conditions(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Passes filter conditions to Qdrant."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])
        filters = {"paper_type": "research_paper"}

        vector_store._search_sync(query_vector=sample_vector, filter_conditions=filters)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] == filters

    def test_search_returns_formatted_results(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Returns list of dicts with id, score, and payload."""
        hit1 = MagicMock()
        hit1.id = "pattern-1"
        hit1.score = 0.95
        hit1.payload = {"summary": "Pattern 1"}

        hit2 = MagicMock()
        hit2.id = "pattern-2"
        hit2.score = 0.87
        hit2.payload = {"summary": "Pattern 2"}

        mock_qdrant_client.query_points.return_value = MagicMock(points=[hit1, hit2])

        results = vector_store._search_sync(query_vector=sample_vector)

        assert len(results) == 2
        assert results[0] == {
            "id": "pattern-1",
            "score": 0.95,
            "payload": {"summary": "Pattern 1"},
        }
        assert results[1] == {
            "id": "pattern-2",
            "score": 0.87,
            "payload": {"summary": "Pattern 2"},
        }

    def test_search_returns_empty_list_when_no_results(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Returns empty list when no patterns match."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        results = vector_store._search_sync(query_vector=sample_vector)

        assert results == []

    async def test_async_search_delegates_to_sync(
        self, vector_store, mock_qdrant_client, sample_vector
    ):
        """Async search() delegates to _search_sync via asyncio.to_thread."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        results = await vector_store.search(query_vector=sample_vector)

        assert results == []
        mock_qdrant_client.query_points.assert_called_once()


# --- VectorStoreService.get_collection_info() Tests ---


class TestVectorStoreServiceGetCollectionInfo:
    """Tests for VectorStoreService.get_collection_info() method."""

    def test_get_collection_info_returns_stats(self, vector_store, mock_qdrant_client):
        """Returns collection statistics."""
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 50
        mock_qdrant_client.get_collection.return_value = mock_info

        result = vector_store.get_collection_info()

        assert result == {
            "name": COLLECTION_NAME,
            "vectors_count": 100,
            "points_count": 50,
        }

    def test_get_collection_info_calls_get_collection(
        self, vector_store, mock_qdrant_client
    ):
        """Calls get_collection with correct collection name."""
        mock_qdrant_client.get_collection.return_value = MagicMock(
            vectors_count=0, points_count=0
        )

        vector_store.get_collection_info()

        mock_qdrant_client.get_collection.assert_called_once_with(COLLECTION_NAME)

    def test_get_collection_info_returns_none_on_unexpected_response(
        self, vector_store, mock_qdrant_client
    ):
        """Returns None when collection doesn't exist or Qdrant returns error."""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"Collection not found",
            headers={},
        )

        result = vector_store.get_collection_info()

        assert result is None


# --- Constants Tests ---


class TestVectorStoreConstants:
    """Tests for VectorStoreService constants."""

    def test_collection_name_is_patterns(self):
        """Collection name should be 'patterns'."""
        assert COLLECTION_NAME == "patterns"

    def test_vector_size_is_1536(self):
        """Vector size should be 1536 (OpenAI text-embedding-3-small)."""
        assert VECTOR_SIZE == 1536
