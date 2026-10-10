"""Tests for EmbeddingService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding_service import (
    BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_CHARS,
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
)


class TestEmbeddingServiceInit:
    """Tests for EmbeddingService initialization."""

    def test_init_with_provided_api_key(self) -> None:
        """Should use provided API key."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            service = EmbeddingService(
                api_key="test-key"  # pragma: allowlist secret
            )
            assert service.api_key == "test-key"  # pragma: allowlist secret

    def test_init_with_settings_api_key(self) -> None:
        """Should fall back to settings API key."""
        with (
            patch("app.services.embedding_service.settings") as mock_settings,
            patch("app.services.embedding_service.openai.AsyncOpenAI"),
        ):
            mock_settings.openai_api_key = "settings-key"  # pragma: allowlist secret
            service = EmbeddingService()
            assert service.api_key == "settings-key"  # pragma: allowlist secret

    def test_init_raises_without_api_key(self) -> None:
        """Should raise error if no API key available."""
        with patch("app.services.embedding_service.settings") as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="OPENAI_API_KEY not configured"):
                EmbeddingService()

    def test_init_uses_default_model(self) -> None:
        """Should use default embedding model."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            service = EmbeddingService(api_key="test-key")  # pragma: allowlist secret
            assert service.model == EMBEDDING_MODEL

    def test_init_accepts_custom_model(self) -> None:
        """Should accept custom model."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            service = EmbeddingService(
                api_key="test-key",  # pragma: allowlist secret
                model="custom-model",
            )
            assert service.model == "custom-model"


class TestPrepareEmbeddingText:
    """Tests for prepare_embedding_text method."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create service with mocked client."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            return EmbeddingService(api_key="test-key")  # pragma: allowlist secret

    def test_combines_all_fields(self, service: EmbeddingService) -> None:
        """Should combine all fields with | separator."""
        pattern = {
            "name": "Test Pattern",
            "claim": "The main claim",
            "evidence": "Supporting evidence",
            "context": "Additional context",
        }
        text = service.prepare_embedding_text(pattern)
        expected = (
            "Test Pattern | The main claim | Supporting evidence | Additional context"
        )
        assert text == expected

    def test_handles_missing_fields(self, service: EmbeddingService) -> None:
        """Should skip missing fields."""
        pattern = {
            "name": "Test Pattern",
            "claim": "The main claim",
        }
        text = service.prepare_embedding_text(pattern)
        assert text == "Test Pattern | The main claim"

    def test_handles_null_fields(self, service: EmbeddingService) -> None:
        """Should skip null fields."""
        pattern = {
            "name": "Test Pattern",
            "claim": None,
            "evidence": "Supporting evidence",
            "context": None,
        }
        text = service.prepare_embedding_text(pattern)
        assert text == "Test Pattern | Supporting evidence"

    def test_handles_empty_string_fields(self, service: EmbeddingService) -> None:
        """Should skip empty string fields."""
        pattern = {
            "name": "Test Pattern",
            "claim": "",
            "evidence": "Supporting evidence",
            "context": "   ",  # whitespace only
        }
        text = service.prepare_embedding_text(pattern)
        assert text == "Test Pattern | Supporting evidence"

    def test_strips_whitespace(self, service: EmbeddingService) -> None:
        """Should strip whitespace from fields."""
        pattern = {
            "name": "  Test Pattern  ",
            "claim": "  The main claim  ",
        }
        text = service.prepare_embedding_text(pattern)
        assert text == "Test Pattern | The main claim"

    def test_truncates_long_text(self, service: EmbeddingService) -> None:
        """Should truncate text exceeding MAX_CHARS."""
        pattern = {
            "name": "Test",
            "claim": "x" * (MAX_CHARS + 100),
        }
        text = service.prepare_embedding_text(pattern)
        assert len(text) == MAX_CHARS

    def test_handles_empty_pattern(self, service: EmbeddingService) -> None:
        """Should return empty string for empty pattern."""
        pattern: dict = {}
        text = service.prepare_embedding_text(pattern)
        assert text == ""

    def test_converts_non_string_values(self, service: EmbeddingService) -> None:
        """Should convert non-string values to strings."""
        pattern = {
            "name": 123,
            "claim": True,
        }
        text = service.prepare_embedding_text(pattern)
        assert text == "123 | True"


class TestGenerateEmbedding:
    """Tests for generate_embedding method."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create mock OpenAI client."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> EmbeddingService:
        """Create service with mocked client."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            svc = EmbeddingService(api_key="test-key")  # pragma: allowlist secret
            svc.client = mock_client
            return svc

    @pytest.mark.asyncio
    async def test_generates_embedding_successfully(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should return EmbeddingResult on success."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
        mock_response.model = EMBEDDING_MODEL
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create.return_value = mock_response

        result = await service.generate_embedding("Test text")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "Test text"
        assert len(result.embedding) == EMBEDDING_DIMENSIONS
        assert result.model == EMBEDDING_MODEL
        assert result.token_count == 10

    @pytest.mark.asyncio
    async def test_raises_for_empty_text(self, service: EmbeddingService) -> None:
        """Should raise EmbeddingError for empty text."""
        with pytest.raises(EmbeddingError, match="empty text"):
            await service.generate_embedding("")

    @pytest.mark.asyncio
    async def test_raises_for_whitespace_text(self, service: EmbeddingService) -> None:
        """Should raise EmbeddingError for whitespace-only text."""
        with pytest.raises(EmbeddingError, match="empty text"):
            await service.generate_embedding("   ")

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should retry on rate limit error."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
        mock_response.model = EMBEDDING_MODEL
        mock_response.usage.total_tokens = 10

        # Fail twice, succeed on third
        mock_client.embeddings.create.side_effect = [
            RateLimitError("rate limited", response=MagicMock(), body=None),
            RateLimitError("rate limited", response=MagicMock(), body=None),
            mock_response,
        ]

        with patch("app.services.embedding_service.asyncio.sleep"):
            result = await service.generate_embedding("Test text")

        assert result.text == "Test text"
        assert mock_client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should retry on connection error."""
        from openai import APIConnectionError

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
        mock_response.model = EMBEDDING_MODEL
        mock_response.usage.total_tokens = 10

        mock_client.embeddings.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_response,
        ]

        with patch("app.services.embedding_service.asyncio.sleep"):
            result = await service.generate_embedding("Test text")

        assert result.text == "Test text"
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_api_error(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should raise immediately on non-retryable API error."""
        from openai import APIError

        mock_client.embeddings.create.side_effect = APIError(
            "bad request", request=MagicMock(), body=None
        )

        with pytest.raises(EmbeddingError, match="OpenAI API error"):
            await service.generate_embedding("Test text")

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should raise after exhausting retries."""
        from openai import RateLimitError

        mock_client.embeddings.create.side_effect = RateLimitError(
            "rate limited", response=MagicMock(), body=None
        )

        with (
            patch("app.services.embedding_service.asyncio.sleep"),
            pytest.raises(EmbeddingError, match="Failed after"),
        ):
            await service.generate_embedding("Test text")


class TestGenerateEmbeddingsBatch:
    """Tests for generate_embeddings_batch method."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create mock OpenAI client."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> EmbeddingService:
        """Create service with mocked client."""
        with patch("app.services.embedding_service.openai.AsyncOpenAI"):
            svc = EmbeddingService(api_key="test-key")  # pragma: allowlist secret
            svc.client = mock_client
            return svc

    @pytest.mark.asyncio
    async def test_processes_patterns_in_batch(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should process patterns in batch."""
        patterns = [
            {"name": "Pattern 1", "claim": "Claim 1"},
            {"name": "Pattern 2", "claim": "Claim 2"},
        ]

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS),
            MagicMock(embedding=[0.2] * EMBEDDING_DIMENSIONS),
        ]
        mock_client.embeddings.create.return_value = mock_response

        results = await service.generate_embeddings_batch(patterns)

        assert len(results) == 2
        assert results[0][0] == patterns[0]
        assert results[1][0] == patterns[1]

    @pytest.mark.asyncio
    async def test_skips_empty_patterns(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should skip patterns with no content."""
        patterns = [
            {"name": "Pattern 1", "claim": "Claim 1"},
            {},  # Empty pattern
            {"name": "Pattern 3", "claim": "Claim 3"},
        ]

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS),
            MagicMock(embedding=[0.3] * EMBEDDING_DIMENSIONS),
        ]
        mock_client.embeddings.create.return_value = mock_response

        results = await service.generate_embeddings_batch(patterns)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_handles_empty_pattern_list(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should handle empty pattern list."""
        results = await service.generate_embeddings_batch([])
        assert results == []
        mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_processes_in_batches(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should process patterns in batches of BATCH_SIZE."""
        # Create more patterns than BATCH_SIZE
        patterns = [{"name": f"Pattern {i}", "claim": f"Claim {i}"} for i in range(25)]

        def create_response(texts: list) -> MagicMock:
            resp = MagicMock()
            resp.data = [
                MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS) for _ in texts
            ]
            return resp

        mock_client.embeddings.create.side_effect = lambda **kwargs: create_response(
            kwargs["input"]
        )

        results = await service.generate_embeddings_batch(patterns)

        assert len(results) == 25
        # Should have made 2 batch calls (20 + 5)
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_individual_on_batch_error(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should fall back to individual embeddings on batch error."""
        from openai import APIError

        patterns = [
            {"name": "Pattern 1", "claim": "Claim 1"},
            {"name": "Pattern 2", "claim": "Claim 2"},
        ]

        # Batch fails, but individual calls succeed
        individual_response = MagicMock()
        individual_response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
        individual_response.model = EMBEDDING_MODEL
        individual_response.usage.total_tokens = 5

        mock_client.embeddings.create.side_effect = [
            APIError("batch failed", request=MagicMock(), body=None),
            individual_response,
            individual_response,
        ]

        results = await service.generate_embeddings_batch(patterns)

        # Should have results from individual fallback
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_skips_failed_individual_embeddings(
        self, service: EmbeddingService, mock_client: AsyncMock
    ) -> None:
        """Should skip patterns that fail individual embedding."""
        from openai import APIError

        patterns = [
            {"name": "Pattern 1", "claim": "Claim 1"},
            {"name": "Pattern 2", "claim": "Claim 2"},
        ]

        individual_response = MagicMock()
        individual_response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
        individual_response.model = EMBEDDING_MODEL
        individual_response.usage.total_tokens = 5

        # Batch fails, first individual succeeds, second fails
        mock_client.embeddings.create.side_effect = [
            APIError("batch failed", request=MagicMock(), body=None),
            individual_response,
            APIError("individual failed", request=MagicMock(), body=None),
        ]

        results = await service.generate_embeddings_batch(patterns)

        # Only first pattern should have embedding
        assert len(results) == 1
        assert results[0][0] == patterns[0]


class TestConstants:
    """Tests for module constants."""

    def test_embedding_model_is_correct(self) -> None:
        """Should use text-embedding-3-small."""
        assert EMBEDDING_MODEL == "text-embedding-3-small"

    def test_embedding_dimensions_is_correct(self) -> None:
        """Should be 1536 dimensions."""
        assert EMBEDDING_DIMENSIONS == 1536

    def test_batch_size_is_reasonable(self) -> None:
        """Batch size should be reasonable."""
        assert 10 <= BATCH_SIZE <= 100

    def test_max_chars_is_reasonable(self) -> None:
        """Max chars should allow ~8K tokens."""
        assert 20_000 <= MAX_CHARS <= 50_000
