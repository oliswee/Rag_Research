"""Tests for LLM client service."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from anthropic import APIConnectionError, APIError, RateLimitError

from app.config import ModelID
from app.services.llm_client import LLMClient, LLMError, LLMResponse

# --- LLMClient Initialization Tests ---


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_init_requires_api_key(self):
        """Client raises error without API key."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            with pytest.raises(ValueError) as exc_info:
                LLMClient()

            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_init_none_api_key_raises_error(self):
        """Client raises error when API key is None."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            with pytest.raises(ValueError):
                LLMClient()

    def test_init_uses_settings_defaults(self, mock_anthropic_client):
        """Client uses settings for defaults."""
        mock_instance, _ = mock_anthropic_client

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()

            assert client.api_key == "test-key"  # pragma: allowlist secret
            assert client.model == "claude-sonnet-4-20250514"
            assert client.max_retries == 3
            assert client.timeout_seconds == 120

    def test_init_accepts_custom_parameters(self, mock_anthropic_client):
        """Client accepts custom initialization parameters."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "default-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient(
                api_key="custom-key",  # pragma: allowlist secret
                model=ModelID.HAIKU_3_5,
                max_retries=5,
                timeout_seconds=60,
            )

            assert client.api_key == "custom-key"  # pragma: allowlist secret
            assert client.model == ModelID.HAIKU_3_5
            assert client.max_retries == 5
            assert client.timeout_seconds == 60

    def test_init_validates_model_id(self, mock_anthropic_client):
        """Client validates model ID on initialization."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            with pytest.raises(ValueError, match="Unknown model ID"):
                LLMClient(model="invalid-model-id")


# --- LLMClient.complete() Tests ---


class TestLLMClientComplete:
    """Tests for LLMClient.complete() method."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, mock_anthropic_client):
        """Complete returns LLMResponse with content."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory("Hello, world!")
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            response = await client.complete(
                system_prompt="You are helpful.",
                user_prompt="Say hello.",
            )

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello, world!"
            assert response.input_tokens == 100
            assert response.output_tokens == 50

    @pytest.mark.asyncio
    async def test_complete_passes_correct_parameters(self, mock_anthropic_client):
        """Complete passes correct parameters to API."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory("Response")
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            await client.complete(
                system_prompt="System message",
                user_prompt="User message",
                max_tokens=2048,
                temperature=0.5,
            )

            call_kwargs = mock_instance.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_kwargs["max_tokens"] == 2048
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["system"] == "System message"
            assert call_kwargs["messages"] == [
                {"role": "user", "content": "User message"}
            ]

    @pytest.mark.asyncio
    async def test_complete_uses_default_parameters(self, mock_anthropic_client):
        """Complete uses default max_tokens and temperature."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory("Response")
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            await client.complete(
                system_prompt="Test",
                user_prompt="Test",
            )

            call_kwargs = mock_instance.messages.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096  # Default
            assert call_kwargs["temperature"] == 0.0  # Default


# --- LLMClient.complete() Retry Tests ---


class TestLLMClientRetry:
    """Tests for LLMClient retry logic."""

    @pytest.mark.asyncio
    async def test_complete_retries_on_rate_limit(self, mock_anthropic_client):
        """Complete retries on RateLimitError."""
        mock_instance, mock_response_factory = mock_anthropic_client

        # First call raises rate limit, second succeeds
        mock_instance.messages.create = AsyncMock(
            side_effect=[
                RateLimitError(
                    message="Rate limited",
                    response=AsyncMock(status_code=429),
                    body=None,
                ),
                mock_response_factory("Success after retry"),
            ]
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient()
                response = await client.complete(
                    system_prompt="Test",
                    user_prompt="Test",
                )

                assert response.content == "Success after retry"
                assert mock_instance.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_retries_on_connection_error(self, mock_anthropic_client):
        """Complete retries on APIConnectionError."""
        mock_instance, mock_response_factory = mock_anthropic_client

        mock_instance.messages.create = AsyncMock(
            side_effect=[
                APIConnectionError(message="Connection failed", request=AsyncMock()),
                mock_response_factory("Success"),
            ]
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient()
                response = await client.complete(
                    system_prompt="Test",
                    user_prompt="Test",
                )

                assert response.content == "Success"

    @pytest.mark.asyncio
    async def test_complete_fails_after_max_retries(self, mock_anthropic_client):
        """Complete raises LLMError after exhausting retries."""
        mock_instance, _ = mock_anthropic_client

        # All calls raise rate limit
        mock_instance.messages.create = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limited",
                response=AsyncMock(status_code=429),
                body=None,
            )
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 2
            mock_settings.extraction_timeout_seconds = 120

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient()

                with pytest.raises(LLMError) as exc_info:
                    await client.complete(
                        system_prompt="Test",
                        user_prompt="Test",
                    )

                assert "Failed after 2 retries" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_no_retry_on_api_error(self, mock_anthropic_client):
        """Complete does not retry on non-retryable APIError."""
        mock_instance, _ = mock_anthropic_client

        mock_instance.messages.create = AsyncMock(
            side_effect=APIError(
                message="Bad request",
                request=AsyncMock(),
                body=None,
            )
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()

            with pytest.raises(LLMError) as exc_info:
                await client.complete(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            assert "Claude API error" in str(exc_info.value)
            # Should only be called once (no retries)
            assert mock_instance.messages.create.call_count == 1


# --- LLMClient.complete_json() Tests ---


class TestLLMClientCompleteJson:
    """Tests for LLMClient.complete_json() method."""

    @pytest.mark.asyncio
    async def test_complete_json_parses_response(self, mock_anthropic_client):
        """Complete_json parses JSON from response."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = json.dumps({"key": "value", "number": 42})
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_complete_json_strips_markdown_json_block(
        self, mock_anthropic_client
    ):
        """Complete_json handles ```json code blocks."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = '```json\n{"key": "value"}\n```'
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_strips_prose_before_markdown(
        self, mock_anthropic_client
    ):
        """Complete_json handles prose before markdown code blocks."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = (
            "I'll analyze the data. Here's my response:\n\n"
            '```json\n{"key": "value"}\n```'
        )
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_handles_plain_code_block(self, mock_anthropic_client):
        """Complete_json handles plain ``` code blocks."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = '```\n{"key": "value"}\n```'
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_handles_raw_json_object(self, mock_anthropic_client):
        """Complete_json handles raw JSON object with leading text."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = 'Here is the data: {"key": "value"}'
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_handles_array(self, mock_anthropic_client):
        """Complete_json handles JSON arrays."""
        mock_instance, mock_response_factory = mock_anthropic_client
        json_content = "[1, 2, 3]"
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory(json_content)
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            result = await client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Give me data.",
            )

            assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_complete_json_invalid_json_raises_error(self, mock_anthropic_client):
        """Complete_json raises LLMError for invalid JSON."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory("not valid json at all")
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()

            with pytest.raises(LLMError) as exc_info:
                await client.complete_json(
                    system_prompt="Return JSON.",
                    user_prompt="Give me data.",
                )

            assert "Failed to parse JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_json_uses_higher_default_max_tokens(
        self, mock_anthropic_client
    ):
        """Complete_json uses 8192 as default max_tokens."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory('{"key": "value"}')
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = "claude-sonnet-4-20250514"
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient()
            await client.complete_json(
                system_prompt="Test",
                user_prompt="Test",
            )

            call_kwargs = mock_instance.messages.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 8192


# --- LLMResponse Tests ---


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_stores_all_fields(self):
        """LLMResponse stores all fields correctly."""
        response = LLMResponse(
            content="test content",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
        )

        assert response.content == "test content"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_response_is_dataclass(self):
        """LLMResponse is a proper dataclass."""
        response = LLMResponse(
            content="test",
            model="model",
            input_tokens=0,
            output_tokens=0,
        )
        # Dataclasses have __dataclass_fields__
        assert hasattr(response, "__dataclass_fields__")


# --- LLMError Tests ---


class TestLLMError:
    """Tests for LLMError exception."""

    def test_error_message(self):
        """LLMError includes message."""
        error = LLMError("Test error")
        assert str(error) == "Test error"

    def test_error_is_exception(self):
        """LLMError is a proper exception."""
        error = LLMError("test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """LLMError can be raised and caught."""
        with pytest.raises(LLMError):
            raise LLMError("test error")


# --- ModelID Tests ---


class TestModelID:
    """Tests for ModelID validation."""

    def test_model_id_constants_exist(self):
        """ModelID class has all required model constants."""
        assert ModelID.HAIKU_3 == "claude-3-haiku-20240307"
        assert ModelID.HAIKU_3_5 == "claude-3-5-haiku-20241022"
        assert ModelID.SONNET_4 == "claude-sonnet-4-20250514"

    def test_model_id_validation_valid(self):
        """ModelID.validate() accepts valid model IDs."""
        # Should not raise
        ModelID.validate(ModelID.HAIKU_3)
        ModelID.validate(ModelID.HAIKU_3_5)
        ModelID.validate(ModelID.SONNET_4)

    def test_model_id_validation_invalid(self):
        """ModelID.validate() rejects invalid model IDs."""
        with pytest.raises(ValueError, match="Unknown model ID"):
            ModelID.validate("invalid-model-id")

    def test_model_id_all_models(self):
        """ModelID.all_models() returns all valid models."""
        models = ModelID.all_models()
        assert ModelID.HAIKU_3 in models
        assert ModelID.HAIKU_3_5 in models
        assert ModelID.SONNET_4 in models
        assert len(models) == 3


# --- Multi-Model Support Tests ---


class TestMultiModelSupport:
    """Tests for using different models."""

    def test_client_with_haiku_model(self, mock_anthropic_client):
        """Client can be initialized with Haiku 3.5."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = ModelID.SONNET_4
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient(model=ModelID.HAIKU_3_5)

            assert client.model == ModelID.HAIKU_3_5

    def test_multiple_clients_different_models(self, mock_anthropic_client):
        """Can create multiple clients with different models."""
        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = ModelID.SONNET_4
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            haiku_client = LLMClient(model=ModelID.HAIKU_3_5)
            sonnet_client = LLMClient(model=ModelID.SONNET_4)

            assert haiku_client.model == ModelID.HAIKU_3_5
            assert sonnet_client.model == ModelID.SONNET_4

    @pytest.mark.asyncio
    async def test_complete_uses_specified_model(self, mock_anthropic_client):
        """Complete sends requests with the specified model."""
        mock_instance, mock_response_factory = mock_anthropic_client
        mock_instance.messages.create = AsyncMock(
            return_value=mock_response_factory("Response")
        )

        with patch("app.services.llm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"  # pragma: allowlist secret
            mock_settings.anthropic_model = ModelID.SONNET_4
            mock_settings.extraction_max_retries = 3
            mock_settings.extraction_timeout_seconds = 120

            client = LLMClient(model=ModelID.HAIKU_3_5)
            await client.complete(
                system_prompt="Test",
                user_prompt="Test",
            )

            call_kwargs = mock_instance.messages.create.call_args.kwargs
            assert call_kwargs["model"] == ModelID.HAIKU_3_5
