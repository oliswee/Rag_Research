"""Claude API client wrapper with retry logic."""

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anthropic
import httpx
import openai
from anthropic import APIConnectionError, APIError, RateLimitError

from app.config import ModelID, settings
from app.services.local_execution import local_only_request

if TYPE_CHECKING:
    from app.services.retrieval_trace import RetrievalTrace


class LLMError(Exception):
    """Raised when LLM call fails after retries."""

    pass


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


def llm_available() -> bool:
    """Whether the configured model can be called without missing credentials."""
    if settings.llm_provider == "ollama":
        return True
    if settings.llm_provider == "deepseek":
        return bool(settings.deepseek_api_key)
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


class LLMClient:
    """Async LLM client; local Ollama is the default provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: int | None = None,
        provider: str | None = None,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. Uses settings if not provided.
            model: Model to use. Uses settings if not provided.
            max_retries: Number of retry attempts for transient errors.
            timeout_seconds: Request timeout.
        """
        configured = getattr(settings, "llm_provider", "anthropic")
        self.provider = provider or (
            "anthropic" if api_key or not isinstance(configured, str) else configured
        )
        if self.provider not in {"anthropic", "ollama", "deepseek"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if self.provider == "deepseek":
            self.api_key = api_key or settings.deepseek_api_key
        elif self.provider == "anthropic":
            self.api_key = api_key or settings.anthropic_api_key
        else:
            self.api_key = ""
        if self.provider == "anthropic":
            self.model = model or settings.anthropic_model
        elif model and model not in ModelID.all_models():
            self.model = model
        else:
            self.model = (
                settings.deepseek_model
                if self.provider == "deepseek"
                else settings.ollama_model
            )
        self.max_retries = max_retries or settings.extraction_max_retries
        self.timeout_seconds = timeout_seconds or settings.extraction_timeout_seconds
        self.trace: RetrievalTrace | None = None

        if self.provider == "anthropic" and not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        if self.provider == "deepseek" and not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not configured")

        # Validate model ID to catch typos/deprecated models early
        if self.provider == "anthropic":
            ModelID.validate(self.model)

        if self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=float(self.timeout_seconds),
                max_retries=0,
            )
        else:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key if self.provider == "deepseek" else "ollama",
                base_url=(
                    "https://api.deepseek.com"
                    if self.provider == "deepseek"
                    else settings.ollama_base_url
                ),
                timeout=float(self.timeout_seconds),
                max_retries=0,
                **(
                    {
                        "http_client": httpx.AsyncClient(
                            trust_env=False, follow_redirects=False
                        )
                    }
                    if local_only_request.get()
                    else {}
                ),
            )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send completion request to Claude with retries.

        Args:
            system_prompt: System message for context/instructions.
            user_prompt: User message with the actual request.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            LLMResponse with content and usage stats.

        Raises:
            LLMError: If all retries fail.
        """
        if self.provider != "anthropic":
            return await self._complete_openai_compatible(
                system_prompt, user_prompt, max_tokens, temperature
            )

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                result = LLMResponse(
                    content=response.content[0].text,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                if self.trace is not None:
                    self.trace.record_llm_usage(
                        result.model, result.input_tokens, result.output_tokens
                    )
                return result

            except RateLimitError as e:
                last_error = e
                if attempt + 1 >= self.max_retries:
                    break
                if self.trace is not None:
                    self.trace.record_llm_retry(type(e).__name__)
                wait_time = 2**attempt  # Exponential backoff
                await asyncio.sleep(wait_time)

            except APIConnectionError as e:
                last_error = e
                if attempt + 1 >= self.max_retries:
                    break
                if self.trace is not None:
                    self.trace.record_llm_retry(type(e).__name__)
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)

            except APIError as e:
                status_code = getattr(e, "status_code", None)
                if status_code not in {408, 409} and not (
                    status_code and status_code >= 500
                ):
                    raise LLMError(f"Claude API error: {e}") from e
                last_error = e
                if attempt + 1 >= self.max_retries:
                    break
                if self.trace is not None:
                    self.trace.record_llm_retry(type(e).__name__)
                await asyncio.sleep(2**attempt)

        raise LLMError(f"Failed after {self.max_retries} retries: {last_error}")

    async def _complete_openai_compatible(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                usage = response.usage
                result = LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                )
                if self.trace is not None:
                    self.trace.record_llm_usage(
                        result.model, result.input_tokens, result.output_tokens
                    )
                return result
            except (openai.RateLimitError, openai.APIConnectionError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                if self.trace is not None:
                    self.trace.record_llm_retry(type(exc).__name__)
                await asyncio.sleep(2**attempt)
            except openai.APIError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code not in {408, 409} and not (
                    status_code and status_code >= 500
                ):
                    raise LLMError(
                        f"{self.provider} API error: {type(exc).__name__}"
                    ) from exc
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                if self.trace is not None:
                    self.trace.record_llm_retry(type(exc).__name__)
                await asyncio.sleep(2**attempt)
        raise LLMError(
            f"{self.provider} failed after {self.max_retries} retries: "
            f"{type(last_error).__name__}"
        )

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send completion request expecting JSON response.

        Args:
            system_prompt: Should instruct Claude to respond with JSON.
            user_prompt: User message with the actual request.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            LLMError: If response is not valid JSON.
        """
        response = await self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            # Handle markdown code blocks (may have prose before/after)
            content = response.content.strip()

            # Look for ```json...``` block anywhere in the response
            json_start = content.find("```json")
            if json_start != -1:
                content = content[json_start + 7 :]  # Skip ```json
                json_end = content.find("```")
                if json_end != -1:
                    content = content[:json_end]
            elif content.startswith("```"):
                # Plain ``` block at start
                content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            else:
                # No markdown - look for raw JSON object or array
                brace_start = content.find("{")
                bracket_start = content.find("[")
                if brace_start != -1 and (
                    bracket_start == -1 or brace_start < bracket_start
                ):
                    content = content[brace_start:]
                elif bracket_start != -1:
                    content = content[bracket_start:]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse JSON response: {e}\nContent: {response.content[:500]}"
            ) from e
