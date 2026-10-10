"""Embedding service for pattern vectors."""

import asyncio
import logging
from dataclasses import dataclass

import openai
from openai import APIConnectionError, APIError, RateLimitError

from app.config import settings

logger = logging.getLogger(__name__)

# OpenAI text-embedding-3-small produces 1536-dimensional vectors
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Batch settings
BATCH_SIZE = 20
MAX_CHARS = 30_000  # ~8K tokens


class EmbeddingError(Exception):
    """Raised when embedding generation fails after retries."""

    pass


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    text: str
    embedding: list[float]
    model: str
    token_count: int


class EmbeddingService:
    """Service for generating pattern embeddings using OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = EMBEDDING_MODEL,
        max_retries: int = 3,
    ) -> None:
        """Initialize embedding service.

        Args:
            api_key: OpenAI API key. Uses settings if not provided.
            model: Embedding model to use.
            max_retries: Number of retry attempts for transient errors.
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    def prepare_embedding_text(self, pattern_data: dict) -> str:
        """Prepare text for embedding from pattern data.

        Combines name, claim, evidence, and context into a single string
        suitable for embedding.

        Args:
            pattern_data: Dict with name, claim, evidence, context fields.

        Returns:
            Combined text for embedding, truncated if necessary.
        """
        parts = []

        # Add fields in order of importance
        for field in ["name", "claim", "evidence", "context"]:
            content = pattern_data.get(field)
            if content and str(content).strip():
                parts.append(str(content).strip())

        text = " | ".join(parts)

        # Truncate if too long
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            logger.warning(f"Embedding text truncated to {MAX_CHARS} characters")

        return text

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector and metadata.

        Raises:
            EmbeddingError: If all retries fail.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )

                return EmbeddingResult(
                    text=text,
                    embedding=response.data[0].embedding,
                    model=response.model,
                    token_count=response.usage.total_tokens,
                )

            except RateLimitError as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)

            except APIConnectionError as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    f"Connection error, waiting {wait_time}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)

            except APIError as e:
                # Non-retryable errors
                raise EmbeddingError(f"OpenAI API error: {e}") from e

        raise EmbeddingError(f"Failed after {self.max_retries} retries: {last_error}")

    async def generate_embeddings_batch(
        self,
        patterns: list[dict],
    ) -> list[tuple[dict, list[float]]]:
        """Generate embeddings for multiple patterns.

        Processes patterns in batches with retry logic.

        Args:
            patterns: List of pattern dicts with name, claim, evidence, context.

        Returns:
            List of (pattern, embedding) tuples.
        """
        results: list[tuple[dict, list[float]]] = []

        # Process in batches
        for i in range(0, len(patterns), BATCH_SIZE):
            batch = patterns[i : i + BATCH_SIZE]

            # Prepare texts for batch
            texts = [self.prepare_embedding_text(p) for p in batch]

            # Filter out empty texts
            valid_items = [
                (p, t) for p, t in zip(batch, texts, strict=False) if t.strip()
            ]

            if not valid_items:
                continue

            valid_patterns, valid_texts = zip(*valid_items, strict=False)

            # Generate embeddings for batch
            try:
                embeddings = await self._generate_batch(list(valid_texts))

                for pattern, embedding in zip(valid_patterns, embeddings, strict=False):
                    results.append((pattern, embedding))

            except EmbeddingError as e:
                logger.error(f"Batch embedding failed: {e}")
                # Try individual embeddings as fallback
                for pattern, text in valid_items:
                    try:
                        result = await self.generate_embedding(text)
                        results.append((pattern, result.embedding))
                    except EmbeddingError as individual_error:
                        logger.error(f"Individual embedding failed: {individual_error}")
                        # Skip this pattern

        return results

    async def generate_text_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed source chunks in bounded requests, preserving order."""
        vectors = []
        for offset in range(0, len(texts), BATCH_SIZE):
            vectors.extend(
                await self._generate_batch(texts[offset : offset + BATCH_SIZE])
            )
        return vectors

    async def _generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.

        Raises:
            EmbeddingError: If batch generation fails.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                # Return embeddings in order
                return [item.embedding for item in response.data]

            except RateLimitError as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)

            except APIConnectionError as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    f"Connection error, waiting {wait_time}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(wait_time)

            except APIError as e:
                raise EmbeddingError(f"OpenAI API error: {e}") from e

        raise EmbeddingError(
            f"Batch failed after {self.max_retries} retries: {last_error}"
        )
