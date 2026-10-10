"""Metadata extraction service using Claude."""

import datetime
import json
import logging

from app.models import PaperType
from app.schemas.paper import PaperMetadataExtracted, PaperMetadataInput
from app.services.llm_client import LLMClient, LLMError
from app.services.prompts.metadata import (
    METADATA_EXTRACTION_SYSTEM,
    METADATA_EXTRACTION_USER,
)

logger = logging.getLogger(__name__)

# Maximum characters to send to LLM (roughly 12k tokens)
MAX_TEXT_CHARS = 50_000


class MetadataExtractor:
    """Extracts metadata from paper text using Claude."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize metadata extractor.

        Args:
            llm_client: LLM client to use. Creates default if not provided.
        """
        self._client = llm_client

    def _ensure_client(self) -> LLMClient:
        """Lazily initialize LLM client."""
        if self._client is None:
            self._client = LLMClient()
        return self._client

    async def extract(
        self,
        text: str,
        pdf_hints: dict[str, str] | None = None,
        user_metadata: PaperMetadataInput | None = None,
    ) -> PaperMetadataExtracted:
        """Extract metadata from paper text.

        Args:
            text: Paper content text.
            pdf_hints: Optional hints from PDF metadata (title, author, etc).
            user_metadata: Optional user-provided overrides.

        Returns:
            Extracted metadata with confidence score.
        """
        # Truncate long text to avoid token limits
        truncated_text = text[:MAX_TEXT_CHARS] if text else ""

        # Format PDF hints for prompt
        hints_str = "None available"
        if pdf_hints:
            hints_str = json.dumps(pdf_hints, indent=2)

        # Build user prompt
        user_prompt = METADATA_EXTRACTION_USER.format(
            text=truncated_text,
            pdf_hints=hints_str,
        )

        try:
            client = self._ensure_client()
            raw_response = await client.complete_json(
                system_prompt=METADATA_EXTRACTION_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.1,  # Low temperature for consistent extraction
            )

            # Parse the response into structured metadata
            extracted = self._parse_response(raw_response)

        except LLMError as e:
            logger.warning(f"Metadata extraction failed: {e}")
            extracted = self._default_metadata()

        # Apply user overrides (user values take precedence)
        if user_metadata:
            extracted = self._apply_overrides(extracted, user_metadata)

        return extracted

    def _parse_response(self, response: dict) -> PaperMetadataExtracted:
        """Parse LLM response into PaperMetadataExtracted."""
        # Parse date string to date object
        published_date = self._parse_date(response.get("published_date"))

        # Parse paper type
        type_str = response.get("type", "research_paper")
        try:
            paper_type = PaperType(type_str)
        except ValueError:
            paper_type = PaperType.research_paper

        return PaperMetadataExtracted(
            title=response.get("title", "Untitled Paper"),
            authors=response.get("authors", []),
            source=response.get("source"),
            published_date=published_date,
            type=paper_type,
            year=response.get("year"),
            one_sentence_summary=response.get("one_sentence_summary"),
            confidence=response.get("confidence", 0.5),
        )

    def _parse_date(self, date_str: str | None) -> datetime.date | None:
        """Parse date string to datetime.date."""
        if not date_str:
            return None
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            return None

    def _default_metadata(self) -> PaperMetadataExtracted:
        """Return default metadata when extraction fails."""
        return PaperMetadataExtracted(
            title="Untitled Paper",
            authors=[],
            source=None,
            published_date=None,
            type=PaperType.research_paper,
            year=None,
            one_sentence_summary=None,
            confidence=0.0,
        )

    def _apply_overrides(
        self,
        extracted: PaperMetadataExtracted,
        user_metadata: PaperMetadataInput,
    ) -> PaperMetadataExtracted:
        """Apply user-provided overrides to extracted metadata."""
        # User values take precedence when explicitly set
        title = user_metadata.title if user_metadata.title else extracted.title
        authors = user_metadata.authors if user_metadata.authors else extracted.authors
        source = user_metadata.source if user_metadata.source else extracted.source
        published_date = (
            user_metadata.published_date
            if user_metadata.published_date
            else extracted.published_date
        )
        # Only override type if user explicitly set a non-default value
        paper_type = (
            user_metadata.type
            if user_metadata.type != PaperType.research_paper
            else extracted.type
        )

        return PaperMetadataExtracted(
            title=title,
            authors=authors,
            source=source,
            published_date=published_date,
            type=paper_type,
            year=extracted.year,
            one_sentence_summary=extracted.one_sentence_summary,
            confidence=extracted.confidence,
        )
