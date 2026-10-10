"""Tests for MetadataExtractor service."""

import datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.models import PaperType
from app.schemas.paper import PaperMetadataExtracted, PaperMetadataInput
from app.services.llm_client import LLMError
from app.services.metadata_extractor import MetadataExtractor


class TestMetadataExtractorExtract:
    """Tests for MetadataExtractor.extract() method."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create extractor with mocked LLM client."""
        with patch("app.services.metadata_extractor.LLMClient") as mock_class:
            mock_client = AsyncMock()
            mock_class.return_value = mock_client
            extractor = MetadataExtractor()
            extractor._client = mock_client
            return extractor

    @pytest.fixture
    def sample_text(self) -> str:
        """Sample paper text for testing."""
        return """
        ReAct: Synergizing Reasoning and Acting in Language Models

        Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du,
        Izhak Shafran, Karthik Narasimhan, Yuan Cao

        Abstract
        While large language models (LLMs) have demonstrated impressive performance
        across tasks in language understanding and interactive decision making, their
        abilities for reasoning and acting are studied separately...

        Published: October 6, 2022
        """

    @pytest.mark.asyncio
    async def test_extract_returns_metadata_from_llm_response(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should return PaperMetadataExtracted from LLM JSON response."""
        extractor._client.complete_json.return_value = {
            "title": "ReAct: Synergizing Reasoning and Acting",
            "authors": ["Shunyu Yao", "Jeffrey Zhao"],
            "published_date": "2022-10-06",
            "year": 2022,
            "type": "research_paper",
            "one_sentence_summary": "A method combining reasoning and acting in LLMs.",
            "source": None,
            "confidence": 0.95,
        }

        result = await extractor.extract(sample_text)

        assert isinstance(result, PaperMetadataExtracted)
        assert result.title == "ReAct: Synergizing Reasoning and Acting"
        assert result.authors == ["Shunyu Yao", "Jeffrey Zhao"]
        assert result.published_date == datetime.date(2022, 10, 6)
        assert result.year == 2022
        assert result.type == PaperType.research_paper
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_extract_handles_missing_optional_fields(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should handle missing optional fields with defaults."""
        extractor._client.complete_json.return_value = {
            "title": "A Paper Title",
            "authors": [],
            "published_date": None,
            "year": None,
            "type": "research_paper",
            "one_sentence_summary": None,
            "source": None,
            "confidence": 0.5,
        }

        result = await extractor.extract(sample_text)

        assert result.title == "A Paper Title"
        assert result.authors == []
        assert result.published_date is None
        assert result.year is None
        assert result.one_sentence_summary is None

    @pytest.mark.asyncio
    async def test_extract_parses_date_string_to_date_object(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should convert date string to datetime.date."""
        extractor._client.complete_json.return_value = {
            "title": "Paper",
            "authors": [],
            "published_date": "2023-05-15",
            "year": 2023,
            "type": "research_paper",
            "one_sentence_summary": None,
            "source": None,
            "confidence": 0.8,
        }

        result = await extractor.extract(sample_text)

        assert result.published_date == datetime.date(2023, 5, 15)

    @pytest.mark.asyncio
    async def test_extract_handles_pdf_hints(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should include PDF hints in the prompt."""
        extractor._client.complete_json.return_value = {
            "title": "Paper from PDF",
            "authors": ["Author Name"],
            "published_date": None,
            "year": None,
            "type": "research_paper",
            "one_sentence_summary": "Summary",
            "source": None,
            "confidence": 0.7,
        }

        pdf_hints = {"title": "PDF Title Hint", "author": "PDF Author"}
        await extractor.extract(sample_text, pdf_hints=pdf_hints)

        # Verify pdf_hints were passed to the prompt
        call_args = extractor._client.complete_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[0][1]
        assert "PDF Title Hint" in user_prompt
        assert "PDF Author" in user_prompt

    @pytest.mark.asyncio
    async def test_extract_uses_user_overrides(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """User-provided metadata should override extracted values."""
        extractor._client.complete_json.return_value = {
            "title": "Extracted Title",
            "authors": ["Extracted Author"],
            "published_date": "2022-01-01",
            "year": 2022,
            "type": "research_paper",
            "one_sentence_summary": "Extracted summary",
            "source": None,
            "confidence": 0.9,
        }

        user_metadata = PaperMetadataInput(
            title="User Title",
            authors=["User Author"],
            published_date=datetime.date(2023, 6, 15),
            type=PaperType.blog_post,
        )

        result = await extractor.extract(sample_text, user_metadata=user_metadata)

        # User values should override extracted values
        assert result.title == "User Title"
        assert result.authors == ["User Author"]
        assert result.published_date == datetime.date(2023, 6, 15)
        assert result.type == PaperType.blog_post
        # Extracted values kept when not overridden
        assert result.one_sentence_summary == "Extracted summary"

    @pytest.mark.asyncio
    async def test_extract_partial_user_overrides(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Partial user overrides should only replace specified fields."""
        extractor._client.complete_json.return_value = {
            "title": "Extracted Title",
            "authors": ["Extracted Author"],
            "published_date": "2022-01-01",
            "year": 2022,
            "type": "research_paper",
            "one_sentence_summary": "Summary",
            "source": "https://example.com",
            "confidence": 0.85,
        }

        # Only override title
        user_metadata = PaperMetadataInput(title="Custom Title")

        result = await extractor.extract(sample_text, user_metadata=user_metadata)

        assert result.title == "Custom Title"
        assert result.authors == ["Extracted Author"]  # Not overridden
        assert result.published_date == datetime.date(2022, 1, 1)  # Not overridden

    @pytest.mark.asyncio
    async def test_extract_handles_llm_error_returns_defaults(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should return default metadata if LLM fails."""
        extractor._client.complete_json.side_effect = LLMError("API Error")

        result = await extractor.extract(sample_text)

        assert result.title == "Untitled Paper"
        assert result.authors == []
        assert result.published_date is None
        assert result.confidence == 0.0  # Zero confidence for defaults

    @pytest.mark.asyncio
    async def test_extract_handles_invalid_json_response(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Extract should return defaults if LLM returns unparseable response."""
        extractor._client.complete_json.side_effect = LLMError("Failed to parse JSON")

        result = await extractor.extract(sample_text)

        assert result.title == "Untitled Paper"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_extract_handles_empty_text(
        self, extractor: MetadataExtractor
    ) -> None:
        """Extract should handle empty or very short text gracefully."""
        extractor._client.complete_json.return_value = {
            "title": "Unknown",
            "authors": [],
            "published_date": None,
            "year": None,
            "type": "research_paper",
            "one_sentence_summary": None,
            "source": None,
            "confidence": 0.1,
        }

        result = await extractor.extract("")

        assert result.confidence == 0.1

    @pytest.mark.asyncio
    async def test_extract_infers_paper_type_from_content(
        self, extractor: MetadataExtractor
    ) -> None:
        """Extract should infer document type from content characteristics."""
        blog_text = "This blog post discusses the latest trends in AI..."

        extractor._client.complete_json.return_value = {
            "title": "AI Trends",
            "authors": ["Blog Author"],
            "published_date": None,
            "year": None,
            "type": "blog_post",
            "one_sentence_summary": "Discussion of AI trends",
            "source": "https://blog.example.com",
            "confidence": 0.75,
        }

        result = await extractor.extract(blog_text)

        assert result.type == PaperType.blog_post

    @pytest.mark.asyncio
    async def test_extract_confidence_reflects_extraction_quality(
        self, extractor: MetadataExtractor, sample_text: str
    ) -> None:
        """Confidence score should reflect how much metadata was extracted."""
        extractor._client.complete_json.return_value = {
            "title": "Well Documented Paper",
            "authors": ["Author One", "Author Two"],
            "published_date": "2023-01-15",
            "year": 2023,
            "type": "research_paper",
            "one_sentence_summary": "A comprehensive paper with all metadata.",
            "source": "https://arxiv.org/abs/1234",
            "confidence": 0.98,
        }

        result = await extractor.extract(sample_text)

        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_extract_truncates_long_text(
        self, extractor: MetadataExtractor
    ) -> None:
        """Extract should truncate very long text to avoid token limits."""
        long_text = "x" * 100_000  # Very long text

        extractor._client.complete_json.return_value = {
            "title": "Long Paper",
            "authors": [],
            "published_date": None,
            "year": None,
            "type": "research_paper",
            "one_sentence_summary": None,
            "source": None,
            "confidence": 0.5,
        }

        await extractor.extract(long_text)

        # Verify text was truncated in the prompt
        call_args = extractor._client.complete_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[0][1]
        # Should be much shorter than 100k
        assert len(user_prompt) < 60_000


class TestMetadataExtractorParseDate:
    """Tests for date parsing helper."""

    def test_parse_valid_date_string(self) -> None:
        """Should parse YYYY-MM-DD format."""
        extractor = MetadataExtractor.__new__(MetadataExtractor)
        result = extractor._parse_date("2023-05-15")
        assert result == datetime.date(2023, 5, 15)

    def test_parse_none_returns_none(self) -> None:
        """Should return None for None input."""
        extractor = MetadataExtractor.__new__(MetadataExtractor)
        result = extractor._parse_date(None)
        assert result is None

    def test_parse_invalid_date_returns_none(self) -> None:
        """Should return None for invalid date strings."""
        extractor = MetadataExtractor.__new__(MetadataExtractor)
        result = extractor._parse_date("not-a-date")
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        """Should return None for empty string."""
        extractor = MetadataExtractor.__new__(MetadataExtractor)
        result = extractor._parse_date("")
        assert result is None
