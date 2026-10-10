"""Tests for Paper Pydantic schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.models import PaperStatus, PaperType
from app.schemas.paper import (
    PaperListResponse,
    PaperMetadataExtracted,
    PaperMetadataInput,
    PaperResponse,
)


class TestPaperMetadataInput:
    """Tests for PaperMetadataInput schema."""

    def test_all_fields_optional(self):
        """All fields should be optional with defaults."""
        metadata = PaperMetadataInput()
        assert metadata.title is None
        assert metadata.authors is None
        assert metadata.source is None
        assert metadata.published_date is None
        assert metadata.type == PaperType.research_paper

    def test_accepts_full_metadata(self):
        """Should accept all fields populated."""
        metadata = PaperMetadataInput(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            source="https://example.com",
            published_date=date(2024, 1, 15),
            type=PaperType.blog_post,
        )
        assert metadata.title == "Test Paper"
        assert metadata.authors == ["Author 1", "Author 2"]
        assert metadata.source == "https://example.com"
        assert metadata.published_date == date(2024, 1, 15)
        assert metadata.type == PaperType.blog_post

    def test_accepts_partial_metadata(self):
        """Should accept partial fields."""
        metadata = PaperMetadataInput(title="Just Title")
        assert metadata.title == "Just Title"
        assert metadata.authors is None
        assert metadata.type == PaperType.research_paper

    def test_type_accepts_all_paper_types(self):
        """Should accept all valid PaperType values."""
        for paper_type in PaperType:
            metadata = PaperMetadataInput(type=paper_type)
            assert metadata.type == paper_type

    def test_type_rejects_invalid_value(self):
        """Should reject invalid paper type."""
        with pytest.raises(ValidationError) as exc_info:
            PaperMetadataInput(type="invalid_type")
        assert "type" in str(exc_info.value)


class TestPaperMetadataExtracted:
    """Tests for PaperMetadataExtracted schema."""

    def test_title_required(self):
        """Title should be required."""
        with pytest.raises(ValidationError) as exc_info:
            PaperMetadataExtracted()
        assert "title" in str(exc_info.value)

    def test_minimal_valid(self):
        """Should accept just title."""
        metadata = PaperMetadataExtracted(title="Test Paper")
        assert metadata.title == "Test Paper"
        assert metadata.authors == []
        assert metadata.source is None
        assert metadata.published_date is None
        assert metadata.type == PaperType.research_paper
        assert metadata.year is None
        assert metadata.one_sentence_summary is None
        assert metadata.confidence == 0.8

    def test_full_metadata(self):
        """Should accept all fields."""
        metadata = PaperMetadataExtracted(
            title="ReAct: Reasoning and Acting",
            authors=["Shunyu Yao", "Jeffrey Zhao"],
            source="https://arxiv.org/abs/2210.03629",
            published_date=date(2022, 10, 6),
            type=PaperType.research_paper,
            year=2022,
            one_sentence_summary="A framework combining reasoning and acting.",
            confidence=0.95,
        )
        assert metadata.title == "ReAct: Reasoning and Acting"
        assert len(metadata.authors) == 2
        assert metadata.year == 2022
        assert metadata.confidence == 0.95

    def test_confidence_default(self):
        """Confidence should default to 0.8."""
        metadata = PaperMetadataExtracted(title="Test")
        assert metadata.confidence == 0.8

    def test_confidence_lower_bound(self):
        """Confidence must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            PaperMetadataExtracted(title="Test", confidence=-0.1)
        assert "confidence" in str(exc_info.value)

    def test_confidence_upper_bound(self):
        """Confidence must be <= 1."""
        with pytest.raises(ValidationError) as exc_info:
            PaperMetadataExtracted(title="Test", confidence=1.5)
        assert "confidence" in str(exc_info.value)

    def test_confidence_at_bounds(self):
        """Confidence at 0 and 1 should be valid."""
        metadata_zero = PaperMetadataExtracted(title="Test", confidence=0.0)
        assert metadata_zero.confidence == 0.0

        metadata_one = PaperMetadataExtracted(title="Test", confidence=1.0)
        assert metadata_one.confidence == 1.0


class TestPaperResponse:
    """Tests for PaperResponse schema."""

    def test_all_required_fields(self):
        """Should require all response fields."""
        response = PaperResponse(
            id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Paper",
            authors=["Author 1"],
            date=date(2024, 1, 15),
            source="https://example.com",
            published_date=date(2024, 1, 1),
            type=PaperType.research_paper,
            status=PaperStatus.active,
            pattern_count=5,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        assert response.id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.title == "Test Paper"
        assert response.pattern_count == 5
        assert response.status == PaperStatus.active

    def test_nullable_fields(self):
        """Should allow null for optional fields."""
        response = PaperResponse(
            id="test-id",
            title="Test Paper",
            authors=[],
            date=None,
            source=None,
            published_date=None,
            type=PaperType.article,
            status=PaperStatus.processing,
            pattern_count=0,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        assert response.date is None
        assert response.source is None
        assert response.published_date is None

    def test_missing_required_field(self):
        """Should raise error for missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            PaperResponse(
                id="test-id",
                title="Test",
                # missing authors, type, status, etc.
            )
        errors = str(exc_info.value)
        assert "authors" in errors or "type" in errors or "status" in errors

    def test_all_status_values(self):
        """Should accept all PaperStatus values."""
        for status in PaperStatus:
            response = PaperResponse(
                id="test-id",
                title="Test",
                authors=[],
                date=None,
                source=None,
                published_date=None,
                type=PaperType.research_paper,
                status=status,
                pattern_count=0,
                created_at="2024-01-15T10:30:00Z",
                updated_at="2024-01-15T10:30:00Z",
            )
            assert response.status == status


class TestPaperListResponse:
    """Tests for PaperListResponse schema."""

    def test_empty_list(self):
        """Should handle empty paper list."""
        response = PaperListResponse(papers=[], total=0)
        assert response.papers == []
        assert response.total == 0

    def test_with_papers(self):
        """Should handle list of papers."""
        paper = PaperResponse(
            id="test-id",
            title="Test Paper",
            authors=["Author"],
            date=None,
            source=None,
            published_date=None,
            type=PaperType.research_paper,
            status=PaperStatus.active,
            pattern_count=3,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        response = PaperListResponse(papers=[paper], total=1)
        assert len(response.papers) == 1
        assert response.total == 1
        assert response.papers[0].title == "Test Paper"

    def test_total_can_exceed_papers_length(self):
        """Total can be greater than papers length (pagination)."""
        paper = PaperResponse(
            id="test-id",
            title="Test",
            authors=[],
            date=None,
            source=None,
            published_date=None,
            type=PaperType.research_paper,
            status=PaperStatus.active,
            pattern_count=0,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        response = PaperListResponse(papers=[paper], total=100)
        assert len(response.papers) == 1
        assert response.total == 100
