"""Tests for Pattern Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.models import PatternStatus
from app.schemas.pattern import (
    ExtractedPattern,
    ExtractedPatternResponse,
    PatternDecision,
    PatternResponse,
    PatternReviewRequest,
    PatternReviewResponse,
    PatternUpdateRequest,
)


class TestExtractedPattern:
    """Tests for ExtractedPattern schema (internal DTO from Claude)."""

    def test_valid_with_claim_only(self) -> None:
        """Pattern with only claim is valid."""
        pattern = ExtractedPattern(
            name="Test Pattern",
            claim="This is the main assertion.",
        )
        assert pattern.name == "Test Pattern"
        assert pattern.claim == "This is the main assertion."
        assert pattern.evidence is None
        assert pattern.context is None

    def test_valid_with_evidence_only(self) -> None:
        """Pattern with only evidence is valid."""
        pattern = ExtractedPattern(
            name="Test Pattern",
            evidence="Supporting quote from the paper.",
        )
        assert pattern.evidence == "Supporting quote from the paper."
        assert pattern.claim is None

    def test_valid_with_context_only(self) -> None:
        """Pattern with only context is valid."""
        pattern = ExtractedPattern(
            name="Test Pattern",
            context="Broader implications of this finding.",
        )
        assert pattern.context == "Broader implications of this finding."
        assert pattern.claim is None
        assert pattern.evidence is None

    def test_valid_with_all_content_fields(self) -> None:
        """Pattern with all content fields is valid."""
        pattern = ExtractedPattern(
            name="Complete Pattern",
            claim="Main assertion",
            evidence="Supporting data",
            context="Implications",
            tags=["methodology", "finding"],
        )
        assert pattern.claim == "Main assertion"
        assert pattern.evidence == "Supporting data"
        assert pattern.context == "Implications"
        assert pattern.tags == ["methodology", "finding"]

    def test_invalid_without_content_fields(self) -> None:
        """Pattern must have at least one content field."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedPattern(name="Empty Pattern")
        assert "at least one" in str(exc_info.value).lower()

    def test_invalid_with_empty_strings(self) -> None:
        """Empty strings don't count as valid content."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedPattern(name="Empty Pattern", claim="", evidence="")
        assert "at least one" in str(exc_info.value).lower()

    def test_content_fields_property_single(self) -> None:
        """content_fields returns single field name."""
        pattern = ExtractedPattern(name="Test", claim="Something")
        assert pattern.content_fields == ["claim"]

    def test_content_fields_property_multiple(self) -> None:
        """content_fields returns all populated field names in order."""
        pattern = ExtractedPattern(
            name="Test",
            claim="Main point",
            context="Implications",
        )
        assert pattern.content_fields == ["claim", "context"]

    def test_content_fields_property_all(self) -> None:
        """content_fields returns all three in order."""
        pattern = ExtractedPattern(
            name="Test",
            claim="Claim",
            evidence="Evidence",
            context="Context",
        )
        assert pattern.content_fields == ["claim", "evidence", "context"]

    def test_name_required(self) -> None:
        """Name field is required."""
        with pytest.raises(ValidationError):
            ExtractedPattern(claim="Some claim")  # type: ignore[call-arg]

    def test_tags_default_to_none(self) -> None:
        """Tags default to None if not provided."""
        pattern = ExtractedPattern(name="Test", claim="Claim")
        assert pattern.tags is None

    def test_tags_can_be_empty_list(self) -> None:
        """Tags can be an empty list."""
        pattern = ExtractedPattern(name="Test", claim="Claim", tags=[])
        assert pattern.tags == []

    def test_invalid_with_whitespace_only_strings(self) -> None:
        """Whitespace-only strings don't count as valid content."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedPattern(name="Test", claim="   ", evidence="\t\n")
        assert "at least one" in str(exc_info.value).lower()

    def test_valid_with_whitespace_padded_content(self) -> None:
        """Content with whitespace padding is valid if has non-whitespace."""
        pattern = ExtractedPattern(name="Test", claim="  Valid claim  ")
        assert pattern.claim == "  Valid claim  "
        assert pattern.content_fields == ["claim"]

    def test_model_dump_serialization(self) -> None:
        """Schema can be serialized to dict."""
        pattern = ExtractedPattern(
            name="Test Pattern",
            claim="Main assertion",
            tags=["tag1"],
        )
        data = pattern.model_dump()
        assert data["name"] == "Test Pattern"
        assert data["claim"] == "Main assertion"
        assert data["evidence"] is None
        assert data["tags"] == ["tag1"]

    def test_model_validate_from_dict(self) -> None:
        """Schema can be created from dict."""
        data = {
            "name": "From Dict",
            "evidence": "Some evidence",
        }
        pattern = ExtractedPattern.model_validate(data)
        assert pattern.name == "From Dict"
        assert pattern.evidence == "Some evidence"


class TestExtractedPatternResponse:
    """Tests for ExtractedPatternResponse schema (upload response)."""

    def test_valid_response(self) -> None:
        """Valid response with temp_id."""
        response = ExtractedPatternResponse(
            temp_id="temp-123",
            name="Pattern Name",
            claim="Main assertion",
            evidence="Supporting data",
            tags=["tag1"],
        )
        assert response.temp_id == "temp-123"
        assert response.name == "Pattern Name"
        assert response.claim == "Main assertion"

    def test_content_fields_computed(self) -> None:
        """content_fields is computed from set fields."""
        response = ExtractedPatternResponse(
            temp_id="temp-123",
            name="Test",
            claim="Claim",
            context="Context",
        )
        assert response.content_fields == ["claim", "context"]

    def test_temp_id_required(self) -> None:
        """temp_id is required."""
        with pytest.raises(ValidationError):
            ExtractedPatternResponse(  # type: ignore[call-arg]
                name="Test",
                claim="Claim",
            )

    def test_tags_default_to_empty_list(self) -> None:
        """Tags default to empty list in response."""
        response = ExtractedPatternResponse(
            temp_id="temp-123",
            name="Test",
            claim="Claim",
        )
        assert response.tags == []

    def test_all_content_fields_optional(self) -> None:
        """All content fields can be None (for response flexibility)."""
        # Note: ExtractedPatternResponse doesn't validate content presence
        # because it's a response schema showing what was extracted
        response = ExtractedPatternResponse(
            temp_id="temp-123",
            name="Test",
        )
        assert response.claim is None
        assert response.evidence is None
        assert response.context is None

    def test_model_dump_serialization(self) -> None:
        """Response can be serialized to dict."""
        response = ExtractedPatternResponse(
            temp_id="temp-abc",
            name="Test Pattern",
            claim="Main claim",
            tags=["tag1", "tag2"],
        )
        data = response.model_dump()
        assert data["temp_id"] == "temp-abc"
        assert data["name"] == "Test Pattern"
        assert data["tags"] == ["tag1", "tag2"]

    def test_name_required(self) -> None:
        """Name is required."""
        with pytest.raises(ValidationError):
            ExtractedPatternResponse(temp_id="temp-123")  # type: ignore[call-arg]


class TestPatternResponse:
    """Tests for PatternResponse schema (GET endpoints)."""

    def test_valid_response(self) -> None:
        """Valid pattern response with all fields."""
        response = PatternResponse(
            id="uuid-123",
            paper_id="paper-uuid",
            name="Pattern Name",
            claim="Main assertion",
            evidence="Supporting data",
            context="Implications",
            tags=["tag1", "tag2"],
            custom_fields={"key": "value"},
            status=PatternStatus.approved,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:00:00Z",
        )
        assert response.id == "uuid-123"
        assert response.paper_id == "paper-uuid"
        assert response.status == PatternStatus.approved

    def test_required_fields(self) -> None:
        """All required fields must be present."""
        with pytest.raises(ValidationError):
            PatternResponse(  # type: ignore[call-arg]
                id="uuid-123",
                # Missing paper_id, name, status, timestamps
            )

    def test_optional_content_fields(self) -> None:
        """Content fields can be None."""
        response = PatternResponse(
            id="uuid-123",
            paper_id="paper-uuid",
            name="Test",
            claim=None,
            evidence=None,
            context=None,
            tags=[],
            status=PatternStatus.pending,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        assert response.claim is None
        assert response.evidence is None
        assert response.context is None

    def test_custom_fields_default_to_none(self) -> None:
        """custom_fields defaults to None."""
        response = PatternResponse(
            id="uuid-123",
            paper_id="paper-uuid",
            name="Test",
            tags=[],
            status=PatternStatus.approved,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        assert response.custom_fields is None

    def test_status_enum_values(self) -> None:
        """Status accepts all PatternStatus values."""
        for status in PatternStatus:
            response = PatternResponse(
                id="uuid",
                paper_id="paper-uuid",
                name="Test",
                tags=[],
                status=status,
                created_at="2024-01-15T10:30:00Z",
                updated_at="2024-01-15T10:30:00Z",
            )
            assert response.status == status

    def test_invalid_status_string(self) -> None:
        """Invalid status string should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            PatternResponse(
                id="uuid",
                paper_id="paper-uuid",
                name="Test",
                tags=[],
                status="invalid_status",  # type: ignore[arg-type]
                created_at="2024-01-15T10:30:00Z",
                updated_at="2024-01-15T10:30:00Z",
            )
        assert "status" in str(exc_info.value).lower()

    def test_model_dump_serialization(self) -> None:
        """Response can be serialized to dict."""
        response = PatternResponse(
            id="uuid-123",
            paper_id="paper-uuid",
            name="Test",
            claim="Main claim",
            tags=["tag1"],
            status=PatternStatus.approved,
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
        )
        data = response.model_dump()
        assert data["id"] == "uuid-123"
        assert data["status"] == "approved"


class TestPatternUpdateRequest:
    """Tests for PatternUpdateRequest schema (PUT endpoint)."""

    def test_update_name_only(self) -> None:
        """Can update just the name."""
        request = PatternUpdateRequest(name="New Name")
        assert request.name == "New Name"
        assert request.claim is None

    def test_update_claim_only(self) -> None:
        """Can update just the claim."""
        request = PatternUpdateRequest(claim="Updated claim")
        assert request.claim == "Updated claim"
        assert request.name is None

    def test_update_evidence_only(self) -> None:
        """Can update just the evidence."""
        request = PatternUpdateRequest(evidence="New evidence")
        assert request.evidence == "New evidence"

    def test_update_context_only(self) -> None:
        """Can update just the context."""
        request = PatternUpdateRequest(context="New context")
        assert request.context == "New context"

    def test_update_tags_only(self) -> None:
        """Can update just the tags."""
        request = PatternUpdateRequest(tags=["new", "tags"])
        assert request.tags == ["new", "tags"]

    def test_update_multiple_fields(self) -> None:
        """Can update multiple fields at once."""
        request = PatternUpdateRequest(
            name="Updated Name",
            claim="Updated claim",
            tags=["updated"],
        )
        assert request.name == "Updated Name"
        assert request.claim == "Updated claim"
        assert request.tags == ["updated"]

    def test_all_fields_optional(self) -> None:
        """All fields are optional (but at least one required)."""
        # Empty update should fail - need at least one field
        with pytest.raises(ValidationError) as exc_info:
            PatternUpdateRequest()
        assert "at least one field" in str(exc_info.value).lower()

    def test_empty_string_name_invalid(self) -> None:
        """Empty string name is invalid."""
        with pytest.raises(ValidationError):
            PatternUpdateRequest(name="")

    def test_empty_string_claim_is_valid(self) -> None:
        """Empty string claim is valid (allows clearing the field)."""
        request = PatternUpdateRequest(claim="")
        assert request.claim == ""

    def test_empty_string_evidence_is_valid(self) -> None:
        """Empty string evidence is valid (allows clearing the field)."""
        request = PatternUpdateRequest(evidence="")
        assert request.evidence == ""

    def test_empty_string_context_is_valid(self) -> None:
        """Empty string context is valid (allows clearing the field)."""
        request = PatternUpdateRequest(context="")
        assert request.context == ""

    def test_empty_tags_list_is_valid(self) -> None:
        """Empty tags list is valid (allows clearing tags)."""
        request = PatternUpdateRequest(tags=[])
        assert request.tags == []


class TestPatternDecision:
    """Tests for PatternDecision schema (review decision)."""

    def test_approve_decision(self) -> None:
        """Can create an approve decision."""
        decision = PatternDecision(temp_id="temp-123", approved=True)
        assert decision.temp_id == "temp-123"
        assert decision.approved is True

    def test_reject_decision(self) -> None:
        """Can create a reject decision."""
        decision = PatternDecision(temp_id="temp-456", approved=False)
        assert decision.temp_id == "temp-456"
        assert decision.approved is False

    def test_temp_id_required(self) -> None:
        """temp_id is required."""
        with pytest.raises(ValidationError):
            PatternDecision(approved=True)  # type: ignore[call-arg]

    def test_approved_required(self) -> None:
        """approved is required."""
        with pytest.raises(ValidationError):
            PatternDecision(temp_id="temp-123")  # type: ignore[call-arg]


class TestPatternReviewRequest:
    """Tests for PatternReviewRequest schema."""

    def test_valid_request_single_decision(self) -> None:
        """Valid request with single decision."""
        request = PatternReviewRequest(
            decisions=[PatternDecision(temp_id="temp-1", approved=True)]
        )
        assert len(request.decisions) == 1

    def test_valid_request_multiple_decisions(self) -> None:
        """Valid request with multiple decisions."""
        request = PatternReviewRequest(
            decisions=[
                PatternDecision(temp_id="temp-1", approved=True),
                PatternDecision(temp_id="temp-2", approved=False),
                PatternDecision(temp_id="temp-3", approved=True),
            ]
        )
        assert len(request.decisions) == 3

    def test_empty_decisions_invalid(self) -> None:
        """Empty decisions list is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            PatternReviewRequest(decisions=[])
        assert "at least 1" in str(exc_info.value).lower()

    def test_decisions_required(self) -> None:
        """decisions field is required."""
        with pytest.raises(ValidationError):
            PatternReviewRequest()  # type: ignore[call-arg]


class TestPatternReviewResponse:
    """Tests for PatternReviewResponse schema."""

    def test_valid_response(self) -> None:
        """Valid review response."""
        response = PatternReviewResponse(
            saved_count=3,
            rejected_count=2,
            patterns=[
                PatternResponse(
                    id="uuid-1",
                    paper_id="paper-uuid",
                    name="Pattern 1",
                    tags=[],
                    status=PatternStatus.approved,
                    created_at="2024-01-15T10:30:00Z",
                    updated_at="2024-01-15T10:30:00Z",
                ),
            ],
        )
        assert response.saved_count == 3
        assert response.rejected_count == 2
        assert len(response.patterns) == 1

    def test_zero_counts_valid(self) -> None:
        """Zero counts are valid (e.g., all rejected)."""
        response = PatternReviewResponse(
            saved_count=0,
            rejected_count=5,
            patterns=[],
        )
        assert response.saved_count == 0
        assert response.rejected_count == 5

    def test_empty_patterns_list_valid(self) -> None:
        """Empty patterns list is valid (all rejected)."""
        response = PatternReviewResponse(
            saved_count=0,
            rejected_count=3,
            patterns=[],
        )
        assert response.patterns == []

    def test_required_fields(self) -> None:
        """All fields are required."""
        with pytest.raises(ValidationError):
            PatternReviewResponse(  # type: ignore[call-arg]
                saved_count=1,
                # Missing rejected_count and patterns
            )

    def test_negative_counts_allowed_by_schema(self) -> None:
        """Schema allows negative counts (API layer should validate)."""
        # Note: Pydantic doesn't restrict int sign by default
        # Business logic validation happens at API layer
        response = PatternReviewResponse(
            saved_count=-1,
            rejected_count=-2,
            patterns=[],
        )
        assert response.saved_count == -1

    def test_model_dump_serialization(self) -> None:
        """Response can be serialized to dict."""
        response = PatternReviewResponse(
            saved_count=2,
            rejected_count=1,
            patterns=[],
        )
        data = response.model_dump()
        assert data["saved_count"] == 2
        assert data["rejected_count"] == 1
        assert data["patterns"] == []
