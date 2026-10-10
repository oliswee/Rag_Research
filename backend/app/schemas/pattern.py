"""Pattern-related request and response schemas."""

from pydantic import BaseModel, Field, model_validator

from app.models import PatternStatus


class ExtractedPattern(BaseModel):
    """Pattern extracted by Claude (internal DTO).

    At least one of claim, evidence, or context must be non-null.
    """

    name: str = Field(
        ...,
        description="3-6 word pattern name",
        json_schema_extra={"example": "Agent Memory Fragility"},
    )
    claim: str | None = Field(
        default=None,
        description="Main assertion or finding",
        json_schema_extra={"example": "Agents lose track of long-term goals..."},
    )
    evidence: str | None = Field(
        default=None,
        description="Supporting quote, data, or example",
        json_schema_extra={"example": "In experiments, performance dropped by 40%..."},
    )
    context: str | None = Field(
        default=None,
        description="Implications, limitations, connections",
        json_schema_extra={"example": "This suggests explicit state management..."},
    )
    tags: list[str] | None = Field(
        default=None,
        description="Classification tags",
        json_schema_extra={"example": ["methodology", "finding"]},
    )

    @model_validator(mode="after")
    def validate_has_content(self) -> "ExtractedPattern":
        """Ensure at least one content field is present."""
        if not any(
            [
                self.claim and self.claim.strip(),
                self.evidence and self.evidence.strip(),
                self.context and self.context.strip(),
            ]
        ):
            raise ValueError(
                "At least one of claim, evidence, or context must be non-null"
            )
        return self

    @property
    def content_fields(self) -> list[str]:
        """Return list of non-null content field names in order."""
        fields = []
        if self.claim:
            fields.append("claim")
        if self.evidence:
            fields.append("evidence")
        if self.context:
            fields.append("context")
        return fields


class ExtractedPatternResponse(BaseModel):
    """Pattern as returned from extraction (before persistence).

    Used in upload response to show what was extracted.
    """

    temp_id: str = Field(
        ...,
        description="Temporary ID for review decisions",
        json_schema_extra={"example": "temp-abc123"},
    )
    name: str = Field(
        ...,
        description="Pattern name (3-6 words)",
        json_schema_extra={"example": "Agent Memory Fragility"},
    )
    claim: str | None = Field(
        default=None,
        description="Main assertion or finding",
    )
    evidence: str | None = Field(
        default=None,
        description="Supporting quote, data, or example",
    )
    context: str | None = Field(
        default=None,
        description="Implications, limitations, connections",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Classification tags",
    )
    confidence: str = Field(
        default="medium",
        description="Extraction confidence (high, medium, low)",
        json_schema_extra={"example": "high"},
    )

    @property
    def content_fields(self) -> list[str]:
        """Return list of non-null content field names in order."""
        fields = []
        if self.claim:
            fields.append("claim")
        if self.evidence:
            fields.append("evidence")
        if self.context:
            fields.append("context")
        return fields


class PatternResponse(BaseModel):
    """Full pattern response for GET endpoints."""

    id: str = Field(..., description="Pattern UUID")
    paper_id: str = Field(..., description="Parent paper UUID")
    name: str = Field(..., description="Pattern name (3-6 words)")
    claim: str | None = Field(default=None, description="Main assertion or finding")
    evidence: str | None = Field(
        default=None, description="Supporting quote, data, or example"
    )
    context: str | None = Field(
        default=None, description="Implications, limitations, connections"
    )
    tags: list[str] = Field(..., description="Classification tags")
    custom_fields: dict | None = Field(
        default=None, description="User-defined extensible fields"
    )
    status: PatternStatus = Field(..., description="Pattern status")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")


class PatternUpdateRequest(BaseModel):
    """Request body for updating a pattern.

    All fields are optional but at least one must be provided.
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        description="Updated pattern name",
    )
    claim: str | None = Field(
        default=None,
        description="Updated claim",
    )
    evidence: str | None = Field(
        default=None,
        description="Updated evidence",
    )
    context: str | None = Field(
        default=None,
        description="Updated context",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Updated tags",
    )

    @model_validator(mode="after")
    def validate_has_field(self) -> "PatternUpdateRequest":
        """Ensure at least one field is provided for update."""
        if not any(
            [
                self.name is not None,
                self.claim is not None,
                self.evidence is not None,
                self.context is not None,
                self.tags is not None,
            ]
        ):
            raise ValueError("At least one field must be provided for update")
        return self


class PatternDecision(BaseModel):
    """Single pattern review decision (approve or reject)."""

    temp_id: str = Field(
        ...,
        description="Temporary ID from extraction response",
    )
    approved: bool = Field(
        ...,
        description="True to approve, False to reject",
    )


class PatternReviewRequest(BaseModel):
    """Request body for submitting pattern review decisions."""

    decisions: list[PatternDecision] = Field(
        ...,
        min_length=1,
        description="List of pattern decisions (at least one required)",
    )

    @model_validator(mode="after")
    def validate_has_decisions(self) -> "PatternReviewRequest":
        """Ensure at least one decision is provided."""
        if not self.decisions:
            raise ValueError("At least one decision is required")
        return self


class PatternReviewResponse(BaseModel):
    """Response from pattern review submission."""

    saved_count: int = Field(..., description="Number of patterns approved and saved")
    rejected_count: int = Field(..., description="Number of patterns rejected")
    patterns: list[PatternResponse] = Field(
        ..., description="Approved patterns with IDs"
    )
