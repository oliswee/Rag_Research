"""Query-related request and response schemas."""

from pydantic import BaseModel, Field

from app.schemas.session import SourceReference


class QueryRequest(BaseModel):
    """Request body for submitting a query."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query",
        json_schema_extra={"example": "What patterns exist about agent memory?"},
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to continue an existing conversation",
    )


class FollowUpRequest(BaseModel):
    """Request body for follow-up queries."""

    query: str = Field(
        ...,
        min_length=1,
        description="Follow-up question",
        json_schema_extra={"example": "Tell me more about the first one"},
    )
    session_id: str = Field(
        ...,
        description="Required session ID from previous query",
    )


class QueryResponse(BaseModel):
    """Response from a query."""

    session_id: str = Field(..., description="Session ID for follow-up queries")
    response: str = Field(..., description="Synthesized answer with citations")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Patterns referenced in the response",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level (high, medium, low)",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions",
    )
