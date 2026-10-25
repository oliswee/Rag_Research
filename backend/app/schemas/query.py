"""Query-related request and response schemas."""

from typing import Literal

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
    paper_ids: list[str] | None = None
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rewrite_query: bool = True
    local_only: bool = False
    only_uploaded: bool = False
    intent: Literal["factual", "compare", "discover", "latest"] | None = None
    latest: bool = False
    year_from: int | None = Field(None, ge=1800, le=2100)
    year_to: int | None = Field(None, ge=1800, le=2100)
    provider: Literal["semantic_scholar", "openalex", "pubmed", "arxiv"] | None = None


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
    paper_ids: list[str] | None = None
    retrieval_mode: Literal["hybrid", "dense", "bm25"] | None = None
    rewrite_query: bool = True
    local_only: bool = False
    only_uploaded: bool = False
    intent: Literal["factual", "compare", "discover", "latest"] | None = None
    latest: bool = False
    year_from: int | None = Field(None, ge=1800, le=2100)
    year_to: int | None = Field(None, ge=1800, le=2100)
    provider: Literal["semantic_scholar", "openalex", "pubmed", "arxiv"] | None = None


class QueryResponse(BaseModel):
    """Response from a query."""

    session_id: str = Field(..., description="Session ID for follow-up queries")
    run_id: str | None = None
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
    evidence: list[dict] = Field(default_factory=list)
    context_coverage: dict[str, dict] = Field(default_factory=dict)
    retrieval_mode: str | None = None
    retrieval_query: str | None = None
    query_count: int = 1
    routing: dict = Field(default_factory=dict)
    web_records: list[dict] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    retrieval_status: str | None = None
    rewrite_status: Literal["disabled", "applied", "unavailable", "failed"] | None = (
        None
    )
