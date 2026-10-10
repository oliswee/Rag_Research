"""Pydantic schemas for conversation session management."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Reference to a pattern used in a response.

    Minimal schema for session management. Full implementation
    with paper details will be added in the query phase.
    """

    pattern_id: str = Field(..., description="UUID of the referenced pattern")
    summary: str | None = Field(default=None, description="Pattern summary")
    relevance_score: float | None = Field(
        default=None,
        description="Similarity score from vector search",
        ge=0.0,
        le=1.0,
    )


class ConversationTurn(BaseModel):
    """A single turn in a conversation (user or assistant message)."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who sent this message",
    )
    content: str = Field(
        ...,
        description="The message content",
    )
    sources: list[SourceReference] | None = Field(
        default=None,
        description="Sources referenced in this turn (assistant only)",
    )
    timestamp: datetime = Field(
        ...,
        description="When this turn was created",
    )


class ConversationSession(BaseModel):
    """A conversation session with history."""

    session_id: str = Field(..., description="Unique session identifier")
    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Conversation history",
    )
    created_at: datetime = Field(..., description="When the session was created")
    last_accessed: datetime = Field(..., description="Last activity timestamp")
