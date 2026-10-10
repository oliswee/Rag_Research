"""Paper-related request and response schemas."""

import datetime

from pydantic import BaseModel, Field

from app.models import PaperStatus, PaperType


class PaperMetadataInput(BaseModel):
    """Optional metadata provided by user during upload."""

    title: str | None = Field(
        default=None,
        description="Paper title (overrides extracted title)",
        json_schema_extra={"example": "ReAct: Synergizing Reasoning and Acting"},
    )
    authors: list[str] | None = Field(
        default=None,
        description="List of author names",
        json_schema_extra={"example": ["Shunyu Yao", "Jeffrey Zhao"]},
    )
    source: str | None = Field(
        default=None,
        description="URL or source reference",
        json_schema_extra={"example": "https://arxiv.org/abs/2210.03629"},
    )
    published_date: datetime.date | None = Field(
        default=None,
        description="Original publication date",
        json_schema_extra={"example": "2022-10-06"},
    )
    type: PaperType = Field(
        default=PaperType.research_paper,
        description="Type of research document",
        json_schema_extra={"example": "research_paper"},
    )


class PaperMetadataExtracted(BaseModel):
    """Metadata extracted by Claude from paper content."""

    title: str = Field(..., description="Extracted paper title")
    authors: list[str] = Field(
        default_factory=list, description="Extracted author names"
    )
    source: str | None = Field(default=None, description="Extracted source URL")
    published_date: datetime.date | None = Field(
        default=None, description="Extracted publication date"
    )
    type: PaperType = Field(
        default=PaperType.research_paper, description="Inferred document type"
    )
    year: int | None = Field(default=None, description="Publication year")
    one_sentence_summary: str | None = Field(
        default=None, description="Brief summary of the paper"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0-1.0)",
    )


class PaperResponse(BaseModel):
    """Full paper response for GET endpoints."""

    id: str = Field(..., description="Paper UUID")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="List of author names")
    date: datetime.date | None = Field(
        default=None, description="User-editable effective date"
    )
    source: str | None = Field(default=None, description="URL or source reference")
    published_date: datetime.date | None = Field(
        default=None, description="Original publication date"
    )
    type: PaperType = Field(..., description="Document type")
    status: PaperStatus = Field(..., description="Processing status")
    pattern_count: int = Field(..., description="Number of patterns extracted")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")


class PaperListResponse(BaseModel):
    """Response for paper listing with pagination."""

    papers: list[PaperResponse] = Field(..., description="List of papers")
    total: int = Field(..., description="Total count (for pagination)")
