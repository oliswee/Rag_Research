"""Paper model for research documents."""

import uuid
from datetime import UTC, date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Date, DateTime, Index, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.pattern import Pattern


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class PaperType(str, Enum):
    """Type of research document."""

    research_paper = "research_paper"
    blog_post = "blog_post"
    article = "article"
    technical_report = "technical_report"
    book_chapter = "book_chapter"


class PaperStatus(str, Enum):
    """Processing status of a paper."""

    processing = "processing"
    ready_for_review = "ready_for_review"
    active = "active"
    archived = "archived"


class Paper(Base):
    """Research document ingested into the system."""

    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String(500))
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    type: Mapped[PaperType] = mapped_column(SQLEnum(PaperType))
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[PaperStatus] = mapped_column(
        SQLEnum(PaperStatus), default=PaperStatus.processing
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_now, onupdate=_utc_now
    )

    # Relationship to patterns
    patterns: Mapped[list["Pattern"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_papers_status_type_date", "status", "type", "published_date"),
    )
