"""Minimal public DOI status provenance; absence never means unretracted."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PublicationState(Base):
    __tablename__ = "publication_states"

    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime)
    refresh_error: Mapped[str | None] = mapped_column(String(100))
