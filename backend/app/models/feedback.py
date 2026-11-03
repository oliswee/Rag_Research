"""Local run identities and feedback; no questions, answers or external full text."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.source import _utc_now, _uuid


class AnswerRun(Base):
    __tablename__ = "answer_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_version: Mapped[str] = mapped_column(String(80), nullable=False)
    scope: Mapped[dict] = mapped_column(JSON, nullable=False)
    sources: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    minimal_trace: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("answer_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    source: Mapped[dict | None] = mapped_column(JSON)
    expected_paper_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class FeedbackRegression(Base):
    __tablename__ = "feedback_regressions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    feedback_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("answer_feedback.id", ondelete="CASCADE"), nullable=False
    )
    fixture: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), default="needs_review", nullable=False
    )
    result: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    fix_reference: Mapped[str | None] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
