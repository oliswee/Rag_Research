"""Traceable source content and online import records."""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.paper import Paper
    from app.models.pattern import Pattern


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


class PaperIdentifier(Base):
    """A provider or publication identifier that resolves to exactly one paper."""

    __tablename__ = "paper_identifiers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[str] = mapped_column(String(500), nullable=False)

    paper: Mapped["Paper"] = relationship(back_populates="identifiers")

    __table_args__ = (UniqueConstraint("kind", "value", name="uq_paper_identifier"),)


class PaperChunk(Base):
    """Original source excerpt that a generated answer may cite."""

    __tablename__ = "paper_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_scope: Mapped[str] = mapped_column(String(20), nullable=False)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    checksum: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    supporting_patterns: Mapped[list["PatternEvidence"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_paper_chunk_index"),
        Index("idx_paper_chunks_paper_id", "paper_id"),
    )


class PatternEvidence(Base):
    """Explicit link from an extracted pattern to the original excerpt."""

    __tablename__ = "pattern_evidence"

    pattern_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patterns.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("paper_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    pattern: Mapped["Pattern"] = relationship(back_populates="source_evidence")
    chunk: Mapped[PaperChunk] = relationship(back_populates="supporting_patterns")


class ImportJob(Base):
    """Persisted progress for an asynchronous batch import."""

    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    items: Mapped[list["ImportItem"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ImportItem(Base):
    """One selected provider record and its import outcome."""

    __tablename__ = "import_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("import_jobs.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(500), nullable=False)
    candidate: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    paper_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )
    content_scope: Mapped[str | None] = mapped_column(String(20), nullable=True)
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped[ImportJob] = relationship(back_populates="items")

    __table_args__ = (
        UniqueConstraint(
            "job_id", "provider", "provider_id", name="uq_import_job_item"
        ),
        Index("idx_import_items_job_id", "job_id"),
    )
