"""Versioned source snapshots and index generations, retained for old citations."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.source import _utc_now


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    document_version: Mapped[str] = mapped_column(String(80), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(150), nullable=False)
    snapshot_digest: Mapped[str] = mapped_column(String(80), nullable=False)
    raw_pages: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    content_scope: Mapped[str] = mapped_column(String(20), nullable=False)
    source_origin: Mapped[str] = mapped_column(
        String(30), default="pdf", nullable=False
    )
    source_url: Mapped[str | None] = mapped_column(String(2000))
    original_pdf: Mapped[bytes | None] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        UniqueConstraint("id", "paper_id", name="uq_source_document_paper"),
    )


class SourceIndexGeneration(Base):
    __tablename__ = "source_index_generations"

    index_version: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(36), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False)
    cleaner_version: Mapped[str] = mapped_column(String(100), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    child_size: Mapped[int] = mapped_column(Integer, default=1200, nullable=False)
    parent_size: Mapped[int] = mapped_column(Integer, default=4800, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
    )
    vector_status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )
    physical_revision: Mapped[str | None] = mapped_column(String(80))
    index_manifest: Mapped[dict | None] = mapped_column(JSON)
    cleaning_manifest: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_type: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "paper_id"],
            ["source_documents.id", "source_documents.paper_id"],
            ondelete="CASCADE",
            name="fk_generation_document_paper",
        ),
        UniqueConstraint(
            "index_version", "paper_id", "document_id", name="uq_generation_source"
        ),
        UniqueConstraint("index_version", "document_id", name="uq_generation_document"),
        Index(
            "uq_active_generation_paper",
            "paper_id",
            unique=True,
            sqlite_where=text("is_active = 1"),
        ),
    )


class SourceParent(Base):
    __tablename__ = "source_parents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation_id: Mapped[str] = mapped_column(String(80), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False)
    section_id: Mapped[str | None] = mapped_column(String(36))
    parent_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_spans: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        ForeignKeyConstraint(
            ["generation_id", "document_id"],
            [
                "source_index_generations.index_version",
                "source_index_generations.document_id",
            ],
            ondelete="CASCADE",
            name="fk_parent_generation_document",
        ),
        UniqueConstraint(
            "id", "generation_id", "document_id", name="uq_parent_generation_source"
        ),
        Index("idx_source_parents_generation", "generation_id"),
    )
