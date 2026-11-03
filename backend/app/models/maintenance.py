"""Local rebuildable embedding memo and durable, dataset-scoped cleanup work."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.source import _utc_now


class EmbeddingMemo(Base):
    __tablename__ = "embedding_memos"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    vector_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class SourceDeletionJob(Base):
    __tablename__ = "source_deletion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    point_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class SourceSyncState(Base):
    __tablename__ = "source_sync_states"

    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    index_version: Mapped[str] = mapped_column(String(80), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(150), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    checked_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    outcome: Mapped[str] = mapped_column(String(30), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(100))
