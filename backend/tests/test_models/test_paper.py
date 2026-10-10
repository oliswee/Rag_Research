"""Tests for Paper model and enums."""

import uuid
from datetime import date, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper, PaperStatus, PaperType


class TestPaperTypeEnum:
    """Tests for PaperType enum."""

    def test_enum_values(self):
        """PaperType should have expected values."""
        assert PaperType.research_paper.value == "research_paper"
        assert PaperType.blog_post.value == "blog_post"
        assert PaperType.article.value == "article"
        assert PaperType.technical_report.value == "technical_report"
        assert PaperType.book_chapter.value == "book_chapter"

    def test_enum_is_string_enum(self):
        """PaperType should be a string enum."""
        assert isinstance(PaperType.research_paper, str)
        assert PaperType.research_paper == "research_paper"

    def test_enum_count(self):
        """PaperType should have exactly 5 values."""
        assert len(PaperType) == 5


class TestPaperStatusEnum:
    """Tests for PaperStatus enum."""

    def test_enum_values(self):
        """PaperStatus should have expected values."""
        assert PaperStatus.processing.value == "processing"
        assert PaperStatus.ready_for_review.value == "ready_for_review"
        assert PaperStatus.active.value == "active"
        assert PaperStatus.archived.value == "archived"

    def test_enum_is_string_enum(self):
        """PaperStatus should be a string enum."""
        assert isinstance(PaperStatus.processing, str)
        assert PaperStatus.processing == "processing"

    def test_enum_count(self):
        """PaperStatus should have exactly 4 values."""
        assert len(PaperStatus) == 4


class TestPaperModel:
    """Tests for Paper SQLAlchemy model."""

    def test_paper_has_correct_tablename(self):
        """Paper should use 'papers' table."""
        assert Paper.__tablename__ == "papers"

    @pytest.mark.asyncio
    async def test_paper_default_id_is_uuid(self, async_db_session: AsyncSession):
        """Paper should generate UUID string for id on insert."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()
        await async_db_session.refresh(paper)

        assert paper.id is not None
        # Validate it's a valid UUID format
        uuid.UUID(paper.id)

    @pytest.mark.asyncio
    async def test_paper_default_status_is_processing(
        self, async_db_session: AsyncSession
    ):
        """Paper should default to processing status on insert."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()
        await async_db_session.refresh(paper)

        assert paper.status == PaperStatus.processing

    @pytest.mark.asyncio
    async def test_paper_authors_defaults_to_empty_list(
        self, async_db_session: AsyncSession
    ):
        """Paper authors should default to empty list on insert."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()
        await async_db_session.refresh(paper)

        assert paper.authors == []

    def test_paper_nullable_fields(self):
        """Paper should allow null for optional fields."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        assert paper.date is None
        assert paper.source is None
        assert paper.published_date is None
        assert paper.content_text is None

    @pytest.mark.asyncio
    async def test_paper_persists_to_db(self, async_db_session: AsyncSession):
        """Paper should persist and retrieve from database."""
        paper = Paper(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            type=PaperType.research_paper,
            source="https://example.com",
            published_date=date(2024, 1, 15),
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Paper).where(Paper.id == paper.id)
        )
        loaded = result.scalar_one()

        assert loaded.title == "Test Paper"
        assert loaded.authors == ["Author 1", "Author 2"]
        assert loaded.type == PaperType.research_paper
        assert loaded.status == PaperStatus.processing
        assert loaded.source == "https://example.com"
        assert loaded.published_date == date(2024, 1, 15)

    @pytest.mark.asyncio
    async def test_paper_timestamps_auto_set(self, async_db_session: AsyncSession):
        """Paper should auto-set created_at and updated_at."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()
        await async_db_session.refresh(paper)

        assert paper.created_at is not None
        assert paper.updated_at is not None
        assert isinstance(paper.created_at, datetime)
        assert isinstance(paper.updated_at, datetime)

    @pytest.mark.asyncio
    async def test_paper_status_update(self, async_db_session: AsyncSession):
        """Paper status can be updated."""
        paper = Paper(title="Test", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        paper.status = PaperStatus.ready_for_review
        await async_db_session.commit()
        await async_db_session.refresh(paper)

        assert paper.status == PaperStatus.ready_for_review

    @pytest.mark.asyncio
    async def test_paper_content_text_stores_large_text(
        self, async_db_session: AsyncSession
    ):
        """Paper should store large content_text."""
        large_text = "x" * 100000  # 100KB
        paper = Paper(
            title="Test",
            type=PaperType.research_paper,
            content_text=large_text,
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Paper).where(Paper.id == paper.id)
        )
        loaded = result.scalar_one()
        assert loaded.content_text == large_text
