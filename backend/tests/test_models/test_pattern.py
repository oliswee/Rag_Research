"""Tests for Pattern model and enum."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Paper, PaperType
from app.models.pattern import Pattern, PatternStatus


class TestPatternStatusEnum:
    """Tests for PatternStatus enum."""

    def test_enum_values(self):
        """PatternStatus should have expected values."""
        assert PatternStatus.pending.value == "pending"
        assert PatternStatus.approved.value == "approved"
        assert PatternStatus.rejected.value == "rejected"

    def test_enum_is_string_enum(self):
        """PatternStatus should be a string enum."""
        assert isinstance(PatternStatus.pending, str)
        assert PatternStatus.pending == "pending"

    def test_enum_count(self):
        """PatternStatus should have exactly 3 values."""
        assert len(PatternStatus) == 3


class TestPatternModel:
    """Tests for Pattern SQLAlchemy model."""

    def test_pattern_has_correct_tablename(self):
        """Pattern should use 'patterns' table."""
        assert Pattern.__tablename__ == "patterns"

    @pytest.mark.asyncio
    async def test_pattern_default_id_is_uuid(self, async_db_session: AsyncSession):
        """Pattern should generate UUID string for id on insert."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.id is not None
        uuid.UUID(pattern.id)

    @pytest.mark.asyncio
    async def test_pattern_default_status_is_pending(
        self, async_db_session: AsyncSession
    ):
        """Pattern should default to pending status on insert."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.status == PatternStatus.pending

    @pytest.mark.asyncio
    async def test_pattern_timestamps_auto_set(self, async_db_session: AsyncSession):
        """Pattern should auto-set created_at and updated_at."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.created_at is not None
        assert pattern.updated_at is not None
        assert isinstance(pattern.created_at, datetime)


class TestPatternContentValidation:
    """Tests for Pattern content field validation."""

    def test_has_valid_content_with_claim_only(self):
        """has_valid_content should return True when claim is set."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            claim="Some claim",
        )
        assert pattern.has_valid_content is True

    def test_has_valid_content_with_evidence_only(self):
        """has_valid_content should return True when evidence is set."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            evidence="Some evidence",
        )
        assert pattern.has_valid_content is True

    def test_has_valid_content_with_context_only(self):
        """has_valid_content should return True when context is set."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            context="Some context",
        )
        assert pattern.has_valid_content is True

    def test_has_valid_content_with_all_fields(self):
        """has_valid_content should return True when all fields set."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            claim="Claim",
            evidence="Evidence",
            context="Context",
        )
        assert pattern.has_valid_content is True

    def test_has_valid_content_with_no_content(self):
        """has_valid_content should return False when no content fields set."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
        )
        assert pattern.has_valid_content is False

    def test_has_valid_content_with_empty_strings(self):
        """has_valid_content should return False for empty strings."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            claim="",
            evidence="",
            context="",
        )
        assert pattern.has_valid_content is False

    def test_has_valid_content_with_whitespace_only(self):
        """has_valid_content should return False for whitespace-only strings."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            claim="   ",
            evidence="\t\n",
            context="  \t  ",
        )
        assert pattern.has_valid_content is False

    def test_has_valid_content_with_mixed_whitespace_and_content(self):
        """has_valid_content should return True if any field has real content."""
        pattern = Pattern(
            paper_id="test-id",
            name="Test",
            claim="   ",  # whitespace only
            evidence="Real content",  # has content
            context=None,
        )
        assert pattern.has_valid_content is True


class TestPatternContentFields:
    """Tests for Pattern content_fields property."""

    def test_content_fields_with_claim_only(self):
        """content_fields should list only claim when set."""
        pattern = Pattern(paper_id="test", name="Test", claim="Claim")
        assert pattern.content_fields == ["claim"]

    def test_content_fields_with_evidence_only(self):
        """content_fields should list only evidence when set."""
        pattern = Pattern(paper_id="test", name="Test", evidence="Evidence")
        assert pattern.content_fields == ["evidence"]

    def test_content_fields_with_context_only(self):
        """content_fields should list only context when set."""
        pattern = Pattern(paper_id="test", name="Test", context="Context")
        assert pattern.content_fields == ["context"]

    def test_content_fields_with_multiple_fields(self):
        """content_fields should list all populated fields in order."""
        pattern = Pattern(
            paper_id="test",
            name="Test",
            claim="Claim",
            context="Context",
        )
        assert pattern.content_fields == ["claim", "context"]

    def test_content_fields_with_all_fields(self):
        """content_fields should list all three when all set."""
        pattern = Pattern(
            paper_id="test",
            name="Test",
            claim="Claim",
            evidence="Evidence",
            context="Context",
        )
        assert pattern.content_fields == ["claim", "evidence", "context"]

    def test_content_fields_empty_when_no_content(self):
        """content_fields should be empty when no content set."""
        pattern = Pattern(paper_id="test", name="Test")
        assert pattern.content_fields == []

    def test_content_fields_excludes_whitespace_only(self):
        """content_fields should exclude whitespace-only strings."""
        pattern = Pattern(
            paper_id="test",
            name="Test",
            claim="   ",  # whitespace only - excluded
            evidence="Real evidence",  # included
            context="\t\n",  # whitespace only - excluded
        )
        assert pattern.content_fields == ["evidence"]

    def test_content_fields_empty_when_all_whitespace(self):
        """content_fields should be empty when all fields are whitespace-only."""
        pattern = Pattern(
            paper_id="test",
            name="Test",
            claim="   ",
            evidence="\t",
            context="  \n  ",
        )
        assert pattern.content_fields == []


class TestPatternPersistence:
    """Tests for Pattern database persistence."""

    @pytest.mark.asyncio
    async def test_pattern_persists_all_fields(self, async_db_session: AsyncSession):
        """Pattern should persist and retrieve all fields from database."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Main assertion here",
            evidence="Supporting quote",
            context="Implications and limitations",
            tags=["methodology", "finding"],
            custom_fields={"confidence": 0.9, "source_page": 5},
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Pattern).where(Pattern.id == pattern.id)
        )
        loaded = result.scalar_one()

        assert loaded.name == "Test Pattern"
        assert loaded.claim == "Main assertion here"
        assert loaded.evidence == "Supporting quote"
        assert loaded.context == "Implications and limitations"
        assert loaded.tags == ["methodology", "finding"]
        assert loaded.custom_fields == {"confidence": 0.9, "source_page": 5}
        assert loaded.status == PatternStatus.pending

    @pytest.mark.asyncio
    async def test_pattern_nullable_fields(self, async_db_session: AsyncSession):
        """Pattern should allow null for optional fields."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Minimal Pattern",
            claim="Just a claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.evidence is None
        assert pattern.context is None
        assert pattern.tags is None
        assert pattern.custom_fields is None
        assert pattern.embedding_id is None

    @pytest.mark.asyncio
    async def test_pattern_status_update(self, async_db_session: AsyncSession):
        """Pattern status can be updated."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        pattern.status = PatternStatus.approved
        pattern.embedding_id = "qdrant-point-id"
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.status == PatternStatus.approved
        assert pattern.embedding_id == "qdrant-point-id"


class TestPatternRelationship:
    """Tests for Pattern-Paper relationship."""

    @pytest.mark.asyncio
    async def test_pattern_has_paper_relationship(self, async_db_session: AsyncSession):
        """Pattern should have access to parent paper."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Pattern).where(Pattern.id == pattern.id)
        )
        loaded = result.scalar_one()

        # Access paper through relationship
        assert loaded.paper_id == paper.id

    @pytest.mark.asyncio
    async def test_paper_has_patterns_relationship(
        self, async_db_session: AsyncSession
    ):
        """Paper should have access to child patterns."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern1 = Pattern(paper_id=paper.id, name="Pattern 1", claim="Claim 1")
        pattern2 = Pattern(paper_id=paper.id, name="Pattern 2", evidence="Evidence 2")
        async_db_session.add_all([pattern1, pattern2])
        await async_db_session.commit()

        # Use selectinload for async eager loading
        result = await async_db_session.execute(
            select(Paper)
            .where(Paper.id == paper.id)
            .options(selectinload(Paper.patterns))
        )
        loaded_paper = result.scalar_one()

        assert len(loaded_paper.patterns) == 2
        assert {p.name for p in loaded_paper.patterns} == {"Pattern 1", "Pattern 2"}

    @pytest.mark.asyncio
    async def test_cascade_delete_patterns_when_paper_deleted(
        self, async_db_session: AsyncSession
    ):
        """Patterns should be deleted when parent paper is deleted."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        pattern = Pattern(
            paper_id=paper.id,
            name="Test Pattern",
            claim="Test claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        pattern_id = pattern.id

        # Delete paper
        await async_db_session.delete(paper)
        await async_db_session.commit()

        # Pattern should be gone
        result = await async_db_session.execute(
            select(Pattern).where(Pattern.id == pattern_id)
        )
        assert result.scalar_one_or_none() is None


class TestPatternCheckConstraint:
    """Tests for Pattern CHECK constraint (ck_patterns_has_content)."""

    @pytest.mark.asyncio
    async def test_insert_pattern_without_content_raises_integrity_error(
        self, async_db_session: AsyncSession
    ):
        """Inserting a pattern with no content fields should raise IntegrityError."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        # Try to insert pattern with no content fields
        pattern = Pattern(
            paper_id=paper.id,
            name="Empty Pattern",
            # No claim, evidence, or context
        )
        async_db_session.add(pattern)

        with pytest.raises(IntegrityError) as exc_info:
            await async_db_session.commit()

        # Verify it's the CHECK constraint that failed
        assert (
            "ck_patterns_has_content" in str(exc_info.value).lower()
            or "check" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_insert_pattern_with_content_succeeds(
        self, async_db_session: AsyncSession
    ):
        """Inserting a pattern with at least one content field should succeed."""
        paper = Paper(title="Test Paper", type=PaperType.research_paper)
        async_db_session.add(paper)
        await async_db_session.commit()

        # Insert pattern with claim only
        pattern = Pattern(
            paper_id=paper.id,
            name="Valid Pattern",
            claim="This is a valid claim",
        )
        async_db_session.add(pattern)
        await async_db_session.commit()
        await async_db_session.refresh(pattern)

        assert pattern.id is not None
        assert pattern.claim == "This is a valid claim"
