"""Tests for review workflow."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import PaperStatus, PatternStatus
from app.schemas.pattern import PatternDecision
from app.workflows.review import ReviewResult, ReviewWorkflow


class TestReviewWorkflow:
    """Tests for ReviewWorkflow."""

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create mock database session."""
        db = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        return db

    @pytest.fixture
    def mock_embedding_service(self) -> MagicMock:
        """Create mock embedding service."""
        service = MagicMock()
        service.prepare_embedding_text.return_value = "prepared text"
        service.generate_embedding = AsyncMock(
            return_value=MagicMock(embedding=[0.1] * 1536)
        )
        return service

    @pytest.fixture
    def mock_vector_store(self) -> AsyncMock:
        """Create mock vector store service."""
        store = AsyncMock()
        store.upsert = AsyncMock()
        store.delete_pattern = MagicMock()
        return store

    @pytest.fixture
    def sample_paper(self) -> MagicMock:
        """Create sample paper mock."""
        paper = MagicMock()
        paper.id = "paper-123"
        paper.title = "Test Paper"
        paper.status = PaperStatus.ready_for_review
        return paper

    @pytest.fixture
    def sample_patterns(self, sample_paper: MagicMock) -> list[MagicMock]:
        """Create sample pattern mocks."""
        pattern1 = MagicMock()
        pattern1.id = "pattern-1"
        pattern1.paper_id = sample_paper.id
        pattern1.name = "Pattern One"
        pattern1.claim = "Claim one"
        pattern1.evidence = "Evidence one"
        pattern1.context = None
        pattern1.tags = ["tag1"]
        pattern1.status = PatternStatus.pending

        pattern2 = MagicMock()
        pattern2.id = "pattern-2"
        pattern2.paper_id = sample_paper.id
        pattern2.name = "Pattern Two"
        pattern2.claim = "Claim two"
        pattern2.evidence = None
        pattern2.context = "Context two"
        pattern2.tags = []
        pattern2.status = PatternStatus.pending

        return [pattern1, pattern2]

    @pytest.fixture
    def workflow(
        self,
        mock_db: AsyncMock,
        mock_embedding_service: MagicMock,
        mock_vector_store: AsyncMock,
    ) -> ReviewWorkflow:
        """Create workflow with mocked dependencies."""
        return ReviewWorkflow(
            db=mock_db,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_approve_patterns_generates_embeddings(
        self,
        workflow: ReviewWorkflow,
        mock_embedding_service: MagicMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should generate embeddings for approved patterns."""
        # Mock internal methods
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.approved_count == 1
        mock_embedding_service.generate_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_patterns_upserts_to_vector_store(
        self,
        workflow: ReviewWorkflow,
        mock_vector_store: AsyncMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should upsert approved patterns to vector store."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.approved_count == 1
        mock_vector_store.upsert.assert_called_once()
        call_kwargs = mock_vector_store.upsert.call_args.kwargs
        assert call_kwargs["pattern_id"] == "pattern-1"
        assert call_kwargs["payload"]["name"] == "Pattern One"

    @pytest.mark.asyncio
    async def test_approve_updates_pattern_status(
        self,
        workflow: ReviewWorkflow,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should update approved pattern status."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        await workflow.run(sample_paper.id, decisions)

        assert sample_patterns[0].status == PatternStatus.approved

    @pytest.mark.asyncio
    async def test_reject_updates_pattern_status(
        self,
        workflow: ReviewWorkflow,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should update rejected pattern status."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="pattern-1", approved=False)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.rejected_count == 1
        assert sample_patterns[0].status == PatternStatus.rejected

    @pytest.mark.asyncio
    async def test_rejecting_previously_approved_pattern_removes_vector(
        self,
        workflow: ReviewWorkflow,
        mock_vector_store: AsyncMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        sample_patterns[0].status = PatternStatus.approved
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        result = await workflow.run(
            sample_paper.id,
            [PatternDecision(temp_id="pattern-1", approved=False)],
        )

        assert result.rejected_count == 1
        assert sample_patterns[0].status == PatternStatus.rejected
        mock_vector_store.delete_pattern.assert_called_once_with("pattern-1")

    @pytest.mark.asyncio
    async def test_reject_does_not_embed(
        self,
        workflow: ReviewWorkflow,
        mock_embedding_service: MagicMock,
        mock_vector_store: AsyncMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should not generate embeddings for rejected patterns."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="pattern-1", approved=False)]

        await workflow.run(sample_paper.id, decisions)

        mock_embedding_service.generate_embedding.assert_not_called()
        mock_vector_store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_decisions(
        self,
        workflow: ReviewWorkflow,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should handle mixed approve/reject decisions."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        # Return different patterns for each call
        workflow._get_pattern = AsyncMock(
            side_effect=[sample_patterns[0], sample_patterns[1]]
        )
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [
            PatternDecision(temp_id="pattern-1", approved=True),
            PatternDecision(temp_id="pattern-2", approved=False),
        ]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.approved_count == 1
        assert result.rejected_count == 1
        assert sample_patterns[0].status == PatternStatus.approved
        assert sample_patterns[1].status == PatternStatus.rejected

    @pytest.mark.asyncio
    async def test_embedding_failure_continues(
        self,
        workflow: ReviewWorkflow,
        mock_embedding_service: MagicMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should continue if embedding fails, but log error."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)
        mock_embedding_service.generate_embedding.side_effect = Exception("API error")

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        # Pattern still approved despite embedding failure
        assert result.approved_count == 1
        assert sample_patterns[0].status == PatternStatus.approved
        assert any("embedding failed" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_vector_store_failure_continues(
        self,
        workflow: ReviewWorkflow,
        mock_vector_store: AsyncMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should continue if vector store fails, but log error."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)
        mock_vector_store.upsert.side_effect = Exception("Connection error")

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        # Pattern still approved despite vector store failure
        assert result.approved_count == 1
        assert sample_patterns[0].status == PatternStatus.approved
        assert any("vector store failed" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_pattern_id_adds_error(
        self,
        workflow: ReviewWorkflow,
        sample_paper: MagicMock,
    ) -> None:
        """Should add error for invalid pattern ID."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=None)
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)

        decisions = [PatternDecision(temp_id="nonexistent", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.approved_count == 0
        assert len(result.errors) >= 1
        assert any("nonexistent" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_paper_not_found_returns_error(
        self,
        workflow: ReviewWorkflow,
    ) -> None:
        """Should return error if paper not found."""
        workflow._get_paper = AsyncMock(return_value=None)

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run("nonexistent", decisions)

        assert result.success is False
        assert any("not found" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_database_commit_failure_returns_error(
        self,
        workflow: ReviewWorkflow,
        mock_db: AsyncMock,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should return error if database commit fails."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=False)
        mock_db.commit.side_effect = Exception("Database error")

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.success is False
        assert any("commit failed" in e for e in result.errors)
        mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_status_updated_when_all_reviewed(
        self,
        workflow: ReviewWorkflow,
        sample_paper: MagicMock,
        sample_patterns: list[MagicMock],
    ) -> None:
        """Should update paper status when all patterns reviewed."""
        workflow._get_paper = AsyncMock(return_value=sample_paper)
        workflow._get_pattern = AsyncMock(return_value=sample_patterns[0])
        workflow._update_paper_status_if_complete = AsyncMock(return_value=True)

        decisions = [PatternDecision(temp_id="pattern-1", approved=True)]

        result = await workflow.run(sample_paper.id, decisions)

        assert result.paper_status_updated is True


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        result = ReviewResult()
        assert result.approved_count == 0
        assert result.rejected_count == 0
        assert result.errors == []
        assert result.paper_status_updated is False
        assert result.success is True

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        result = ReviewResult(
            approved_count=5,
            rejected_count=2,
            errors=["error1"],
            paper_status_updated=True,
            success=True,
        )
        assert result.approved_count == 5
        assert result.rejected_count == 2
        assert result.errors == ["error1"]
        assert result.paper_status_updated is True
