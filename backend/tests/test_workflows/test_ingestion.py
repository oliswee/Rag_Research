"""Tests for ingestion workflow."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import PaperStatus, PatternStatus
from app.schemas.paper import PaperMetadataInput
from app.schemas.pattern import ExtractedPatternResponse
from app.services import PDFParseError
from app.workflows.ingestion import IngestionInput, IngestionResult, IngestionWorkflow


class TestIngestionWorkflow:
    """Tests for IngestionWorkflow."""

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create mock database session."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        return db

    @pytest.fixture
    def mock_pdf_parser(self) -> MagicMock:
        """Create mock PDF parser."""
        parser = MagicMock()
        parser.parse.return_value = MagicMock(
            text="Sample paper text content",
            page_count=10,
            metadata={"title": "PDF Title"},
        )
        return parser

    @pytest.fixture
    def mock_metadata_extractor(self) -> AsyncMock:
        """Create mock metadata extractor."""
        extractor = AsyncMock()
        extractor.extract.return_value = MagicMock(
            title="Extracted Title",
            authors=["Author One", "Author Two"],
            source="arXiv",
            published_date=None,
            type=None,
        )
        return extractor

    @pytest.fixture
    def mock_pattern_extractor(self) -> AsyncMock:
        """Create mock pattern extractor."""
        extractor = AsyncMock()
        # Create pattern mocks with proper attribute access
        # Note: MagicMock(name=...) doesn't set .name attribute, it names the mock
        pattern1 = MagicMock()
        pattern1.name = "Pattern 1"
        pattern1.claim = "Claim 1"
        pattern1.evidence = "Evidence 1"
        pattern1.context = "Context 1"
        pattern1.tags = ["finding"]
        pattern1.confidence = MagicMock(value="high")

        pattern2 = MagicMock()
        pattern2.name = "Pattern 2"
        pattern2.claim = "Claim 2"
        pattern2.evidence = None
        pattern2.context = "Context 2"
        pattern2.tags = []
        pattern2.confidence = MagicMock(value="medium")

        extractor.extract.return_value = MagicMock(
            patterns=MagicMock(patterns=[pattern1, pattern2]),
            warnings=[],
            is_successful=True,
            pattern_count=2,
        )
        return extractor

    @pytest.fixture
    def workflow(
        self,
        mock_db: AsyncMock,
        mock_pdf_parser: MagicMock,
        mock_metadata_extractor: AsyncMock,
        mock_pattern_extractor: AsyncMock,
        monkeypatch,
    ) -> IngestionWorkflow:
        """Create workflow with mocked dependencies."""
        from types import SimpleNamespace

        # These orchestration tests use a mock session; real source transactions
        # and vector-failure persistence are covered in test_structured_ingestion.
        monkeypatch.setattr(
            "app.workflows.ingestion.stage_generation",
            AsyncMock(return_value=SimpleNamespace(children=[])),
        )
        return IngestionWorkflow(
            db=mock_db,
            pdf_parser=mock_pdf_parser,
            metadata_extractor=mock_metadata_extractor,
            pattern_extractor=mock_pattern_extractor,
        )

    @pytest.mark.asyncio
    async def test_happy_path_creates_paper_and_patterns(
        self,
        workflow: IngestionWorkflow,
        mock_db: AsyncMock,
    ) -> None:
        """Should create paper and patterns on successful ingestion."""
        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.success is True
        assert result.pattern_count == 2
        assert len(result.patterns) == 2
        assert result.error is None
        # Verify database operations
        assert mock_db.add.call_count >= 3  # 1 paper + 2 patterns
        assert mock_db.commit.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_paper_id(
        self,
        workflow: IngestionWorkflow,
    ) -> None:
        """Should return valid paper ID."""
        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.paper_id is not None
        assert len(result.paper_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_returns_pattern_responses_for_review(
        self,
        workflow: IngestionWorkflow,
    ) -> None:
        """Should return ExtractedPatternResponse objects for UI."""
        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert len(result.patterns) == 2
        assert all(isinstance(p, ExtractedPatternResponse) for p in result.patterns)
        assert result.patterns[0].name == "Pattern 1"
        assert result.patterns[0].confidence == "high"

    @pytest.mark.asyncio
    async def test_pdf_parse_failure_returns_error(
        self,
        workflow: IngestionWorkflow,
        mock_pdf_parser: MagicMock,
    ) -> None:
        """Should return error result if PDF parsing fails."""
        mock_pdf_parser.parse.side_effect = PDFParseError("Invalid PDF")

        input_data = IngestionInput(pdf_bytes=b"invalid pdf")

        result = await workflow.run(input_data)

        assert result.success is False
        assert "PDF parsing failed" in result.error
        assert result.pattern_count == 0

    @pytest.mark.asyncio
    async def test_metadata_extraction_failure_continues(
        self,
        workflow: IngestionWorkflow,
        mock_metadata_extractor: AsyncMock,
    ) -> None:
        """Should continue with defaults if metadata extraction fails."""
        mock_metadata_extractor.extract.side_effect = Exception("LLM error")

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.success is True
        assert any("Metadata extraction failed" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_pattern_extraction_failure_continues(
        self,
        workflow: IngestionWorkflow,
        mock_pattern_extractor: AsyncMock,
        mock_db: AsyncMock,
    ) -> None:
        """Should still create paper if pattern extraction fails."""
        mock_pattern_extractor.extract.side_effect = Exception("Extraction error")

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.success is True
        assert result.pattern_count == 0
        assert any("Pattern extraction failed" in w for w in result.warnings)
        assert mock_db.commit.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_patterns_is_valid(
        self,
        workflow: IngestionWorkflow,
        mock_pattern_extractor: AsyncMock,
    ) -> None:
        """Should handle case with no patterns extracted."""
        mock_pattern_extractor.extract.return_value = MagicMock(
            patterns=MagicMock(patterns=[]),
            warnings=[],
            is_successful=True,
            pattern_count=0,
        )

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.success is True
        assert result.pattern_count == 0
        assert result.patterns == []

    @pytest.mark.asyncio
    async def test_database_commit_failure_returns_error(
        self,
        workflow: IngestionWorkflow,
        mock_db: AsyncMock,
    ) -> None:
        """Should return error if database commit fails."""
        mock_db.commit.side_effect = Exception("Database error")

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert result.success is False
        assert "Database commit failed" in result.error
        mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_propagates_extraction_warnings(
        self,
        workflow: IngestionWorkflow,
        mock_pattern_extractor: AsyncMock,
    ) -> None:
        """Should propagate warnings from pattern extraction."""
        mock_pattern_extractor.extract.return_value = MagicMock(
            patterns=MagicMock(patterns=[]),
            warnings=["Citation E5 not found", "Unused evidence"],
            is_successful=True,
            pattern_count=0,
        )

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert "Citation E5 not found" in result.warnings
        assert "Unused evidence" in result.warnings

    @pytest.mark.asyncio
    async def test_user_metadata_takes_precedence(
        self,
        workflow: IngestionWorkflow,
        mock_metadata_extractor: AsyncMock,
        mock_db: AsyncMock,
    ) -> None:
        """User-provided metadata should take precedence over extraction."""
        user_metadata = PaperMetadataInput(
            title="User Title",
            authors=["User Author"],
        )

        input_data = IngestionInput(
            pdf_bytes=b"fake pdf content",
            user_metadata=user_metadata,
        )

        await workflow.run(input_data)

        # Check that metadata extractor was called with user_metadata
        mock_metadata_extractor.extract.assert_called_once()
        call_kwargs = mock_metadata_extractor.extract.call_args.kwargs
        assert call_kwargs["user_metadata"] == user_metadata

    @pytest.mark.asyncio
    async def test_patterns_have_pending_status(
        self,
        workflow: IngestionWorkflow,
        mock_db: AsyncMock,
    ) -> None:
        """Created patterns should have pending status."""
        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        await workflow.run(input_data)

        # Check Pattern objects added to database
        pattern_adds = [
            call
            for call in mock_db.add.call_args_list
            if hasattr(call.args[0], "status") and hasattr(call.args[0], "claim")
        ]
        for call in pattern_adds:
            pattern = call.args[0]
            assert pattern.status == PatternStatus.pending

    @pytest.mark.asyncio
    async def test_paper_has_ready_for_review_status(
        self,
        workflow: IngestionWorkflow,
        mock_db: AsyncMock,
    ) -> None:
        """Created paper should have ready_for_review status."""
        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        await workflow.run(input_data)

        # Find Paper object in add calls
        paper_adds = [
            call
            for call in mock_db.add.call_args_list
            if hasattr(call.args[0], "content_text")
        ]
        assert len(paper_adds) >= 1
        paper = paper_adds[0].args[0]
        assert paper.status == PaperStatus.ready_for_review

    @pytest.mark.asyncio
    async def test_unsuccessful_extraction_adds_warning(
        self,
        workflow: IngestionWorkflow,
        mock_pattern_extractor: AsyncMock,
    ) -> None:
        """Should add warning if extraction verification failed."""
        mock_pattern_extractor.extract.return_value = MagicMock(
            patterns=MagicMock(patterns=[]),
            warnings=[],
            is_successful=False,  # Verification failed
            pattern_count=0,
        )

        input_data = IngestionInput(pdf_bytes=b"fake pdf content")

        result = await workflow.run(input_data)

        assert any("manual review" in w.lower() for w in result.warnings)


class TestIngestionInput:
    """Tests for IngestionInput dataclass."""

    def test_creates_with_pdf_bytes_only(self) -> None:
        """Should create with just PDF bytes."""
        input_data = IngestionInput(pdf_bytes=b"content")
        assert input_data.pdf_bytes == b"content"
        assert input_data.user_metadata is None

    def test_creates_with_user_metadata(self) -> None:
        """Should accept optional user metadata."""
        metadata = PaperMetadataInput(title="Test")
        input_data = IngestionInput(pdf_bytes=b"content", user_metadata=metadata)
        assert input_data.user_metadata == metadata


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_creates_success_result(self) -> None:
        """Should create successful result."""
        result = IngestionResult(
            paper_id="abc123",
            pattern_count=5,
            patterns=[],
            warnings=["warning"],
            success=True,
        )
        assert result.paper_id == "abc123"
        assert result.pattern_count == 5
        assert result.success is True
        assert result.error is None

    def test_creates_error_result(self) -> None:
        """Should create error result."""
        result = IngestionResult(
            paper_id="abc123",
            pattern_count=0,
            success=False,
            error="Something failed",
        )
        assert result.success is False
        assert result.error == "Something failed"

    def test_defaults_to_empty_lists(self) -> None:
        """Should default to empty lists."""
        result = IngestionResult(
            paper_id="abc123",
            pattern_count=0,
        )
        assert result.patterns == []
        assert result.warnings == []
