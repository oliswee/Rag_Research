"""Tests for PatternExtractor service (3-pass pipeline)."""

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from app.services.llm_client import LLMError
from app.services.pattern_extractor import (
    PatternExtractionResult,
    PatternExtractor,
)
from app.services.prompts.verification import VerificationStatus


class TestPatternExtractor:
    """Tests for PatternExtractor service."""

    @pytest.fixture
    def mock_haiku_client(self) -> AsyncMock:
        """Create mock Haiku client."""
        return AsyncMock()

    @pytest.fixture
    def mock_sonnet_client(self) -> AsyncMock:
        """Create mock Sonnet client."""
        return AsyncMock()

    @pytest.fixture
    def extractor(
        self, mock_haiku_client: AsyncMock, mock_sonnet_client: AsyncMock
    ) -> PatternExtractor:
        """Create extractor with mocked clients."""
        extractor = PatternExtractor(
            haiku_client=mock_haiku_client,
            sonnet_client=mock_sonnet_client,
        )
        return extractor

    @pytest.fixture
    def sample_evidence_response(self) -> dict:
        """Sample evidence inventory response."""
        return {
            "paper_title": "Test Paper",
            "evidence_items": [
                {
                    "id": "E1",
                    "quote": "First evidence quote",
                    "location": "Abstract",
                    "type": "finding",
                    "importance": "high",
                },
                {
                    "id": "E2",
                    "quote": "Second evidence quote",
                    "location": "Methods",
                    "type": "methodology",
                    "importance": "medium",
                },
            ],
            "paper_type": "empirical",
            "core_contribution_summary": "Test contribution",
        }

    @pytest.fixture
    def sample_extraction_response(self) -> dict:
        """Sample pattern extraction response."""
        return {
            "patterns": [
                {
                    "rank": 1,
                    "name": "Test Pattern",
                    "claim": "Test claim [E1]",
                    "evidence": "Test evidence [E2]",
                    "context": "Test context [E1, E2]",
                    "tags": ["finding"],
                    "evidence_ids": ["E1", "E2"],
                    "confidence": "high",
                }
            ],
            "total_evidence_used": 2,
            "gaps_identified": [],
        }

    @pytest.fixture
    def sample_verification_response(self) -> dict:
        """Sample verification response."""
        return {
            "verification_status": "pass",
            "citation_issues": [],
            "unused_evidence": [],
            "accuracy_concerns": [],
            "overall_quality": "high",
            "improvement_suggestions": [],
        }

    @pytest.mark.asyncio
    async def test_extract_full_pipeline_success(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
        sample_verification_response: dict,
    ) -> None:
        """Should complete full 3-pass pipeline successfully."""
        # Setup mocks
        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,  # Pass 1
            sample_verification_response,  # Pass 3
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        result = await extractor.extract("Test paper text")

        assert isinstance(result, PatternExtractionResult)
        assert result.is_successful is True
        assert result.pattern_count == 1
        assert len(result.evidence_inventory.evidence_items) == 2
        assert result.verification.verification_status == VerificationStatus.passed

    @pytest.mark.asyncio
    async def test_extract_uses_correct_models(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
        sample_verification_response: dict,
    ) -> None:
        """Should use Haiku for Pass 1/3 and Sonnet for Pass 2."""
        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,
            sample_verification_response,
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        await extractor.extract("Test paper text")

        # Haiku called twice (Pass 1 and Pass 3)
        assert mock_haiku_client.complete_json.call_count == 2
        # Sonnet called once (Pass 2)
        assert mock_sonnet_client.complete_json.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_pass1_failure_returns_failed_result(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
    ) -> None:
        """Should return failed result if Pass 1 fails."""
        mock_haiku_client.complete_json.side_effect = LLMError("API Error")

        result = await extractor.extract("Test paper text")

        assert result.is_successful is False
        assert result.pattern_count == 0
        assert "Extraction failed" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_extract_pass2_failure_returns_failed_result(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
    ) -> None:
        """Should return failed result if Pass 2 fails."""
        mock_haiku_client.complete_json.return_value = sample_evidence_response
        mock_sonnet_client.complete_json.side_effect = LLMError("API Error")

        result = await extractor.extract("Test paper text")

        assert result.is_successful is False
        assert result.pattern_count == 0
        # Evidence was still extracted
        assert len(result.evidence_inventory.evidence_items) == 2

    @pytest.mark.asyncio
    async def test_extract_pass3_failure_returns_patterns_with_warning(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
    ) -> None:
        """Should return patterns with warning if Pass 3 fails."""
        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,  # Pass 1 succeeds
            LLMError("Verification failed"),  # Pass 3 fails
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        result = await extractor.extract("Test paper text")

        # Patterns should still be returned
        assert result.pattern_count == 1
        # But with warning status
        assert result.verification.verification_status == VerificationStatus.warn
        assert any("Verification failed" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_extract_uses_chunking_for_large_documents(
        self,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
    ) -> None:
        """Should chunk large documents and add warning."""
        from app.services.document_chunker import DocumentChunker

        # Use a small token limit to force chunking
        chunker = DocumentChunker(token_limit=100)  # 400 chars

        extractor = PatternExtractor(
            haiku_client=mock_haiku_client,
            sonnet_client=mock_sonnet_client,
            chunker=chunker,
        )

        # Mock responses for chunked extraction (multiple calls per chunk)
        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,  # Chunk 1 Pass 1
            sample_evidence_response,  # Chunk 2 Pass 1
            {"merge": [], "remove": [], "gaps": []},  # Synthesis pass
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        long_text = "x" * 2000  # ~500 tokens, will be chunked at 100 token limit

        result = await extractor.extract(long_text)

        # Should have warning about chunking
        assert any("chunk" in w.lower() for w in result.warnings)
        assert result.was_chunked is True

    @pytest.mark.asyncio
    async def test_extract_includes_metadata_context(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
        sample_verification_response: dict,
    ) -> None:
        """Should include metadata context in extraction prompt."""
        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,
            sample_verification_response,
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        await extractor.extract("Paper text", metadata_context="Title: Test Paper")

        # Check that metadata was included in Pass 2 prompt
        call_args = mock_sonnet_client.complete_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt", "")
        assert "Title: Test Paper" in user_prompt

    @pytest.mark.asyncio
    async def test_extract_builds_warnings_from_verification(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
    ) -> None:
        """Should build warnings from verification issues."""
        verification_with_issues = {
            "verification_status": "warn",
            "citation_issues": [
                {
                    "pattern_rank": 1,
                    "field": "claim",
                    "issue": "Citation E99 not found",
                    "severity": "warning",
                }
            ],
            "unused_evidence": ["E5", "E6", "E7", "E8"],
            "accuracy_concerns": [],
            "overall_quality": "medium",
            "improvement_suggestions": [],
        }

        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,
            verification_with_issues,
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        result = await extractor.extract("Test text")

        assert any("E99 not found" in w for w in result.warnings)
        assert any("Unused evidence" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_extract_overrides_llm_status_with_computed(
        self,
        extractor: PatternExtractor,
        mock_haiku_client: AsyncMock,
        mock_sonnet_client: AsyncMock,
        sample_evidence_response: dict,
        sample_extraction_response: dict,
    ) -> None:
        """Should override LLM status with computed status when different."""
        # LLM says pass but has error-level issues
        verification_with_errors = {
            "verification_status": "pass",  # LLM incorrectly says pass
            "citation_issues": [
                {
                    "pattern_rank": 1,
                    "field": "claim",
                    "issue": "Invalid citation",
                    "severity": "error",  # But has error
                }
            ],
            "unused_evidence": [],
            "accuracy_concerns": [],
            "overall_quality": "medium",
            "improvement_suggestions": [],
        }

        mock_haiku_client.complete_json.side_effect = [
            sample_evidence_response,
            verification_with_errors,
        ]
        mock_sonnet_client.complete_json.return_value = sample_extraction_response

        result = await extractor.extract("Test text")

        # Should be overridden to fail
        assert result.verification.verification_status == VerificationStatus.fail


class TestPatternExtractionResult:
    """Tests for PatternExtractionResult dataclass."""

    def test_is_successful_true_for_pass(self) -> None:
        """is_successful should return True for pass status."""
        from app.services.prompts.evidence_inventory import EvidenceInventory
        from app.services.prompts.extraction import ExtractionResult
        from app.services.prompts.verification import VerificationResult

        result = PatternExtractionResult(
            patterns=ExtractionResult(),
            evidence_inventory=EvidenceInventory(paper_title="Test"),
            verification=VerificationResult(
                verification_status=VerificationStatus.passed
            ),
            warnings=[],
        )
        assert result.is_successful is True

    def test_is_successful_true_for_warn(self) -> None:
        """is_successful should return True for warn status."""
        from app.services.prompts.evidence_inventory import EvidenceInventory
        from app.services.prompts.extraction import ExtractionResult
        from app.services.prompts.verification import VerificationResult

        result = PatternExtractionResult(
            patterns=ExtractionResult(),
            evidence_inventory=EvidenceInventory(paper_title="Test"),
            verification=VerificationResult(
                verification_status=VerificationStatus.warn
            ),
            warnings=[],
        )
        assert result.is_successful is True

    def test_is_successful_false_for_fail(self) -> None:
        """is_successful should return False for fail status."""
        from app.services.prompts.evidence_inventory import EvidenceInventory
        from app.services.prompts.extraction import ExtractionResult
        from app.services.prompts.verification import VerificationResult

        result = PatternExtractionResult(
            patterns=ExtractionResult(),
            evidence_inventory=EvidenceInventory(paper_title="Test"),
            verification=VerificationResult(
                verification_status=VerificationStatus.fail
            ),
            warnings=[],
        )
        assert result.is_successful is False

    def test_pattern_count(self) -> None:
        """pattern_count should return number of patterns."""
        from app.services.prompts.evidence_inventory import EvidenceInventory
        from app.services.prompts.extraction import (
            ExtractedPatternRaw,
            ExtractionResult,
        )
        from app.services.prompts.verification import VerificationResult

        result = PatternExtractionResult(
            patterns=ExtractionResult(
                patterns=[
                    ExtractedPatternRaw(rank=1, name="Pattern 1"),
                    ExtractedPatternRaw(rank=2, name="Pattern 2"),
                ]
            ),
            evidence_inventory=EvidenceInventory(paper_title="Test"),
            verification=VerificationResult(),
            warnings=[],
        )
        assert result.pattern_count == 2


class TestPatternExtractorLazyInit:
    """Tests for lazy client initialization."""

    @pytest.mark.asyncio
    async def test_creates_haiku_client_on_demand(self) -> None:
        """Should create Haiku client when first needed."""
        with patch("app.services.pattern_extractor.LLMClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.complete_json.side_effect = LLMError("Test")
            mock_class.return_value = mock_client

            extractor = PatternExtractor()

            # Client not created yet
            assert extractor._haiku_client is None

            # Trigger client creation
            with contextlib.suppress(Exception):
                await extractor.extract("Test")

            # Client should be created with Haiku model
            mock_class.assert_called()

    @pytest.mark.asyncio
    async def test_creates_sonnet_client_on_demand(self) -> None:
        """Should create Sonnet client when first needed."""
        with patch("app.services.pattern_extractor.LLMClient") as mock_class:
            mock_haiku = AsyncMock()
            mock_haiku.complete_json.return_value = {
                "paper_title": "Test",
                "evidence_items": [],
            }

            mock_sonnet = AsyncMock()
            mock_sonnet.complete_json.side_effect = LLMError("Test")

            # Return different mocks for different calls
            mock_class.side_effect = [mock_haiku, mock_sonnet]

            extractor = PatternExtractor()

            with contextlib.suppress(Exception):
                await extractor.extract("Test")

            # Both clients should be created
            assert mock_class.call_count == 2
