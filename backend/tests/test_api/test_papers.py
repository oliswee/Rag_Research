"""Tests for paper API routes."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from sqlalchemy import select

from app.models import Paper, PaperStatus, PaperType, Pattern, PatternStatus
from app.schemas.pattern import ExtractedPatternResponse
from app.workflows import IngestionResult


class TestUploadPaper:
    """Tests for POST /api/papers."""

    def test_upload_pdf_success(self, client_with_db, async_db_session):
        """Upload PDF returns 201 with paper_id and patterns."""
        # Create a minimal valid PDF
        pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"

        mock_result = IngestionResult(
            paper_id="test-uuid-123",
            pattern_count=2,
            patterns=[
                ExtractedPatternResponse(
                    temp_id="pattern-1",
                    name="Test Pattern",
                    claim="Test claim",
                    evidence=None,
                    context=None,
                    tags=["test"],
                    confidence="high",
                ),
            ],
            warnings=[],
            success=True,
        )

        with patch("app.routers.papers.IngestionWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
            response = client_with_db.post("/api/papers", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["paper_id"] == "test-uuid-123"
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["name"] == "Test Pattern"

    def test_upload_text_success(self, client_with_db):
        """Upload text returns 201 with paper_id and patterns."""
        mock_result = IngestionResult(
            paper_id="test-uuid-456",
            pattern_count=1,
            patterns=[
                ExtractedPatternResponse(
                    temp_id="pattern-2",
                    name="Text Pattern",
                    claim="From text",
                    evidence=None,
                    context=None,
                    tags=[],
                    confidence="medium",
                ),
            ],
            warnings=[],
            success=True,
        )

        with patch("app.routers.papers.IngestionWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers",
                data={"text": "This is a research paper about AI agents..."},
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["paper_id"] == "test-uuid-456"

    def test_upload_with_metadata(self, client_with_db):
        """Upload with user metadata includes it in response."""
        mock_result = IngestionResult(
            paper_id="test-uuid-789",
            pattern_count=0,
            patterns=[],
            warnings=[],
            success=True,
        )

        with patch("app.routers.papers.IngestionWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers",
                data={
                    "text": "Paper content here...",
                    "title": "Custom Title",
                    "authors": "Author One, Author Two",
                    "source": "https://arxiv.org/abs/1234",
                    "type": "research_paper",
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        # Verify workflow was called with user metadata
        MockWorkflow.return_value.run.assert_called_once()

    def test_upload_no_file_or_text_returns_400(self, client_with_db):
        """Upload without file or text returns validation error."""
        response = client_with_db.post("/api/papers", data={})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["detail"]["code"] == "VALIDATION_ERROR"

    def test_upload_workflow_failure_returns_500(self, client_with_db):
        """Workflow failure returns 500 with error details."""
        mock_result = IngestionResult(
            paper_id="",
            pattern_count=0,
            patterns=[],
            warnings=[],
            success=False,
            error="PDF parsing failed: Invalid format",
        )

        with patch("app.routers.papers.IngestionWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers",
                data={"text": "Some content"},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["detail"]["code"] == "EXTRACTION_FAILED"

    def test_upload_returns_warnings(self, client_with_db):
        """Upload returns extraction warnings in response."""
        mock_result = IngestionResult(
            paper_id="test-uuid",
            pattern_count=1,
            patterns=[],
            warnings=["Metadata extraction failed", "Low confidence patterns"],
            success=True,
        )

        with patch("app.routers.papers.IngestionWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers",
                data={"text": "Paper content"},
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert len(data["extraction_warnings"]) == 2


class TestListPapers:
    """Tests for GET /api/papers."""

    @pytest.mark.asyncio
    async def test_list_papers_empty(self, client_with_db, async_db_session):
        """List papers returns empty list when no papers exist."""
        response = client_with_db.get("/api/papers")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["papers"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_papers_returns_papers(self, client_with_db, async_db_session):
        """List papers returns all papers."""
        # Create test papers
        paper1 = Paper(
            id="paper-1",
            title="First Paper",
            authors=["Author A"],
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        paper2 = Paper(
            id="paper-2",
            title="Second Paper",
            authors=["Author B"],
            status=PaperStatus.ready_for_review,
            type=PaperType.blog_post,
        )
        async_db_session.add(paper1)
        async_db_session.add(paper2)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["papers"]) == 2

    @pytest.mark.asyncio
    async def test_list_papers_with_status_filter(
        self, client_with_db, async_db_session
    ):
        """List papers filters by status."""
        paper1 = Paper(
            id="paper-1",
            title="Active Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        paper2 = Paper(
            id="paper-2",
            title="Review Paper",
            status=PaperStatus.ready_for_review,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper1)
        async_db_session.add(paper2)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers?status=active")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["papers"][0]["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_papers_with_type_filter(self, client_with_db, async_db_session):
        """List papers filters by type."""
        paper1 = Paper(
            id="paper-1",
            title="Research Paper",
            type=PaperType.research_paper,
            status=PaperStatus.active,
        )
        paper2 = Paper(
            id="paper-2",
            title="Blog Post",
            type=PaperType.blog_post,
            status=PaperStatus.active,
        )
        async_db_session.add(paper1)
        async_db_session.add(paper2)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers?type=blog_post")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["papers"][0]["type"] == "blog_post"

    @pytest.mark.asyncio
    async def test_list_papers_with_search(self, client_with_db, async_db_session):
        """List papers searches in title."""
        paper1 = Paper(
            id="paper-1",
            title="Agent Memory Research",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        paper2 = Paper(
            id="paper-2",
            title="LLM Prompting",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper1)
        async_db_session.add(paper2)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers?search=Agent")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert "Agent" in data["papers"][0]["title"]

    @pytest.mark.asyncio
    async def test_list_papers_with_pagination(self, client_with_db, async_db_session):
        """List papers supports pagination."""
        # Create 5 papers
        for i in range(5):
            paper = Paper(
                id=f"paper-{i}",
                title=f"Paper {i}",
                status=PaperStatus.active,
                type=PaperType.research_paper,
            )
            async_db_session.add(paper)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers?limit=2&offset=2")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 5
        assert len(data["papers"]) == 2

    @pytest.mark.asyncio
    async def test_list_papers_includes_pattern_count(
        self, client_with_db, async_db_session
    ):
        """List papers includes pattern count for each paper."""
        paper = Paper(
            id="paper-1",
            title="Paper with patterns",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        # Add patterns
        for i in range(3):
            pattern = Pattern(
                id=f"pattern-{i}",
                paper_id="paper-1",
                name=f"Pattern {i}",
                claim="Test claim",
                status=PatternStatus.approved,
            )
            async_db_session.add(pattern)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["papers"][0]["pattern_count"] == 3


class TestGetPaper:
    """Tests for GET /api/papers/{paper_id}."""

    @pytest.mark.asyncio
    async def test_get_paper_success(self, client_with_db, async_db_session):
        """Get paper returns full paper details."""
        paper = Paper(
            id="paper-123",
            title="Test Paper",
            authors=["Author A", "Author B"],
            source="https://example.com",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers/paper-123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "paper-123"
        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Author A", "Author B"]
        assert data["status"] == "active"

    def test_get_paper_not_found(self, client_with_db):
        """Get nonexistent paper returns 404."""
        response = client_with_db.get("/api/papers/nonexistent-id")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"]["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_get_paper_includes_pattern_count(
        self, client_with_db, async_db_session
    ):
        """Get paper includes pattern count."""
        paper = Paper(
            id="paper-456",
            title="Paper with patterns",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        for i in range(2):
            pattern = Pattern(
                id=f"pattern-{i}",
                paper_id="paper-456",
                name=f"Pattern {i}",
                claim="Test",
                status=PatternStatus.approved,
            )
            async_db_session.add(pattern)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers/paper-456")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["pattern_count"] == 2


class TestDeletePaper:
    """Tests for DELETE /api/papers/{paper_id}."""

    @pytest.mark.asyncio
    async def test_delete_paper_success(self, client_with_db, async_db_session):
        """Delete paper returns 204 and removes paper."""
        paper = Paper(
            id="paper-to-delete",
            title="Delete Me",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        with patch("app.routers.papers.VectorStoreService") as MockVectorStore:
            mock_vs = MagicMock()
            mock_vs.delete_by_paper_id = AsyncMock()
            MockVectorStore.return_value = mock_vs

            response = client_with_db.delete("/api/papers/paper-to-delete")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify paper is deleted
        result = await async_db_session.execute(
            select(Paper).where(Paper.id == "paper-to-delete")
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_paper_removes_patterns(
        self, client_with_db, async_db_session
    ):
        """Delete paper also removes associated patterns."""
        paper = Paper(
            id="paper-with-patterns",
            title="Has Patterns",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-to-delete",
            paper_id="paper-with-patterns",
            name="Pattern",
            claim="Claim",
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        with patch("app.routers.papers.VectorStoreService") as MockVectorStore:
            mock_vs = MagicMock()
            mock_vs.delete_by_paper_id = AsyncMock()
            MockVectorStore.return_value = mock_vs

            response = client_with_db.delete("/api/papers/paper-with-patterns")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify patterns are deleted
        result = await async_db_session.execute(
            select(Pattern).where(Pattern.paper_id == "paper-with-patterns")
        )
        assert list(result.scalars().all()) == []

    def test_delete_paper_not_found(self, client_with_db):
        """Delete nonexistent paper returns 404."""
        response = client_with_db.delete("/api/papers/nonexistent-id")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"]["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_delete_paper_removes_from_vector_store(
        self, client_with_db, async_db_session
    ):
        """Delete paper removes embeddings from Qdrant."""
        paper = Paper(
            id="paper-vec",
            title="Vector Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        with patch("app.routers.papers.VectorStoreService") as MockVectorStore:
            mock_vs = MagicMock()
            mock_vs.delete_by_paper_id = AsyncMock()
            MockVectorStore.return_value = mock_vs

            response = client_with_db.delete("/api/papers/paper-vec")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_vs.delete_by_paper_id.assert_called_once_with("paper-vec")
