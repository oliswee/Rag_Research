"""Tests for pattern API routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from app.models import Paper, PaperStatus, PaperType, Pattern, PatternStatus
from app.workflows import ReviewResult


class TestReviewPatterns:
    """Tests for POST /api/papers/{paper_id}/patterns."""

    @pytest.mark.asyncio
    async def test_review_approve_success(self, client_with_db, async_db_session):
        """Review with approvals updates patterns and returns counts."""
        # Create paper with patterns
        paper = Paper(
            id="paper-1",
            title="Test Paper",
            status=PaperStatus.ready_for_review,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-1",
            paper_id="paper-1",
            name="Test Pattern",
            claim="Test claim",
            status=PatternStatus.pending,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        mock_result = ReviewResult(
            approved_count=1,
            rejected_count=0,
            errors=[],
            paper_status_updated=True,
            success=True,
        )

        with patch("app.routers.patterns.ReviewWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers/paper-1/patterns",
                json={"decisions": [{"temp_id": "pattern-1", "approved": True}]},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["saved_count"] == 1
        assert data["rejected_count"] == 0

    @pytest.mark.asyncio
    async def test_review_reject_success(self, client_with_db, async_db_session):
        """Review with rejections updates patterns."""
        paper = Paper(
            id="paper-2",
            title="Test Paper",
            status=PaperStatus.ready_for_review,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-2",
            paper_id="paper-2",
            name="Pattern to Reject",
            claim="Test",
            status=PatternStatus.pending,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        mock_result = ReviewResult(
            approved_count=0,
            rejected_count=1,
            errors=[],
            success=True,
        )

        with patch("app.routers.patterns.ReviewWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers/paper-2/patterns",
                json={"decisions": [{"temp_id": "pattern-2", "approved": False}]},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["rejected_count"] == 1

    @pytest.mark.asyncio
    async def test_review_mixed_decisions(self, client_with_db, async_db_session):
        """Review with mixed decisions returns correct counts."""
        paper = Paper(
            id="paper-3",
            title="Test Paper",
            status=PaperStatus.ready_for_review,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)
        await async_db_session.commit()

        mock_result = ReviewResult(
            approved_count=2,
            rejected_count=1,
            errors=[],
            success=True,
        )

        with patch("app.routers.patterns.ReviewWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers/paper-3/patterns",
                json={
                    "decisions": [
                        {"temp_id": "p1", "approved": True},
                        {"temp_id": "p2", "approved": True},
                        {"temp_id": "p3", "approved": False},
                    ]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["saved_count"] == 2
        assert data["rejected_count"] == 1

    def test_review_paper_not_found(self, client_with_db):
        """Review on nonexistent paper returns 404."""
        mock_result = ReviewResult(
            success=False,
            errors=["Paper nonexistent not found"],
        )

        with patch("app.routers.patterns.ReviewWorkflow") as MockWorkflow:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_result
            MockWorkflow.return_value = mock_instance

            response = client_with_db.post(
                "/api/papers/nonexistent/patterns",
                json={"decisions": [{"temp_id": "p1", "approved": True}]},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_review_empty_decisions_returns_400(self, client_with_db):
        """Review with empty decisions returns validation error."""
        response = client_with_db.post(
            "/api/papers/paper-1/patterns",
            json={"decisions": []},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestListPatterns:
    """Tests for GET /api/papers/{paper_id}/patterns."""

    @pytest.mark.asyncio
    async def test_list_patterns_success(self, client_with_db, async_db_session):
        """List patterns returns all patterns for paper."""
        paper = Paper(
            id="paper-list-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        for i in range(3):
            pattern = Pattern(
                id=f"pattern-{i}",
                paper_id="paper-list-1",
                name=f"Pattern {i}",
                claim=f"Claim {i}",
                status=PatternStatus.approved,
            )
            async_db_session.add(pattern)
        await async_db_session.commit()

        response = client_with_db.get("/api/papers/paper-list-1/patterns")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["patterns"]) == 3

    @pytest.mark.asyncio
    async def test_list_patterns_with_status_filter(
        self, client_with_db, async_db_session
    ):
        """List patterns filters by status."""
        paper = Paper(
            id="paper-filter-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        approved = Pattern(
            id="approved-1",
            paper_id="paper-filter-1",
            name="Approved",
            claim="Test",
            status=PatternStatus.approved,
        )
        rejected = Pattern(
            id="rejected-1",
            paper_id="paper-filter-1",
            name="Rejected",
            claim="Test",
            status=PatternStatus.rejected,
        )
        async_db_session.add(approved)
        async_db_session.add(rejected)
        await async_db_session.commit()

        response = client_with_db.get(
            "/api/papers/paper-filter-1/patterns?status=approved"
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["status"] == "approved"

    def test_list_patterns_paper_not_found(self, client_with_db):
        """List patterns on nonexistent paper returns 404."""
        response = client_with_db.get("/api/papers/nonexistent/patterns")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestUpdatePattern:
    """Tests for PUT /api/patterns/{pattern_id}."""

    @pytest.mark.asyncio
    async def test_update_pattern_success(self, client_with_db, async_db_session):
        """Update pattern modifies fields."""
        paper = Paper(
            id="paper-upd-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-upd-1",
            paper_id="paper-upd-1",
            name="Original Name",
            claim="Original claim",
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        with patch("app.routers.patterns.EmbeddingService") as MockEmbed:
            mock_embed = MagicMock()
            mock_embed.prepare_embedding_text.return_value = "text"
            mock_embed.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1536)
            )
            MockEmbed.return_value = mock_embed

            with patch("app.routers.patterns.VectorStoreService") as MockVS:
                mock_vs = MagicMock()
                mock_vs.upsert = AsyncMock()
                MockVS.return_value = mock_vs

                response = client_with_db.put(
                    "/api/patterns/pattern-upd-1",
                    json={"name": "Updated Name", "claim": "Updated claim"},
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["claim"] == "Updated claim"

    def test_update_pattern_not_found(self, client_with_db):
        """Update nonexistent pattern returns 404."""
        response = client_with_db.put(
            "/api/patterns/nonexistent",
            json={"name": "New Name"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_pattern_no_fields_returns_422(self, client_with_db):
        """Update with no fields returns validation error."""
        response = client_with_db.put(
            "/api/patterns/pattern-1",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_update_pattern_re_embeds_on_content_change(
        self, client_with_db, async_db_session
    ):
        """Update that changes content triggers re-embedding."""
        paper = Paper(
            id="paper-embed-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-embed-1",
            paper_id="paper-embed-1",
            name="Name",
            claim="Original",
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        with patch("app.routers.patterns.EmbeddingService") as MockEmbed:
            mock_embed = MagicMock()
            mock_embed.prepare_embedding_text.return_value = "text"
            mock_embed.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1536)
            )
            MockEmbed.return_value = mock_embed

            with patch("app.routers.patterns.VectorStoreService") as MockVS:
                mock_vs = MagicMock()
                mock_vs.upsert = AsyncMock()
                MockVS.return_value = mock_vs

                response = client_with_db.put(
                    "/api/patterns/pattern-embed-1",
                    json={"claim": "Updated claim"},
                )

        assert response.status_code == status.HTTP_200_OK
        mock_embed.generate_embedding.assert_called_once()
        mock_vs.upsert.assert_called_once()


class TestGetPattern:
    """Tests for GET /api/patterns/{pattern_id}."""

    @pytest.mark.asyncio
    async def test_get_pattern_success(self, client_with_db, async_db_session):
        """Get pattern returns full details with paper title."""
        paper = Paper(
            id="paper-get-1",
            title="Test Paper Title",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-get-1",
            paper_id="paper-get-1",
            name="Test Pattern",
            claim="Test claim",
            evidence="Test evidence",
            context="Test context",
            tags=["tag1", "tag2"],
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        response = client_with_db.get("/api/patterns/pattern-get-1")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "pattern-get-1"
        assert data["paper_id"] == "paper-get-1"
        assert data["paper_title"] == "Test Paper Title"
        assert data["name"] == "Test Pattern"
        assert data["claim"] == "Test claim"
        assert data["evidence"] == "Test evidence"
        assert data["context"] == "Test context"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["status"] == "approved"

    @pytest.mark.asyncio
    async def test_get_pattern_with_some_null_fields(
        self, client_with_db, async_db_session
    ):
        """Get pattern handles partially null fields."""
        paper = Paper(
            id="paper-get-partial-1",
            title="Paper With Partial Nulls",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        # Pattern must have at least one content field due to CHECK constraint
        pattern = Pattern(
            id="pattern-get-partial-1",
            paper_id="paper-get-partial-1",
            name="Minimal Pattern",
            claim="This is the claim",  # At least one content field required
            status=PatternStatus.pending,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        response = client_with_db.get("/api/patterns/pattern-get-partial-1")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["claim"] == "This is the claim"
        assert data["evidence"] is None
        assert data["context"] is None
        assert data["tags"] == []

    def test_get_pattern_not_found(self, client_with_db):
        """Get nonexistent pattern returns 404."""
        response = client_with_db.get("/api/patterns/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"]["code"] == "NOT_FOUND"


class TestDeletePattern:
    """Tests for DELETE /api/patterns/{pattern_id}."""

    @pytest.mark.asyncio
    async def test_delete_pattern_success(self, client_with_db, async_db_session):
        """Delete pattern returns 204."""
        paper = Paper(
            id="paper-del-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-del-1",
            paper_id="paper-del-1",
            name="Delete Me",
            claim="Test",
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        with patch("app.routers.patterns.VectorStoreService") as MockVS:
            mock_vs = MagicMock()
            mock_vs.delete_pattern = MagicMock()
            MockVS.return_value = mock_vs

            response = client_with_db.delete("/api/patterns/pattern-del-1")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_pattern_not_found(self, client_with_db):
        """Delete nonexistent pattern returns 404."""
        response = client_with_db.delete("/api/patterns/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_pattern_removes_from_vector_store(
        self, client_with_db, async_db_session
    ):
        """Delete pattern removes embedding from Qdrant."""
        paper = Paper(
            id="paper-vec-del-1",
            title="Test Paper",
            status=PaperStatus.active,
            type=PaperType.research_paper,
        )
        async_db_session.add(paper)

        pattern = Pattern(
            id="pattern-vec-del-1",
            paper_id="paper-vec-del-1",
            name="Pattern",
            claim="Test",
            status=PatternStatus.approved,
        )
        async_db_session.add(pattern)
        await async_db_session.commit()

        with patch("app.routers.patterns.VectorStoreService") as MockVS:
            mock_vs = MagicMock()
            mock_vs.delete_pattern = MagicMock()
            MockVS.return_value = mock_vs

            response = client_with_db.delete("/api/patterns/pattern-vec-del-1")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_vs.delete_pattern.assert_called_once_with("pattern-vec-del-1")
