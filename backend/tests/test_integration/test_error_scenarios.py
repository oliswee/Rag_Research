"""Integration tests for error handling scenarios.

These tests verify graceful error handling throughout the pipeline.
"""

from unittest.mock import patch

from fastapi import status

from app.services import PDFParseError


class TestUploadErrors:
    """Test error handling during paper upload."""

    def test_upload_no_file_or_text_returns_400(self, integration_client):
        """Upload without file or text returns validation error."""
        response = integration_client.post("/api/papers", data={})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["detail"]["code"] == "VALIDATION_ERROR"

    def test_upload_empty_text_returns_400(self, integration_client):
        """Upload with empty text returns validation error."""
        response = integration_client.post(
            "/api/papers",
            data={"text": "   "},  # Whitespace only
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_pdf_parse_error_returns_500(self, integration_client):
        """PDF parse failure returns error with details."""
        # Override the mock to raise an error
        with patch("app.workflows.ingestion.PDFParserService") as MockParser:
            mock_parser = MockParser.return_value
            mock_parser.parse.side_effect = PDFParseError("Invalid PDF format")

            response = integration_client.post(
                "/api/papers",
                files={"file": ("test.pdf", b"invalid pdf content", "application/pdf")},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["detail"]["code"] == "EXTRACTION_FAILED"
        assert "PDF parsing failed" in data["detail"]["message"]

    def test_upload_metadata_extraction_failure_continues(
        self,
        integration_client,
        mock_pattern_extractor,
    ):
        """Metadata extraction failure still creates paper with user defaults."""
        # Override metadata extractor to fail
        with patch("app.workflows.ingestion.MetadataExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value

            async def failing_extract(*args, **kwargs):
                raise Exception("LLM API error")

            mock_extractor.extract = failing_extract

            # Also need to patch pattern extractor to work
            with patch(
                "app.workflows.ingestion.PatternExtractor",
                return_value=mock_pattern_extractor,
            ):
                # Provide user metadata to avoid NULL type constraint
                response = integration_client.post(
                    "/api/papers",
                    data={
                        "text": "Research paper content...",
                        "type": "research_paper",  # Required when metadata fails
                    },
                )

        # Should still succeed with warnings
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["paper_id"] is not None
        assert any(
            "Metadata extraction failed" in w for w in data["extraction_warnings"]
        )

    def test_upload_pattern_extraction_failure_still_creates_paper(
        self,
        integration_client,
        mock_metadata_extractor,
    ):
        """Pattern extraction failure still creates paper (partial success)."""
        with (
            patch(
                "app.workflows.ingestion.MetadataExtractor",
                return_value=mock_metadata_extractor,
            ),
            patch("app.workflows.ingestion.PatternExtractor") as MockExtractor,
        ):
            mock_extractor = MockExtractor.return_value

            async def failing_extract(*args, **kwargs):
                raise Exception("Extraction service unavailable")

            mock_extractor.extract = failing_extract

            response = integration_client.post(
                "/api/papers",
                data={"text": "Research paper content..."},
            )

        # Should succeed but with no patterns and warnings
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["paper_id"] is not None
        assert len(data["patterns"]) == 0
        assert any(
            "Pattern extraction failed" in w for w in data["extraction_warnings"]
        )


class TestPaperAccessErrors:
    """Test error handling for paper access."""

    def test_get_nonexistent_paper_returns_404(self, integration_client):
        """Getting a nonexistent paper returns 404."""
        response = integration_client.get("/api/papers/nonexistent-uuid")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"]["code"] == "NOT_FOUND"

    def test_delete_nonexistent_paper_returns_404(self, integration_client):
        """Deleting a nonexistent paper returns 404."""
        response = integration_client.delete("/api/papers/nonexistent-uuid")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_patterns_for_nonexistent_paper_returns_404(self, integration_client):
        """Getting patterns for nonexistent paper returns 404."""
        response = integration_client.get("/api/papers/nonexistent-uuid/patterns")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestReviewErrors:
    """Test error handling during pattern review."""

    def test_review_nonexistent_paper_returns_404(self, integration_client):
        """Reviewing patterns for nonexistent paper returns 404."""
        response = integration_client.post(
            "/api/papers/nonexistent-uuid/patterns",
            json={"decisions": [{"temp_id": "pattern-1", "approved": True}]},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_review_nonexistent_pattern_logs_error(self, integration_client):
        """Reviewing nonexistent pattern ID logs error but continues."""
        # First upload a paper
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]

        # Try to review with invalid pattern ID
        response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": [{"temp_id": "nonexistent-pattern", "approved": True}]},
        )

        # Review continues with partial success (logs error, returns 200)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # No patterns were actually approved
        assert data["saved_count"] == 0

    def test_review_empty_decisions_returns_validation_error(self, integration_client):
        """Reviewing with empty decisions list returns validation error."""
        # Upload paper
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]

        # Review with empty decisions - API validates this
        response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": []},
        )

        # Empty decisions return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPatternAccessErrors:
    """Test error handling for pattern access."""

    def test_update_nonexistent_pattern_returns_404(self, integration_client):
        """Updating a nonexistent pattern returns 404."""
        response = integration_client.put(
            "/api/patterns/nonexistent-uuid",
            json={"claim": "Updated claim"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_nonexistent_pattern_returns_404(self, integration_client):
        """Deleting a nonexistent pattern returns 404."""
        response = integration_client.delete("/api/patterns/nonexistent-uuid")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestQueryErrors:
    """Test error handling during queries."""

    def test_query_empty_string_returns_422(self, integration_client):
        """Query with empty string returns validation error."""
        response = integration_client.post(
            "/api/query",
            json={"query": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_followup_without_session_returns_422(self, integration_client):
        """Follow-up without session_id returns validation error."""
        response = integration_client.post(
            "/api/query/followup",
            json={"query": "Tell me more"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_followup_empty_query_returns_422(self, integration_client):
        """Follow-up with empty query returns validation error."""
        response = integration_client.post(
            "/api/query/followup",
            json={"query": "", "session_id": "session-123"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_embedding_failure_falls_back_to_bm25(
        self,
        integration_client,
        mock_vector_store,
    ):
        """Source embedding failure remains visible and preserves keyword search."""
        # Add some data first
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]
        decisions = [{"temp_id": p["temp_id"], "approved": True} for p in patterns]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Now make embedding service fail
        with patch("app.services.embedding_service.EmbeddingService") as MockService:
            mock_service = MockService.return_value

            async def failing_embed(text):
                raise Exception("Embedding API unavailable")

            mock_service.generate_text_batch = failing_embed

            response = integration_client.post(
                "/api/query",
                json={"query": "Paper content"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["retrieval_mode"] == "bm25_fallback"
        assert data["evidence"]

    def test_query_vector_search_failure_falls_back_to_bm25(
        self,
        integration_client,
        mock_embedding_service,
    ):
        """Qdrant failure preserves source evidence with explicit fallback."""
        # Add some data
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]
        decisions = [{"temp_id": p["temp_id"], "approved": True} for p in patterns]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Make vector store fail
        with (
            patch(
                "app.services.embedding_service.EmbeddingService",
                return_value=mock_embedding_service,
            ),
            patch("app.workflows.source_query.SourceVectorIndex") as MockStore,
        ):
            mock_store = MockStore.return_value

            def failing_search(*args, **kwargs):
                raise Exception("Qdrant connection failed")

            mock_store.search = failing_search

            response = integration_client.post(
                "/api/query",
                json={"query": "Paper content"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["retrieval_mode"] == "bm25_fallback"
        assert response.json()["evidence"]


class TestInvalidRequestFormats:
    """Test handling of malformed requests."""

    def test_review_invalid_json_returns_422(self, integration_client):
        """Review with invalid JSON structure returns 422."""
        # Upload paper first
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]

        # Send malformed request
        response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": "not-a-list"},  # Should be a list
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_missing_query_field_returns_422(self, integration_client):
        """Query request missing 'query' field returns 422."""
        response = integration_client.post(
            "/api/query",
            json={"session_id": "session-123"},  # Missing 'query'
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_update_pattern_invalid_tags_type_returns_422(self, integration_client):
        """Updating pattern with invalid tags type returns 422."""
        # Create a pattern first
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]
        pattern_id = patterns[0]["temp_id"]

        # Approve it
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": [{"temp_id": pattern_id, "approved": True}]},
        )

        # Try to update with invalid tags
        response = integration_client.put(
            f"/api/patterns/{pattern_id}",
            json={"tags": "not-a-list"},  # Should be a list
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
