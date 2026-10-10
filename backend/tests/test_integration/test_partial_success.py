"""Integration tests for partial success scenarios.

These tests verify the system handles partial failures gracefully,
continuing where possible rather than failing entirely.
"""

from unittest.mock import MagicMock, patch

from fastapi import status


class TestIngestionPartialSuccess:
    """Test partial success during paper ingestion."""

    def test_metadata_fails_patterns_succeed(
        self,
        integration_client,
        mock_pattern_extractor,
    ):
        """Paper created with patterns even if metadata extraction fails."""
        with patch("app.workflows.ingestion.MetadataExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value

            async def failing_extract(*args, **kwargs):
                raise Exception("Metadata service down")

            mock_extractor.extract = failing_extract

            with patch(
                "app.workflows.ingestion.PatternExtractor",
                return_value=mock_pattern_extractor,
            ):
                # Provide type to avoid NULL constraint violation
                response = integration_client.post(
                    "/api/papers",
                    data={
                        "text": "Research paper about AI agents...",
                        "type": "research_paper",
                    },
                )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Paper was created
        assert data["paper_id"] is not None

        # Patterns were extracted
        assert len(data["patterns"]) == 2

        # Warning about metadata failure
        assert any("Metadata" in w for w in data["extraction_warnings"])

    def test_patterns_fail_paper_still_created(
        self,
        integration_client,
        mock_metadata_extractor,
    ):
        """Paper created even if pattern extraction fails completely."""
        with (
            patch(
                "app.workflows.ingestion.MetadataExtractor",
                return_value=mock_metadata_extractor,
            ),
            patch("app.workflows.ingestion.PatternExtractor") as MockExtractor,
        ):
            mock_extractor = MockExtractor.return_value

            async def failing_extract(*args, **kwargs):
                raise Exception("Pattern extraction crashed")

            mock_extractor.extract = failing_extract

            response = integration_client.post(
                "/api/papers",
                data={"text": "Research paper content..."},
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Paper exists
        assert data["paper_id"] is not None

        # No patterns
        assert len(data["patterns"]) == 0

        # Warning about extraction failure
        warnings = data["extraction_warnings"]
        assert any("Pattern extraction failed" in w for w in warnings)

        # Paper is still accessible
        paper_resp = integration_client.get(f"/api/papers/{data['paper_id']}")
        assert paper_resp.status_code == status.HTTP_200_OK

    def test_extraction_verification_warns_but_succeeds(
        self,
        integration_client,
        mock_metadata_extractor,
    ):
        """Extraction with verification warnings still returns patterns."""
        with (
            patch(
                "app.workflows.ingestion.MetadataExtractor",
                return_value=mock_metadata_extractor,
            ),
            patch("app.workflows.ingestion.PatternExtractor") as MockExtractor,
        ):
            mock_extractor = MockExtractor.return_value

            # Create patterns with low confidence
            pattern = MagicMock()
            pattern.name = "Uncertain Pattern"
            pattern.claim = "Maybe this is true"
            pattern.evidence = None
            pattern.context = None
            pattern.tags = []
            pattern.confidence = MagicMock(value="low")

            async def extract_with_warnings(*args, **kwargs):
                result = MagicMock()
                result.patterns = MagicMock(patterns=[pattern])
                result.warnings = [
                    "Citation E3 not found in evidence",
                    "Pattern has low confidence",
                ]
                result.is_successful = False  # Verification failed
                result.pattern_count = 1
                return result

            mock_extractor.extract = extract_with_warnings

            response = integration_client.post(
                "/api/papers",
                data={"text": "Ambiguous research paper..."},
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Patterns returned despite warnings
        assert len(data["patterns"]) == 1

        # Warnings propagated
        assert any("Citation" in w for w in data["extraction_warnings"])
        assert any("manual review" in w.lower() for w in data["extraction_warnings"])


class TestReviewPartialSuccess:
    """Test partial success during pattern review."""

    def test_embedding_fails_pattern_still_approved(
        self,
        integration_client,
        mock_vector_store,
    ):
        """Pattern approved even if embedding generation fails."""
        # Upload paper
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]

        # Make embedding fail but let review succeed
        with patch("app.workflows.review.EmbeddingService") as MockService:
            mock_service = MockService.return_value
            mock_service.prepare_embedding_text.return_value = "text"

            async def failing_embed(text):
                raise Exception("Embedding API timeout")

            mock_service.generate_embedding = failing_embed

            response = integration_client.post(
                f"/api/papers/{paper_id}/patterns",
                json={
                    "decisions": [{"temp_id": patterns[0]["temp_id"], "approved": True}]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Pattern still approved
        assert data["saved_count"] == 1

        # Pattern accessible
        patterns_resp = integration_client.get(
            f"/api/papers/{paper_id}/patterns?status=approved"
        )
        assert len(patterns_resp.json()["patterns"]) == 1

    def test_vector_store_fails_pattern_still_approved(
        self,
        integration_client,
        mock_embedding_service,
    ):
        """Pattern approved even if vector store upsert fails."""
        # Upload paper
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]

        # Make vector store fail
        with (
            patch(
                "app.workflows.review.EmbeddingService",
                return_value=mock_embedding_service,
            ),
            patch("app.workflows.review.VectorStoreService") as MockStore,
        ):
            mock_store = MockStore.return_value

            async def failing_upsert(*args, **kwargs):
                raise Exception("Qdrant cluster unavailable")

            mock_store.upsert = failing_upsert

            response = integration_client.post(
                f"/api/papers/{paper_id}/patterns",
                json={
                    "decisions": [{"temp_id": patterns[0]["temp_id"], "approved": True}]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Pattern approved despite vector store failure
        assert data["saved_count"] == 1

    def test_mixed_success_failure_in_batch(
        self,
        integration_client,
        mock_embedding_service,
    ):
        """Batch review with some failures still processes valid patterns."""
        # Upload paper
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]

        # Include one valid and one invalid pattern ID
        decisions = [
            {"temp_id": patterns[0]["temp_id"], "approved": True},
            {"temp_id": "nonexistent-pattern-id", "approved": True},
        ]

        response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Workflow continues with partial success (logs errors, returns 200)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Valid pattern was approved
        assert data["saved_count"] == 1

        # Invalid pattern was skipped (logged as error internally)


class TestQueryPartialSuccess:
    """Test partial success during queries."""

    def test_synthesis_fails_gracefully(
        self,
        integration_client,
        mock_embedding_service,
        mock_vector_store,
        mock_session_manager,
    ):
        """Synthesis failure returns error but doesn't crash."""
        # Add data to query
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

        # Make LLM synthesis fail
        with (
            patch(
                "app.workflows.query.EmbeddingService",
                return_value=mock_embedding_service,
            ),
            patch(
                "app.workflows.query.VectorStoreService",
                return_value=mock_vector_store,
            ),
            patch(
                "app.workflows.query.SessionManager",
                return_value=mock_session_manager,
            ),
            patch("app.workflows.query.LLMClient") as MockLLM,
        ):
            mock_llm = MockLLM.return_value

            async def failing_synthesis(*args, **kwargs):
                raise Exception("Claude API rate limited")

            mock_llm.complete_json = failing_synthesis

            response = integration_client.post(
                "/api/query",
                json={"query": "What patterns exist?"},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "synthesis failed" in data["detail"]["message"].lower()


class TestDeletionPartialSuccess:
    """Test partial success during deletions."""

    def test_vector_store_delete_fails_paper_still_deleted(
        self,
        integration_client,
    ):
        """Paper deleted from DB even if vector store delete fails."""
        # Upload and review paper
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

        # Make vector store delete fail
        with patch("app.routers.papers.VectorStoreService") as MockStore:
            mock_store = MockStore.return_value

            async def failing_delete(paper_id):
                raise Exception("Qdrant delete failed")

            mock_store.delete_by_paper_id = failing_delete

            response = integration_client.delete(f"/api/papers/{paper_id}")

        # Paper still deleted (vector store failure is logged but not blocking)
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Confirm paper is gone
        get_resp = integration_client.get(f"/api/papers/{paper_id}")
        assert get_resp.status_code == status.HTTP_404_NOT_FOUND


class TestUpdatePartialSuccess:
    """Test partial success during pattern updates."""

    def test_reembedding_fails_update_still_saved(
        self,
        integration_client,
    ):
        """Pattern update saved even if re-embedding fails."""
        # Upload and approve pattern
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]
        pattern_id = patterns[0]["temp_id"]

        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": [{"temp_id": pattern_id, "approved": True}]},
        )

        # Make embedding fail during update
        with patch("app.routers.patterns.EmbeddingService") as MockService:
            mock_service = MockService.return_value
            mock_service.prepare_embedding_text.return_value = "text"

            async def failing_embed(text):
                raise Exception("Embedding API down")

            mock_service.generate_embedding = failing_embed

            response = integration_client.put(
                f"/api/patterns/{pattern_id}",
                json={"claim": "Updated claim that should be saved"},
            )

        # Update still succeeds (embedding failure is logged)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["claim"] == "Updated claim that should be saved"
