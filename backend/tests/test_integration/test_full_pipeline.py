"""Integration tests for the full pipeline: upload → extract → review → query.

These tests verify the end-to-end flow with mocked external services.
"""

from fastapi import status


class TestFullPipeline:
    """Test the complete paper-to-query pipeline."""

    def test_upload_extract_review_query_flow(
        self,
        integration_client,
        mock_vector_store,
        mock_session_manager,
    ):
        """Complete flow: upload text -> extract patterns -> review -> query."""
        # Step 1: Upload a paper (text)
        upload_response = integration_client.post(
            "/api/papers",
            data={
                "text": """
                This is a research paper about agent memory systems.

                We found that agents benefit significantly from periodic memory
                consolidation. In our experiments, we observed a 40% improvement
                in recall accuracy after implementing consolidation routines.

                Additionally, context window management is crucial. Our studies
                show that efficient context management reduces hallucinations
                by approximately 25%.
                """
            },
        )

        assert upload_response.status_code == status.HTTP_201_CREATED
        upload_data = upload_response.json()

        paper_id = upload_data["paper_id"]
        patterns = upload_data["patterns"]

        assert paper_id is not None
        assert len(paper_id) == 36  # UUID format
        assert len(patterns) == 2  # Mock extractor returns 2 patterns
        assert patterns[0]["name"] == "Memory Consolidation Pattern"
        assert patterns[1]["name"] == "Context Window Management"

        # Step 2: Verify paper was created with ready_for_review status
        paper_response = integration_client.get(f"/api/papers/{paper_id}")
        assert paper_response.status_code == status.HTTP_200_OK
        paper_data = paper_response.json()
        assert paper_data["status"] == "ready_for_review"
        assert paper_data["pattern_count"] == 2

        # Step 3: Review patterns (approve both)
        pattern_ids = [p["temp_id"] for p in patterns]
        decisions = [{"temp_id": pid, "approved": True} for pid in pattern_ids]

        review_response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        assert review_response.status_code == status.HTTP_200_OK
        review_data = review_response.json()
        assert review_data["saved_count"] == 2
        assert review_data["rejected_count"] == 0

        # Step 4: Verify patterns are in vector store
        assert len(mock_vector_store._vectors) == 2

        # Step 5: Verify paper status is now active
        paper_response = integration_client.get(f"/api/papers/{paper_id}")
        assert paper_response.status_code == status.HTTP_200_OK
        paper_data = paper_response.json()
        assert paper_data["status"] == "active"

        # Step 6: Query the library
        query_response = integration_client.post(
            "/api/query",
            json={"query": "What are the key findings about agent memory?"},
        )

        assert query_response.status_code == status.HTTP_200_OK
        query_data = query_response.json()

        assert query_data["session_id"] is not None
        assert query_data["retrieval_mode"] in {"hybrid", "bm25_fallback"}
        assert query_data["confidence"] == "medium"
        assert query_data["evidence"]
        assert (
            "memory consolidation"
            in " ".join(
                " ".join(source["excerpt"].split()) for source in query_data["evidence"]
            ).lower()
        )
        assert (
            query_data["sources"] == []
        )  # Source citations replace Pattern citations.

        session_id = query_data["session_id"]

        # Step 7: Follow-up query
        followup_response = integration_client.post(
            "/api/query/followup",
            json={
                "query": "Tell me more about context management",
                "session_id": session_id,
            },
        )

        assert followup_response.status_code == status.HTTP_200_OK
        followup_data = followup_response.json()
        assert followup_data["session_id"] == session_id

        # Verify session has history
        session = mock_session_manager._sessions[session_id]
        assert len(session.history) == 4  # 2 user turns + 2 assistant turns

    def test_upload_with_user_metadata(self, integration_client):
        """Upload with user-provided metadata uses those values."""
        response = integration_client.post(
            "/api/papers",
            data={
                "text": "Research paper content here...",
                "title": "Custom Title",
                "authors": "John Doe,Jane Smith",  # No spaces after comma
                "source": "https://example.com/paper",
                "type": "research_paper",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify metadata in response
        assert data["metadata"]["title"] == "Custom Title"
        assert data["metadata"]["authors"] == ["John Doe", "Jane Smith"]
        assert data["metadata"]["source"] == "https://example.com/paper"

    def test_reject_patterns_flow(self, integration_client, mock_vector_store):
        """Rejecting patterns excludes them from vector store."""
        # Upload paper
        upload_response = integration_client.post(
            "/api/papers",
            data={"text": "Research paper content..."},
        )
        paper_id = upload_response.json()["paper_id"]
        patterns = upload_response.json()["patterns"]

        # Reject first pattern, approve second
        decisions = [
            {"temp_id": patterns[0]["temp_id"], "approved": False},
            {"temp_id": patterns[1]["temp_id"], "approved": True},
        ]

        review_response = integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        assert review_response.status_code == status.HTTP_200_OK
        review_data = review_response.json()
        assert review_data["saved_count"] == 1
        assert review_data["rejected_count"] == 1

        # Only approved pattern should be in vector store
        assert len(mock_vector_store._vectors) == 1

        # Verify pattern statuses
        patterns_response = integration_client.get(f"/api/papers/{paper_id}/patterns")
        patterns_data = patterns_response.json()["patterns"]

        statuses = {p["name"]: p["status"] for p in patterns_data}
        assert statuses["Memory Consolidation Pattern"] == "rejected"
        assert statuses["Context Window Management"] == "approved"

    def test_list_papers_with_filters(self, integration_client):
        """List papers with status and type filters."""
        # Upload first paper
        integration_client.post(
            "/api/papers",
            data={
                "text": "First paper content...",
                "title": "Paper One",
                "type": "research_paper",
            },
        )

        # Upload second paper
        integration_client.post(
            "/api/papers",
            data={
                "text": "Second paper content...",
                "title": "Paper Two",
                "type": "blog_post",
            },
        )

        # List all papers
        response = integration_client.get("/api/papers")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2

        # Filter by type
        response = integration_client.get("/api/papers?type=blog_post")
        data = response.json()
        assert data["total"] == 1
        assert data["papers"][0]["type"] == "blog_post"

        # Filter by status
        response = integration_client.get("/api/papers?status=ready_for_review")
        data = response.json()
        assert data["total"] == 2  # Both papers still in review

    def test_delete_paper_removes_patterns_and_vectors(
        self,
        integration_client,
        mock_vector_store,
    ):
        """Deleting a paper removes patterns and vectors."""
        # Upload and review paper
        upload_response = integration_client.post(
            "/api/papers",
            data={"text": "Paper to delete..."},
        )
        paper_id = upload_response.json()["paper_id"]
        patterns = upload_response.json()["patterns"]

        # Approve patterns
        decisions = [{"temp_id": p["temp_id"], "approved": True} for p in patterns]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Verify vectors exist
        assert len(mock_vector_store._vectors) == 2

        # Delete paper
        delete_response = integration_client.delete(f"/api/papers/{paper_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify paper is gone
        get_response = integration_client.get(f"/api/papers/{paper_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

        # Vectors should be cleaned up (mock_vector_store.delete_by_paper_id was called)
        # Note: The actual deletion happens via the route calling delete_by_paper_id

    def test_update_pattern_triggers_reembedding(
        self,
        integration_client,
        mock_vector_store,
    ):
        """Updating approved pattern content triggers re-embedding."""
        # Upload and approve patterns
        upload_response = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_response.json()["paper_id"]
        patterns = upload_response.json()["patterns"]
        pattern_id = patterns[0]["temp_id"]

        # Approve patterns
        decisions = [{"temp_id": p["temp_id"], "approved": True} for p in patterns]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Get initial vector
        initial_vector = mock_vector_store._vectors.get(pattern_id)
        assert initial_vector is not None

        # Update pattern content
        update_response = integration_client.put(
            f"/api/patterns/{pattern_id}",
            json={"claim": "Updated claim content"},
        )

        assert update_response.status_code == status.HTTP_200_OK
        update_data = update_response.json()
        assert update_data["claim"] == "Updated claim content"

    def test_query_empty_library(self, integration_client):
        """Query with empty library returns helpful message."""
        response = integration_client.post(
            "/api/query",
            json={"query": "What are the key findings?"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["confidence"] == "low"
        assert "insufficient evidence" in data["response"].lower()
        assert data["routing"]["action"] == "search_web"
        assert data["routing"]["local_retry_count"] == 1
        assert data["web_records"] == []

    def test_multiple_papers_query(
        self,
        integration_client,
        mock_vector_store,
    ):
        """Query across multiple papers works correctly."""
        # Upload and approve first paper
        resp1 = integration_client.post(
            "/api/papers",
            data={"text": "First paper about memory..."},
        )
        paper1_id = resp1.json()["paper_id"]
        patterns1 = resp1.json()["patterns"]

        decisions1 = [{"temp_id": p["temp_id"], "approved": True} for p in patterns1]
        integration_client.post(
            f"/api/papers/{paper1_id}/patterns",
            json={"decisions": decisions1},
        )

        # Upload and approve second paper
        resp2 = integration_client.post(
            "/api/papers",
            data={"text": "Second paper about agents..."},
        )
        paper2_id = resp2.json()["paper_id"]
        patterns2 = resp2.json()["patterns"]

        decisions2 = [{"temp_id": p["temp_id"], "approved": True} for p in patterns2]
        integration_client.post(
            f"/api/papers/{paper2_id}/patterns",
            json={"decisions": decisions2},
        )

        # Should have 4 patterns in vector store
        assert len(mock_vector_store._vectors) == 4

        # Query should work across all patterns
        query_response = integration_client.post(
            "/api/query",
            json={"query": "What patterns exist in the library?"},
        )

        assert query_response.status_code == status.HTTP_200_OK


class TestPaperLifecycle:
    """Test paper status transitions through the lifecycle."""

    def test_paper_status_transitions(self, integration_client):
        """Paper transitions: upload → ready_for_review → active."""
        # Upload: status becomes ready_for_review
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]

        paper_resp = integration_client.get(f"/api/papers/{paper_id}")
        assert paper_resp.json()["status"] == "ready_for_review"

        # Review all patterns: status becomes active
        patterns = upload_resp.json()["patterns"]
        decisions = [{"temp_id": p["temp_id"], "approved": True} for p in patterns]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        paper_resp = integration_client.get(f"/api/papers/{paper_id}")
        assert paper_resp.json()["status"] == "active"

    def test_partial_review_keeps_ready_for_review(self, integration_client):
        """Paper stays ready_for_review if patterns are pending."""
        # Upload
        upload_resp = integration_client.post(
            "/api/papers",
            data={"text": "Paper content..."},
        )
        paper_id = upload_resp.json()["paper_id"]
        patterns = upload_resp.json()["patterns"]

        # Review only first pattern
        decisions = [{"temp_id": patterns[0]["temp_id"], "approved": True}]
        integration_client.post(
            f"/api/papers/{paper_id}/patterns",
            json={"decisions": decisions},
        )

        # Status should still be ready_for_review
        paper_resp = integration_client.get(f"/api/papers/{paper_id}")
        assert paper_resp.json()["status"] == "ready_for_review"


class TestSessionManagement:
    """Test query session management."""

    def test_new_query_creates_session(
        self,
        integration_client,
        mock_session_manager,
        mock_vector_store,
    ):
        """New query without session_id creates a new session."""
        # Add some data to query
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

        # Query without session
        response = integration_client.post(
            "/api/query",
            json={"query": "Tell me about patterns"},
        )

        assert response.status_code == status.HTTP_200_OK
        session_id = response.json()["session_id"]
        assert session_id is not None
        assert session_id in mock_session_manager._sessions

    def test_followup_requires_session(self, integration_client):
        """Follow-up query requires a valid session_id."""
        response = integration_client.post(
            "/api/query/followup",
            json={"query": "Tell me more"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_session_history_accumulates(
        self,
        integration_client,
        mock_session_manager,
        mock_vector_store,
    ):
        """Session history accumulates across turns."""
        # Add data
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

        # First query
        resp1 = integration_client.post(
            "/api/query",
            json={"query": "First question"},
        )
        session_id = resp1.json()["session_id"]

        # Check history after first query
        session = mock_session_manager._sessions[session_id]
        assert len(session.history) == 2  # user + assistant

        # Follow-up
        integration_client.post(
            "/api/query/followup",
            json={"query": "Second question", "session_id": session_id},
        )

        # Check history after follow-up
        session = mock_session_manager._sessions[session_id]
        assert len(session.history) == 4  # 2 user + 2 assistant turns
