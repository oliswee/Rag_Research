import pytest

from app.config import settings
from app.models import Paper, PaperChunk, PaperStatus, PaperType
from app.routers.query import follow_up_query, submit_query
from app.schemas.query import FollowUpRequest, QueryRequest


@pytest.mark.asyncio
async def test_source_query_and_followup_use_local_evidence(
    async_db_session, monkeypatch
):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    paper = Paper(
        title="Source",
        authors=[],
        type=PaperType.research_paper,
        status=PaperStatus.ready_for_review,
        import_source="openalex",
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    async_db_session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text="Active retrieval improves answers.",
            content_scope="abstract",
            checksum="x",
        )
    )
    await async_db_session.commit()
    first = await submit_query(
        QueryRequest(query="active retrieval", retrieval_mode="dense"), async_db_session
    )
    assert first["retrieval_mode"] == "bm25_fallback"
    assert first["evidence"][0]["paper_id"] == paper.id
    assert first["evidence"][0]["page_start"] is None
    second = await follow_up_query(
        FollowUpRequest(query="What improves?", session_id=first["session_id"]),
        async_db_session,
    )
    assert second["session_id"] == first["session_id"]
    assert second["evidence"]
