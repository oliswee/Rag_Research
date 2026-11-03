import pytest

from app.config import settings
from app.routers.query import submit_query
from app.schemas.query import QueryRequest
from tests.test_api.test_m6_query import generation_model as generation_model_fixture
from tests.test_services.test_evidence_registry import local_source

generation_model = generation_model_fixture


@pytest.mark.asyncio
async def test_real_qa_response_registers_local_feedback_manifest(
    async_db_session, client_with_db, monkeypatch, tmp_path, generation_model
):
    paper, source = await local_source(async_db_session)

    async def retrieve(*args, **kwargs):
        return [source], "bm25"

    monkeypatch.setattr("app.routers.query.retrieve", retrieve)
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = await submit_query(
        QueryRequest(
            query="memory accuracy",
            retrieval_mode="bm25",
            paper_ids=[paper.id],
            rewrite_query=False,
        ),
        async_db_session,
    )
    assert response["feedback_available"] is True
    manifest = client_with_db.get(f"/api/feedback/runs/{response['run_id']}")
    assert manifest.status_code == 200
    saved = client_with_db.post(
        "/api/feedback",
        json={
            "run_id": response["run_id"],
            "reason": "unsupported_citation",
            "source_id": source["chunk_id"],
        },
    )
    assert (
        saved.status_code == 201 and saved.json()["source"]["source_origin"] == "local"
    )
