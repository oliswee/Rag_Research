"""M5 supersedes the Pattern-default API with the local/Web evidence workflow."""

import json

import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.context_packing import pack_context
from app.services.session_manager import session_manager
from app.workflows.local_web import LocalRun


@pytest.fixture
def evidence_api(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    calls = []

    async def local(db, policy, history, trace, **kwargs):
        calls.append((policy, len(history)))
        return LocalRun(
            pack_context([], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    monkeypatch.setattr("app.routers.query.local_retrieval", local)
    return path, calls


def test_query_creates_session_and_records_abstention(client_with_db, evidence_api):
    path, calls = evidence_api
    response = client_with_db.post("/api/query", json={"query": "memory"})
    assert response.status_code == 200
    data = response.json()
    assert session_manager.get_session(data["session_id"]) is not None
    assert data["evidence"] == [] and data["sources"] == []
    assert data["confidence"] == "low" and "Insufficient" in data["response"]
    trace = json.loads(path.read_text().splitlines()[0])
    assert trace["run_id"] == data["run_id"] and trace["support_status"] == "abstain"
    assert calls[0][0].retrieval_mode == "hybrid"


@pytest.mark.parametrize("endpoint", ["/api/query", "/api/query/followup"])
def test_query_continues_real_session(client_with_db, evidence_api, endpoint):
    first = client_with_db.post("/api/query", json={"query": "memory"}).json()
    response = client_with_db.post(
        endpoint, json={"query": "latency", "session_id": first["session_id"]}
    )
    assert (
        response.status_code == 200
        and response.json()["session_id"] == first["session_id"]
    )
    assert evidence_api[1][-1][1] == 2


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("/api/query", {"query": ""}),
        ("/api/query/followup", {"query": "memory"}),
        ("/api/query/followup", {"query": "", "session_id": "s"}),
    ],
)
def test_invalid_query_is_422(client_with_db, endpoint, body):
    assert client_with_db.post(endpoint, json=body).status_code == 422


def test_unexpected_workflow_failure_is_traced(
    client_with_db, evidence_api, monkeypatch
):
    async def fail(*args, **kwargs):
        raise HTTPException(status_code=500, detail={"code": "QUERY_FAILED"})

    monkeypatch.setattr("app.routers.query.local_retrieval", fail)
    response = client_with_db.post("/api/query", json={"query": "memory"})
    assert response.status_code == 500
    assert json.loads(evidence_api[0].read_text())["support_status"] == "error"


def test_no_results_has_notices_and_no_fabricated_suggestions(
    client_with_db, evidence_api
):
    data = client_with_db.post("/api/query", json={"query": "memory"}).json()
    assert data["notices"] and data["follow_up_suggestions"] == []
    assert data["routing"]["local_retry_count"] == 1


@pytest.mark.parametrize("endpoint", ["/api/query", "/api/query/followup"])
def test_unknown_session_is_404(client_with_db, evidence_api, endpoint):
    assert (
        client_with_db.post(
            endpoint, json={"query": "memory", "session_id": "missing"}
        ).status_code
        == 404
    )
