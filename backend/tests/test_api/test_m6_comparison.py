import json

import pytest

from app.config import settings
from app.services.context_packing import pack_context
from app.workflows.local_web import LocalRun
from tests.test_services.test_evidence_registry import local_source


def test_local_only_is_enforced_before_paper_lookup(client_with_db, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    response = client_with_db.post(
        "/api/research/compare", json={"paper_ids": ["a", "b"], "local_only": True}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "LOCAL_ONLY_REMOTE_PROVIDER"


@pytest.mark.asyncio
async def test_compare_api_returns_verified_claims_and_resolvable_local_citations(
    async_db_session, client_with_db, monkeypatch, tmp_path
):
    samples = [await local_source(async_db_session) for _ in range(2)]
    sources = {p.id: s for p, s in samples}

    async def local(db, policy, history, trace, **kwargs):
        return LocalRun(
            pack_context([sources[policy.paper_ids[0]]], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    class Model:
        async def complete_json(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "evidence_blocks" in payload:
                block = json.loads(payload["evidence_blocks"].split("\n\n")[0])
                source_id, paper_id, quote = (
                    block["citations"][0]["id"],
                    block["paper_ids"][0],
                    block["text"],
                )
                return {
                    "claims": [
                        {
                            "paper_id": paper_id,
                            "dimension": dim,
                            "text": quote.strip(),
                            "source_ids": [source_id],
                            "supporting_quotes": {source_id: quote},
                        }
                        for dim in payload["dimensions"]
                    ]
                }
            return {
                "verdicts": [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "supported",
                        "characteristics": {},
                    }
                    for claim in payload["claims"]
                ]
            }

        async def close(self):
            pass

    monkeypatch.setattr("app.workflows.comparison.local_retrieval", local)
    monkeypatch.setattr("app.services.llm_client.LLMClient", lambda **kwargs: Model())
    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    monkeypatch.setattr(settings, "retrieval_trace_path", str(tmp_path / "trace.jsonl"))
    response = client_with_db.post(
        "/api/research/compare", json={"paper_ids": list(sources)}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] and data["verification"]["calls"] == 4
    for row in data["rows"]:
        for cell in row["cells"]:
            assert cell["support_status"] == "supported"
            assert cell["claims"][0]["paper_id"] == cell["paper_id"]
            citation = cell["citations"][0]
            assert citation["raw_quote"] == sources[cell["paper_id"]]["excerpt"]
            assert citation["pdf_page"] is None
            resolved = client_with_db.get(citation["source_url"])
            assert resolved.status_code == 200
        assert row["comparisons"][0]["status"] == "insufficient_evidence"
    trace = json.loads((tmp_path / "trace.jsonl").read_text())
    assert "Memory accuracy is 95 percent." not in str(trace)


@pytest.mark.parametrize("ids", [["a"], ["a", "a"], ["a", "b", "c", "d", "e", "f"]])
def test_comparison_scope_validation(client_with_db, ids):
    assert (
        client_with_db.post(
            "/api/research/compare", json={"paper_ids": ids}
        ).status_code
        == 422
    )
