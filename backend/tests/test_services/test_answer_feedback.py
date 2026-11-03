import importlib.util

import pytest
from sqlalchemy import select

from app.services.retrieval_trace import RetrievalTrace
from tests.test_services.test_evidence_registry import local_source, web_source

REASONS = [
    "wrong_paper",
    "missed_evidence",
    "unsupported_citation",
    "incorrect_interpretation",
    "outdated_conflicting",
    "web_result_irrelevant",
]


def api():
    assert importlib.util.find_spec("app.services.answer_feedback"), (
        "Local source-aware answer feedback is missing"
    )
    from app.services import answer_feedback

    return answer_feedback


async def run(db):
    service = api()
    paper, source = await local_source(db)
    external = web_source()
    external["paper_id"] = paper.id
    external["source_url"] = (
        "https://example.org/paper?api_key=development-canary&lang=en"
    )
    trace = RetrievalTrace(
        "private research question must not be persisted", {}, corpus_version="fixture"
    )
    response = {
        "run_id": trace.run_id,
        "evidence": [source, external],
        "citations": [],
        "web_records": [],
        "support_status": "evidence_only",
    }
    await service.record_run(db, trace, response, paper_ids=[paper.id])
    return service, trace.run_id, paper, source, external


@pytest.mark.asyncio
async def test_run_manifest_keeps_identity_without_private_text_or_url_credentials(
    async_db_session,
):
    service, run_id, paper, source, external = await run(async_db_session)
    manifest = await service.get_run(async_db_session, run_id)
    assert manifest["scope"]["paper_ids"] == [paper.id]
    assert {s["source_id"] for s in manifest["sources"]} == {
        source["chunk_id"],
        external["chunk_id"],
    }
    assert "Memory accuracy" not in str(manifest)
    assert "private research" not in str(manifest)
    assert "development-canary" not in str(manifest)
    assert (
        next(s for s in manifest["sources"] if s["source_origin"] == "web")[
            "source_url"
        ]
        == "https://example.org/paper"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", REASONS)
async def test_all_feedback_reasons_persist_bound_run_and_source(
    async_db_session, reason
):
    service, run_id, _, source, _ = await run(async_db_session)
    saved = await service.create_feedback(
        async_db_session, run_id=run_id, reason=reason, source_id=source["chunk_id"]
    )
    assert saved["run_id"] == run_id and saved["reason"] == reason
    assert saved["source"]["source_id"] == source["chunk_id"]
    assert saved["config_version"] and saved["created_at"]


@pytest.mark.asyncio
async def test_unknown_run_or_foreign_source_cannot_create_feedback(async_db_session):
    service, run_id, _, _, _ = await run(async_db_session)
    with pytest.raises(LookupError):
        await service.create_feedback(
            async_db_session, run_id="missing", reason="missed_evidence"
        )
    with pytest.raises(ValueError):
        await service.create_feedback(
            async_db_session,
            run_id=run_id,
            reason="unsupported_citation",
            source_id="unseen-source",
        )
    with pytest.raises(ValueError):
        await service.create_feedback(
            async_db_session, run_id=run_id, reason="arbitrary instruction"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["wrong_paper", "unsupported_citation", "web_result_irrelevant"]
)
async def test_three_synthetic_feedback_regression_fix_roundtrips(
    async_db_session, reason
):
    service, run_id, _, source, external = await run(async_db_session)
    selected = external if reason == "web_result_irrelevant" else source
    saved = await service.create_feedback(
        async_db_session, run_id=run_id, reason=reason, source_id=selected["chunk_id"]
    )
    regression = await service.create_regression(
        async_db_session, saved["id"], synthetic=True
    )
    assert regression["fixture"]["synthetic"] is True
    result = await service.run_regression(async_db_session, regression["id"])
    assert result["status"] == "passed"
    fixed = await service.record_fix(
        async_db_session, regression["id"], "commit:development-fixture"
    )
    assert fixed["fix_reference"] == "commit:development-fixture"
    assert "Memory accuracy" not in str(fixed)


@pytest.mark.asyncio
async def test_real_feedback_does_not_become_automatic_training_or_gold(
    async_db_session,
):
    service, run_id, _, source, _ = await run(async_db_session)
    saved = await service.create_feedback(
        async_db_session,
        run_id=run_id,
        reason="unsupported_citation",
        source_id=source["chunk_id"],
    )
    regression = await service.create_regression(async_db_session, saved["id"])
    result = await service.run_regression(async_db_session, regression["id"])
    assert result["status"] == "needs_review" and not result["fixture"]["synthetic"]
    with pytest.raises(ValueError):
        await service.record_fix(
            async_db_session, regression["id"], "commit:unreviewed"
        )


@pytest.mark.asyncio
async def test_feedback_identity_survives_source_paper_deletion(async_db_session):
    service, run_id, paper, source, _ = await run(async_db_session)
    saved = await service.create_feedback(
        async_db_session,
        run_id=run_id,
        reason="wrong_paper",
        source_id=source["chunk_id"],
    )
    await async_db_session.delete(paper)
    await async_db_session.commit()
    from app.models.feedback import AnswerFeedback

    stored = (
        await async_db_session.scalars(
            select(AnswerFeedback).where(AnswerFeedback.id == saved["id"])
        )
    ).one()
    assert stored.run_id == run_id
