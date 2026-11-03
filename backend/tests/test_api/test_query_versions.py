"""Runtime trace versions must follow real stored content without exposing it."""

import json

import pytest

from app.config import settings
from app.models import Paper, PaperChunk, PaperStatus, PaperType, Pattern, PatternStatus
from app.routers.query import follow_up_query, submit_query
from app.schemas.query import FollowUpRequest, QueryRequest


@pytest.mark.asyncio
async def test_corpus_snapshot_fetches_large_tables_in_bounded_batches(
    async_db_session, monkeypatch
):
    from sqlalchemy.dialects.sqlite.aiosqlite import AsyncAdapt_aiosqlite_ss_cursor

    from app.services.trace_versions import corpus_version

    paper = Paper(title="Paper", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    async_db_session.add_all(
        [
            PaperChunk(
                paper_id=paper.id,
                chunk_index=index,
                text="source" * 100,
                checksum=str(index),
                content_scope="full_text",
            )
            for index in range(600)
        ]
    )
    await async_db_session.commit()
    batch_sizes = []
    cursor_class = AsyncAdapt_aiosqlite_ss_cursor
    original = cursor_class.fetchmany

    def reject_fetchall(self):
        raise AssertionError("Unbounded corpus fetch")

    def bounded_fetchmany(self, size=None):
        assert size is not None and 0 < size <= 256
        batch_sizes.append(size)
        return original(self, size)

    with monkeypatch.context() as patch:
        patch.setattr(cursor_class, "fetchall", reject_fetchall)
        patch.setattr(cursor_class, "fetchmany", bounded_fetchmany)
        first = await corpus_version(async_db_session)
        second = await corpus_version(async_db_session)
    assert first == second
    assert batch_sizes.count(256) >= 4


async def query_trace(db, path):
    result = await submit_query(
        QueryRequest(query="private question", retrieval_mode="bm25"), db
    )
    trace = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    return result, trace


@pytest.mark.asyncio
async def test_empty_library_has_stable_private_runtime_versions(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    _, first = await query_trace(async_db_session, path)
    _, second = await query_trace(async_db_session, path)
    assert first["corpus_version"].startswith("sqlite:sha256:")
    assert first["config_version"].startswith("sha256:")
    assert first["index_version"].startswith("logical:sha256:")
    for field in ("corpus_version", "config_version", "index_version"):
        assert first[field] == second[field]
    assert first["physical_index_revision"] is None
    assert first["index_locator"] == "sqlite_fts5"
    assert first["stage_ms"]["corpus_snapshot"] >= 0
    assert "private question" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_version_changes_on_content_anchor_and_deletion(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    paper = Paper(
        title="Private title", type=PaperType.research_paper, status=PaperStatus.active
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        text="private source alpha",
        checksum="stale-checksum",
        chunk_index=0,
        content_scope="full_text",
        page_start=1,
        page_end=1,
    )
    async_db_session.add(chunk)
    await async_db_session.commit()
    _, first = await query_trace(async_db_session, path)
    chunk.text = "private source beta"
    await async_db_session.commit()
    _, changed = await query_trace(async_db_session, path)
    assert changed["corpus_version"] != first["corpus_version"]
    assert changed["index_version"] != first["index_version"]
    assert changed["config_version"] == first["config_version"]
    chunk.page_start = 2
    await async_db_session.commit()
    _, moved = await query_trace(async_db_session, path)
    assert moved["corpus_version"] != changed["corpus_version"]
    await async_db_session.delete(chunk)
    await async_db_session.commit()
    _, deleted = await query_trace(async_db_session, path)
    assert deleted["corpus_version"] != moved["corpus_version"]
    saved = path.read_text(encoding="utf-8")
    assert "private source" not in saved
    assert "Private title" not in saved


@pytest.mark.asyncio
async def test_runtime_config_changes_for_models_but_excludes_credentials(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    _, first = await query_trace(async_db_session, path)
    monkeypatch.setattr(settings, "ncbi_api_key", "new-private-secret-value")
    monkeypatch.setattr(settings, "ncbi_email", "private@example.test")
    _, rotated = await query_trace(async_db_session, path)
    assert rotated["config_version"] == first["config_version"]
    monkeypatch.setattr(settings, "anthropic_model", "different-model")
    _, model_changed = await query_trace(async_db_session, path)
    assert model_changed["config_version"] != first["config_version"]
    assert model_changed["corpus_version"] == first["corpus_version"]
    saved = path.read_text(encoding="utf-8")
    assert "new-private-secret-value" not in saved
    assert "private@example.test" not in saved
    assert "api_key" not in str(model_changed["config_manifest"])


@pytest.mark.asyncio
async def test_followup_versions_capture_pattern_edits(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    paper = Paper(
        title="Paper", type=PaperType.research_paper, status=PaperStatus.active
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    pattern = Pattern(
        paper_id=paper.id,
        name="Private pattern",
        claim="Private claim",
        status=PatternStatus.approved,
    )
    async_db_session.add(pattern)
    await async_db_session.commit()
    result, first = await query_trace(async_db_session, path)
    pattern.claim = "Changed private claim"
    await async_db_session.commit()
    await follow_up_query(
        FollowUpRequest(
            query="followup", session_id=result["session_id"], retrieval_mode="bm25"
        ),
        async_db_session,
    )
    followup = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert followup["corpus_version"] != first["corpus_version"]
    assert followup["corpus_version"].startswith("sqlite:sha256:")
    assert "Private claim" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_config_distinguishes_endpoints_without_persisting_them(
    async_db_session, monkeypatch, tmp_path
):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "retrieval_trace_path", str(path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    _, first = await query_trace(async_db_session, path)
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11435/private-endpoint"
    )
    _, second = await query_trace(async_db_session, path)
    assert second["config_version"] != first["config_version"]
    assert "private-endpoint" not in path.read_text(encoding="utf-8")
