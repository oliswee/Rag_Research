"""Citation URLs resolve retained sources and fail explicitly on corrupt anchors."""

import json

import fitz
import pytest
from httpx import ASGITransport, AsyncClient

from app.db import get_db
from app.main import app
from app.models import Paper, PaperChunk, PaperType
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService


@pytest.fixture
async def source_client(async_db_session):
    async def session():
        yield async_db_session

    app.dependency_overrides[get_db] = session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_old_citation_keeps_original_parent_spans_and_pdf_after_rechunk(
    async_db_session,
    source_client,
):
    pdf = fitz.open()
    for _ in range(2):
        pdf.new_page().insert_text(
            (50, 80), "Repeated original evidence on a real page."
        )
    payload = pdf.tobytes()
    pdf.close()
    parsed = PDFParserService().parse(payload)
    paper = Paper(title="Source jump", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    first = await stage_generation(
        async_db_session,
        paper,
        parsed,
        child_size=64,
        parent_size=256,
        source_bytes=payload,
    )
    await async_db_session.commit()
    citation = f"/api/sources/chunks/{first.children[0].id}"
    original = await source_client.get(citation)
    assert original.status_code == 200
    body = original.json()
    assert body["status"] == "active"
    assert body["excerpt"] == first.children[0].text
    assert body["parent"]["raw_text"] == first.parents[0].raw_text
    assert body["source_spans"] == json.loads(
        json.dumps(first.children[0].source_spans)
    )
    stored_pdf = await source_client.get(body["original_pdf_url"])
    assert stored_pdf.status_code == 200 and stored_pdf.content == payload
    assert stored_pdf.headers["content-type"] == "application/pdf"
    await stage_generation(
        async_db_session,
        paper,
        parsed,
        child_size=100,
        parent_size=256,
    )
    await async_db_session.commit()
    retired = await source_client.get(citation)
    assert retired.status_code == 200
    assert retired.json()["status"] == "retired"
    for key in ("excerpt", "parent", "source_spans", "document_version"):
        assert retired.json()[key] == body[key]
    first.children[0].text = "Tampered source text"
    await async_db_session.commit()
    corrupt = await source_client.get(citation)
    assert corrupt.status_code == 409
    assert "Tampered" not in corrupt.text


@pytest.mark.asyncio
async def test_legacy_citation_is_explicit_and_missing_source_is_404(
    async_db_session,
    source_client,
):
    paper = Paper(title="Legacy", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    child = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        text="Historical evidence",
        checksum="old",
        content_scope="abstract",
    )
    async_db_session.add(child)
    await async_db_session.commit()
    response = await source_client.get(f"/api/sources/chunks/{child.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "legacy"
    assert response.json()["parent"] is None
    assert response.json()["source_spans"] == []
    assert response.json()["original_pdf_url"] is None
    assert (await source_client.get("/api/sources/chunks/missing")).status_code == 404
