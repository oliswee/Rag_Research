"""Actual import transactions persist structured sources before vector publication."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import pytest
from sqlalchemy import select

from app.models import PaperChunk, SourceDocument, SourceIndexGeneration
from app.services.pdf_parser import PDFParserService
from app.workflows.ingestion import IngestionInput, IngestionWorkflow


def test_plain_text_has_real_null_page_and_full_text_origin():
    parsed = PDFParserService().parse_text("A supplied full paper.")
    assert parsed.content_scope == "full_text"
    assert parsed.pages[0].number is None
    assert parsed.pages[0].blocks[0].source_origin == "provided_text"
    assert parsed.pages[0].blocks[0].kind == "paragraph"


@pytest.mark.asyncio
@pytest.mark.parametrize("pdf", [False, True])
async def test_upload_saves_original_generation_despite_vector_failure(
    async_db_session,
    monkeypatch,
    pdf,
):
    from app.config import settings
    from app.workflows import ingestion

    text = "Original retrieval evidence."
    payload = None
    if pdf:
        doc = fitz.open()
        doc.new_page().insert_text((50, 80), text)
        payload = doc.tobytes()
        doc.close()
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    publish = AsyncMock(side_effect=RuntimeError("failed offline index"))
    monkeypatch.setattr(ingestion, "publish_generation", publish)
    metadata = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("offline")))
    patterns = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("offline")))
    result = await IngestionWorkflow(
        async_db_session,
        metadata_extractor=metadata,
        pattern_extractor=patterns,
    ).run(IngestionInput(pdf_bytes=payload) if pdf else IngestionInput(text=text))
    assert result.success
    publish.assert_awaited_once()
    document = (await async_db_session.scalars(select(SourceDocument))).one()
    generation = (await async_db_session.scalars(select(SourceIndexGeneration))).one()
    child = (await async_db_session.scalars(select(PaperChunk))).one()
    assert document.paper_id == result.paper_id
    assert document.original_pdf == payload
    assert child.text.strip() == text
    assert child.page_start == (1 if pdf else None)
    assert child.document_id == document.id
    assert generation.is_active
    assert any("Vector indexing pending" in warning for warning in result.warnings)
