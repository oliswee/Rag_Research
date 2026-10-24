"""Automatic imports keep source scope and deduplicate provider records."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from app.models import ImportItem, ImportJob, Paper, PaperChunk, SourceDocument
from app.services.fulltext_locations import FullTextLocation
from app.workflows import discovery_import
from app.workflows.discovery_import import import_one


@pytest.mark.asyncio
async def test_open_pdf_import_reports_unreadable_pages(async_db_session, monkeypatch):
    import fitz

    document = fitz.open()
    document.new_page().insert_text((50, 80), "Readable methods.")
    document.new_page()
    data = document.tobytes()
    document.close()

    class OpenDiscovery(FakeDiscovery):
        async def get_candidate(self, provider, provider_id):
            result = await super().get_candidate(provider, provider_id)
            result.update(pdf_url="https://example.org/open.pdf", open_access=True)
            return result

    async def download(url):
        return data

    monkeypatch.setattr(discovery_import, "download_pdf", download)
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    job = ImportJob()
    async_db_session.add(job)
    await async_db_session.flush()
    item = ImportItem(
        job_id=job.id, provider="openalex", provider_id="W123", candidate={}
    )
    async_db_session.add(item)
    await async_db_session.flush()
    await import_one(async_db_session, item, OpenDiscovery())
    assert item.status == "done"
    assert "PDF page 2: no_extractable_text" in (item.warning or "")
    chunks = (await async_db_session.execute(select(PaperChunk))).scalars().all()
    assert chunks and {chunk.page_start for chunk in chunks} == {1}
    assert chunks[0].parent_id is not None
    source = (await async_db_session.scalars(select(SourceDocument))).one()
    assert source.original_pdf == data


class FakeDiscovery:
    async def get_candidate(self, provider, provider_id):
        return {
            "provider": provider,
            "provider_id": provider_id,
            "title": "A sourced paper",
            "authors": ["A. Author"],
            "year": 2024,
            "abstract": "The study found a measurable result.",
            "source_url": "https://example.org/paper",
            "pdf_url": None,
            "open_access": False,
            "doi": "10.1234/source",
            "arxiv_id": None,
        }


@pytest.mark.asyncio
async def test_abstract_import_and_duplicate(async_db_session, monkeypatch):
    class NoOpenCopies:
        async def find(self, candidate):
            return [], []

        async def close(self):
            pass

    monkeypatch.setattr(discovery_import, "FullTextResolver", NoOpenCopies)
    db = async_db_session
    job = ImportJob()
    db.add(job)
    await db.flush()
    first = ImportItem(
        job_id=job.id, provider="openalex", provider_id="W123", candidate={}
    )
    db.add(first)
    await db.flush()
    await import_one(db, first, FakeDiscovery())
    assert first.status == "done"
    assert first.content_scope == "abstract"
    chunks = (await db.execute(select(PaperChunk))).scalars().all()
    assert len(chunks) == 1 and chunks[0].page_start is None
    assert chunks[0].parent_id is not None
    duplicate = ImportItem(
        job_id=job.id, provider="openalex", provider_id="W456", candidate={}
    )
    db.add(duplicate)
    await db.flush()
    await import_one(db, duplicate, FakeDiscovery())
    assert duplicate.status == "duplicate"
    assert duplicate.paper_id == first.paper_id
    assert (await db.execute(select(func.count(Paper.id)))).scalar() == 1


@pytest.mark.asyncio
async def test_provider_error_does_not_expose_request_credentials(
    async_db_session, caplog
):
    class FailingDiscovery:
        async def get_candidate(self, provider, provider_id):
            raise RuntimeError("api_key=private-value")

    job = ImportJob()
    async_db_session.add(job)
    await async_db_session.flush()
    item = ImportItem(
        job_id=job.id, provider="pubmed", provider_id="12345", candidate={}
    )
    async_db_session.add(item)
    await async_db_session.flush()
    await import_one(async_db_session, item, FailingDiscovery())
    assert item.status == "failed"
    assert item.error == "Import failed: RuntimeError"
    assert "private-value" not in caplog.text


@pytest.mark.asyncio
async def test_import_keeps_abstract_when_all_open_locations_fail(
    async_db_session, monkeypatch
):
    class FakeResolver:
        async def find(self, candidate):
            return ([], ["Unpaywall: 429 rate limit reached"])

        async def close(self):
            pass

    monkeypatch.setattr(discovery_import, "FullTextResolver", FakeResolver)
    job = ImportJob()
    async_db_session.add(job)
    await async_db_session.flush()
    item = ImportItem(job_id=job.id, provider="pubmed", provider_id="123", candidate={})
    async_db_session.add(item)
    await async_db_session.flush()
    await import_one(async_db_session, item, FakeDiscovery())
    assert item.content_scope == "abstract"
    assert "Unpaywall: 429" in item.warning


@pytest.mark.asyncio
async def test_import_uses_unpaywall_pdf_with_page_evidence(
    async_db_session, monkeypatch
):
    class FakeResolver:
        async def find(self, candidate):
            return (
                [
                    FullTextLocation(
                        "Unpaywall",
                        "https://repo.example.org/open.pdf",
                        "https://repo.example.org/open.pdf",
                    )
                ],
                [],
            )

        async def close(self):
            pass

    import fitz

    document = fitz.open()
    document.new_page().insert_text((50, 80), "Open paper methods.")
    payload = document.tobytes()
    document.close()

    monkeypatch.setattr(discovery_import, "FullTextResolver", FakeResolver)
    monkeypatch.setattr(
        discovery_import, "download_pdf", AsyncMock(return_value=payload)
    )
    job = ImportJob()
    async_db_session.add(job)
    await async_db_session.flush()
    item = ImportItem(
        job_id=job.id, provider="openalex", provider_id="W1", candidate={}
    )
    async_db_session.add(item)
    await async_db_session.flush()
    await import_one(async_db_session, item, FakeDiscovery())
    chunk = (await async_db_session.execute(select(PaperChunk))).scalar_one()
    assert item.content_scope == "full_text"
    assert chunk.page_start == 1
    assert chunk.source_url == "https://repo.example.org/open.pdf"
    assert "Unpaywall" in item.warning


@pytest.mark.asyncio
async def test_staging_error_survives_session_rollback(async_db_session, monkeypatch):
    class NoOpenCopies:
        async def find(self, candidate):
            return [], []

        async def close(self):
            pass

    monkeypatch.setattr(discovery_import, "FullTextResolver", NoOpenCopies)
    monkeypatch.setattr(
        discovery_import,
        "stage_generation",
        AsyncMock(side_effect=ValueError("private")),
    )
    job = ImportJob()
    async_db_session.add(job)
    await async_db_session.flush()
    item = ImportItem(
        job_id=job.id, provider="openalex", provider_id="W1", candidate={}
    )
    async_db_session.add(item)
    await async_db_session.flush()
    await import_one(async_db_session, item, FakeDiscovery())
    assert item.status == "failed"
    assert item.error == "Import failed: ValueError"
