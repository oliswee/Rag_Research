"""Automatic imports keep source scope and deduplicate provider records."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from app.models import ImportItem, ImportJob, Paper, PaperChunk
from app.services.fulltext_locations import FullTextLocation
from app.workflows import discovery_import
from app.workflows.discovery_import import import_one


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
async def test_import_uses_europe_pmc_open_xml_when_provider_has_no_pdf(
    async_db_session, monkeypatch
):
    class FakeResolver:
        async def find(self, candidate):
            return (
                [
                    FullTextLocation(
                        "Europe PMC",
                        "xml",
                        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML",
                        "https://europepmc.org/articles/PMC123",
                    )
                ],
                [],
            )

        async def fetch_xml(self, location):
            return "The full article describes the methods and results."

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
    chunk = (await async_db_session.execute(select(PaperChunk))).scalar_one()
    assert item.content_scope == "full_text"
    assert chunk.source_url == "https://europepmc.org/articles/PMC123"
    assert chunk.page_start is None


@pytest.mark.asyncio
async def test_import_keeps_abstract_when_all_open_locations_fail(
    async_db_session, monkeypatch
):
    class FakeResolver:
        async def find(self, candidate):
            return ([], ["CORE: 429 rate limit reached"])

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
    assert "CORE: 429" in item.warning


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
                        "pdf",
                        "https://repo.example.org/open.pdf",
                        "https://repo.example.org/open.pdf",
                    )
                ],
                [],
            )

        async def close(self):
            pass

    class FakeParser:
        def __init__(self, max_pages):
            pass

        def parse(self, payload):
            assert payload == b"%PDF-valid"
            return SimpleNamespace(
                pages=[SimpleNamespace(text="Open paper methods.", number=1)]
            )

    monkeypatch.setattr(discovery_import, "FullTextResolver", FakeResolver)
    monkeypatch.setattr(
        discovery_import, "download_pdf", AsyncMock(return_value=b"%PDF-valid")
    )
    monkeypatch.setattr(discovery_import, "PDFParserService", FakeParser)
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
