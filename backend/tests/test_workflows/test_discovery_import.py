"""Automatic imports keep source scope and deduplicate provider records."""

import pytest
from sqlalchemy import func, select

from app.models import ImportItem, ImportJob, Paper, PaperChunk
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
async def test_abstract_import_and_duplicate(async_db_session):
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
