from datetime import UTC, datetime

import pytest

from app.models import PaperIdentifier, PublicationState
from tests.test_services.test_evidence_registry import local_source, web_source


@pytest.mark.asyncio
async def test_publication_status_get_is_unknown_until_checked(
    async_db_session, client_with_db
):
    paper, _ = await local_source(async_db_session)
    await async_db_session.commit()
    response = client_with_db.get(f"/api/papers/{paper.id}/publication")
    assert response.status_code == 200
    assert response.json()["status"] == "unknown"
    assert response.json()["checked_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["local", "web"])
async def test_verified_publication_notice_is_bound_into_original_citation(
    async_db_session, origin
):
    from app.services.evidence_registry import build_registry, seal_web_sources

    paper, source = await local_source(async_db_session)
    async_db_session.add(
        PaperIdentifier(paper_id=paper.id, kind="doi", value="10.1234/article")
    )
    async_db_session.add(
        PublicationState(
            paper_id=paper.id,
            checked_at=datetime.now(UTC),
            details={
                "doi": "10.1234/article",
                "status": "retracted",
                "source_url": "https://api.crossref.org/works/10.1234/article",
            },
        )
    )
    await async_db_session.flush()
    snapshots = ()
    if origin == "web":
        source = web_source(paper_id=paper.id, doi="10.1234/article")
        source["source_spans"] = [
            {**s, "paper_id": paper.id} for s in source["source_spans"]
        ]
        snapshots = seal_web_sources([source])
    registry = await build_registry(async_db_session, [source], web_snapshots=snapshots)
    citation = registry.cite(source["chunk_id"], source["excerpt"])
    assert citation.publication_status == "retracted"
    assert (
        citation.publication_source_url
        == "https://api.crossref.org/works/10.1234/article"
    )
    assert citation.publication_checked_at


@pytest.mark.asyncio
async def test_cached_status_for_a_different_doi_cannot_label_this_paper(
    async_db_session,
):
    from app.services.publication_status import get_status

    paper, _ = await local_source(async_db_session)
    async_db_session.add(
        PaperIdentifier(paper_id=paper.id, kind="doi", value="10.1234/article")
    )
    async_db_session.add(
        PublicationState(
            paper_id=paper.id,
            checked_at=datetime.now(UTC),
            details={
                "doi": "10.1234/other",
                "status": "retracted",
                "source_url": "https://api.crossref.org/works/10.1234/other",
            },
        )
    )
    await async_db_session.flush()
    result = await get_status(async_db_session, paper.id)
    assert result["status"] == "unknown" and result["checked_at"] is None
