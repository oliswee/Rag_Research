"""DOI context makes one optional Crossref lookup."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.discovery import DiscoveryService
from app.services.metadata_context import get_paper_context


@pytest.mark.asyncio
async def test_context_uses_only_crossref_for_doi():
    service = DiscoveryService()
    service._get = AsyncMock(
        return_value={
            "message": {
                "DOI": "10.1234/example",
                "container-title": ["Research Journal"],
                "publisher": "Example Press",
                "references-count": 6,
            }
        }
    )
    try:
        context = await get_paper_context(service, "10.1234/example")
    finally:
        await service.close()
    assert context["metadata"]["venue"] == "Research Journal"
    assert context["metadata"]["source"] == "Crossref"
    assert context["errors"] == []
    service._get.assert_awaited_once_with(
        "https://api.crossref.org/works/10.1234/example"
    )


@pytest.mark.asyncio
async def test_crossref_404_is_missing_metadata_and_429_is_sanitized():
    service = DiscoveryService()
    try:
        for status in (404, 429):
            response = httpx.Response(
                status,
                request=httpx.Request(
                    "GET", "https://api.crossref.org/works/10.1234/example"
                ),
            )
            service._get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "private request detail",
                    request=response.request,
                    response=response,
                )
            )
            context = await get_paper_context(service, "10.1234/example")
            assert context["metadata"] is None
            assert context["errors"] == (
                []
                if status == 404
                else ["Crossref: 429 rate limit reached; retry later"]
            )
    finally:
        await service.close()
