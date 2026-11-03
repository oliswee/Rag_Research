import importlib.util

import httpx
import pytest

from app.models import Paper, PaperIdentifier, PaperType


def api():
    assert importlib.util.find_spec("app.services.publication_status"), (
        "Verified publication status metadata is missing"
    )
    from app.services import publication_status

    return publication_status


def article(doi="10.1234/article", **fields):
    return {"DOI": doi, **fields}


def notice(kind="retraction", target="10.1234/article", doi="10.1234/notice"):
    return {
        "DOI": doi,
        "update-to": [
            {"DOI": target, "type": kind, "updated": {"date-parts": [[2025, 2, 3]]}}
        ],
    }


async def paper(db, *, with_doi=True):
    result = Paper(
        title="Retraction in the title is not a status", type=PaperType.research_paper
    )
    db.add(result)
    await db.flush()
    if with_doi:
        db.add(PaperIdentifier(paper_id=result.id, kind="doi", value="10.1234/article"))
    await db.flush()
    return result


def test_missing_status_is_unknown_even_when_title_suggests_retraction():
    result = api().status_from_records(
        "10.1234/article", article(title=["Retracted paper"]), []
    )
    assert result["status"] == "unknown" and not result["updates"]


def test_preprint_relation_keeps_verified_related_doi():
    result = api().status_from_records(
        "10.1234/article",
        article(
            relation={"is-preprint-of": [{"id-type": "doi", "id": "10.1234/formal"}]}
        ),
        [],
    )
    assert result["relations"] == [
        {
            "kind": "is-preprint-of",
            "doi": "10.1234/formal",
            "url": "https://doi.org/10.1234/formal",
        }
    ]


def test_explicit_editorial_notice_must_target_this_article():
    result = api().status_from_records("10.1234/article", article(), [notice()])
    assert result["status"] == "retracted"
    assert result["updates"][0]["notice_doi"] == "10.1234/notice"
    assert result["updates"][0]["date"] == "2025-02-03"


@pytest.mark.parametrize(
    "record,updates",
    [
        (article("10.1234/wrong"), [notice()]),
        (article(), [notice(target="10.1234/other")]),
        (article(**{"update-to": notice()["update-to"]}), []),
        (
            article(),
            [
                {
                    "DOI": "10.1234/notice",
                    "update-to": [{"DOI": "10.1234/article", "type": "retraction"}],
                }
            ],
        ),
    ],
)
def test_wrong_doi_update_direction_or_undated_event_never_grants_status(
    record, updates
):
    assert (
        api().status_from_records("10.1234/article", record, updates)["status"]
        == "unknown"
    )


def test_reinstatement_needs_review_instead_of_assuming_current_retraction():
    result = api().status_from_records(
        "10.1234/article",
        article(),
        [notice(), notice("reinstatement", doi="10.1234/reinstated")],
    )
    assert (
        result["status"] == "unknown"
        and "conflicting_status_events" in result["notices"]
    )


@pytest.mark.asyncio
async def test_local_get_and_local_only_refresh_do_not_access_web(async_db_session):
    service = api()
    item = await paper(async_db_session)
    cached = await service.get_status(async_db_session, item.id)
    assert cached["status"] == "unknown" and cached["checked_at"] is None

    async def forbidden(request):
        pytest.fail("Local-only publication status used the network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        result = await service.refresh_status(
            async_db_session, item.id, client=client, local_only=True
        )
    assert result["status"] == "unknown" and "local_only" in result["notices"]


@pytest.mark.asyncio
async def test_refresh_uses_two_bounded_official_requests_and_stores_minimum_metadata(
    async_db_session,
):
    service = api()
    item = await paper(async_db_session)
    calls = []

    async def transport(request):
        calls.append(request)
        assert request.url.host == "api.crossref.org"
        assert request.method == "GET"
        if request.url.path == "/works":
            assert request.url.params["filter"] == "updates:10.1234/article"
            return httpx.Response(200, json={"message": {"items": [notice()]}})
        return httpx.Response(
            200, json={"message": article(abstract="do not persist external full text")}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await service.refresh_status(async_db_session, item.id, client=client)
    assert result["status"] == "retracted" and len(calls) == 2
    assert result["checked_at"] and result["source_url"].startswith(
        "https://api.crossref.org/works/"
    )
    assert "do not persist" not in str(
        await service.get_status(async_db_session, item.id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["429", "timeout", "oversized"])
async def test_provider_failure_remains_unknown_without_retry_loop(
    async_db_session, failure
):
    service = api()
    item = await paper(async_db_session)
    calls = []

    async def transport(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private provider body")
        if failure == "oversized":
            return httpx.Response(200, content=b"x" * 262145)
        return httpx.Response(429, text="private provider body")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await service.refresh_status(async_db_session, item.id, client=client)
    assert result["status"] == "unknown" and result["refresh_error"]
    assert len(calls) == 1 and "private provider body" not in str(result)
