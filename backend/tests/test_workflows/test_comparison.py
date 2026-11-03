import importlib.util

import pytest

from app.config import settings
from app.models import Paper, PaperIdentifier, PaperType
from app.schemas.evidence import GroundedAnswer
from app.services.academic_search import SearchResult, WebBudget
from app.services.context_packing import pack_context
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.local_web import LocalRun
from tests.test_services.test_evidence_registry import local_source


def api():
    assert importlib.util.find_spec("app.workflows.comparison"), (
        "Scoped comparison workflow is missing"
    )
    from app.schemas.research import CompareRequest
    from app.workflows.comparison import run_comparison

    return CompareRequest, run_comparison


@pytest.fixture
def endpoints(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "fastembed")
    monkeypatch.setattr(settings, "llm_provider", "ollama")


async def no_model(question, registry, packed, trace, **kwargs):
    return GroundedAnswer(
        text="Original evidence only.",
        evidence=registry.source_dicts(),
        support_status="evidence_only",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [2, 3, 5])
async def test_every_selected_paper_has_independent_local_evidence_before_any_web(
    async_db_session, count, endpoints
):
    Request, run = api()
    samples = [await local_source(async_db_session) for _ in range(count)]
    events = []
    by_id = {paper.id: source for paper, source in samples}

    async def local(policy, history, trace, *, correction=False):
        assert len(policy.paper_ids) == 1
        events.append(("local", policy.paper_ids[0]))
        return LocalRun(
            pack_context([by_id[policy.paper_ids[0]]], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    class Search:
        async def search(self, policy, identities):
            events.append(("web", policy.paper_ids[0]))
            return SearchResult()

    trace = RetrievalTrace("compare", {}, corpus_version="fixture")
    result = await run(
        async_db_session,
        Request(paper_ids=list(by_id)),
        trace,
        local_runner=local,
        searcher=Search(),
        generator=no_model,
    )
    assert [e[0] for e in events[:count]] == ["local"] * count
    assert len(result.papers) == count and len(result.rows) == 4
    for row in result.rows:
        assert {cell.paper_id for cell in row.cells} == set(by_id)
        assert all(
            all(s["paper_id"] == cell.paper_id for s in cell.evidence)
            for cell in row.cells
        )
    assert all(result.coverage[paper_id]["candidate_count"] == 1 for paper_id in by_id)


@pytest.mark.asyncio
async def test_missing_local_paper_uses_only_its_verified_web_identity(
    async_db_session, endpoints
):
    Request, run = api()
    paper, source = await local_source(async_db_session)
    missing = Paper(
        title=paper.title,
        type=PaperType.research_paper,
        content_scope="abstract",
        import_source="openalex",
    )
    async_db_session.add(missing)
    await async_db_session.flush()
    async_db_session.add(
        PaperIdentifier(paper_id=missing.id, kind="doi", value="10.1234/missing")
    )
    await async_db_session.flush()

    async def local(policy, history, trace, *, correction=False):
        sources = [source] if policy.paper_ids == (paper.id,) else []
        return LocalRun(
            pack_context(sources, trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    class Search:
        async def search(self, policy, identities):
            if policy.paper_ids != (missing.id,):
                return SearchResult()
            assert identities[0].doi == "10.1234/missing"
            return SearchResult(
                [
                    {
                        "provider": "openalex",
                        "provider_id": "W2",
                        "doi": "10.1234/missing",
                        "title": "Same title, distinct identity",
                        "abstract": "Memory accuracy is 95 percent.",
                        "source_url": "https://doi.org/10.1234/missing",
                    }
                ]
            )

    result = await run(
        async_db_session,
        Request(paper_ids=[paper.id, missing.id]),
        RetrievalTrace("compare", {}, corpus_version="fixture"),
        local_runner=local,
        searcher=Search(),
        generator=no_model,
    )
    assert result.coverage[missing.id]["context_count"] > 0
    cell = next(c for c in result.rows[0].cells if c.paper_id == missing.id)
    assert (
        cell.evidence[0]["source_origin"] == "web"
        and cell.evidence[0]["page_start"] is None
    )
    assert all(s["paper_id"] == missing.id for s in cell.evidence)


@pytest.mark.asyncio
async def test_no_legal_text_and_model_timeout_remain_unknown(
    async_db_session, endpoints
):
    Request, run = api()
    papers = [
        Paper(title="Duplicate title", type=PaperType.research_paper) for _ in range(2)
    ]
    async_db_session.add_all(papers)
    await async_db_session.flush()

    async def local(policy, history, trace, *, correction=False):
        return LocalRun(
            pack_context([], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    async def fail(*args, **kwargs):
        raise TimeoutError("private model body")

    result = await run(
        async_db_session,
        Request(paper_ids=[p.id for p in papers], local_only=True),
        RetrievalTrace("compare", {}, corpus_version="fixture"),
        local_runner=local,
        generator=fail,
    )
    assert all(
        cell.support_status == "unknown" for row in result.rows for cell in row.cells
    )
    assert all(
        "private model body" not in cell.text
        for row in result.rows
        for cell in row.cells
    )
    assert all(
        c.status == "insufficient_evidence"
        for row in result.rows
        for c in row.comparisons
    )


@pytest.mark.asyncio
async def test_uploaded_scope_cannot_use_selected_imported_paper(
    async_db_session, endpoints
):
    Request, run = api()
    uploaded, source = await local_source(async_db_session)
    imported = Paper(
        title="Imported", type=PaperType.research_paper, import_source="openalex"
    )
    async_db_session.add(imported)
    await async_db_session.flush()

    async def local(policy, history, trace, *, correction=False):
        assert policy.paper_ids in {(uploaded.id,), ()}
        return LocalRun(
            pack_context([source] if policy.paper_ids else [], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    class Forbidden:
        async def search(self, *args):
            pytest.fail("External access prohibited")

    result = await run(
        async_db_session,
        Request(paper_ids=[uploaded.id, imported.id], only_uploaded=True, latest=True),
        RetrievalTrace("compare", {}, corpus_version="fixture"),
        local_runner=local,
        searcher=Forbidden(),
        generator=no_model,
    )
    assert result.coverage[imported.id]["context_count"] == 0
    assert all(
        not c.evidence
        for row in result.rows
        for c in row.cells
        if c.paper_id == imported.id
    )


@pytest.mark.asyncio
async def test_all_columns_share_one_web_attempt_budget(monkeypatch):
    import httpx

    from app.schemas.query import QueryRequest
    from app.services.academic_search import AcademicSearch
    from app.services.evidence_routing import parse_policy

    monkeypatch.setattr(settings, "web_max_attempts", 2)
    budget = WebBudget()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": []})

    assert "budget" in __import__("inspect").signature(AcademicSearch).parameters, (
        "Shared Web budget missing"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        searcher = AcademicSearch(client, budget=budget)
        for _ in range(5):
            await searcher.search(
                parse_policy(QueryRequest(query="memory", provider="openalex"))
            )
    assert len(calls) == budget.attempts == 2
