from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select

from app.models import Paper, PaperChunk, PaperIdentifier, PaperType
from app.schemas.query import QueryRequest
from app.services.academic_search import AcademicSearch, SearchResult
from app.services.context_packing import pack_context
from app.services.evidence_routing import parse_policy
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.local_web import LocalRun, corpus_facts, run_local_web


def source():
    return {
        "paper_id": "a",
        "chunk_id": "c",
        "title": "Memory",
        "excerpt": "Memory accuracy improves.",
        "content_scope": "full_text",
        "source_origin": "pdf",
        "page_start": 1,
        "page_end": 1,
        "source_spans": [
            {
                "paper_id": "a",
                "document_id": "d",
                "document_version": "v1",
                "page": 1,
                "block_id": "b",
                "raw_start": 0,
                "raw_end": 25,
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,has_local,expected",
    [
        ({}, True, False),
        ({}, False, True),
        ({"latest": True}, True, True),
        ({"intent": "discover"}, True, True),
        ({"local_only": True}, False, False),
        ({"only_uploaded": True, "latest": True}, False, False),
        ({"paper_ids": ["absent"]}, False, True),
        ({"retrieval_mode": "bm25"}, False, False),
    ],
)
async def test_local_first_route_matrix(async_db_session, fields, has_local, expected):
    events = []

    async def local(policy, history, trace, *, correction=False):
        events.append("correction" if correction else "local")
        return LocalRun(
            pack_context([source()] if has_local else [], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    class Search:
        async def search(self, policy, identities):
            events.append("web")
            return SearchResult()

    policy = parse_policy(QueryRequest(query="memory accuracy", **fields))
    trace = RetrievalTrace(policy.question, {}, corpus_version="fixture")
    result = await run_local_web(
        async_db_session, policy, [], trace, local_runner=local, searcher=Search()
    )
    assert events[0] == "local"
    assert ("web" in events) is expected
    assert events.count("correction") <= 1
    assert result.packed.token_units <= 8192
    assert result.decision.action == ("search_web" if expected else "answer_local")


@pytest.mark.asyncio
async def test_web_fault_preserves_local_context_and_no_writes(async_db_session):
    async def local(policy, history, trace, *, correction=False):
        return LocalRun(
            pack_context([source()], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    searcher = AsyncMock()
    searcher.search.side_effect = TimeoutError("private failure")
    trace = RetrievalTrace("latest memory", {}, corpus_version="fixture")
    before = (
        await async_db_session.scalar(select(func.count(Paper.id))),
        await async_db_session.scalar(select(func.count(PaperChunk.id))),
    )
    result = await run_local_web(
        async_db_session,
        parse_policy(QueryRequest(query="latest memory")),
        [],
        trace,
        local_runner=local,
        searcher=searcher,
    )
    after = (
        await async_db_session.scalar(select(func.count(Paper.id))),
        await async_db_session.scalar(select(func.count(PaperChunk.id))),
    )
    assert before == after == (0, 0)
    assert result.packed.sources[0]["chunk_id"] == "c"
    assert "private failure" not in str(trace.record)
    assert result.notices


@pytest.mark.asyncio
async def test_metadata_never_enters_context_and_abstract_is_graded(async_db_session):
    async def local(policy, history, trace, *, correction=False):
        return LocalRun(
            pack_context([], trace=trace), "hybrid", 1, "disabled", "fixture"
        )

    searcher = AsyncMock()
    searcher.search.return_value = SearchResult(
        [
            {
                "provider": "openalex",
                "provider_id": "W1",
                "title": "Memory",
                "source_url": "https://openalex.org/W1",
                "abstract": "Memory accuracy improves.",
            },
            {
                "provider": "openalex",
                "provider_id": "W2",
                "title": "Bibliography only",
                "source_url": "https://openalex.org/W2",
            },
        ]
    )
    trace = RetrievalTrace("find papers", {}, corpus_version="fixture")
    result = await run_local_web(
        async_db_session,
        parse_policy(QueryRequest(query="find papers")),
        [],
        trace,
        local_runner=local,
        searcher=searcher,
    )
    assert len(result.web_records) == 2
    assert all(s["evidence_level"] == "abstract_only" for s in result.packed.sources)
    assert "Bibliography only" not in result.packed.text
    assert not trace.record["prompt_evidence_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["429", "timeout", "blocked", "network", "oversize", "redirect", "malformed"],
)
async def test_joint_provider_faults_keep_local_without_private_context_outbound(
    async_db_session, fault
):
    calls = []
    local_source = {**source(), "excerpt": "PRIVATE LIBRARY EXCERPT"}

    async def local(policy, history, trace, *, correction=False):
        return LocalRun(
            pack_context([local_source], trace=trace),
            "hybrid",
            1,
            "disabled",
            "fixture",
        )

    def handler(request):
        calls.append(request)
        assert (
            "PRIVATE LIBRARY EXCERPT" not in str(request.url) + request.content.decode()
        )
        if request.url.host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Bibliography",
                        }
                    ]
                },
            )
        if fault == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        if fault == "network":
            raise httpx.ConnectError("secret", request=request)
        if fault == "oversize":
            return httpx.Response(200, content=b"x" * 524289)
        if fault == "redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        if fault == "malformed":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(429 if fault == "429" else 403)

    policy = parse_policy(QueryRequest(query="find papers on memory"))
    trace = RetrievalTrace(policy.question, {}, corpus_version="fixture")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_local_web(
            async_db_session,
            policy,
            [],
            trace,
            local_runner=local,
            searcher=AcademicSearch(client, retry_delay=0),
        )
    assert result.packed.sources[0]["excerpt"] == "PRIVATE LIBRARY EXCERPT"
    assert result.web_records[0]["evidence_level"] == "metadata_only"
    assert len(calls) <= 3 and "secret" not in str(trace.record)
    assert not async_db_session.new and not async_db_session.dirty


@pytest.mark.asyncio
async def test_pending_native_document_binds_version_before_index_publication(
    async_db_session,
):
    import fitz

    from app.services.parent_child import stage_generation
    from app.services.pdf_parser import PDFParserService

    paper = Paper(
        title="Native pending", type=PaperType.research_paper, content_scope="full_text"
    )
    async_db_session.add(paper)
    await async_db_session.flush()
    async_db_session.add(
        PaperIdentifier(paper_id=paper.id, kind="doi", value="10.1234/native")
    )
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Memory accuracy improves.")
    payload = doc.tobytes()
    doc.close()
    parsed = PDFParserService().parse(payload)
    staged = await stage_generation(
        async_db_session, paper, parsed, embedding_model_id="local:test"
    )
    staged.generation.is_active = False
    await async_db_session.flush()
    assert not staged.generation.is_active
    facts, identities = await corpus_facts(
        async_db_session,
        parse_policy(QueryRequest(query="memory", paper_ids=[paper.id])),
    )
    assert identities[0].document_version == parsed.document_version
