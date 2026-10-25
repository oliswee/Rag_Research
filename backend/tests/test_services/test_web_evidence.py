import hashlib

import fitz
import pytest

from app.schemas.query import QueryRequest
from app.services.academic_search import PaperIdentity
from app.services.evidence_routing import parse_policy
from app.services.pdf_parser import PDFParserService
from app.services.web_evidence import build_web_evidence, parse_temporary_pdf


def candidate(**fields):
    return {
        "provider": "openalex",
        "provider_id": "W1",
        "doi": "10.1234/a",
        "title": "Memory",
        "source_url": "https://doi.org/10.1234/a",
        "abstract": "Memory accuracy is 95 percent.",
        "retrieved_at": "2026-10-25T00:00:00Z",
        **fields,
    }


def pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Memory accuracy is 95 percent.")
    result = doc.tobytes()
    doc.close()
    return result


@pytest.mark.asyncio
async def test_abstract_and_metadata_have_no_pages_and_metadata_not_factual():
    pool = await build_web_evidence(
        [candidate(), candidate(provider_id="W2", doi=None, abstract=None)],
        parse_policy(QueryRequest(query="memory")),
        (),
    )
    assert [r["evidence_level"] for r in pool.records] == [
        "abstract_only",
        "metadata_only",
    ]
    assert len(pool.sources) == 1
    assert pool.sources[0]["excerpt"] == candidate()["abstract"]
    assert all(r["page_start"] is None and r["page_end"] is None for r in pool.records)
    assert pool.sources[0]["source_url"].startswith("https://doi.org/")


@pytest.mark.asyncio
async def test_only_known_license_pdf_is_fetched_once_with_original_anchors():
    data = pdf_bytes()
    calls = []

    async def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return data

    async def parser(payload):
        return PDFParserService().parse(payload, source_origin="web")

    pool = await build_web_evidence(
        [
            candidate(
                open_access=True, pdf_url="https://example.org/a.pdf", license="cc-by"
            ),
            candidate(
                provider_id="W2",
                doi="10.1234/b",
                open_access=True,
                pdf_url="https://example.org/b.pdf",
                license="cc-by",
            ),
        ],
        parse_policy(QueryRequest(query="memory")),
        (),
        fetcher=fetch,
        parser=parser,
    )
    assert len(calls) == 1 and calls[0][1]["same_host"] is True
    source = pool.sources[0]
    assert source["evidence_level"] == "full_text" and source["source_origin"] == "web"
    assert source["document_version"] == "sha256:" + hashlib.sha256(data).hexdigest()
    parsed = await parser(data)
    anchor = source["source_spans"][0]
    assert (
        parsed.pages[0].text[anchor["raw_start"] : anchor["raw_end"]]
        == source["excerpt"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("license", [None, "unknown"])
async def test_unknown_permission_and_local_only_do_not_call_fetcher(license):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    c = candidate(
        open_access=True, pdf_url="https://example.org/a.pdf", license=license
    )
    pool = await build_web_evidence(
        [c], parse_policy(QueryRequest(query="memory")), (), fetcher=forbidden
    )
    assert pool.records[0]["evidence_level"] == "abstract_only"
    pool = await build_web_evidence(
        [c],
        parse_policy(QueryRequest(query="memory", local_only=True)),
        (),
        fetcher=forbidden,
    )
    assert not pool.sources and not pool.records


@pytest.mark.asyncio
async def test_scoped_version_mismatch_cannot_use_other_fulltext():
    data = pdf_bytes()

    async def fetch(*args, **kwargs):
        return data

    async def parser(payload):
        return PDFParserService().parse(payload, source_origin="web")

    pool = await build_web_evidence(
        [
            candidate(
                local_paper_id="a",
                open_access=True,
                pdf_url="https://example.org/a.pdf",
                license="cc-by",
            )
        ],
        parse_policy(QueryRequest(query="memory", paper_ids=["a"])),
        [PaperIdentity("a", doi="10.1234/a", document_version="sha256:other")],
        fetcher=fetch,
        parser=parser,
    )
    assert not pool.sources
    assert pool.records[0]["evidence_level"] == "metadata_only"
    assert pool.records[0]["version_status"] == "mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [ValueError("malicious secret"), TimeoutError("secret")]
)
async def test_bad_or_timed_out_pdf_falls_back_to_abstract_without_error_text(fault):
    async def fail(*args, **kwargs):
        raise fault

    pool = await build_web_evidence(
        [
            candidate(
                open_access=True, pdf_url="https://example.org/a.pdf", license="cc-by"
            )
        ],
        parse_policy(QueryRequest(query="memory")),
        (),
        fetcher=fail,
    )
    assert pool.sources[0]["evidence_level"] == "abstract_only"
    assert "secret" not in str(pool.events)


@pytest.mark.asyncio
async def test_native_parser_process_valid_and_malicious_bytes():
    parsed = await parse_temporary_pdf(pdf_bytes())
    assert parsed.pages[0].number == 1
    with pytest.raises(ValueError):
        await parse_temporary_pdf(b"not pdf")


@pytest.mark.asyncio
async def test_scoped_unknown_identity_never_becomes_factual():
    pool = await build_web_evidence(
        [candidate()], parse_policy(QueryRequest(query="memory", paper_ids=["a"])), ()
    )
    assert not pool.sources and not pool.records
