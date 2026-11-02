import importlib.util
from dataclasses import FrozenInstanceError

import pytest

from app.models import Paper, PaperChunk, PaperStatus, PaperType
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.source_anchors import resolve_child


def api():
    assert importlib.util.find_spec("app.services.evidence_registry"), (
        "M6 evidence registry is missing"
    )
    from app.services.evidence_registry import build_registry, seal_web_sources

    return build_registry, seal_web_sources


async def local_source(db, *, native=False):
    paper = Paper(
        title="Memory study",
        type=PaperType.research_paper,
        content_scope="full_text" if native else "abstract",
    )
    db.add(paper)
    await db.flush()
    parsed = (
        PDFParserService().parse_text("Memory accuracy is 95 percent.")
        if native
        else PDFParserService().parse_abstract("Memory accuracy is 95 percent.")
    )
    build = await stage_generation(db, paper, parsed, embedding_model_id="local:test")
    source = await resolve_child(db, build.children[0], allow_parent_failure=True)
    return paper, {**source, "title": paper.title}


def web_source(**changes):
    import hashlib

    text = "Memory accuracy is 95 percent."
    version = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    return {
        "chunk_id": "web:one",
        "paper_id": "p",
        "title": "Web study",
        "doi": "10.1234/a",
        "source_origin": "web",
        "content_scope": "abstract",
        "evidence_level": "abstract_only",
        "excerpt": text,
        "document_version": version,
        "source_url": "https://example.org/paper",
        "retrieved_at": "2026-11-02T00:00:00Z",
        "page_start": None,
        "page_end": None,
        "source_spans": [
            {
                "paper_id": "p",
                "document_id": "p",
                "document_version": version,
                "page": None,
                "block_id": "abstract",
                "raw_start": 0,
                "raw_end": len(text),
                "excerpt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        **changes,
    }


@pytest.mark.asyncio
async def test_registry_rechecks_sqlite_and_generates_resolvable_original_citation(
    async_db_session,
):
    build_registry, _ = api()
    paper, source = await local_source(async_db_session)
    registry = await build_registry(async_db_session, [source], paper_ids=[paper.id])
    assert list(registry.sources) == [source["chunk_id"]]
    citation = registry.cite(source["chunk_id"], "Memory accuracy is 95 percent.")
    assert citation.paper_id == paper.id and citation.source_origin == "local"
    assert citation.evidence_type == "abstract_only" and citation.pdf_page is None
    assert citation.source_url == f"/api/sources/chunks/{source['chunk_id']}"
    assert citation.raw_quote == source["excerpt"] and citation.span
    with pytest.raises((FrozenInstanceError, ValueError, TypeError)):
        registry.sources[source["chunk_id"]].raw_text = "Changed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        {"excerpt": "A fabricated result."},
        {"paper_id": "other"},
        {"document_version": "sha256:wrong"},
        {"source_spans": []},
        {"page_start": 99},
    ],
)
async def test_mutated_same_id_is_not_trusted(async_db_session, mutation):
    build_registry, _ = api()
    paper, source = await local_source(async_db_session)
    registry = await build_registry(
        async_db_session, [{**source, **mutation}], paper_ids=[paper.id]
    )
    assert not registry.sources
    assert registry.rejections


@pytest.mark.asyncio
async def test_archived_or_wrong_scope_source_is_rejected(async_db_session):
    build_registry, _ = api()
    paper, source = await local_source(async_db_session)
    assert not (
        await build_registry(async_db_session, [source], paper_ids=["other"])
    ).sources
    paper.status = PaperStatus.archived
    await async_db_session.flush()
    assert not (await build_registry(async_db_session, [source])).sources


@pytest.mark.asyncio
async def test_legacy_local_text_is_citable_but_pdf_page_not_guessed(async_db_session):
    build_registry, _ = api()
    paper = Paper(title="Legacy", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    child = PaperChunk(
        paper_id=paper.id,
        text="Memory accuracy is 95 percent.",
        content_scope="full_text",
        checksum="historical",
        page_start=42,
        page_end=42,
    )
    async_db_session.add(child)
    await async_db_session.flush()
    source = {**(await resolve_child(async_db_session, child)), "title": paper.title}
    registry = await build_registry(async_db_session, [source])
    citation = registry.cite(child.id, child.text)
    assert citation.pdf_page is None and citation.version.startswith("sha256:")


@pytest.mark.asyncio
async def test_web_requires_current_sealed_snapshot_and_preserves_version_origin(
    async_db_session,
):
    build_registry, seal = api()
    source = web_source()
    assert not (await build_registry(async_db_session, [source])).sources
    snapshots = seal([source])
    source["title"] = "Mutable display label"
    registry = await build_registry(async_db_session, [source], web_snapshots=snapshots)
    citation = registry.cite("web:one", "Memory accuracy is 95 percent.")
    assert citation.source_origin == "web" and citation.pdf_page is None
    assert citation.source_url == "https://example.org/paper"
    assert citation.version == snapshots[0].version
    assert not (
        await build_registry(
            async_db_session, [source], web_snapshots=snapshots, local_only=True
        )
    ).sources
    source["excerpt"] = "Fabricated after sealing."
    assert not (
        await build_registry(async_db_session, [source], web_snapshots=snapshots)
    ).sources


@pytest.mark.asyncio
async def test_metadata_and_abstract_fake_page_never_become_factual(async_db_session):
    build_registry, seal = api()
    metadata = web_source(evidence_level="metadata_only")
    abstract = web_source(page_start=12, page_end=12)
    assert not (
        await build_registry(
            async_db_session, [metadata], web_snapshots=seal([metadata])
        )
    ).sources
    with pytest.raises(ValueError):
        seal([abstract])


@pytest.mark.asyncio
async def test_quote_must_be_in_child_not_only_parent(async_db_session):
    build_registry, _ = api()
    paper, source = await local_source(async_db_session)
    source["parent"] = {"raw_text": "Parent-only unsupported conclusion."}
    registry = await build_registry(async_db_session, [source])
    assert (
        registry.source_dicts()[0]["parent"]["raw_text"] != source["parent"]["raw_text"]
    )
    with pytest.raises(ValueError):
        registry.cite(source["chunk_id"], "Parent-only unsupported conclusion.")


@pytest.mark.asyncio
async def test_canonical_parent_still_expands_bounded_context_after_registry(
    async_db_session,
):
    from app.services.context_packing import pack_context

    build_registry, _ = api()
    paper = Paper(title="Long parent", type=PaperType.research_paper)
    async_db_session.add(paper)
    await async_db_session.flush()
    parsed = PDFParserService().parse_abstract(
        "Memory accuracy improves with retrieval. " * 80
    )
    build = await stage_generation(
        async_db_session,
        paper,
        parsed,
        child_size=120,
        parent_size=1200,
        embedding_model_id="local:test",
    )
    source = await resolve_child(
        async_db_session, build.children[0], allow_parent_failure=True
    )
    registry = await build_registry(async_db_session, [source])
    packed = pack_context(registry.source_dicts(), budget=8192, parent_budget=2048)
    assert packed.blocks[0]["expansion_status"] == "parent_window"
    assert len(packed.blocks[0]["text"]) > len(source["excerpt"])
