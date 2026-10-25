"""The real source pipeline preserves baseline and prompt attribution contracts."""

import json

import pytest

from app.models import Paper, PaperChunk, PaperType
from app.services.retrieval_branches import BranchResult
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.source_query import answer_from_evidence, retrieve


@pytest.mark.asyncio
async def test_candidate_pool_is_larger_than_baseline_and_reserves_requested_papers(
    async_db_session, monkeypatch
):
    papers = [
        Paper(title=f"Paper {i}", type=PaperType.research_paper) for i in range(2)
    ]
    async_db_session.add_all(papers)
    await async_db_session.flush()
    children = [
        PaperChunk(
            paper_id=papers[0 if i < 35 else 1].id,
            chunk_index=i,
            text=f"Result {i}.",
            checksum="legacy",
            content_scope="abstract",
        )
        for i in range(36)
    ]
    async_db_session.add_all(children)
    await async_db_session.commit()

    async def branches(*args, **kwargs):
        return [
            BranchResult(
                "Q0", "bm25", [(c.id, -i) for i, c in enumerate(children)], "ok", 1
            )
        ]

    monkeypatch.setattr("app.workflows.source_query.collect_rankings", branches)
    baseline, _ = await retrieve(async_db_session, "result", mode="bm25")
    assert len(baseline) == 10
    recorder = RetrievalTrace("result", {}, corpus_version="test")
    candidates, _ = await retrieve(
        async_db_session,
        "result",
        mode="bm25",
        paper_ids=[p.id for p in papers],
        candidate_limit=30,
        trace=recorder,
    )
    assert len(candidates) == 30
    assert {s["paper_id"] for s in candidates} == {p.id for p in papers}
    assert recorder.record["retrieval_budget"]["candidate_k"] == 30


@pytest.mark.asyncio
async def test_prepared_context_explains_selection_and_missing_paper(monkeypatch):
    from app.services.evidence_context import prepare_evidence

    class Scorer:
        def score(self, question, texts):
            assert question == "original numerical question"
            return list(range(len(texts)))

    sources = [
        {
            "chunk_id": f"c{i}",
            "paper_id": f"p{i % 2}",
            "excerpt": f"Result {i}.",
            "title": "Paper",
            "content_scope": "abstract",
            "page_start": None,
            "page_end": None,
            "parent": None,
        }
        for i in range(15)
    ]
    monkeypatch.setattr("app.services.child_reranker._default_scorer", lambda: Scorer())
    recorder = RetrievalTrace("q", {}, corpus_version="test")
    packed = await prepare_evidence(
        "original numerical question",
        sources,
        paper_ids=["p0", "p1", "missing"],
        trace=recorder,
    )
    assert len(packed.sources) == 10
    assert packed.sources[0]["chunk_id"] == "c14"
    assert packed.coverage["missing"]["context_status"] == "unknown"
    decisions = recorder.record["context_packing"]["decisions"]
    assert len(decisions) == 15
    assert sum(d["reason"] == "below_selection_limit" for d in decisions) == 5
    assert recorder.record["prompt_evidence_ids"] == []


@pytest.mark.asyncio
async def test_parent_context_cannot_be_cited_as_child_and_prompt_ids_are_exact(
    monkeypatch,
):
    from app.services.context_packing import PackedContext

    source = {
        "chunk_id": "child",
        "paper_id": "p",
        "excerpt": "Latency is 1.5 ms.",
        "title": "Paper",
        "content_scope": "full_text",
        "page_start": 2,
    }
    packed = PackedContext(
        [source], [], "Latency is 1.5 ms. Background says 9.9 ms.", {}, 55
    )

    class LLM:
        async def complete_json(self, **kwargs):
            assert "Background says 9.9 ms." in kwargs["user_prompt"]
            return {
                "claims": [
                    {
                        "text": "Latency is 9.9 ms.",
                        "source_ids": ["child"],
                        "supporting_quotes": {"child": "Background says 9.9 ms."},
                    }
                ]
            }

    monkeypatch.setattr("app.services.llm_client.llm_available", lambda: True)
    monkeypatch.setattr("app.services.llm_client.LLMClient", LLM)
    recorder = RetrievalTrace("q", {}, corpus_version="test")
    answer, used = await answer_from_evidence(
        "q", [source], recorder, packed_context=packed
    )
    assert used == [] and "Insufficient" in answer
    assert recorder.record["prompt_evidence_ids"] == ["child"]


@pytest.mark.asyncio
async def test_generation_failure_retains_actually_sent_ids(monkeypatch):
    from app.services.context_packing import pack_context

    source = {
        "chunk_id": "child",
        "paper_id": "p",
        "excerpt": "A valid result.",
        "title": "Paper",
        "content_scope": "abstract",
        "page_start": None,
    }
    packed = pack_context([source])

    class LLM:
        async def complete_json(self, **kwargs):
            raise RuntimeError("private-generation-payload")

    monkeypatch.setattr("app.services.llm_client.llm_available", lambda: True)
    monkeypatch.setattr("app.services.llm_client.LLMClient", LLM)
    recorder = RetrievalTrace("q", {}, corpus_version="test")
    with pytest.raises(RuntimeError):
        await answer_from_evidence("q", [source], recorder, packed_context=packed)
    assert recorder.record["prompt_evidence_ids"] == ["child"]
    assert "private-generation-payload" not in json.dumps(recorder.record)
    assert packed.coverage["p"]["prompt_status"] == "included"
