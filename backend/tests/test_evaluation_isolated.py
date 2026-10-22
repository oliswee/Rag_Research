import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.isolated_corpus import (
    evaluate_bm25,
    prepare_vector_index,
    validate_corpus,
)


@pytest.mark.asyncio
async def test_benchmark_preflights_all_ids_before_any_write(monkeypatch):
    chunks = [{"id": str(i), "checksum": str(i)} for i in range(201)]
    index = MagicMock()
    index.client.retrieve.side_effect = [
        [],
        [SimpleNamespace(payload={"dataset_id": "library"})],
    ]
    monkeypatch.setattr(
        "evaluation.isolated_corpus.SourceVectorIndex", lambda **_: index
    )
    embedder = SimpleNamespace(model="local", generate_text_batch=AsyncMock())
    with pytest.raises(ValueError, match="collides"):
        await prepare_vector_index(chunks, "test", embedder)
    assert index.client.retrieve.call_count == 2
    embedder.generate_text_batch.assert_not_called()
    index.upsert_many.assert_not_called()
    index.client.close.assert_called_once()


def sample_corpus():
    papers = [
        {
            "provider": "arxiv",
            "id": "paper-a",
            "title": "Aptamer study",
            "source_url": "https://example.org/a",
            "status": "parsed",
        },
        {
            "provider": "arxiv",
            "id": "paper-b",
            "title": "Climate study",
            "source_url": "https://example.org/b",
            "status": "parsed",
        },
    ]
    chunks = [
        {
            "id": "chunk-a",
            "paper_id": "arxiv:paper-a",
            "chunk_index": 0,
            "text": "Aptamers are short DNA or RNA molecules with high specificity.",
            "content_scope": "full_text",
            "page_start": 1,
            "page_end": 1,
            "source_url": "https://example.org/a",
            "checksum": "a",
        },
        {
            "id": "chunk-b",
            "paper_id": "arxiv:paper-b",
            "chunk_index": 0,
            "text": "Climate projections affect power grid planning.",
            "content_scope": "full_text",
            "page_start": 2,
            "page_end": 2,
            "source_url": "https://example.org/b",
            "checksum": "b",
        },
    ]
    labels = [
        {
            "query": "What are aptamers?",
            "group": "direct",
            "split": "development",
            "relevant_chunk_ids": ["chunk-a"],
            "label_status": "draft_requires_independent_review",
        },
        {
            "query": "What was the 2029 clinical outcome?",
            "group": "no_answer",
            "split": "test",
            "relevant_chunk_ids": [],
            "label_status": "draft_requires_independent_review",
        },
    ]
    return papers, chunks, labels


@pytest.mark.asyncio
async def test_bm25_uses_isolated_app_retrieval_and_separates_no_answer():
    papers, chunks, labels = sample_corpus()
    report = await evaluate_bm25(papers, chunks, labels)

    assert report["paper_count"] == 2
    assert report["chunk_count"] == 2
    assert report["answerable_count"] == 1
    assert report["no_answer_count"] == 1
    assert report["metrics"]["recall_at_10"] == 1.0
    assert report["metrics"]["hit_at_10"] == 1.0
    assert report["groups"]["no_answer"]["recall_at_10"] is None
    assert report["results"][0]["ranked_chunk_ids"][0] == "chunk-a"
    assert all(case["actual_mode"] == "bm25" for case in report["results"])
    assert report["labels_declared_reviewed"] is False
    assert report["quality_gate_met"] is False


@pytest.mark.asyncio
async def test_isolated_benchmark_writes_versioned_retrieval_trace(tmp_path):
    papers, chunks, labels = sample_corpus()
    output = tmp_path / "retrieval_trace.jsonl"
    await evaluate_bm25(
        papers,
        chunks,
        labels,
        trace_output=output,
        corpus_version="fixture-v1",
        corpus_public=True,
    )
    traces = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(traces) == 2
    assert traces[0]["corpus_version"] == "fixture-v1"
    assert traces[0]["rankings"]["bm25"][0]["child_id"] == "chunk-a"
    assert traces[0]["rankings"]["selected"][0]["child_id"] == "chunk-a"
    assert "rrf" not in traces[0]["rankings"]
    assert traces[0]["prompt_evidence_ids"] == []
    assert traces[0]["support_status"] == "retrieval_only"
    assert "Aptamers are short" not in output.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_isolated_benchmark_records_retrieval_failure(tmp_path, monkeypatch):
    async def broken_retrieve(*_args, **_kwargs):
        raise TimeoutError("private failure detail")

    monkeypatch.setattr("evaluation.isolated_corpus.retrieve", broken_retrieve)
    papers, chunks, labels = sample_corpus()
    output = tmp_path / "trace.jsonl"
    with pytest.raises(TimeoutError):
        await evaluate_bm25(papers, chunks, labels, trace_output=output)
    saved = json.loads(output.read_text().strip())
    assert saved["support_status"] == "error"
    assert saved["failures"] == [{"stage": "retrieve", "error_type": "TimeoutError"}]
    assert "private failure detail" not in str(saved)


@pytest.mark.asyncio
async def test_review_status_requires_reviewer_provenance():
    papers, chunks, labels = sample_corpus()
    for case in labels:
        case["label_status"] = "independently_reviewed"
    assert not (await evaluate_bm25(papers, chunks, labels))["labels_declared_reviewed"]

    for case in labels:
        case["reviewer"] = "external-reviewer"
        case["review_date"] = "2026-10-13"
    report = await evaluate_bm25(papers, chunks, labels)
    assert report["labels_declared_reviewed"]
    assert not report["quality_gate_met"]


@pytest.mark.asyncio
async def test_paper_scope_limits_answerable_and_no_answer_retrieval():
    papers, chunks, labels = sample_corpus()
    chunks[1]["text"] = (
        "Aptamers are used here. Aptamers occur repeatedly. Climate is also discussed."
    )
    labels[0]["paper_ids"] = ["arxiv:paper-a"]
    labels[1]["query"] = "Where are aptamers discussed?"
    labels[1]["paper_ids"] = ["arxiv:paper-b"]
    labels[1]["relevant_chunk_ids"] = ["chunk-b"]

    report = await evaluate_bm25(papers, chunks, labels)

    assert report["results"][0]["paper_ids"] == ["arxiv:paper-a"]
    assert report["results"][0]["ranked_chunk_ids"] == ["chunk-a"]
    assert report["results"][1]["ranked_chunk_ids"] == ["chunk-b"]


@pytest.mark.asyncio
async def test_paper_scoped_no_answer_excludes_other_papers():
    papers, chunks, labels = sample_corpus()
    labels[1]["query"] = "What climate projections affect power grid planning?"
    labels[1]["paper_ids"] = ["arxiv:paper-a"]

    report = await evaluate_bm25(papers, chunks, labels)

    assert report["results"][1]["ranked_chunk_ids"] == []
    assert report["results"][1]["paper_ids"] == ["arxiv:paper-a"]


@pytest.mark.parametrize("paper_ids", [[], ["arxiv:missing"], "arxiv:paper-a"])
def test_validation_rejects_invalid_paper_scope(paper_ids):
    papers, chunks, labels = sample_corpus()
    labels[0]["paper_ids"] = paper_ids
    with pytest.raises(ValueError, match="paper_ids"):
        validate_corpus(papers, chunks, labels)


def test_validation_rejects_gold_chunk_outside_paper_scope():
    papers, chunks, labels = sample_corpus()
    labels[0]["paper_ids"] = ["arxiv:paper-b"]
    with pytest.raises(ValueError, match="outside paper_ids"):
        validate_corpus(papers, chunks, labels)


def test_validation_rejects_unknown_gold_ids_and_duplicate_chunks():
    papers, chunks, labels = sample_corpus()
    labels[0]["relevant_chunk_ids"] = ["missing"]
    with pytest.raises(ValueError, match="Unknown relevant chunk ID"):
        validate_corpus(papers, chunks, labels)

    labels[0]["relevant_chunk_ids"] = ["chunk-a"]
    chunks.append(dict(chunks[0]))
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        validate_corpus(papers, chunks, labels)


def test_validation_rejects_missing_or_duplicate_paper_ids():
    papers, chunks, labels = sample_corpus()
    chunks[0]["paper_id"] = "arxiv:missing"
    with pytest.raises(ValueError, match="Unknown paper ID"):
        validate_corpus(papers, chunks, labels)

    chunks[0]["paper_id"] = "arxiv:paper-a"
    papers.append(dict(papers[0]))
    with pytest.raises(ValueError, match="Duplicate paper ID"):
        validate_corpus(papers, chunks, labels)
