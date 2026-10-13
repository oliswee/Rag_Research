import pytest

from evaluation.isolated_corpus import evaluate_bm25, validate_corpus


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
