"""Saved retrieval replay cannot invent new retrieval measurements or evidence."""

import pytest


def test_replay_counts_only_real_branch_lists_and_enforces_scope():
    from evaluation.ranking_packing_benchmark import replay_candidates

    label = {"query": "q", "paper_ids": ["p"]}
    trace = {
        "original_query": "q",
        "retrieval_branches": [
            {"query_id": "Q0", "retriever": "bm25"},
            {"query_id": "Q0", "retriever": "dense"},
        ],
        "rankings": {
            "Q0.bm25": [{"child_id": "a"}, {"child_id": "b"}],
            "Q0.dense": [{"child_id": "b"}],
            "bm25": [{"child_id": "a"}] * 10,
        },
    }
    chunks = {c: {"id": c, "paper_id": "p"} for c in ["a", "b"]}
    assert replay_candidates(label, trace, chunks) == ["b", "a"]
    chunks["b"]["paper_id"] = "foreign"
    with pytest.raises(ValueError, match="scope"):
        replay_candidates(label, trace, chunks)


def test_development_parents_preserve_public_text_but_invent_no_pdf_pages():
    from evaluation.ranking_packing_benchmark import development_sources

    chunks = [
        {
            "id": f"c{i}",
            "paper_id": "p",
            "text": f"Result {i}.",
            "section": "Results",
            "chunk_index": i,
        }
        for i in range(4)
    ]
    sources = development_sources(chunks, [{"paper_id": "p", "title": "Paper"}])
    assert sources["c1"]["parent"]["raw_text"] == "Result 0.\n\nResult 1.\n\nResult 2."
    assert sources["c1"]["excerpt"] == "Result 1."
    assert sources["c1"]["source_origin"] == "qasper_public_text"
    assert sources["c1"]["page_start"] is None
    assert sources["c3"]["parent"]["id"] != sources["c1"]["parent"]["id"]


def test_ranking_metrics_remain_separate_from_background_span_coverage():
    from evaluation.ranking_packing_benchmark import ranking_metrics, span_coverage

    label = {"relevant_chunk_ids": ["evidence"]}
    assert ranking_metrics(["irrelevant", "evidence"], label)["mrr_at_10"] == 0.5
    assert ranking_metrics(["irrelevant"], label)["recall_at_10"] == 0
    blocks = [
        {"source_spans": [{"block_id": "evidence", "raw_start": 0, "raw_end": 12}]}
    ]
    chunks = {"evidence": {"text": "Exact quote."}}
    assert span_coverage(blocks, label, chunks) == 1
    blocks[0]["source_spans"][0]["raw_end"] = 5
    assert span_coverage(blocks, label, chunks) == 0
