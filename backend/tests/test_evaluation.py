from evaluation.evaluate import (
    build_report,
    citations_valid,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ranked_metrics_and_citation_integrity():
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=1) == 0
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=2) == 1
    assert 0 < ndcg_at_k(["irrelevant", "relevant"], {"relevant"}) < 1
    assert citations_valid("Claim [C:known]", [{"chunk_id": "known"}])
    assert not citations_valid("Claim [C:invented]", [{"chunk_id": "known"}])


def test_relevance_metrics_skip_unanswerable_cases_and_duplicate_results():
    assert recall_at_k(["unrelated"], set()) is None
    assert ndcg_at_k(["unrelated"], set()) is None
    assert ndcg_at_k(["relevant", "relevant"], {"relevant"}) == 1.0
    assert ndcg_at_k(["relevant"], {"relevant"}, k=0) == 0.0


def test_hit_and_mrr_reflect_first_relevant_rank_within_cutoff():
    ranked = ["irrelevant", "relevant", "relevant"]
    assert hit_at_k(ranked, {"relevant"}, k=1) == 0.0
    assert hit_at_k(ranked, {"relevant"}, k=2) == 1.0
    assert mrr_at_k(ranked, {"relevant"}, k=1) == 0.0
    assert mrr_at_k(ranked, {"relevant"}, k=2) == 0.5
    assert hit_at_k(ranked, set()) is None
    assert mrr_at_k(ranked, set()) is None


def test_report_keeps_no_answer_cases_out_of_relevance_means():
    answerable = {
        "query": "answerable",
        "group": "exact_terms",
        "answerable": True,
        "citation_ids_valid": True,
        "modes": {
            "dense": {
                "actual_mode": "dense",
                "recall_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "hit_at_10": 0.0,
                "mrr_at_10": 0.0,
                "latency_ms": 10.0,
            },
            "bm25": {
                "actual_mode": "bm25",
                "recall_at_10": 1.0,
                "ndcg_at_10": 0.63,
                "hit_at_10": 1.0,
                "mrr_at_10": 0.5,
                "latency_ms": 5.0,
            },
            "hybrid": {
                "actual_mode": "hybrid",
                "recall_at_10": 1.0,
                "ndcg_at_10": 0.63,
                "hit_at_10": 1.0,
                "mrr_at_10": 0.5,
                "latency_ms": 12.0,
            },
        },
    }
    no_answer = {
        "query": "unanswerable",
        "group": "no_answer",
        "answerable": False,
        "citation_ids_valid": True,
        "modes": {
            mode: {
                "actual_mode": mode,
                "recall_at_10": None,
                "ndcg_at_10": None,
                "hit_at_10": None,
                "mrr_at_10": None,
                "latency_ms": 20.0,
            }
            for mode in ("dense", "bm25", "hybrid")
        },
    }

    report = build_report([answerable, no_answer])
    assert report["answerable_count"] == 1
    assert report["no_answer_count"] == 1
    assert report["hybrid"]["recall_at_10"] == 1.0
    assert report["hybrid"]["hit_at_10"] == 1.0
    assert report["hybrid"]["mrr_at_10"] == 0.5
    assert report["hybrid"]["latency_ms"] == 16.0
    assert report["exact_terms_recall_improvement"] == 1.0
    assert not report["quality_gate_met"]

    no_answer_only = build_report([no_answer])
    assert no_answer_only["answerable_count"] == 0
    assert no_answer_only["no_answer_count"] == 1
    assert no_answer_only["dense"]["recall_at_10"] is None
    assert no_answer_only["dense"]["hit_at_10"] is None
    assert not no_answer_only["quality_gate_met"]


def test_citation_check_rejects_uncited_answer_but_allows_explicit_abstention():
    evidence = [{"chunk_id": "known"}]
    assert not citations_valid(
        "A factual answer without a source.", evidence, answerable=True
    )
    assert citations_valid("A sourced fact [C:known]", evidence, answerable=True)
    assert not citations_valid(
        "An invented source [C:made-up]", evidence, answerable=True
    )
    assert citations_valid(
        "Insufficient evidence in the local library. "
        "Discover or import relevant papers first.",
        [],
        answerable=False,
    )
    assert not citations_valid(
        "An uncited claim despite no matching papers.", [], answerable=False
    )
    assert not citations_valid(
        "Insufficient evidence in the local library. But this treatment works.",
        [],
        answerable=False,
    )
