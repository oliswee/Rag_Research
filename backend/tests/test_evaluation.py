import json

import pytest

from evaluation.evaluate import citations_valid, ndcg_at_k, recall_at_k


def test_ranked_metrics_and_citation_integrity():
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=1) == 0
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=2) == 1
    assert 0 < ndcg_at_k(["irrelevant", "relevant"], {"relevant"}) < 1
    assert citations_valid("Claim [C:known]", [{"chunk_id": "known"}])
    assert not citations_valid("Claim [C:invented]", [{"chunk_id": "known"}])
    assert not citations_valid("Claim without citation", [{"chunk_id": "known"}])


def test_no_answer_queries_do_not_receive_perfect_recall():
    assert recall_at_k(["irrelevant"], set()) is None
    assert ndcg_at_k(["irrelevant"], set()) is None


def test_labels_reject_duplicates_and_unknown_chunk_ids(tmp_path):
    from evaluation.labels import load_cases

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {
                    "query": "What is active retrieval?",
                    "group": "exact_terms",
                    "relevant_chunk_ids": ["known"],
                },
                {
                    "query": "  what is active retrieval?  ",
                    "group": "semantic_rephrase",
                    "relevant_chunk_ids": ["known"],
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate query"):
        load_cases(labels, {"known"})

    labels.write_text(
        json.dumps(
            {
                "query": "Does it improve latency?",
                "group": "exact_terms",
                "relevant_chunk_ids": ["missing"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown chunk ID"):
        load_cases(labels, {"known"})


def test_no_answer_labels_must_have_no_relevant_chunks(tmp_path):
    from evaluation.labels import load_cases

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "query": "What is not covered?",
                "group": "no_answer",
                "relevant_chunk_ids": ["known"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No-answer"):
        load_cases(labels, {"known"})


def test_quality_gate_needs_independent_labels_and_all_retrieval_modes():
    from evaluation.evaluate import build_report

    groups = ["exact_terms", "semantic_rephrase", "cross_paper", "no_answer"]
    results = []
    for index in range(40):
        group = groups[index % len(groups)]
        recall = None if group == "no_answer" else 0.8
        hybrid_recall = (
            None if group == "no_answer" else (0.9 if group == "exact_terms" else 0.8)
        )
        results.append(
            {
                "query": f"Independent question {index}",
                "group": group,
                "citation_ids_valid": True,
                "modes": {
                    "dense": {
                        "actual_mode": "dense",
                        "recall_at_10": recall,
                        "ndcg_at_10": recall,
                        "latency_ms": 10,
                    },
                    "bm25": {
                        "actual_mode": "bm25",
                        "recall_at_10": recall,
                        "ndcg_at_10": recall,
                        "latency_ms": 5,
                    },
                    "hybrid": {
                        "actual_mode": "hybrid",
                        "recall_at_10": hybrid_recall,
                        "ndcg_at_10": hybrid_recall,
                        "latency_ms": 15,
                    },
                },
            }
        )

    assert not build_report(results, human_test_labels_confirmed=False)[
        "quality_gate_met"
    ]
    assert build_report(results, human_test_labels_confirmed=True)["quality_gate_met"]
    results[0]["modes"]["hybrid"]["actual_mode"] = "bm25_fallback"
    assert not build_report(results, human_test_labels_confirmed=True)[
        "quality_gate_met"
    ]
