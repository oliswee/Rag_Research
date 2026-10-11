from evaluation.evaluate import citations_valid, ndcg_at_k, recall_at_k


def test_ranked_metrics_and_citation_integrity():
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=1) == 0
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=2) == 1
    assert 0 < ndcg_at_k(["irrelevant", "relevant"], {"relevant"}) < 1
    assert citations_valid("Claim [C:known]", [{"chunk_id": "known"}])
    assert not citations_valid("Claim [C:invented]", [{"chunk_id": "known"}])
