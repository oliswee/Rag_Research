from evaluation.evaluate import citations_valid, ndcg_at_k, recall_at_k


def test_ranked_metrics_and_citation_integrity():
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=1) == 0
    assert recall_at_k(["irrelevant", "relevant"], {"relevant"}, k=2) == 1
    assert 0 < ndcg_at_k(["irrelevant", "relevant"], {"relevant"}) < 1
    assert citations_valid("Claim [C:known]", [{"chunk_id": "known"}])
    assert not citations_valid("Claim [C:invented]", [{"chunk_id": "known"}])


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
        "Insufficient evidence in the local library. Discover papers first.",
        [],
        answerable=False,
    )
    assert not citations_valid(
        "An uncited claim despite no matching papers.", [], answerable=False
    )
