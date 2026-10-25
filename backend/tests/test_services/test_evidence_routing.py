from dataclasses import FrozenInstanceError

import pytest

from app.schemas.query import QueryRequest
from app.services.evidence_routing import (
    LocalFacts,
    assess_local,
    parse_policy,
    route_after_local,
)


@pytest.mark.parametrize(
    "question,fields,intent,blocked",
    [
        ("Explain memory", {}, "factual", False),
        ("Compare memory methods", {}, "compare", False),
        ("Find papers on memory", {}, "discover", False),
        ("最新的记忆研究", {}, "latest", False),
        ("Explain memory", {"latest": True}, "latest", False),
        ("Latest papers", {"local_only": True}, "latest", True),
        ("Find papers", {"only_uploaded": True}, "discover", True),
        ("只用上传的论文，比较方法", {}, "compare", True),
        ("Only use local papers; latest memory", {}, "latest", True),
    ],
)
def test_literal_policy(question, fields, intent, blocked):
    policy = parse_policy(QueryRequest(query=question, **fields))
    assert policy.intent == intent
    assert policy.web_prohibited is blocked
    with pytest.raises(FrozenInstanceError):
        policy.intent = "discover"


def test_hard_scope_is_immutable():
    request = QueryRequest(
        query="Find papers", paper_ids=["a"], year_from=2020, provider="arxiv"
    )
    policy = parse_policy(request)
    request.paper_ids.append("b")
    assert policy.paper_ids == ("a",)
    assert (policy.year_from, policy.provider) == (2020, "arxiv")


def test_sufficient_requires_original_span_and_content_coverage():
    policy = parse_policy(QueryRequest(query="memory accuracy"))
    sources = [
        {
            "paper_id": "a",
            "excerpt": "Memory accuracy improves.",
            "evidence_level": "full_text",
            "raw_start": 0,
            "raw_end": 25,
        }
    ]
    result = assess_local(policy, sources, LocalFacts(), "hybrid", "applied")
    assert result.state == "sufficient"
    assert route_after_local(policy, result, 0).action == "answer_local"
    assert (
        assess_local(
            policy,
            [{**sources[0], "excerpt": "Unrelated material"}],
            LocalFacts(),
            "hybrid",
            "applied",
        ).state
        == "uncertain"
    )


def test_unreturned_is_not_known_absent_and_correction_is_bounded():
    policy = parse_policy(QueryRequest(query="memory accuracy", paper_ids=["a"]))
    facts = LocalFacts(present_papers=frozenset({"a"}), indexed_papers=frozenset({"a"}))
    result = assess_local(policy, [], facts, "hybrid", "unavailable")
    assert result.state == "uncertain"
    assert "suspected_retrieval_miss" in result.reasons
    assert route_after_local(policy, result, 0).action == "correct_local"
    assert route_after_local(policy, result, 1).action == "search_web"


@pytest.mark.parametrize(
    "facts,reason",
    [
        (LocalFacts(), "known_missing_paper"),
        (LocalFacts(present_papers=frozenset({"a"})), "known_index_gap"),
        (
            LocalFacts(
                present_papers=frozenset({"a"}),
                indexed_papers=frozenset({"a"}),
                abstract_only_papers=frozenset({"a"}),
            ),
            "known_abstract_only",
        ),
    ],
)
def test_known_gaps_only_from_observed_facts(facts, reason):
    policy = parse_policy(QueryRequest(query="memory results", paper_ids=["a"]))
    result = assess_local(policy, [], facts, "hybrid", "applied")
    assert result.state == "corpus_gap"
    assert reason in result.reasons


def test_privacy_wins_and_latest_forces_web():
    for fields, action in [
        ({"latest": True}, "search_web"),
        ({"latest": True, "local_only": True}, "answer_local"),
    ]:
        policy = parse_policy(QueryRequest(query="memory", **fields))
        assessment = assess_local(
            policy, [], LocalFacts(), "retrieval_failed", "failed"
        )
        decision = route_after_local(policy, assessment, 0)
        assert decision.action == action
        if fields.get("local_only"):
            assert "privacy_overrides_freshness" in decision.reasons


def test_numbers_subquestions_and_abstract_are_conservative():
    policy = parse_policy(QueryRequest(query="accuracy 95 and latency 20"))
    result = assess_local(
        policy,
        [{"excerpt": "Accuracy 95", "evidence_level": "abstract_only"}],
        LocalFacts(),
        "hybrid",
        "applied",
    )
    assert result.state == "uncertain"
    assert "missing_numeric_support" in result.reasons
    assert "abstract_only_evidence" in result.reasons
    assert "incomplete_question_coverage" in result.reasons


def test_retrieval_failure_does_not_mean_corpus_gap():
    policy = parse_policy(QueryRequest(query="memory"))
    result = assess_local(policy, [], LocalFacts(), "retrieval_failed", "failed")
    assert result.state == "retrieval_failure"
    assert "known_corpus_gap" not in result.reasons
