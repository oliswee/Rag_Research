import importlib.util
from types import SimpleNamespace

from app.schemas.query import QueryRequest
from app.services.evidence_routing import parse_policy


def clarify(question, *, titles=(), paper_ids=None, history=(), latest=False):
    assert importlib.util.find_spec("app.services.query_clarification"), (
        "Scope clarification helper is missing"
    )
    from app.services.query_clarification import clarify_scope

    policy = parse_policy(
        QueryRequest(query=question, paper_ids=paper_ids, latest=latest)
    )
    return clarify_scope(
        question,
        [
            SimpleNamespace(id=str(i), title=t, content_scope="abstract")
            for i, t in enumerate(titles)
        ],
        list(history),
        policy,
    )


def test_duplicate_named_papers_need_identity_not_guess():
    result = clarify(
        "What does Memory Study report?", titles=["Memory Study", "Memory Study"]
    )
    assert result.question and result.code == "ambiguous_paper_title"
    assert (
        clarify(
            "What does Memory Study report?",
            titles=["Memory Study", "Memory Study"],
            paper_ids=["0"],
        ).question
        is None
    )


def test_latest_uses_a_visible_bounded_assumption_instead_of_guessing_time_window():
    result = clarify("latest memory papers", latest=True)
    assert result.question is None and result.assumptions


def test_known_abstract_only_scope_links_discovery_without_import():
    result = clarify(
        "What are the full-text experiments?", titles=["Memory Study"], paper_ids=["0"]
    )
    assert result.question is None and result.discovery_url == "/discover"
    assert "abstract" in " ".join(result.assumptions).lower()
