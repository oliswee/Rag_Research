"""Deterministic routing hints. Scores are not calibrated probabilities."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPolicy:
    question: str
    intent: str = "factual"
    paper_ids: tuple[str, ...] | None = None
    year_from: int | None = None
    year_to: int | None = None
    provider: str | None = None
    local_only: bool = False
    only_uploaded: bool = False
    retrieval_mode: str = "hybrid"
    rewrite_query: bool = True

    @property
    def web_prohibited(self):
        return self.local_only or self.only_uploaded or self.retrieval_mode != "hybrid"


def parse_policy(request) -> QueryPolicy:
    text = request.query.lower()
    uploaded = request.only_uploaded or bool(
        re.search(
            r"只(?:用|查|看|使用).{0,8}上传|仅.{0,4}上传|only\s+(?:use\s+)?(?:the\s+)?uploaded",
            text,
        )
    )
    local = request.local_only or bool(
        re.search(
            r"仅.{0,4}本地|只(?:用|查|看|使用).{0,8}本地|only\s+(?:use\s+)?local|do not (?:search|use) (?:the )?(?:web|internet)",
            text,
        )
    )
    intent = request.intent
    if request.latest or re.search(r"\blatest\b|\brecent\b|最新|近年", text):
        intent = "latest"
    elif intent is None:
        if re.search(r"\bcompar(?:e|ison)\b|比较|对比", text):
            intent = "compare"
        elif re.search(
            r"\b(?:find|discover|search)\s+(?:\w+\s+){0,2}papers\b|找.{0,6}论文|搜索.{0,6}文献",
            text,
        ):
            intent = "discover"
        else:
            intent = "factual"
    return QueryPolicy(
        request.query,
        intent,
        tuple(request.paper_ids) if request.paper_ids is not None else None,
        request.year_from,
        request.year_to,
        request.provider,
        bool(local),
        bool(uploaded),
        request.retrieval_mode,
        request.rewrite_query,
    )


@dataclass(frozen=True)
class LocalFacts:
    present_papers: frozenset[str] = frozenset()
    indexed_papers: frozenset[str] = frozenset()
    abstract_only_papers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Sufficiency:
    state: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    reasons: tuple[str, ...]


_STOP = set(
    "what which how does do is are the a an of for in to and or with explain compare methods results papers paper use only local uploaded latest find on".split()
)


def content_terms(text):
    return {
        w
        for w in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
        if w not in _STOP
    }


def assess_local(
    policy, sources, facts, retrieval_mode, rerank_status=None
) -> Sufficiency:
    if retrieval_mode in {"retrieval_failed", "failed"}:
        return Sufficiency("retrieval_failure", ("local_retrieval_failed",))
    reasons = []
    scoped = set(policy.paper_ids or ())
    if scoped - facts.present_papers:
        reasons.append("known_missing_paper")
    if scoped & facts.present_papers - facts.indexed_papers:
        reasons.append("known_index_gap")
    if scoped & facts.abstract_only_papers:
        reasons.append("known_abstract_only")
    if reasons:
        return Sufficiency("corpus_gap", tuple(reasons))
    factual = [s for s in sources if s.get("evidence_level") != "metadata_only"]
    if not factual or scoped - {s.get("paper_id") for s in factual}:
        reasons.append("suspected_retrieval_miss")
    text = " ".join(s.get("excerpt", "") for s in factual).lower()
    terms = content_terms(policy.question)
    if not terms or len(terms & content_terms(text)) / len(terms) < 0.65:
        reasons.append("incomplete_question_coverage")
    # Independently check conjunctions; a long first subquestion cannot hide the second.
    parts = re.split(r"\band\b|[;；？?]|以及|并且", policy.question.lower())
    if any(
        content_terms(p) and not content_terms(p) & content_terms(text) for p in parts
    ):
        reasons.append("incomplete_question_coverage")
    if set(re.findall(r"\d+(?:\.\d+)?", policy.question)) - set(
        re.findall(r"\d+(?:\.\d+)?", text)
    ):
        reasons.append("missing_numeric_support")
    if factual and all(
        s.get("evidence_level", s.get("content_scope")) in {"abstract", "abstract_only"}
        for s in factual
    ):
        reasons.append("abstract_only_evidence")
    if factual and not all(
        s.get("source_spans")
        or (s.get("raw_start") is not None and s.get("raw_end") is not None)
        for s in factual
    ):
        reasons.append("unverified_original_span")
    if any(s.get("conflicting_evidence") for s in factual):
        reasons.append("conflicting_evidence")
    return Sufficiency(
        "uncertain" if reasons else "sufficient",
        tuple(dict.fromkeys(reasons)) or ("original_span_question_coverage",),
    )


def route_after_local(policy, assessment, correction_count=0):
    if policy.web_prohibited:
        reasons = ["external_access_prohibited"]
        if policy.intent in {"latest", "discover"}:
            reasons.append("privacy_overrides_freshness")
        return RoutingDecision("answer_local", tuple(reasons))
    if policy.intent in {"latest", "discover"}:
        return RoutingDecision("search_web", ("freshness_required",))
    if assessment.state == "sufficient":
        return RoutingDecision("answer_local", assessment.reasons)
    if assessment.state in {"uncertain", "retrieval_failure"} and correction_count < 1:
        return RoutingDecision("correct_local", assessment.reasons)
    return RoutingDecision("search_web", assessment.reasons)
