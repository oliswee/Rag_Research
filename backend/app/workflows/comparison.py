"""Independent paper retrieval, scoped Web fallback and verified matrix columns."""

import copy
import hashlib
from dataclasses import replace
from itertools import combinations

from fastapi import HTTPException
from sqlalchemy import select

from app.models import Paper
from app.schemas.evidence import GroundedAnswer, StudyCharacteristics
from app.schemas.query import QueryRequest
from app.schemas.research import ComparisonCell, ComparisonResponse, ComparisonRow
from app.services.academic_search import AcademicSearch, WebBudget
from app.services.evidence_context import prepare_evidence
from app.services.evidence_registry import build_registry
from app.services.evidence_routing import RoutingDecision, Sufficiency, parse_policy
from app.services.grounded_generation import VerificationBudget, generate_grounded
from app.services.local_execution import local_only_request
from app.services.retrieval_trace import RetrievalTrace
from app.services.study_comparison import compare_studies
from app.workflows.local_web import (
    LocalRun,
    RoutingResult,
    local_retrieval,
    run_local_web,
)

DIMENSIONS = [
    "Common findings",
    "Different conclusions",
    "Methods and samples",
    "Limitations and open questions",
]
DEFAULT_QUESTION = (
    "Compare main findings, conclusions, methods and samples, "
    "limitations and open questions across the selected papers."
)


async def collect_papers(db, policy, trace, *, local_runner=None, searcher=None):
    """All selected local stages finish before constructing an academic client."""
    ids = list(policy.paper_ids or ())
    if not 2 <= len(ids) <= 5 or len(set(ids)) != len(ids):
        raise ValueError("Require2–5 distinct paper identities")
    found = {
        p.id: p
        for p in (await db.scalars(select(Paper).where(Paper.id.in_(ids)))).all()
    }
    if len(found) != len(ids):
        raise HTTPException(404, "One or more selected papers not found")

    async def runner(p, history, t, *, correction=False):
        return await local_retrieval(db, p, history, t, correction=correction)

    runner = local_runner or runner
    runs = []
    for paper_id in ids:
        paper = found[paper_id]
        scope = (
            ()
            if policy.only_uploaded and paper.import_source is not None
            else (paper_id,)
        )
        per_paper = replace(policy, paper_ids=scope, rewrite_query=False)
        child = RetrievalTrace(
            policy.question,
            {
                "paper_ids": list(scope),
                "local_only": policy.local_only,
                "only_uploaded": policy.only_uploaded,
            },
            corpus_version=trace.record["corpus_version"],
        )
        local = await runner(per_paper, [], child)
        runs.append((paper, per_paper, child, local))
    budget = WebBudget()
    searcher = searcher or AcademicSearch(budget=budget)
    results = []
    for paper, p, child, local in runs:
        result = await run_local_web(
            db,
            p,
            [],
            child,
            local_runner=runner,
            searcher=searcher,
            initial_local=local,
        )
        results.append((paper, result))
        trace.record.setdefault("paper_runs", []).append(
            {
                "paper_sha256": hashlib.sha256(paper.id.encode()).hexdigest(),
                "run_id": child.run_id,
                "retrieval_mode": result.local.mode,
                "local_retry_count": child.record["local_retry_count"],
                "rankings": child.record["rankings"],
                "retrieval_budget": child.record.get("retrieval_budget", {}),
                "local_routing_attempts": child.record.get(
                    "local_routing_attempts", []
                ),
                "web_decision": child.record["web_decision"],
                "web_events": child.record["web_events"],
                "stage_ms": child.record["stage_ms"],
                "failures": child.record["failures"],
            }
        )
    trace.record["comparison_web_attempts"] = budget.attempts
    return results


async def collect_scoped_query(db, policy, trace):
    runs = await collect_papers(db, policy, trace)
    sources = [source for _, result in runs for source in result.packed.sources]
    packed = await prepare_evidence(
        policy.question, sources, paper_ids=policy.paper_ids, trace=trace
    )
    searched = any(result.decision.action == "search_web" for _, result in runs)
    reasons = tuple(
        dict.fromkeys(
            reason for _, result in runs for reason in result.decision.reasons
        )
    )
    modes = {result.local.mode for _, result in runs}
    local = LocalRun(
        packed,
        "hybrid" if modes == {"hybrid"} else "hybrid_degraded",
        sum(result.local.query_count for _, result in runs),
        "disabled",
        "independent_paper_scope",
    )
    trace.record["local_retry_count"] = sum(
        run["local_retry_count"] for run in trace.record["paper_runs"]
    )
    trace.record["web_decision"] = "search_web" if searched else "answer_local"
    return RoutingResult(
        packed=packed,
        decision=RoutingDecision(trace.record["web_decision"], reasons),
        local=local,
        assessment=Sufficiency(
            "sufficient"
            if all(result.assessment.state == "sufficient" for _, result in runs)
            else "uncertain",
            reasons,
        ),
        web_records=[r for _, result in runs for r in result.web_records],
        notices=list(
            dict.fromkeys(notice for _, result in runs for notice in result.notices)
        ),
        web_snapshots=tuple(s for _, result in runs for s in result.web_snapshots),
    )


def _characteristics(claims):
    result = StudyCharacteristics()
    supported = [c for c in claims if c.support_status == "supported"]
    for key in StudyCharacteristics.model_fields:
        values = [getattr(c.characteristics, key) for c in supported]
        if key in {"year", "version", "doi"}:
            nonempty = {v for v in values if v is not None}
            if len(nonempty) == 1:
                setattr(result, key, next(iter(nonempty)))
        else:
            fields = [v for v in values if v.support_status == "supported" and v.value]
            if fields and len({v.value.casefold() for v in fields}) == 1:
                setattr(result, key, fields[0])
    return result


async def run_comparison(
    db, request, trace, *, local_runner=None, searcher=None, generator=None
):
    from app.routers.query import _enforce_local_only

    question = request.question or DEFAULT_QUESTION
    policy = parse_policy(
        QueryRequest(
            query=question,
            intent="compare",
            rewrite_query=False,
            **request.model_dump(exclude={"question"}),
        )
    )
    _enforce_local_only(policy.local_only or policy.only_uploaded)
    token = local_only_request.set(policy.local_only or policy.only_uploaded)
    try:
        runs = await collect_papers(
            db, policy, trace, local_runner=local_runner, searcher=searcher
        )
        budget, answers, notices, coverage, web_records = (
            VerificationBudget(),
            {},
            [],
            {},
            [],
        )
        for paper, result in runs:
            registry = await build_registry(
                db,
                result.packed.sources,
                web_snapshots=result.web_snapshots,
                paper_ids=[paper.id],
                local_only=policy.local_only or policy.only_uploaded,
            )
            try:
                answer = await (generator or generate_grounded)(
                    question,
                    registry,
                    result.packed,
                    trace,
                    dimensions=DIMENSIONS,
                    budget=budget,
                )
            except Exception as exc:
                trace.record_failure("comparison_generation", exc)
                answer = GroundedAnswer(
                    text="Insufficient verified evidence found",
                    evidence=registry.source_dicts(),
                    uncertainty=["verification_unavailable"],
                )
            answers[paper.id] = answer
            trace.record.setdefault("verification_columns", []).append(
                {
                    "paper_sha256": hashlib.sha256(paper.id.encode()).hexdigest(),
                    "verification": copy.deepcopy(trace.record.get("verification", {})),
                }
            )
            coverage[paper.id] = {
                **result.packed.coverage.get(
                    paper.id,
                    {
                        "candidate_count": 0,
                        "context_count": 0,
                        "context_status": "unknown",
                    },
                ),
                "supported_count": sum(
                    c.support_status == "supported" for c in answer.claims
                ),
                "prompt_count": sum(
                    s["chunk_id"] in trace.record["prompt_evidence_ids"]
                    for s in registry.source_dicts()
                ),
            }
            notices.extend(result.notices)
            web_records.extend(result.web_records)
        rows = []
        for label in DIMENSIONS:
            cells = []
            for paper, _ in runs:
                answer = answers[paper.id]
                claims = [c for c in answer.claims if c.dimension == label]
                supported = [
                    c for c in claims if c.support_status == "supported" and c.text
                ]
                citations = list(
                    {
                        c.citation_id: c for claim in supported for c in claim.citations
                    }.values()
                )
                ids = {c.source_id for c in citations}
                cells.append(
                    ComparisonCell(
                        paper_id=paper.id,
                        text="\n\n".join(c.text for c in supported)
                        if supported
                        else "Insufficient verified evidence found",
                        evidence=[s for s in answer.evidence if s["chunk_id"] in ids]
                        if supported
                        else answer.evidence,
                        claims=claims,
                        citations=citations,
                        support_status="supported"
                        if supported
                        else "unsupported"
                        if any(
                            c.support_status in {"unsupported", "contradicted"}
                            for c in claims
                        )
                        else "unknown",
                        uncertainty=answer.uncertainty
                        + [reason for c in claims for reason in c.uncertainty],
                    )
                )
            assessments = [
                compare_studies(
                    _characteristics(left.claims),
                    _characteristics(right.claims),
                    paper_ids=(left.paper_id, right.paper_id),
                )
                for left, right in combinations(cells, 2)
            ]
            rows.append(
                ComparisonRow(label=label, cells=cells, comparisons=assessments)
            )
        # Aggregate coverage cannot use a last-column context denominator.
        trace.record.pop("context_packing", None)
        trace.record["comparison_coverage"] = {
            hashlib.sha256(k.encode()).hexdigest(): value
            for k, value in coverage.items()
        }
        used = list(
            dict.fromkeys(
                c.source_id
                for row in rows
                for cell in row.cells
                for c in cell.citations
            )
        )
        trace.finish(
            "supported"
            if used
            else "evidence_only"
            if any(a.evidence for a in answers.values())
            else "abstain",
            trace.record["prompt_evidence_ids"],
            used,
        )
        return ComparisonResponse(
            run_id=trace.run_id,
            papers=[{"id": p.id, "title": p.title} for p, _ in runs],
            rows=rows,
            coverage=coverage,
            notices=list(dict.fromkeys(notices)),
            web_records=web_records,
            verification={
                "calls": budget.calls,
                "reserved_output_tokens": budget.reserved_output_tokens,
                "input_bytes": budget.input_bytes,
            },
        )
    finally:
        local_only_request.reset(token)
