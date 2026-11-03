"""Bounded local → optional correction → academic Web workflow, not an agent."""

import copy
from dataclasses import dataclass, field, replace

from sqlalchemy import select

from app.config import settings
from app.models import (
    Paper,
    PaperChunk,
    PaperIdentifier,
    SourceDocument,
    SourceIndexGeneration,
)
from app.services.academic_search import AcademicSearch, PaperIdentity
from app.services.context_packing import PackedContext, pack_context
from app.services.evidence_context import prepare_evidence
from app.services.evidence_routing import (
    LocalFacts,
    RoutingDecision,
    assess_local,
    content_terms,
    route_after_local,
)
from app.services.web_evidence import build_web_evidence
from app.workflows.query_expansion import QueryPlan, RetrievalQuery, expand_queries
from app.workflows.source_query import retrieve


@dataclass
class LocalRun:
    packed: PackedContext
    mode: str
    query_count: int
    rewrite_status: str
    retrieval_status: str


@dataclass
class RoutingResult:
    packed: PackedContext
    decision: RoutingDecision
    local: LocalRun
    assessment: object
    web_records: list[dict] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


async def corpus_facts(db, policy):
    if policy.paper_ids is None:
        return LocalFacts(), []
    papers = list(
        (await db.scalars(select(Paper).where(Paper.id.in_(policy.paper_ids)))).all()
    )
    identifiers = list(
        (
            await db.scalars(
                select(PaperIdentifier).where(
                    PaperIdentifier.paper_id.in_(policy.paper_ids)
                )
            )
        ).all()
    )
    documents = list(
        (
            await db.execute(
                select(
                    SourceDocument.paper_id,
                    SourceDocument.document_version,
                    SourceIndexGeneration.is_active,
                )
                .outerjoin(
                    SourceIndexGeneration,
                    SourceIndexGeneration.document_id == SourceDocument.id,
                )
                .where(
                    SourceDocument.paper_id.in_(policy.paper_ids),
                    SourceDocument.content_scope == "full_text",
                )
            )
        ).all()
    )
    known_versions, active_versions = {}, {}
    for paper_id, version, active in documents:
        known_versions.setdefault(paper_id, set()).add(version)
        if active:
            active_versions.setdefault(paper_id, set()).add(version)
    versions = {}
    for paper in papers:
        candidates = active_versions.get(paper.id) or known_versions.get(
            paper.id, set()
        )
        if len(candidates) == 1:
            versions[paper.id] = next(iter(candidates))
        elif candidates or paper.content_scope == "full_text":
            # No external bytes/abstract may impersonate an unresolved native version.
            versions[paper.id] = "unresolved_native_version"
    indexed = frozenset(
        (
            await db.scalars(
                select(PaperChunk.paper_id)
                .where(PaperChunk.paper_id.in_(policy.paper_ids))
                .distinct()
            )
        ).all()
    )
    identities = []
    for paper in papers:
        ids = {i.kind: i.value for i in identifiers if i.paper_id == paper.id}
        identities.append(
            PaperIdentity(
                paper.id,
                doi=ids.get("doi"),
                arxiv_id=ids.get("arxiv"),
                provider=paper.import_source,
                provider_id=ids.get(paper.import_source),
                document_version=versions.get(paper.id),
            )
        )
    return LocalFacts(
        frozenset(p.id for p in papers),
        indexed,
        frozenset(p.id for p in papers if p.content_scope == "abstract"),
    ), identities


async def local_retrieval(
    db, policy, history, trace, *, correction=False, retriever=None
):
    retriever = retriever or retrieve
    if policy.retrieval_mode == "hybrid" and not correction:
        plan = await expand_queries(
            policy.question, history, policy.rewrite_query, trace=trace
        )
    else:
        queries = [RetrievalQuery("Q0", "original", policy.question, policy.question)]
        if correction:
            # Preserve literal Q0; add one deterministic content-first query.
            terms = sorted(content_terms(policy.question))
            if terms:
                queries.append(
                    RetrievalQuery("Q1", "correction", " ".join(terms), policy.question)
                )
        plan = QueryPlan(
            queries,
            "disabled",
            "targeted_local_correction" if correction else "explicit_baseline",
        )
        trace.record_query_plan(plan)
    trace.record_query(policy.question, plan.status)
    evidence, mode = await retriever(
        db,
        policy.question,
        list(policy.paper_ids) if policy.paper_ids is not None else None,
        policy.retrieval_mode,
        year_from=policy.year_from,
        year_to=policy.year_to,
        provider=policy.provider,
        trace=trace,
        queries=plan.queries,
        **(
            {"candidate_limit": settings.reranker_candidate_limit}
            if policy.retrieval_mode == "hybrid"
            else {}
        ),
    )
    if policy.retrieval_mode == "hybrid":
        packed = await prepare_evidence(
            policy.question, evidence, paper_ids=policy.paper_ids, trace=trace
        )
    else:
        packed = pack_context(
            evidence, paper_ids=policy.paper_ids, expand=False, trace=trace
        )
    return LocalRun(packed, mode, len(plan.queries), plan.status, plan.reason)


async def run_local_web(
    db, policy, history, trace, *, local_runner=None, searcher=None
):
    if policy.only_uploaded:
        statement = select(Paper.id).where(Paper.import_source.is_(None))
        if policy.paper_ids is not None:
            statement = statement.where(Paper.id.in_(policy.paper_ids))
        allowed = tuple((await db.scalars(statement)).all())
        policy = replace(policy, paper_ids=allowed)

    async def default_runner(p, h, t, *, correction=False):
        return await local_retrieval(db, p, h, t, correction=correction)

    runner = local_runner or default_runner
    # This must happen before even constructing a provider client.
    local = await runner(policy, history, trace)
    facts, identities = await corpus_facts(db, policy)
    assessment = assess_local(
        policy,
        local.packed.sources,
        facts,
        local.mode,
        trace.record.get("reranking", {}).get("status"),
    )
    decision = route_after_local(policy, assessment, 0)
    attempts = [
        {
            "assessment": assessment.state,
            "reasons": list(assessment.reasons),
            "rankings": copy.deepcopy(trace.record["rankings"]),
            "query_variants": copy.deepcopy(trace.record.get("query_variants", [])),
            "reranking": copy.deepcopy(trace.record.get("reranking", {})),
        }
    ]
    if decision.action == "correct_local":
        trace.record["local_retry_count"] = 1
        initial = local
        # Narrow correction to an uncovered requested paper known to exist.
        covered = {s["paper_id"] for s in local.packed.sources}
        missing = [
            p
            for p in policy.paper_ids or ()
            if p in facts.indexed_papers and p not in covered
        ]
        correction_policy = (
            replace(policy, paper_ids=(missing[0],)) if missing else policy
        )
        try:
            corrected = await runner(correction_policy, history, trace, correction=True)
            combined = list(
                {
                    s["chunk_id"]: s
                    for s in [*initial.packed.sources, *corrected.packed.sources]
                }.values()
            )
            local = LocalRun(
                pack_context(combined, paper_ids=policy.paper_ids, trace=trace),
                corrected.mode,
                initial.query_count + corrected.query_count,
                initial.rewrite_status,
                initial.retrieval_status,
            )
        except Exception as exc:
            trace.record_failure("local_correction", exc)
            local = initial
        assessment = assess_local(
            policy,
            local.packed.sources,
            facts,
            local.mode,
            trace.record.get("reranking", {}).get("status"),
        )
        attempts.append(
            {
                "assessment": assessment.state,
                "reasons": list(assessment.reasons),
                "rankings": copy.deepcopy(trace.record["rankings"]),
                "query_variants": copy.deepcopy(trace.record.get("query_variants", [])),
                "reranking": copy.deepcopy(trace.record.get("reranking", {})),
            }
        )
        decision = route_after_local(policy, assessment, 1)
    trace.record["local_routing_attempts"] = attempts
    trace.record["web_decision"] = decision.action
    trace.record["routing_reasons"] = list(decision.reasons)
    result = RoutingResult(local.packed, decision, local, assessment)
    if "privacy_overrides_freshness" in decision.reasons:
        result.notices.append("本轮仅使用本地资料，未联网；无法保证覆盖最新论文。")
    elif policy.web_prohibited:
        result.notices.append("本轮禁止学术 Web 外访，仅使用允许范围内的本地证据。")
    if assessment.state != "sufficient":
        result.notices.append("本地证据尚不充分：" + ", ".join(assessment.reasons))
    if decision.action != "search_web":
        return result
    try:
        search = await (searcher or AcademicSearch()).search(policy, identities)
        trace.record["web_events"].extend(search.events)
        pool = await build_web_evidence(
            search.candidates, policy, identities, trace=trace, budget=search.budget
        )
        result.web_records = pool.records
        if pool.sources:
            # One common context budget. Local verifiable excerpts retain reservation.
            web = pool.sources[: max(0, 50 - len(local.packed.sources))]
            result.packed = await prepare_evidence(
                policy.question,
                [*local.packed.sources, *web],
                paper_ids=policy.paper_ids,
                trace=trace,
            )
        if not pool.sources:
            result.notices.append(
                "学术 Web 未提供可用正文或摘要证据；保留本地结果，"
                "书目信息不支持实验结论。"
            )
        else:
            result.notices.append(
                "Web 证据仅临时用于本轮，未导入论文库；摘要证据不等同于全文。"
            )
    except Exception as exc:
        trace.record_failure("academic_web", exc)
        result.notices.append(
            "学术 Web 暂不可用；保留可核对的本地证据，缺失结论保持未知。"
        )
    return result
