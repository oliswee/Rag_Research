"""Production QA adapter for the common M6 evidence/claim verifier."""

from app.services.context_packing import pack_context
from app.services.evidence_registry import build_registry
from app.services.grounded_generation import generate_grounded


async def answer_grounded(
    db,
    question,
    packed,
    web_snapshots,
    trace,
    *,
    paper_ids=None,
    local_only=False,
    only_uploaded=False,
):
    registry = await build_registry(
        db,
        packed.sources,
        web_snapshots=web_snapshots,
        paper_ids=paper_ids,
        local_only=local_only,
        only_uploaded=only_uploaded,
    )
    trace.record["source_revalidation"] = {
        "accepted_count": len(registry.sources),
        "rejections": list(registry.rejections),
    }
    canonical = pack_context(registry.source_dicts(), paper_ids=paper_ids, trace=trace)
    answer = await generate_grounded(question, registry, canonical, trace)
    assembled = {
        source_id
        for block in trace.record.get("context_packing", {}).get("blocks", [])
        for source_id in block.get("child_ids", [])
    }
    sent = set(trace.record["prompt_evidence_ids"])
    for paper_id in dict.fromkeys([*(paper_ids or []), *packed.coverage]):
        ids = {s.source_id for s in registry.sources.values() if s.paper_id == paper_id}
        entry = packed.coverage.setdefault(
            paper_id, {"candidate_count": 0, "candidate_status": "unknown"}
        )
        supported = sum(
            c.paper_id == paper_id and c.support_status == "supported"
            for c in answer.claims
        )
        entry.update(
            context_count=len(ids & assembled),
            context_status="included" if ids & assembled else "unknown",
            prompt_count=len(ids & sent),
            prompt_status="included" if ids & sent else "not_sent",
            supported_count=supported,
            support_status="supported" if supported else "unknown",
        )
    return answer
