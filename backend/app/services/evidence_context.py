"""Rerank, reserve paper coverage, and attribute every candidate selection."""

import hashlib

from app.config import settings
from app.services.child_reranker import rerank_children, select_children
from app.services.context_packing import pack_context


async def prepare_evidence(question, sources, *, paper_ids=None, trace=None):
    body = [
        s
        for s in sources
        if s.get("evidence_role") not in {"bibliography", "page_marker"}
    ]
    ranked = await rerank_children(question, body, trace=trace)
    selected = select_children(ranked, settings.reranker_selected_limit, paper_ids)
    packed = pack_context(selected, paper_ids=paper_ids, trace=trace)
    for paper_id in dict.fromkeys(
        [*(paper_ids or []), *(s["paper_id"] for s in sources)]
    ):
        count = sum(s["paper_id"] == paper_id for s in sources)
        entry = packed.coverage.setdefault(
            paper_id,
            {
                "prompt_count": 0,
                "prompt_status": "not_sent",
                "context_count": 0,
                "context_status": "unknown",
                "support_status": "unknown",
            },
        )
        entry.update(
            candidate_count=count, candidate_status="available" if count else "unknown"
        )
    if trace is not None:
        trace.add_ranked(
            "selected_children",
            [(s["chunk_id"], None) for s in selected],
            duration_ms=0,
        )
        record = trace.record["context_packing"]
        selected_ids = {s["chunk_id"] for s in selected}
        record["decisions"].extend(
            {
                "child_id": s["chunk_id"],
                "reason": "non_body_evidence"
                if s not in body
                else "below_selection_limit",
            }
            for s in sources
            if s["chunk_id"] not in selected_ids
        )
        record["candidate_count"] = len(sources)
        record["assembled_evidence_coverage"] = (
            len(packed.sources) / len(sources) if sources else None
        )
        record["paper_coverage"] = {
            hashlib.sha256(k.encode()).hexdigest(): v
            for k, v in packed.coverage.items()
        }
    return packed
