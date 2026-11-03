"""Actual frozen-source retrieval plus explicitly labelled context-only ablations."""

import argparse
import asyncio
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base, Paper, PaperType
from app.services.child_reranker import (
    LocalChildScorer,
    rerank_children,
    select_children,
)
from app.services.context_packing import pack_context
from app.services.embedding_service import EmbeddingService
from app.services.incremental_index import sync_generation
from app.services.local_execution import local_only_request
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.request_budget import RequestBudget, current_budget
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_anchors import resolve_child
from app.services.source_index import CHUNK_COLLECTION, SourceVectorIndex
from app.workflows.query_expansion import QueryPlan, RetrievalQuery, expand_queries
from app.workflows.source_query import retrieve
from evaluation.evaluate import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.final_contracts import (
    freeze_manifest,
    load_jsonl,
    verify_freeze,
    verify_models,
    verify_runtime,
    write_jsonl,
)
from evaluation.final_statistics import (
    evidence_coverage,
    failure_stage,
    paired_summary,
    quantile,
)

VARIANTS = (
    "bm25_flat",
    "dense_flat",
    "single_hybrid_flat",
    "multi_hybrid_flat",
    "multi_parent",
    "multi_rerank_flat",
    "multi_rerank_parent",
)


def metrics(ids, relevant, answerable):
    if answerable and not relevant:
        return {"recall_at_10": 0, "hit_at_10": 0, "mrr_at_10": 0, "ndcg_at_10": 0}
    return {
        "recall_at_10": recall_at_k(ids, relevant),
        "hit_at_10": hit_at_k(ids, relevant),
        "mrr_at_10": mrr_at_k(ids, relevant),
        "ndcg_at_10": ndcg_at_k(ids, relevant),
    }


def spans(sources):
    return [span for source in sources for span in source.get("source_spans", [])]


async def run(root, *, replay=False):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    verify_runtime(manifest)
    verify_models(root, manifest)
    if settings.embedding_provider != "fastembed" or settings.llm_provider != "ollama":
        raise ValueError("Use frozen local BGE/Ollama configuration")
    output = root / ("retrieval_replay" if replay else "retrieval")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve prior run; select a fresh output")
    output.mkdir(exist_ok=True)
    database = root / "retrieval/benchmark.db"
    if database.exists() and not replay:
        raise ValueError("Preserve existing run; use a fresh frozen corpus directory")
    cached = {}
    if replay:
        if not database.is_file():
            raise ValueError("Native source database is required")
        original = root / "retrieval"
        cached_rows = load_jsonl(original / "query_plans.jsonl")
        cached = {row["case_id"]: row for row in cached_rows}
        if len(cached) != len(cached_rows):
            raise ValueError("Duplicate cached plans")
        artifact_freeze = freeze_manifest(
            original,
            [original / "query_plans.jsonl", original / "source_catalog.jsonl"],
            {"source_freeze": manifest["manifest_sha256"]},
        )
        (output / "replay_input_freeze.json").write_text(
            json.dumps(artifact_freeze, indent=2)
        )
    papers = load_jsonl(root / "papers.jsonl")
    references = load_jsonl(root / "development_reference.jsonl")
    documents = {p["paper_id"]: p for p in papers}
    index = SourceVectorIndex(
        dataset_id_override="m8-" + manifest["manifest_sha256"][:16]
    )
    index.ensure_collection()
    embedder = EmbeddingService()
    scorer = LocalChildScorer()
    engine = create_async_engine("sqlite+aiosqlite:///" + database.resolve().as_posix())
    token = local_only_request.set(True)
    setup_started = time.perf_counter()
    rows, contexts, plans = [], [], []
    by_paper = {}
    if replay:
        for source in load_jsonl(root / "retrieval/source_catalog.jsonl"):
            by_paper.setdefault(source["paper_id"], []).append(source)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            for p in [] if replay else papers:
                paper = Paper(
                    id=p["paper_id"],
                    title=p["title"],
                    type=PaperType.research_paper,
                    content_scope="full_text",
                    source=p["source_url"],
                )
                db.add(paper)
                await db.flush()
                build = await stage_generation(
                    db,
                    paper,
                    PDFParserService().parse_text(
                        p["text"], source_origin="qasper_provided_text"
                    ),
                )
                ids = [c.id for c in build.children]
                for start in range(0, len(ids), 100):
                    existing = index.client.retrieve(
                        CHUNK_COLLECTION, ids[start : start + 100], with_payload=True
                    )
                    if any(
                        point.payload.get("dataset_id") != index.dataset_id
                        for point in existing
                    ):
                        raise ValueError("Refusing collision with another dataset")
                await sync_generation(db, build, embedder=embedder, vector_index=index)
                by_paper[p["paper_id"]] = [
                    {
                        **await resolve_child(db, child, allow_parent_failure=True),
                        "title": p["title"],
                    }
                    for child in build.children
                ]
                await db.commit()
            setup_ms = (time.perf_counter() - setup_started) * 1000
            await asyncio.to_thread(scorer.load)
            write_jsonl(
                output / "source_catalog.jsonl",
                [source for sources in by_paper.values() for source in sources],
            )
            for position, reference in enumerate(references):
                question, paper_id = reference["question"], reference["paper_id"]
                all_sources = by_paper[paper_id]
                relevant = {
                    s["chunk_id"]
                    for s in all_sources
                    if (evidence_coverage(reference, s["source_spans"]) or 0) > 0
                }
                full_parse_coverage = evidence_coverage(reference, spans(all_sources))
                trace = RetrievalTrace(
                    question,
                    {"paper_ids": [paper_id], "local_only": True},
                    corpus_version=manifest["manifest_sha256"],
                )
                budget_token = current_budget.set(RequestBudget())
                try:
                    if replay:
                        saved = cached[reference["case_id"]]
                        queries = [
                            RetrievalQuery(**value) for value in saved["queries"]
                        ]
                        if (
                            not queries
                            or len(queries) > 3
                            or queries[0].text != question
                            or any(q.parent_question != question for q in queries)
                        ):
                            raise ValueError("Cached query provenance mismatch")
                        plan = QueryPlan(
                            queries,
                            saved["status"],
                            saved["reason"],
                            saved.get("rejected_count", 0),
                        )
                        trace.record_query_plan(plan)
                    else:
                        plan = await expand_queries(
                            question, [], enabled=True, trace=trace
                        )
                finally:
                    current_budget.reset(budget_token)
                plans.append(
                    {
                        "case_id": reference["case_id"],
                        **asdict(plan),
                        "stage_ms": trace.record["stage_ms"],
                        "llm_calls": trace.record["llm_calls"],
                        "token_usage": trace.record["token_usage"],
                    }
                )
                with (output / "query_plans.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(plans[-1]) + "\n")
                retrieved = {}
                for label, mode, queries in (
                    ("bm25_flat", "bm25", None),
                    ("dense_flat", "dense", None),
                    ("single_hybrid_flat", "hybrid", None),
                    ("multi_hybrid_flat", "hybrid", plan.queries),
                ):
                    recorder = RetrievalTrace(
                        question,
                        {"paper_ids": [paper_id], "local_only": True},
                        corpus_version=manifest["manifest_sha256"],
                    )
                    started = time.perf_counter()
                    found, actual_mode = await retrieve(
                        db,
                        question,
                        paper_ids=[paper_id],
                        mode=mode,
                        vector_index=index,
                        queries=queries,
                        candidate_limit=30,
                        trace=recorder,
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    retrieved[label] = (found, recorder, elapsed, actual_mode)
                multi, multi_trace, multi_ms, actual_mode = retrieved[
                    "multi_hybrid_flat"
                ]
                ranked = await rerank_children(
                    question, multi, scorer=scorer, trace=trace
                )
                selected = select_children(ranked, 10, [paper_id])
                packs = {
                    name: pack_context(data[0][:10], paper_ids=[paper_id], expand=False)
                    for name, data in retrieved.items()
                }
                packs.update(
                    multi_parent=pack_context(multi[:10], paper_ids=[paper_id]),
                    multi_rerank_flat=pack_context(
                        selected, paper_ids=[paper_id], expand=False
                    ),
                    multi_rerank_parent=pack_context(selected, paper_ids=[paper_id]),
                )
                for name in VARIANTS:
                    packed = packs[name]
                    found, recorded, elapsed, mode = retrieved.get(
                        name, retrieved["multi_hybrid_flat"]
                    )
                    ids = [s["chunk_id"] for s in packed.sources]
                    visible_coverage = evidence_coverage(
                        reference,
                        [
                            span
                            for block in packed.blocks
                            for span in block["source_spans"]
                        ],
                    )
                    prompt_coverage = evidence_coverage(
                        reference, spans(packed.sources)
                    )
                    ranked_sources = (
                        selected if name.startswith("multi_rerank") else found[:10]
                    )
                    observed = {
                        "paper_present": paper_id in documents,
                        "parse_valid": full_parse_coverage == 1
                        if reference["answerable_local"]
                        else None,
                        "candidate_found": bool(
                            set(s["chunk_id"] for s in found) & relevant
                        ),
                        "ranked_found": bool(
                            set(s["chunk_id"] for s in ranked_sources) & relevant
                        ),
                        "packed_found": prompt_coverage == 1,
                    }
                    row = {
                        "case_id": reference["case_id"],
                        "variant": name,
                        "provenance": "qasper_reference_development_not_human_gold",
                        "answerable_reference": reference["answerable_local"],
                        **metrics(ids, relevant, reference["answerable_local"]),
                        "paper_hit_at_10": float(
                            any(s["paper_id"] == paper_id for s in packed.sources)
                        )
                        if reference["answerable_local"]
                        else None,
                        "prompt_evidence_coverage": prompt_coverage,
                        "visible_context_evidence_coverage": visible_coverage,
                        "reference_context_precision": len(set(ids) & relevant)
                        / len(ids)
                        if ids and relevant
                        else None,
                        "candidate_count": len(found),
                        "packed_count": len(packed.sources),
                        "packed_token_units": packed.token_units,
                        "retrieval_ms": round(elapsed, 3),
                        "retrieval_timing_shared_with": "multi_hybrid_flat"
                        if name not in retrieved
                        else None,
                        "actual_retrieval_mode": mode,
                        "rerank_ms": trace.record.get("reranking", {}).get(
                            "duration_ms"
                        )
                        if name.startswith("multi_rerank")
                        else None,
                        "stage_ms": recorded.record["stage_ms"],
                        "branches": recorded.record.get("retrieval_branches"),
                        "source_ids": ids,
                        "candidate_ids": [s["chunk_id"] for s in found],
                        "ranked_top10_ids": [s["chunk_id"] for s in ranked_sources],
                        "ranking_before_packing": metrics(
                            [s["chunk_id"] for s in ranked_sources],
                            relevant,
                            reference["answerable_local"],
                        ),
                        "r0_trace": recorded.record,
                        "reference_child_ids": sorted(relevant),
                        "funnel": failure_stage(observed)
                        if reference["answerable_local"]
                        else "unanswerable_reference_control",
                        "funnel_observed": observed,
                        "generation_support": None,
                        "human_claim_support": None,
                    }
                    rows.append(row)
                    with (output / "case_results.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(row) + "\n")
                    contexts.append(
                        {
                            "case_id": reference["case_id"],
                            "variant": name,
                            **asdict(packed),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "case": position + 1,
                            "plan": plan.status,
                            "variants": len(plan.queries),
                            "parent_prompt_coverage": rows[-1][
                                "prompt_evidence_coverage"
                            ],
                            "funnel": rows[-1]["funnel"],
                        }
                    ),
                    flush=True,
                )
            write_jsonl(output / "contexts.jsonl", contexts)
            write_jsonl(
                output / "per_case_sources.jsonl",
                [
                    {
                        "case_id": r["case_id"],
                        "paper_id": r["paper_id"],
                        "all_source_ids": [
                            s["chunk_id"] for s in by_paper[r["paper_id"]]
                        ],
                    }
                    for r in references
                ],
            )
            summary = {
                "cases": len(references),
                "answerable": sum(r["answerable_local"] for r in references),
                "project_human_gold": 0,
                "corpus_sha256": manifest["manifest_sha256"],
                "setup_ms": setup_ms,
                "native_children": sum(map(len, by_paper.values())),
                "embedding_model": embedder.model,
                "reranker_revision": scorer.revision,
                "query_plan_status": dict(Counter(p["status"] for p in plans)),
                "rewrite_calls": sum(p["llm_calls"] for p in plans),
                "replayed_fixed_query_plans": replay,
                "original_rewrite_calls": sum(
                    row["llm_calls"] for row in cached.values()
                )
                if replay
                else None,
                "variants": {},
                "paired": {},
            }
            baseline = [r for r in rows if r["variant"] == "single_hybrid_flat"]
            columns = [
                "recall_at_10",
                "hit_at_10",
                "mrr_at_10",
                "ndcg_at_10",
                "paper_hit_at_10",
                "prompt_evidence_coverage",
                "visible_context_evidence_coverage",
                "reference_context_precision",
            ]
            for name in VARIANTS:
                group = [r for r in rows if r["variant"] == name]
                means = {
                    metric: sum(r[metric] for r in group if r[metric] is not None)
                    / sum(r[metric] is not None for r in group)
                    if any(r[metric] is not None for r in group)
                    else None
                    for metric in columns
                }
                summary["variants"][name] = {
                    **means,
                    "retrieval_p50_ms": quantile(
                        [r["retrieval_ms"] for r in group], 0.5
                    ),
                    "retrieval_p95_ms": quantile(
                        [r["retrieval_ms"] for r in group], 0.95
                    ),
                    "funnel": dict(Counter(r["funnel"] for r in group)),
                }
                summary["paired"][name] = {
                    metric: paired_summary(baseline, group, metric)
                    for metric in (
                        "recall_at_10",
                        "mrr_at_10",
                        "prompt_evidence_coverage",
                    )
                }
            with (output / "ablation.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "variant",
                        *columns,
                        "retrieval_p50_ms",
                        "retrieval_p95_ms",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "variant": name,
                        **{k: v for k, v in values.items() if k != "funnel"},
                    }
                    for name, values in summary["variants"].items()
                )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "completed": len(references),
                        "children": summary["native_children"],
                    }
                ),
                flush=True,
            )
    finally:
        local_only_request.reset(token)
        index.client.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.root, replay=args.replay))
