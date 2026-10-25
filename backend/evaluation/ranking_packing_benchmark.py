"""Actual local reranking/packing over frozen public saved retrieval lists.

Retrieval is replayed, not re-timed. Adjacent public text parents are explicitly
development adapters, not evidence about the native PDF parser's quality.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from app.config import settings
from app.services.child_reranker import (
    LocalChildScorer,
    rerank_children,
    select_children,
)
from app.services.context_packing import pack_context
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_index import rrf_merge
from app.services.trace_versions import configuration_snapshot, digest
from evaluation.evaluate import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.isolated_corpus import load_jsonl, mean
from evaluation.isolated_corpus import paper_id as corpus_paper_id
from evaluation.multiquery_benchmark import CORPUS, HASHES


def replay_candidates(label, trace, chunks):
    if label["query"] != trace["original_query"]:
        raise ValueError("Replay question mismatch")
    lists, seen = [], set()
    for branch in trace["retrieval_branches"]:
        key = f"{branch['query_id']}.{branch['retriever']}"
        if key in seen:
            raise ValueError("Duplicate retrieval branch")
        seen.add(key)
        ranked = [row["child_id"] for row in trace["rankings"][key]][:30]
        for identifier in ranked:
            if identifier not in chunks or (
                label.get("paper_ids") is not None
                and chunks[identifier]["paper_id"] not in label["paper_ids"]
            ):
                raise ValueError("Saved retrieval scope mismatch")
        lists.append(ranked)
    if not 1 <= len(lists) <= 6:
        raise ValueError("Invalid saved branch count")
    return rrf_merge(lists, k=60, limit=30)


def development_sources(chunks, papers):
    """Non-overlapping three-child parents within one public paper/section."""
    sections, result = defaultdict(list), {}
    titles = {corpus_paper_id(paper): paper["title"] for paper in papers}
    corpus_digest = digest([(c["id"], c["text"]) for c in chunks])
    document_version = "qasper-text:" + corpus_digest
    for chunk in chunks:
        sections[(chunk["paper_id"], chunk.get("section"))].append(chunk)
    for (paper_id, section), children in sections.items():
        children.sort(key=lambda c: c["chunk_index"])
        document_id = "qasper-public:" + paper_id
        for left in range(0, len(children), 3):
            group = children[left : left + 3]
            parent_id = "qasper-adjacent:" + digest([c["id"] for c in group])
            spans = [
                {
                    "paper_id": paper_id,
                    "document_id": document_id,
                    "document_version": document_version,
                    "page": None,
                    "block_id": c["id"],
                    "raw_start": 0,
                    "raw_end": len(c["text"]),
                    "bbox": None,
                }
                for c in group
            ]
            parent = {
                "id": parent_id,
                "raw_text": "\n\n".join(c["text"] for c in group),
                "source_spans": spans,
                "page_start": None,
                "page_end": None,
            }
            for chunk, span in zip(group, spans, strict=True):
                result[chunk["id"]] = {
                    "chunk_id": chunk["id"],
                    "paper_id": paper_id,
                    "excerpt": chunk["text"],
                    "title": titles[paper_id],
                    "section": section,
                    "content_scope": chunk.get("content_scope", "full_text"),
                    "page_start": None,
                    "page_end": None,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "document_version": document_version,
                    "source_origin": "qasper_public_text",
                    "source_spans": [span],
                    "parent": parent,
                    "parent_provenance": "adjacent_public_chunks_v1",
                }
    return result


def ranking_metrics(ranked, label):
    relevant = set(label["relevant_chunk_ids"])
    return {
        "recall_at_10": recall_at_k(ranked, relevant),
        "hit_at_10": hit_at_k(ranked, relevant),
        "mrr_at_10": mrr_at_k(ranked, relevant),
        "ndcg_at_10": ndcg_at_k(ranked, relevant),
    }


def span_coverage(blocks, label, chunks):
    relevant = label["relevant_chunk_ids"]
    if not relevant:
        return None
    complete = 0
    for identifier in relevant:
        ranges = sorted(
            (s["raw_start"], s["raw_end"])
            for block in blocks
            for s in block["source_spans"]
            if s["block_id"] == identifier
        )
        cursor = 0
        for left, right in ranges:
            if left > cursor:
                break
            cursor = max(cursor, right)
        complete += cursor >= len(chunks[identifier]["text"])
    return complete / len(relevant)


def percentile(values, fraction):
    ordered = sorted(values)
    return (
        ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
        if ordered
        else None
    )


async def run(destination, saved_trace):
    for name, expected in HASHES.items():
        if hashlib.sha256((CORPUS / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen public corpus hash mismatch")
    chunks = load_jsonl(CORPUS / "chunks.jsonl")
    labels = load_jsonl(CORPUS / "labels.jsonl")
    papers = json.loads((CORPUS / "papers.json").read_text())
    traces = load_jsonl(saved_trace)
    if len(labels) != len(traces):
        raise ValueError("Saved trace count mismatch")
    by_id = {c["id"]: c for c in chunks}
    sources = development_sources(chunks, papers)
    model = LocalChildScorer()
    start = time.perf_counter()
    await asyncio.to_thread(model.load)
    cold_load_ms = (time.perf_counter() - start) * 1000
    destination.mkdir(parents=True, exist_ok=True)
    rows, context_records = [], []
    trace_path = destination / "trace_ranking_packing.jsonl"
    if trace_path.exists():
        raise ValueError(
            "Destination already has a run; preserve it and choose a fresh directory"
        )
    for index, (label, saved) in enumerate(zip(labels, traces, strict=True)):
        candidate_ids = replay_candidates(label, saved, by_id)
        candidates = [sources[identifier] for identifier in candidate_ids]
        recorder = RetrievalTrace(
            label["query"],
            {"paper_ids": label.get("paper_ids")},
            corpus_version="qasper-v0.3-dev-40",
            corpus_public=True,
        )
        ranked = await rerank_children(
            label["query"], candidates, scorer=model, trace=recorder
        )
        selected = select_children(ranked, 10, label.get("paper_ids"))
        variants = {
            "rrf_flat": pack_context(
                candidates[:10], expand=False, paper_ids=label.get("paper_ids")
            ),
            "rerank_flat": pack_context(
                selected, expand=False, paper_ids=label.get("paper_ids")
            ),
            "rerank_parent": pack_context(
                selected, paper_ids=label.get("paper_ids"), trace=recorder
            ),
        }
        row = {
            "case_id": label["id"],
            "query": label["query"],
            "candidate_ids": candidate_ids,
            "reranked_ids": [s["chunk_id"] for s in ranked],
            "rerank_status": recorder.record["reranking"]["status"],
            "rerank_ms": recorder.record["reranking"]["duration_ms"],
            "retrieval_latency_ms": None,
            "rrf_top10": ranking_metrics(candidate_ids[:10], label),
            "rerank_top10": ranking_metrics([s["chunk_id"] for s in selected], label),
            "contexts": {},
        }
        for name, packed in variants.items():
            row["contexts"][name] = {
                **ranking_metrics([s["chunk_id"] for s in packed.sources], label),
                "child_ids": [s["chunk_id"] for s in packed.sources],
                "fully_visible_evidence_fraction": span_coverage(
                    packed.blocks, label, by_id
                ),
                "assembled_child_fraction": len(packed.sources)
                / min(10, len(candidates))
                if candidates
                else None,
                "token_units": packed.token_units,
                "parent_window_count": sum(
                    b["expansion_status"] == "parent_window" for b in packed.blocks
                ),
            }
        recorder.finish("packing_only", [])
        recorder.append(trace_path)
        rows.append(row)
        context_records.append(
            {
                "case_id": label["id"],
                "query": label["query"],
                "contexts": {
                    name: [b["text"] for b in packed.blocks]
                    for name, packed in variants.items()
                },
            }
        )
        print(f"case {index + 1}/{len(labels)}: {row['rerank_status']}", flush=True)
    report = {
        "evaluation_kind": "actual_local_rerank_packing_on_saved_retrieval",
        "retrieval_latency_ms": None,
        "corpus_hashes": HASHES,
        "saved_trace_sha256": hashlib.sha256(saved_trace.read_bytes()).hexdigest(),
        "saved_retrieval_pipeline_fingerprints": sorted(
            {t["config_manifest"]["pipeline_revision"] for t in traces}
        ),
        "config": configuration_snapshot()[1],
        "parent_provenance": "adjacent_public_chunks_v1_not_pdf",
        "model": model.model_id,
        "model_revision": model.revision,
        "device": "cpu_threads2",
        "cold_load_ms": cold_load_ms,
        "rerank_p50_ms": percentile([r["rerank_ms"] for r in rows], 0.5),
        "rerank_p95_ms": percentile([r["rerank_ms"] for r in rows], 0.95),
        "case_count": len(rows),
        "answerable_count": sum(bool(label["relevant_chunk_ids"]) for label in labels),
        "rank_metrics": {
            name: {
                metric: mean([r[name] for r in rows], metric)
                for metric in ranking_metrics([], labels[0])
            }
            for name in ("rrf_top10", "rerank_top10")
        },
        "context_metrics": {
            name: {
                metric: mean([r["contexts"][name] for r in rows], metric)
                for metric in (
                    *ranking_metrics([], labels[0]),
                    "fully_visible_evidence_fraction",
                    "assembled_child_fraction",
                    "token_units",
                )
            }
            for name in ("rrf_flat", "rerank_flat", "rerank_parent")
        },
        "results": rows,
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (destination / "public_contexts.json").write_text(
        json.dumps(context_records), encoding="utf-8"
    )
    return report


async def judge(destination):
    """One public-reference case, three contexts, two metrics; local smoke only."""
    from app.services.local_execution import loopback_url

    if settings.llm_provider != "ollama" or not loopback_url(settings.ollama_base_url):
        raise ValueError("Judge requires explicitly local Ollama")
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    os.environ["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "0"
    from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase

    from evaluation.deepeval_adapter import JudgeConfig, _judge_model

    reference_path = CORPUS / "qasper_generation_reference_inputs.jsonl"
    references = {
        r["case_id"]: r
        for r in load_jsonl(reference_path)
        if r.get("expected_output") and r["trace"].get("corpus_public")
    }
    contexts = json.loads((destination / "public_contexts.json").read_text())
    selected = next(row for row in contexts if row["case_id"] in references)
    reference = references[selected["case_id"]]
    if reference["trace"]["original_query"] != selected["query"]:
        raise ValueError("Public reference question mismatch")
    output = []
    for name, blocks in selected["contexts"].items():
        for metric_name, metric_type in (
            ("contextual_relevancy", ContextualRelevancyMetric),
            ("contextual_precision", ContextualPrecisionMetric),
        ):
            row = {
                "case_id": selected["case_id"],
                "mode": name,
                "metric": metric_name,
                "model": settings.ollama_model,
                "provider": "ollama",
                "temperature": 0,
                "deepeval_version": version("deepeval"),
                "threshold": 0.5,
                "prompt_template": "deepeval_default",
                "context_limit": 1,
                "status": "not_evaluated",
                "score": None,
                "error_type": None,
                "cost": None,
                "api_call_count": None,
                "token_usage": None,
                "reference_input_sha256": hashlib.sha256(
                    reference_path.read_bytes()
                ).hexdigest(),
            }
            started = time.perf_counter()
            try:
                async with asyncio.timeout(60):
                    model = _judge_model(
                        JudgeConfig("ollama", settings.ollama_model, "")
                    )
                    metric = metric_type(model=model, threshold=0.5, async_mode=True)
                    await metric.a_measure(
                        LLMTestCase(
                            input=selected["query"],
                            actual_output="",
                            retrieval_context=blocks[:1],
                            expected_output=reference["expected_output"],
                        )
                    )
                    row.update(
                        status="evaluated",
                        score=metric.score,
                        cost=getattr(metric, "evaluation_cost", None),
                    )
            except Exception as exc:
                row["error_type"] = type(exc).__name__
            row["latency_ms"] = (time.perf_counter() - started) * 1000
            output.append(row)
            (destination / "judge_smoke.json").write_text(
                json.dumps(output, indent=2), encoding="utf-8"
            )
            print(f"judge {name} {metric_name}: {row['status']}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/evaluation/m3_verified_20261024/trace_multi_hybrid.jsonl"),
    )
    parser.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()
    if args.judge_only:
        await judge(args.output)
    else:
        report = await run(args.output, args.trace)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("rank_metrics", "context_metrics", "rerank_p95_ms")
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
