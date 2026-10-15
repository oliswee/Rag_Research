"""Evaluate labelled source chunks through BM25 in an isolated SQLite database.

Usage from backend/:
    uv run python -m evaluation.isolated_corpus \
      --papers manifest.json --chunks chunks.jsonl --labels labels.jsonl \
      --output isolated-report.json

The paper manifest is a JSON array. Each paper needs an ID (or provider + id),
title, and source URL. Each chunk JSONL row needs an ID, paper ID, index, text,
and checksum. Each label JSONL row needs a query and relevant_chunk_ids. An
optional paper_ids list limits retrieval to those papers for that question.
"""

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, Paper, PaperChunk, PaperStatus, PaperType
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_index import ensure_fts
from app.workflows.source_query import retrieve
from evaluation.evaluate import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k

METRICS = ("recall_at_10", "ndcg_at_10", "hit_at_10", "mrr_at_10")


def paper_id(record: dict) -> str:
    if record.get("paper_id"):
        return str(record["paper_id"])
    if record.get("provider") and record.get("id"):
        return f"{record['provider']}:{record['id']}"
    raise ValueError("Paper needs paper_id or provider and id")


def validate_corpus(papers: list[dict], chunks: list[dict], labels: list[dict]) -> None:
    """Reject invalid references before creating or querying any database."""
    if not papers or not chunks or not labels:
        raise ValueError("Papers, chunks, and labels must all be non-empty")
    paper_ids = [paper_id(record) for record in papers]
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Duplicate paper ID")
    if any(not record.get("title") for record in papers):
        raise ValueError("Paper title is required")
    if any(record.get("status", "parsed") != "parsed" for record in papers):
        raise ValueError("All input papers must have parsed content")
    chunk_ids = [record.get("id") for record in chunks]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("Chunk ID is required")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk ID")
    known_papers = set(paper_ids)
    known_chunks = set(chunk_ids)
    chunk_papers = {record["id"]: record.get("paper_id") for record in chunks}
    positions = set()
    for chunk in chunks:
        if chunk.get("paper_id") not in known_papers:
            raise ValueError(f"Unknown paper ID: {chunk.get('paper_id')}")
        if not chunk.get("text") or not chunk.get("checksum"):
            raise ValueError("Chunk text and checksum are required")
        position = (chunk["paper_id"], chunk.get("chunk_index"))
        if position[1] is None or position in positions:
            raise ValueError("Missing or duplicate chunk index within paper")
        positions.add(position)
    for case in labels:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("Label query is required")
        relevant = case.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or any(
            not isinstance(item, str) for item in relevant
        ):
            raise ValueError("Label relevant_chunk_ids must be a list of strings")
        unknown = set(relevant) - known_chunks
        if unknown:
            raise ValueError(f"Unknown relevant chunk ID: {sorted(unknown)[0]}")
        if "paper_ids" in case:
            scope = case["paper_ids"]
            if (
                not isinstance(scope, list)
                or not scope
                or any(not isinstance(item, str) or not item for item in scope)
                or len(scope) != len(set(scope))
                or not set(scope) <= known_papers
            ):
                raise ValueError(
                    "paper_ids must be a non-empty list of known paper IDs"
                )
            if any(chunk_papers[chunk_id] not in scope for chunk_id in relevant):
                raise ValueError("Relevant chunk is outside paper_ids")


def mean(results: list[dict], metric: str) -> float | None:
    values = [result[metric] for result in results if result[metric] is not None]
    return sum(values) / len(values) if values else None


def summary(results: list[dict]) -> dict:
    return {
        "count": len(results),
        **{metric: mean(results, metric) for metric in METRICS},
    }


async def evaluate_bm25(
    papers: list[dict],
    chunks: list[dict],
    labels: list[dict],
    *,
    trace_output: Path | None = None,
    corpus_version: str = "unversioned",
    corpus_public: bool = False,
) -> dict:
    """Use the app's BM25 retrieve path without touching configured app storage."""
    validate_corpus(papers, chunks, labels)
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text("", encoding="utf-8")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as db:
            for record in papers:
                db.add(
                    Paper(
                        id=paper_id(record),
                        title=record["title"],
                        authors=record.get("authors") or [],
                        source=record.get("source_url"),
                        type=PaperType.research_paper,
                        import_source=record.get("provider"),
                        open_access=record.get("open_access"),
                        content_scope="full_text",
                        status=PaperStatus.active,
                    )
                )
            await db.flush()
            for record in chunks:
                db.add(
                    PaperChunk(
                        id=record["id"],
                        paper_id=record["paper_id"],
                        chunk_index=record["chunk_index"],
                        text=record["text"],
                        content_scope=record.get("content_scope", "full_text"),
                        page_start=record.get("page_start"),
                        page_end=record.get("page_end"),
                        start_char=record.get("start_char"),
                        end_char=record.get("end_char"),
                        source_url=record.get("source_url"),
                        checksum=record["checksum"],
                    )
                )
            await db.commit()
            await ensure_fts(db)

            results = []
            for case in labels:
                trace = (
                    RetrievalTrace(
                        case["query"],
                        {"paper_ids": case.get("paper_ids"), "requested_mode": "bm25"},
                        corpus_version=corpus_version,
                        corpus_public=corpus_public,
                    )
                    if trace_output is not None
                    else None
                )
                if trace is not None:
                    trace.record_query(case["query"], "disabled")
                start = time.perf_counter()
                evidence, actual_mode = await retrieve(
                    db,
                    case["query"],
                    paper_ids=case.get("paper_ids"),
                    mode="bm25",
                    trace=trace,
                )
                if trace is not None:
                    trace.finish("retrieval_only", [])
                    trace.append(trace_output)
                latency_ms = (time.perf_counter() - start) * 1000
                if actual_mode != "bm25":
                    raise RuntimeError(f"Expected BM25, got {actual_mode}")
                ranked = [source["chunk_id"] for source in evidence]
                relevant = set(case["relevant_chunk_ids"])
                results.append(
                    {
                        "query": case["query"],
                        "paper_ids": case.get("paper_ids"),
                        "group": case.get("group", "other"),
                        "split": case.get("split", "unspecified"),
                        "actual_mode": actual_mode,
                        "recall_at_10": recall_at_k(ranked, relevant),
                        "ndcg_at_10": ndcg_at_k(ranked, relevant),
                        "hit_at_10": hit_at_k(ranked, relevant),
                        "mrr_at_10": mrr_at_k(ranked, relevant),
                        "latency_ms": latency_ms,
                        "ranked_chunk_ids": ranked,
                    }
                )
        latencies = sorted(result["latency_ms"] for result in results)
        groups = sorted({result["group"] for result in results})
        splits = sorted({result["split"] for result in results})
        return {
            "scope": "isolated_app_bm25",
            "paper_count": len(papers),
            "chunk_count": len(chunks),
            "label_count": len(labels),
            "minimum_40_met": len(labels) >= 40,
            "answerable_count": sum(
                bool(case["relevant_chunk_ids"]) for case in labels
            ),
            "no_answer_count": sum(not case["relevant_chunk_ids"] for case in labels),
            "labels_declared_reviewed": all(
                case.get("label_status") == "independently_reviewed"
                and isinstance(case.get("reviewer"), str)
                and bool(case["reviewer"].strip())
                and isinstance(case.get("review_date"), str)
                and bool(case["review_date"].strip())
                for case in labels
            ),
            "metrics": {metric: mean(results, metric) for metric in METRICS},
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "groups": {
                group: summary(
                    [result for result in results if result["group"] == group]
                )
                for group in groups
            },
            "splits": {
                split: summary(
                    [result for result in results if result["split"] == split]
                )
                for split in splits
            },
            "results": results,
            "quality_gate_met": False,
            "limitations": [
                "BM25-only; no dense or hybrid comparison, answer generation, "
                "or citation review.",
                "Review status requires a reviewer and review date; "
                "metadata alone cannot prove independent review.",
            ],
        }
    finally:
        await engine.dispose()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("isolated-report.json"))
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--corpus-version", default="unversioned")
    parser.add_argument("--public-corpus", action="store_true")
    args = parser.parse_args()
    report = await evaluate_bm25(
        json.loads(args.papers.read_text(encoding="utf-8")),
        load_jsonl(args.chunks),
        load_jsonl(args.labels),
        trace_output=args.trace_output,
        corpus_version=args.corpus_version,
        corpus_public=args.public_corpus,
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}; quality gate met: {report['quality_gate_met']}")


if __name__ == "__main__":
    asyncio.run(main())
