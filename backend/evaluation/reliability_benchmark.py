"""Opt-in actual local-hardware runtime probe on isolated synthetic documents.

Uses production QA and retrieval, local SQLite/FTS and real in-memory Qdrant.
The explicit storage adapter avoids Docker/user-library writes; it is not a
Qdrant server load test or an independent answer-quality evaluation.
"""

import argparse
import asyncio
import json
import math
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import fitz
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base, Paper, PaperType
from app.routers.query import submit_query
from app.schemas.query import QueryRequest
from app.services.embedding_service import EmbeddingService
from app.services.incremental_index import sync_generation
from app.services.local_execution import local_only_request, loopback_url
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.source_index import SourceVectorIndex, ensure_fts
from app.services.trace_versions import configuration_snapshot
from app.workflows.source_query import retrieve


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return round(
        values[lower] + (values[upper] - values[lower]) * (position - lower), 3
    )


def summaries(rows):
    markers = {"selected", "selected_children"}
    stages = sorted(
        {stage for row in rows for stage in row.get("stage_ms", {})} - markers
    )
    result = {}
    for stage in ["observed_wall_ms", "total_ms", *stages]:
        samples = [
            row.get(stage)
            if stage in {"total_ms", "observed_wall_ms"}
            else row.get("stage_ms", {}).get(stage)
            for row in rows
        ]
        samples = [value for value in samples if isinstance(value, (int, float))]
        result[stage] = {
            "n": len(samples),
            "p50_ms": percentile(samples, 0.5),
            "p95_ms": percentile(samples, 0.95),
        }
    return result


async def run(output, count):
    if count < 30 or count > 100:
        raise ValueError("Use 30 to 100 requests")
    if (
        settings.embedding_provider != "fastembed"
        or settings.llm_provider != "ollama"
        or not loopback_url(settings.ollama_base_url)
    ):
        raise ValueError("Benchmark requires local BGE and loopback Ollama")
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces.jsonl"
    if traces.exists():
        raise ValueError("Choose a fresh output directory")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    index = SourceVectorIndex.__new__(SourceVectorIndex)
    index.client = QdrantClient(":memory:")
    index.dataset_id = "m7-isolated-synthetic-runtime"
    embedder = EmbeddingService()
    index.dimensions = embedder.dimensions
    token = local_only_request.set(True)
    original_trace = settings.retrieval_trace_path
    settings.retrieval_trace_path = str(traces.resolve())
    rows, corpus = [], []
    setup_started = time.perf_counter()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            await ensure_fts(db)
            for i in range(6):
                quote = f"Memory accuracy is {95 - i} percent."
                paper = Paper(
                    title=f"Synthetic memory study {chr(65 + i)}",
                    type=PaperType.research_paper,
                    content_scope="full_text" if i % 2 == 0 else "abstract",
                )
                db.add(paper)
                await db.flush()
                parser = PDFParserService()
                if i % 2 == 0:
                    document = fitz.open()
                    page = document.new_page()
                    page.insert_textbox(
                        fitz.Rect(50, 50, 550, 750), "Results\n" + quote
                    )
                    parsed = parser.parse(document.tobytes())
                    document.close()
                else:
                    parsed = parser.parse_abstract(quote)
                build = await stage_generation(db, paper, parsed)
                await sync_generation(db, build, embedder=embedder, vector_index=index)
                corpus.append(
                    {
                        "paper_id": paper.id,
                        "content_scope": paper.content_scope,
                        "quote": quote,
                    }
                )
            await db.commit()
            setup_ms = (time.perf_counter() - setup_started) * 1000

            async def isolated_retrieve(*args, **kwargs):
                return await retrieve(*args, **kwargs, vector_index=index)

            # Only storage location is adapted. Retrieval, BGE, reranker and model
            # responses are actual implementations, with no latency/answer stubs.
            with patch("app.routers.query.retrieve", isolated_retrieve):
                for i in range(count):
                    item = corpus[i % len(corpus)]
                    mode = ("bm25", "dense", "hybrid")[i % 3]
                    started = time.perf_counter()
                    response, error_type = None, None
                    try:
                        response = await submit_query(
                            QueryRequest(
                                query="What memory accuracy is reported in this study?",
                                paper_ids=[item["paper_id"]],
                                retrieval_mode=mode,
                                rewrite_query=False,
                                local_only=True,
                            ),
                            db,
                        )
                    except Exception as exc:
                        error_type = type(exc).__name__
                        await db.rollback()
                    trace = (
                        json.loads(traces.read_text(encoding="utf-8").splitlines()[-1])
                        if traces.exists()
                        else {}
                    )
                    row = {
                        "request": i + 1,
                        "mode": mode,
                        "cold_first_request": i == 0,
                        "source_kind": item["content_scope"],
                        "observed_wall_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
                        "error_type": error_type,
                        "support_status": (response or {}).get("support_status"),
                        "citations": len((response or {}).get("citations", [])),
                        "fake_pdf_pages": sum(
                            c.get("pdf_page") is not None
                            for c in (response or {}).get("citations", [])
                            if c["evidence_type"] != "full_text"
                        ),
                        "expected_quote_present": item["quote"]
                        in (response or {}).get("response", ""),
                        **{
                            key: trace.get(key)
                            for key in (
                                "run_id",
                                "total_ms",
                                "stage_ms",
                                "stage_usage",
                                "llm_calls",
                                "token_usage",
                                "request_budget",
                                "failures",
                                "reranking",
                                "retrieval_branches",
                            )
                        },
                    }
                    rows.append(row)
                    with (output / "requests.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "request",
                                    "mode",
                                    "support_status",
                                    "observed_wall_ms",
                                    "error_type",
                                )
                            }
                        ),
                        flush=True,
                    )
            config_version, config = configuration_snapshot()
            result = {
                "schema": "m7_actual_local_runtime_v1",
                "created_at": datetime.now(UTC).isoformat(),
                "provenance": "synthetic_runtime_not_human_gold",
                "adapter": "production_QA_real_local_Qdrant_SQLite_FTS",
                "hardware": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "setup_ms": round(setup_ms, 3),
                "requests": count,
                "successful_responses": sum(row["error_type"] is None for row in rows),
                "support_status": dict(Counter(row["support_status"] for row in rows)),
                "per_stage": summaries(rows),
                "warm_per_stage": summaries(rows[1:]),
                "unmeasured_ranking_markers": ["selected", "selected_children"],
                "model_calls": sum(row["llm_calls"] or 0 for row in rows),
                "external_attempts": sum(
                    (row["request_budget"] or {}).get("external_attempts", 0)
                    for row in rows
                ),
                "fake_pdf_pages": sum(row["fake_pdf_pages"] for row in rows),
                "config_version": config_version,
                "config": config,
                "limitations": [
                    "serial synthetic requests; no concurrency capacity inference",
                    "query rewriting disabled",
                    "Qdrant local engine, not server transport",
                    "response availability does not establish answer correctness",
                ],
            }
            (output / "summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "completed": count,
                        "successful_responses": result["successful_responses"],
                    }
                ),
                flush=True,
            )
    finally:
        settings.retrieval_trace_path = original_trace
        local_only_request.reset(token)
        index.client.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run(args.output, args.count))
