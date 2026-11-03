"""Labelled development routes, separate from Gold and retrieval quality scores.

uv run python -m evaluation.local_web_benchmark --output data/evaluation/m5/summary.json
Optional --live runs two fixed public metadata searches, never PDFs or paid models.
"""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.base import Base
from app.schemas.query import QueryRequest
from app.services.academic_search import AcademicSearch, SearchResult
from app.services.context_packing import pack_context
from app.services.evidence_routing import parse_policy
from app.services.retrieval_trace import RetrievalTrace
from app.services.trace_versions import configuration_snapshot
from app.workflows.local_web import LocalRun, run_local_web

TEXT = "Memory accuracy improves with retrieval."


def source(text=TEXT, **fields):
    return {
        "chunk_id": "fixture-child",
        "paper_id": "fixture-paper",
        "title": "Development fixture",
        "excerpt": text,
        "content_scope": "full_text",
        "source_origin": "pdf",
        "page_start": 1,
        "page_end": 1,
        "source_spans": [
            {
                "paper_id": "fixture-paper",
                "document_id": "fixture-document",
                "document_version": "fixture-v1",
                "page": 1,
                "block_id": "fixture-block",
                "raw_start": 0,
                "raw_end": len(text),
            }
        ],
        **fields,
    }


# Labels describe intended behaviour, including two deliberate heuristic challenges.
CASES = [
    ("sufficient", "memory accuracy", {}, [source()], False),
    ("empty_local", "memory accuracy", {}, [], True),
    ("latest", "latest memory papers", {}, [source()], True),
    ("discover", "find papers on memory", {}, [source()], True),
    ("local_latest_conflict", "latest memory papers", {"local_only": True}, [], False),
    (
        "uploaded_latest_conflict",
        "latest memory papers",
        {"only_uploaded": True},
        [],
        False,
    ),
    ("literal_privacy", "不要联网，查最新论文", {}, [], False),
    ("known_missing", "memory accuracy", {"paper_ids": ["absent"]}, [], True),
    ("abstract_only", "memory accuracy", {}, [source(content_scope="abstract")], True),
    ("numbers_missing", "memory accuracy 99", {}, [source()], True),
    ("second_question_missing", "memory accuracy and latency", {}, [source()], True),
    ("unanchored", "memory accuracy", {}, [source(source_spans=[])], True),
    ("explicit_baseline", "memory accuracy", {"retrieval_mode": "bm25"}, [], False),
    (
        "conflicting",
        "memory accuracy",
        {},
        [source(), source("Memory accuracy does not improve.", chunk_id="opposed")],
        True,
    ),
    (
        "semantic_synonym_challenge",
        "recollection correctness",
        {},
        [
            source(
                "Recall performance, or correctness of remembered information, "
                "improves."
            )
        ],
        False,
    ),
    (
        "lexical_without_finding_challenge",
        "memory accuracy",
        {},
        [source("Memory accuracy is a heading in our proposed future evaluation.")],
        True,
    ),
]


async def run_routes():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    results = []
    try:
        async with async_sessionmaker(engine)() as db:
            for name, question, fields, sources, expected in CASES:
                events = []

                async def local(
                    policy,
                    history,
                    trace,
                    *,
                    correction=False,
                    events=events,
                    sources=sources,
                ):
                    events.append("correction" if correction else "local")
                    return LocalRun(
                        pack_context(sources, trace=trace),
                        "hybrid",
                        1,
                        "disabled",
                        "fixture",
                    )

                class Search:
                    async def search(self, policy, identities, events=events):
                        events.append("web")
                        return SearchResult()

                trace = RetrievalTrace(
                    question,
                    fields,
                    corpus_version="m5-labelled-development-v1",
                    corpus_public=True,
                )
                started = time.perf_counter()
                result = await run_local_web(
                    db,
                    parse_policy(QueryRequest(query=question, **fields)),
                    [],
                    trace,
                    local_runner=local,
                    searcher=Search(),
                )
                actual = "web" in events
                results.append(
                    {
                        "case": name,
                        "expected_web": expected,
                        "actual_web": actual,
                        "events": events,
                        "sufficiency": result.assessment.state,
                        "reasons": list(result.assessment.reasons),
                        "local_first": events[0] == "local",
                        "corrections": events.count("correction"),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        "prompt_evidence_count": len(
                            trace.record["prompt_evidence_ids"]
                        ),
                    }
                )
    finally:
        await engine.dispose()
    return results


async def run_faults():
    results = []
    for fault in (
        "429",
        "timeout",
        "blocked",
        "network",
        "oversize",
        "redirect",
        "malformed",
    ):
        calls = []

        def handler(request, calls=calls, fault=fault):
            calls.append(request.url.host)
            # A second provider remains available: partial failure must preserve it.
            if request.url.host == "api.openalex.org":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "https://openalex.org/W1",
                                "display_name": "Public fixture",
                                "publication_year": 2025,
                            }
                        ]
                    },
                )
            if fault == "timeout":
                raise httpx.ReadTimeout("untrusted provider error", request=request)
            if fault == "network":
                raise httpx.ConnectError("untrusted provider error", request=request)
            if fault == "oversize":
                return httpx.Response(
                    200, content=b"x" * (settings.web_metadata_bytes + 1)
                )
            if fault == "redirect":
                return httpx.Response(
                    302, headers={"location": "http://127.0.0.1/private"}
                )
            if fault == "malformed":
                return httpx.Response(200, content=b"not-json")
            return httpx.Response(429 if fault == "429" else 403)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await AcademicSearch(client, retry_delay=0).search(
                parse_policy(QueryRequest(query="memory accuracy"))
            )
        results.append(
            {
                "fault": fault,
                "attempts": len(calls),
                "partial_candidates": len(result.candidates),
                "events": result.events,
                "passed": len(result.candidates) == 1
                and len(calls) <= 3
                and "untrusted provider error" not in json.dumps(result.events),
            }
        )
    return results


async def live_connectivity():
    results = []
    previous = settings.web_max_attempts
    settings.web_max_attempts = 1
    try:
        for provider in ("openalex", "semantic_scholar"):
            started = time.perf_counter()
            result = await AcademicSearch().search(
                parse_policy(
                    QueryRequest(
                        query="retrieval augmented generation", provider=provider
                    )
                )
            )
            results.append(
                {
                    "provider": provider,
                    "candidate_count": len(result.candidates),
                    "events": result.events,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "purpose": "connectivity_only",
                }
            )
    finally:
        settings.web_max_attempts = previous
    return results


async def judge_smoke(destination):
    """Two local-only judge diagnostics, not a project answer-quality estimate."""
    from importlib.metadata import version

    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from app.services.local_execution import loopback_url
    from evaluation.deepeval_adapter import JudgeConfig, _judge_model

    if not loopback_url(settings.ollama_base_url):
        raise ValueError("Local judge endpoint required")
    results = []
    for name, answer in (
        ("supported_abstract", "The abstract reports memory accuracy of 95 percent."),
        (
            "unsupported_number_control",
            "The abstract reports memory accuracy of 99 percent.",
        ),
    ):
        row = {
            "case": name,
            "status": "not_evaluated",
            "score": None,
            "model": settings.ollama_model,
            "provider": "ollama",
            "temperature": 0,
            "threshold": 0.5,
            "metric": "faithfulness",
            "deepeval_version": version("deepeval"),
            "template": "deepeval_default",
            "api_call_count": None,
            "token_usage": None,
            "cost": None,
            "purpose": "synthetic_development_diagnostic",
        }
        started = time.perf_counter()
        try:
            async with asyncio.timeout(60):
                metric = FaithfulnessMetric(
                    model=_judge_model(
                        JudgeConfig("ollama", settings.ollama_model, "")
                    ),
                    threshold=0.5,
                    async_mode=True,
                )
                await metric.a_measure(
                    LLMTestCase(
                        input="What memory accuracy does the abstract report?",
                        actual_output=answer,
                        retrieval_context=[
                            "The abstract reports memory accuracy of 95 percent."
                        ],
                        context=["The abstract reports memory accuracy of 95 percent."],
                    )
                )
                row.update(
                    status="evaluated",
                    score=metric.score,
                    cost=getattr(metric, "evaluation_cost", None),
                )
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        results.append(row)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"local judge {name}: {row['status']}", flush=True)
    return results


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()
    if args.judge_only:
        await judge_smoke(args.output)
        return
    routes, faults = await run_routes(), await run_faults()
    fp = sum(r["actual_web"] and not r["expected_web"] for r in routes)
    fn = sum(not r["actual_web"] and r["expected_web"] for r in routes)
    config, manifest = configuration_snapshot()
    result = {
        "schema": "m5_development_v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "config_version": config,
        "config_manifest": manifest,
        "fixture_sha256": hashlib.sha256(
            json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "limitations": (
            "Hand-authored development fixtures with injected local "
            "retrieval; not Gold, real Web gain or latency SLA."
        ),
        "routes": routes,
        "faults": faults,
        "summary": {
            "route_count": len(routes),
            "wrong_web_triggers": fp,
            "missed_web_triggers": fn,
            "local_first_count": sum(r["local_first"] for r in routes),
            "max_corrections": max(r["corrections"] for r in routes),
            "faults_passed": sum(f["passed"] for f in faults),
            "fault_count": len(faults),
        },
        "live": await live_connectivity() if args.live else [],
        "generation_calls": 0,
        "judge_calls": 0,
        "paid_cloud_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    asyncio.run(main())
