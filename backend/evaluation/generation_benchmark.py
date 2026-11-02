"""Opt-in M6 public development diagnostics; never independent Gold.

Run: uv run python -m evaluation.generation_benchmark --output data/evaluation/m6
Add --live for bounded local Ollama generation, ablation and DeepEval.
"""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base, Paper, PaperType
from app.services.claim_verification import check_proposal
from app.services.context_packing import pack_context
from app.services.evidence_registry import build_registry, seal_web_sources
from app.services.grounded_generation import VerificationBudget, generate_grounded
from app.services.local_execution import loopback_url
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_anchors import resolve_child
from app.services.trace_versions import configuration_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "m6_claims.json"


def web_source(text, *, full_text=False):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "chunk_id": "web:development",
        "paper_id": "web-study",
        "title": "Synthetic public study",
        "doi": "10.1234/development",
        "source_origin": "web",
        "excerpt": text,
        "evidence_level": "full_text" if full_text else "abstract_only",
        "content_scope": "full_text" if full_text else "abstract",
        "document_version": "sha256:" + digest,
        "source_url": "https://example.org/public-development-study",
        "retrieved_at": "2026-11-02T00:00:00Z",
        "license": "CC-BY-4.0" if full_text else None,
        "page_start": 1 if full_text else None,
        "page_end": 1 if full_text else None,
        "source_spans": [
            {
                "paper_id": "web-study",
                "document_id": "development",
                "document_version": "sha256:" + digest,
                "block_id": "original",
                "page": 1 if full_text else None,
                "raw_start": 0,
                "raw_end": len(text),
                "excerpt_sha256": digest,
            }
        ],
    }


def proposal(row, source):
    return {
        "text": row["claim"],
        "paper_id": source["paper_id"],
        "source_ids": [source["chunk_id"]],
        "supporting_quotes": {source["chunk_id"]: row["quote"]},
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def judge(answer, question, context):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from evaluation.deepeval_adapter import JudgeConfig, _judge_model

    rows = []
    for name, metric_class in (
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
    ):
        row = {
            "metric": name,
            "status": "not_evaluated",
            "score": None,
            "deepeval_version": version("deepeval"),
            "model": settings.ollama_model,
            "provider": "ollama",
            "temperature": 0,
            "threshold": 0.5,
            "template": "deepeval_default",
            "api_call_count": None,
            "token_usage": None,
            "framework_cost": None,
            "total_cost": None,
        }
        started = time.perf_counter()
        try:
            async with asyncio.timeout(60):
                metric = metric_class(
                    model=_judge_model(
                        JudgeConfig("ollama", settings.ollama_model, "")
                    ),
                    threshold=0.5,
                    async_mode=True,
                )
                await metric.a_measure(
                    LLMTestCase(
                        input=question,
                        actual_output=answer,
                        context=context,
                        retrieval_context=context,
                    )
                )
                row.update(
                    status="evaluated",
                    score=metric.score,
                    framework_cost=getattr(metric, "evaluation_cost", None),
                )
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        row["latency_ms"] = round(1000 * (time.perf_counter() - started), 3)
        rows.append(row)
    return rows


class ReplayGeneration:
    """Fixed first generation; semantic call goes to the actual local model."""

    def __init__(self, candidate):
        from app.services.llm_client import LLMClient

        self.candidate = candidate
        self.calls = self.real_calls = 0
        self.client = LLMClient(max_retries=1, timeout_seconds=30)

    async def complete_json(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"claims": [self.candidate]}
        self.real_calls += 1
        return await self.client.complete_json(**kwargs)

    async def close(self):
        await self.client.client.close()

    @property
    def trace(self):
        return self.client.trace

    @trace.setter
    def trace(self, value):
        self.client.trace = value


async def matrix_demo(db, destination):
    """Real orchestration with fixed retrieval/model boundary, no model or Web I/O."""
    from app.schemas.research import CompareRequest
    from app.workflows.comparison import run_comparison
    from app.workflows.local_web import LocalRun

    originals = {}
    ids = []
    for i in range(5):
        paper = Paper(
            title=f"Development study {i + 1}",
            type=PaperType.research_paper,
            content_scope="abstract",
        )
        db.add(paper)
        await db.flush()
        ids.append(paper.id)
        if i == 4:
            continue
        parsed = PDFParserService().parse_abstract(
            f"The sample included {200 + i} adults. "
            "Treatment was associated with lower risk."
        )
        build = await stage_generation(
            db, paper, parsed, embedding_model_id="local:fixture"
        )
        originals[paper.id] = await resolve_child(
            db, build.children[0], allow_parent_failure=True
        )

    async def local(policy, history, trace, **kwargs):
        source = originals.get(policy.paper_ids[0])
        return LocalRun(
            pack_context([source] if source else [], trace=trace),
            "hybrid",
            0,
            "disabled",
            "injected_public_fixture",
        )

    class Model:
        async def complete_json(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "evidence_blocks" in payload:
                block = json.loads(payload["evidence_blocks"].split("\n\n")[0])
                quote = block["text"].strip().split(". ")[0] + "."
                return {
                    "claims": [
                        {
                            "paper_id": block["paper_ids"][0],
                            "dimension": dim,
                            "text": quote,
                            "source_ids": [block["citations"][0]["id"]],
                            "supporting_quotes": {block["citations"][0]["id"]: quote},
                        }
                        for dim in payload["dimensions"]
                    ]
                }
            return {
                "verdicts": [
                    {
                        "claim_id": c["claim_id"],
                        "status": "supported",
                        "characteristics": {},
                    }
                    for c in payload["claims"]
                ]
            }

    async def generator(*args, **kwargs):
        return await generate_grounded(*args, **kwargs, client=Model())

    trace = RetrievalTrace(
        "Compare development studies",
        {},
        corpus_version="m6-public",
        corpus_public=True,
    )
    response = await run_comparison(
        db,
        CompareRequest(paper_ids=ids, local_only=True),
        trace,
        local_runner=local,
        generator=generator,
    )
    save(
        destination / "matrix_demo.json",
        {
            "boundary": "fixed_public_retrieval_and_model_not_live_quality",
            "result": response.model_dump(),
            "trace": trace.record,
        },
    )


async def run(destination, live=False):
    if live and (
        settings.llm_provider != "ollama" or not loopback_url(settings.ollama_base_url)
    ):
        raise ValueError("Only explicit loopback Ollama is allowed by this runner")
    fixture_bytes = FIXTURES.read_bytes()
    cases = json.loads(fixture_bytes)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    config, manifest = configuration_snapshot()
    result = {
        "schema": "m6_development_v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "config_version": config,
        "config_manifest": manifest,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "purpose": "public_synthetic_development_not_gold",
        "paid_cloud_calls": 0,
        "deterministic": [],
        "ablation": [],
        "generated": [],
    }
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            await matrix_demo(db, destination)
            for case in cases:
                source = web_source(case["quote"])
                registry = await build_registry(
                    db, [source], web_snapshots=seal_web_sources([source])
                )
                candidate = proposal(case, source)
                checked = check_proposal(candidate, registry)
                result["deterministic"].append(
                    {
                        "case": case["id"],
                        "label_supported": case["supported"],
                        "status": checked.support_status,
                        "reasons": checked.uncertainty,
                        "model_calls": 0,
                    }
                )
                if live:
                    for mode in ("whole_review", "atomic_batch"):
                        client = ReplayGeneration(candidate)
                        trace = RetrievalTrace(
                            "What does the original report?",
                            {},
                            corpus_version="m6-public",
                            corpus_public=True,
                        )
                        started = time.perf_counter()
                        try:
                            answer = await generate_grounded(
                                "What does the original report?",
                                registry,
                                pack_context(registry.source_dicts()),
                                trace,
                                client=client,
                                verification_mode=mode,
                            )
                        finally:
                            await client.close()
                        supported = any(
                            c.support_status == "supported" for c in answer.claims
                        )
                        result["ablation"].append(
                            {
                                "case": case["id"],
                                "mode": mode,
                                "label_supported": case["supported"],
                                "published": supported,
                                "false_rejection": case["supported"] and not supported,
                                "false_acceptance": not case["supported"] and supported,
                                "replayed_generation_calls": 1,
                                "real_semantic_calls": client.real_calls,
                                "interface_calls": client.calls,
                                "latency_ms": round(
                                    1000 * (time.perf_counter() - started), 3
                                ),
                                "verification": trace.record.get("verification"),
                                "observed_model_calls": trace.record["llm_calls"],
                                "token_usage": trace.record["token_usage"],
                                "failures": trace.record["failures"],
                            }
                        )
                        save(destination / "summary.json", result)
            paper = Paper(
                title="Synthetic local native PDF",
                type=PaperType.research_paper,
                content_scope="full_text",
            )
            db.add(paper)
            await db.flush()
            import fitz

            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "Memory accuracy is 95 percent.")
            pdf_bytes = pdf.tobytes()
            parsed = PDFParserService().parse(pdf_bytes)
            pdf.close()
            build = await stage_generation(
                db,
                paper,
                parsed,
                embedding_model_id="local:fixture",
                source_bytes=pdf_bytes,
            )
            local = await resolve_child(
                db, build.children[0], allow_parent_failure=True
            )
            generation_cases = [
                ("local_native_pdf", "What memory accuracy was reported?", [local]),
                (
                    "web_oa_fixture",
                    "What memory accuracy was reported?",
                    [web_source("Memory accuracy is 95 percent.", full_text=True)],
                ),
                (
                    "web_abstract_partial",
                    "What accuracy and sample size were reported?",
                    [web_source("Memory accuracy is 95 percent.")],
                ),
                ("no_answer", "What was survival?", []),
            ]
            disagreement = web_source("Memory accuracy is 90 percent.")
            disagreement.update(
                chunk_id="web:second",
                paper_id="second-study",
                doi="10.1234/second-development",
            )
            disagreement["source_spans"] = [
                {**span, "paper_id": "second-study"}
                for span in disagreement["source_spans"]
            ]
            generation_cases.append(
                (
                    "conflicting_originals",
                    "Report each study's memory accuracy separately.",
                    [web_source("Memory accuracy is 95 percent."), disagreement],
                )
            )
            for name, question, sources in generation_cases:
                if not live and sources:
                    continue
                snapshots = seal_web_sources(
                    [s for s in sources if s.get("source_origin") == "web"]
                )
                registry = await build_registry(db, sources, web_snapshots=snapshots)
                trace = RetrievalTrace(
                    question, {}, corpus_version="m6-public", corpus_public=True
                )
                budget = VerificationBudget()
                started = time.perf_counter()
                answer = await generate_grounded(
                    question,
                    registry,
                    pack_context(registry.source_dicts()),
                    trace,
                    budget=budget,
                )
                row = {
                    "case": name,
                    "question": question,
                    "actual_output": answer.model_dump(),
                    "latency_ms": round(1000 * (time.perf_counter() - started), 3),
                    "interface_calls": budget.calls,
                    "trace": trace.record,
                    "metrics": [],
                    "source_boundary": "in_memory_public_fixture_not_live_web",
                }
                if live and answer.support_status in {"supported", "partial"}:
                    row["metrics"] = await judge(
                        answer.text,
                        question,
                        [s.raw_text for s in registry.sources.values()],
                    )
                elif sources:
                    row["metrics"] = [
                        {
                            "metric": n,
                            "status": "not_evaluated",
                            "score": None,
                            "reason": "no_verified_generated_answer",
                        }
                        for n in ("faithfulness", "answer_relevancy")
                    ]
                result["generated"].append(row)
                save(destination / "summary.json", result)
                with (destination / "generation_deepeval_results.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"generation {name}: {answer.support_status}", flush=True)
    finally:
        await engine.dispose()
    save(destination / "summary.json", result)
    print(
        json.dumps(
            {
                "deterministic": len(result["deterministic"]),
                "ablation": len(result["ablation"]),
                "generated": len(result["generated"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="Explicit local diagnostic model override")
    args = parser.parse_args()
    if args.model:
        settings.ollama_model = args.model
    asyncio.run(run(args.output, args.live))
