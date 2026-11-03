"""Bounded local generation/Judge diagnostics; no human calibration inferred."""

import argparse
import asyncio
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "true"
os.environ["CONFIDENT_TRACE_FLUSH"] = "false"

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.context_packing import pack_context  # noqa: E402
from app.services.evidence_registry import (  # noqa: E402
    build_registry,
)
from app.services.grounded_generation import generate_grounded  # noqa: E402
from app.services.request_budget import RequestBudget, current_budget  # noqa: E402
from app.services.retrieval_trace import RetrievalTrace  # noqa: E402
from evaluation.deepeval_adapter import (  # noqa: E402
    METRICS,
    REFERENCE_METRICS,
    JudgeConfig,
    _judge_model,
)
from evaluation.final_contracts import (  # noqa: E402
    load_jsonl,
    verify_freeze,
    verify_runtime,
    write_jsonl,
)


def metric_names(expected):
    return METRICS + (REFERENCE_METRICS if expected else ())


def citation_truth(answer, registry, *, local_only):
    errors, fake_pages, forbidden_web = [], 0, 0
    if answer.support_status in {"supported", "partial"} and not answer.citations:
        errors.append("supported_answer_without_citation")
    for citation in answer.citations:
        try:
            original = registry.cite(citation.source_id, citation.raw_quote)
            for field in (
                "paper_id",
                "doi",
                "source_origin",
                "evidence_type",
                "source_url",
                "pdf_page",
                "version",
                "span",
            ):
                if getattr(original, field) != getattr(citation, field):
                    errors.append("original_citation_field_mismatch")
            if citation.evidence_type != "full_text" and citation.pdf_page is not None:
                fake_pages += 1
        except (ValueError, TypeError):
            errors.append("unresolvable_citation")
        forbidden_web += local_only and citation.source_origin == "web"
    return {
        "citation_integrity": not errors and not fake_pages and not forbidden_web,
        "citation_errors": errors,
        "fake_pdf_pages": fake_pages,
        "local_only_web_citations": int(forbidden_web),
        "supported_claim_count": sum(
            c.support_status == "supported" for c in answer.claims
        ),
        "human_claim_support": None,
    }


async def judge_answer(question, actual, context, expected, *, timeout=45):
    import ollama
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    classes = dict(
        zip(
            (*METRICS, *REFERENCE_METRICS),
            (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
            ),
            strict=True,
        )
    )
    results = []
    for name in metric_names(expected):
        row = {
            "metric": name,
            "status": "not_evaluated",
            "score": None,
            "provider": "ollama",
            "model": settings.ollama_model,
            "temperature": 0,
            "threshold": 0.5,
            "template": "deepeval_default",
            "deepeval_version": version("deepeval"),
            "judge_attempts": 0,
            "completed_responses": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "framework_cost": None,
            "provider_api_currency": 0,
            "total_resource_cost": None,
        }
        if not context and name != "answer_relevancy":
            row.update(status="not_applicable", reason="no_admitted_context")
            results.append(row)
            continue
        original = ollama.AsyncClient.chat

        async def counted(client, *args, _row=row, _original=original, **kwargs):
            if _row["judge_attempts"] >= 6:
                raise RuntimeError("Judge attempt ceiling")
            _row["judge_attempts"] += 1
            try:
                response = await _original(client, *args, **kwargs)
            finally:
                await client._client.aclose()
            _row["completed_responses"] += 1
            for source, target in (
                ("prompt_eval_count", "input_tokens"),
                ("eval_count", "output_tokens"),
            ):
                value = getattr(response, source, None)
                _row[target] = (
                    _row[target] + value
                    if _row[target] is not None and value is not None
                    else None
                )
            return response

        started = time.perf_counter()
        try:
            model = _judge_model(JudgeConfig("ollama", settings.ollama_model, ""))
            model.generation_kwargs.update(num_predict=512, num_ctx=4096)
            metric = classes[name](model=model, threshold=0.5, async_mode=True)
            with patch("ollama.AsyncClient.chat", counted):
                async with asyncio.timeout(timeout):
                    await metric.a_measure(
                        LLMTestCase(
                            input=question,
                            actual_output=actual,
                            context=context,
                            retrieval_context=context,
                            expected_output=expected,
                        )
                    )
            row.update(
                status="evaluated",
                score=metric.score,
                framework_cost=getattr(metric, "evaluation_cost", None),
            )
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            if row["judge_attempts"] > row["completed_responses"]:
                row["input_tokens"] = row["output_tokens"] = None
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        results.append(row)
    return results


async def evaluate_case(
    db, case, sources, *, web_snapshots=(), expected=None, run_judge=True
):
    registry = await build_registry(db, sources, web_snapshots=web_snapshots)
    trace = RetrievalTrace(case["question"], {}, corpus_version=case["provenance"])
    budget = RequestBudget()
    token = current_budget.set(budget)
    try:
        answer = await generate_grounded(
            case["question"], registry, pack_context(registry.source_dicts()), trace
        )
    finally:
        current_budget.reset(token)
    result = {
        "case_id": case["case_id"],
        "provenance": case["provenance"],
        "support_status": answer.support_status,
        "answer": answer.model_dump(),
        "pipeline_llm_calls": trace.record["llm_calls"],
        "pipeline_token_usage": trace.record["token_usage"],
        "stage_ms": trace.record["stage_ms"],
        "pipeline_budget": budget.snapshot(),
        "pipeline_failures": trace.record["failures"],
        "deterministic": citation_truth(
            answer, registry, local_only=not bool(web_snapshots)
        ),
        "metric_target": "verified_answer"
        if answer.support_status in {"supported", "partial"}
        else "safe_abstention_or_evidence_only",
        "human_review_status": "not_reviewed",
        "metrics": [],
    }
    if run_judge:
        result["metrics"] = await judge_answer(
            case["question"],
            answer.text,
            [s.raw_text for s in registry.sources.values()],
            expected,
        )
    return result


async def run(root, *, judge_limit=4):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    verify_runtime(manifest)
    output = root / "generation"
    output.mkdir(exist_ok=True)
    target = output / "results.jsonl"
    if target.exists():
        raise ValueError("Preserve existing generation results")
    contexts = {
        r["case_id"]: r
        for r in load_jsonl(root / "retrieval/contexts.jsonl")
        if r["variant"] == "multi_rerank_parent"
    }
    references = load_jsonl(root / "development_reference.jsonl")
    engine = create_async_engine(
        "sqlite+aiosqlite:///" + (root / "retrieval/benchmark.db").resolve().as_posix()
    )
    results = []
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            chosen = references[:6] + references[-2:]
            for i, ref in enumerate(chosen):
                case = {
                    "case_id": ref["case_id"],
                    "question": ref["question"],
                    "provenance": "qasper_train_reference_development_not_human_gold",
                }
                row = await evaluate_case(
                    db,
                    case,
                    contexts[ref["case_id"]]["sources"],
                    expected="; ".join(ref["expected_answer_points"]) or None,
                    run_judge=i < judge_limit,
                )
                row["answerable_reference"] = ref["answerable_local"]
                if not row["metrics"]:
                    row["judge_status"] = "not_selected_for_bounded_judge_subset"
                results.append(row)
                with target.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            "case": i + 1,
                            "support_status": row["support_status"],
                            "judge_evaluated": sum(
                                m["status"] == "evaluated" for m in row["metrics"]
                            ),
                        }
                    ),
                    flush=True,
                )
            # Whole-paper eligibility respects the same serialized context ceiling.
            catalog = load_jsonl(root / "retrieval/source_catalog.jsonl")
            long_rows = []
            for ref in references:
                all_sources = [s for s in catalog if s["paper_id"] == ref["paper_id"]]
                packed = pack_context(
                    all_sources, paper_ids=[ref["paper_id"]], expand=False
                )
                fits = {s["chunk_id"] for s in packed.sources} == {
                    s["chunk_id"] for s in all_sources
                }
                long_rows.append(
                    {
                        "case_id": ref["case_id"],
                        "full_source_children": len(all_sources),
                        "packed_children": len(packed.sources),
                        "eligible_all_source_children_fit": fits,
                        "status": "eligible_not_generated"
                        if fits
                        else "not_applicable_context_budget",
                        "generation_quality": None,
                        "human_claim_support": None,
                    }
                )
            write_jsonl(output / "long_context_eligibility.jsonl", long_rows)
            for row in long_rows:
                if row["eligible_all_source_children_fit"]:
                    ref = next(
                        ref for ref in references if ref["case_id"] == row["case_id"]
                    )
                    source_list = [
                        s for s in catalog if s["paper_id"] == ref["paper_id"]
                    ]
                    evaluated = await evaluate_case(
                        db,
                        {
                            "case_id": "long:" + ref["case_id"],
                            "question": ref["question"],
                            "provenance": "all_native_children_not_raw_full_paper",
                        },
                        source_list,
                        run_judge=False,
                    )
                    evaluated["baseline_limit"] = (
                        "Native parser may omit non-body text; not raw full-paper."
                    )
                    with (output / "long_context_results.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(evaluated) + "\n")
    finally:
        await engine.dispose()
    summary = {
        "cases": len(results),
        "provenance": "public_reference_development_not_human_gold",
        "pipeline_calls": sum(r["pipeline_llm_calls"] for r in results),
        "judge_attempts": sum(
            m["judge_attempts"] for r in results for m in r["metrics"]
        ),
        "evaluated_metrics": sum(
            m["status"] == "evaluated" for r in results for m in r["metrics"]
        ),
        "failed_metrics": sum(
            m["status"] == "not_evaluated" for r in results for m in r["metrics"]
        ),
        "fake_pdf_pages": sum(r["deterministic"]["fake_pdf_pages"] for r in results),
        "human_claim_support": None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--judge-limit", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args.root, judge_limit=args.judge_limit))
