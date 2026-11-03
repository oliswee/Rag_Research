"""Replay actual local query plans on the frozen public QASPER development set."""

import argparse
import ast
import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from app.config import settings
from app.services.local_execution import local_only_request, loopback_host, loopback_url
from app.services.retrieval_trace import RetrievalTrace
from app.services.source_index import rrf_merge
from app.services.trace_versions import configuration_snapshot, digest
from app.workflows.query_expansion import (
    QueryPlan,
    RetrievalQuery,
    expand_queries,
    safe_variant,
)
from evaluation.evaluate import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.isolated_corpus import METRICS, evaluate_bm25, load_jsonl, mean

CORPUS = Path("data/evaluation/qasper_benchmark")
HASHES = {
    "papers.json": "6780f81a5fc506b0862791980d810e9004307b4207107499638900f55645f462",
    "chunks.jsonl": "bae57496445d2b5dce8bc3aa7f5a761a6ed5a43fd57fab43a1116e26faeb7ace",
    "labels.jsonl": "496751dc21dafddba93db95280848147eba17b58ecb584299820c9ef5dd58080",
}


def plan_version() -> str:
    import app.workflows.query_expansion as implementation

    return digest(
        [
            ast.dump(
                ast.parse(Path(implementation.__file__).read_text(encoding="utf-8"))
            ),
            settings.ollama_model,
            settings.query_variant_limit,
            settings.query_rewrite_timeout_seconds,
        ]
    )


def restore_plan(row: dict, question: str, expected_version: str) -> QueryPlan:
    if row.get("question") != question or row.get("plan_version") != expected_version:
        raise ValueError("Cached plan input/config mismatch")
    queries = [RetrievalQuery(**item) for item in row["queries"]]
    if (
        not queries
        or len(queries) > 3
        or queries[0].query_id != "Q0"
        or queries[0].text != question
    ):
        raise ValueError("Cached plan lacks original query")
    if any(q.parent_question != question for q in queries) or len(
        {q.query_id for q in queries}
    ) != len(queries):
        raise ValueError("Cached plan provenance mismatch")
    if any(not safe_variant(question, q.text, []) for q in queries[1:]):
        raise ValueError("Unsafe cached variant")
    return QueryPlan(
        queries, row["status"], row["reason"], row.get("rejected_count", 0)
    )


def paired_comparison(single: dict, multi: dict) -> dict:
    pairs = list(zip(single["results"], multi["results"], strict=True))
    if any(a["query"] != b["query"] for a, b in pairs):
        raise ValueError("Paired question mismatch")
    eligible = [
        (a, b)
        for a, b in pairs
        if a["recall_at_10"] is not None and b["recall_at_10"] is not None
    ]
    deltas = [b["recall_at_10"] - a["recall_at_10"] for a, b in eligible]
    return {
        "answerable_count": len(eligible),
        "mean_recall_delta": sum(deltas) / len(deltas) if deltas else None,
        "mean_mrr_delta": sum(b["mrr_at_10"] - a["mrr_at_10"] for a, b in eligible)
        / len(eligible)
        if eligible
        else None,
        "recall_win_tie_loss": {
            "win": sum(d > 0 for d in deltas),
            "tie": sum(d == 0 for d in deltas),
            "loss": sum(d < 0 for d in deltas),
        },
    }


async def prepare_plans(
    labels: list[dict], destination: Path
) -> tuple[dict, list[dict]]:
    cache = destination / "query_plans.jsonl"
    records = load_jsonl(cache) if cache.exists() else []
    fingerprint = plan_version()
    by_question = {}
    allowed = {case["query"] for case in labels}
    for row in records:
        question = row["question"]
        if question not in allowed or question in by_question:
            raise ValueError("Cached plan dataset mismatch")
        by_question[question] = restore_plan(row, question, fingerprint)
    for index, label in enumerate(labels):
        question = label["query"]
        if question in by_question:
            continue
        trace = RetrievalTrace(
            question,
            {"paper_ids": label["paper_ids"], "local_only": True},
            corpus_version="qasper-v0.3-dev-40",
            corpus_public=True,
        )
        start = time.perf_counter()
        plan = await expand_queries(question, [], trace=trace)
        trace.finish("query_planning_only", [])
        row = {
            "question": question,
            "plan_version": fingerprint,
            **asdict(plan),
            "planning_ms": round((time.perf_counter() - start) * 1000, 3),
            "trace": trace.as_dict(),
        }
        with cache.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_question[question] = plan
        records.append(row)
        print(
            f"Planned {index + 1}/{len(labels)}: "
            f"{len(plan.queries)} queries, {plan.reason}",
            flush=True,
        )
    return by_question, records


def replay_fusion(
    labels: list[dict], traces: list[dict], *, top_k: int, rrf_k: int
) -> dict:
    """Sweep fusion using saved validated lists; no retrieval latency is claimed."""
    if len(labels) != len(traces) or not 1 <= top_k <= 30 or not 1 <= rrf_k <= 60:
        raise ValueError("Invalid replay inputs")
    results = []
    for label, trace in zip(labels, traces, strict=True):
        if label["query"] != trace["original_query"]:
            raise ValueError("Replay question mismatch")
        rankings = [
            [
                item["child_id"]
                for item in trace["rankings"][
                    f"{branch['query_id']}.{branch['retriever']}"
                ][:top_k]
            ]
            for branch in trace["retrieval_branches"]
        ]
        ranked = rrf_merge(rankings, k=rrf_k, limit=10)
        relevant = set(label["relevant_chunk_ids"])
        results.append(
            {
                "query": label["query"],
                "paper_ids": label.get("paper_ids"),
                "ranked_chunk_ids": ranked,
                "recall_at_10": recall_at_k(ranked, relevant),
                "hit_at_10": hit_at_k(ranked, relevant),
                "mrr_at_10": mrr_at_k(ranked, relevant),
                "ndcg_at_10": ndcg_at_k(ranked, relevant),
                "latency_ms": None,
            }
        )
    return {
        "evaluation_kind": "saved_rank_list_replay",
        "per_query_k": top_k,
        "rrf_k": rrf_k,
        "results": results,
        "metrics": {metric: mean(results, metric) for metric in METRICS},
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "vector_dataset_id": None,
    }


async def judge_contexts(report: dict, chunks: list[dict]) -> list[dict]:
    """Two bounded local Contextual Relevancy smoke cases, not answer evaluation."""
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    os.environ["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "0"
    from deepeval.metrics import ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase

    from evaluation.deepeval_adapter import JudgeConfig, _judge_model

    by_id = {row["id"]: row["text"] for row in chunks}
    outcomes = []
    for case in report["results"][:2]:
        outcome = {
            "query": case["query"],
            "metric": "contextual_relevancy",
            "provider": "ollama",
            "model": settings.ollama_model,
            "deepeval_version": version("deepeval"),
            "threshold": 0.5,
            "temperature": 0,
            "prompt_template": "deepeval_default",
            "context_limit": 1,
            "status": "not_evaluated",
            "score": None,
            "cost": None,
            "api_call_count": None,
            "token_usage": None,
            "error_type": None,
        }
        start = time.perf_counter()
        try:
            async with asyncio.timeout(60):
                model = _judge_model(JudgeConfig("ollama", settings.ollama_model, ""))
                metric = ContextualRelevancyMetric(
                    model=model, threshold=0.5, async_mode=True
                )
                await metric.a_measure(
                    LLMTestCase(
                        input=case["query"],
                        actual_output="",
                        retrieval_context=[
                            by_id[c] for c in case["ranked_chunk_ids"][:1]
                        ],
                    )
                )
                outcome.update(
                    status="evaluated",
                    score=metric.score,
                    cost=getattr(metric, "evaluation_cost", None),
                )
        except Exception as exc:
            outcome["error_type"] = type(exc).__name__
        outcome["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        outcomes.append(outcome)
    return outcomes


async def run(
    destination: Path, *, plans_only: bool = False, judge: bool = False
) -> None:
    if (
        settings.llm_provider != "ollama"
        or settings.embedding_provider != "fastembed"
        or not loopback_url(settings.ollama_base_url)
        or not loopback_host(settings.qdrant_host)
    ):
        raise ValueError("Benchmark requires local Ollama/FastEmbed/loopback Qdrant")
    local_only_request.set(True)
    for filename, expected in HASHES.items():
        if hashlib.sha256((CORPUS / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen public corpus hash mismatch")
    papers = json.loads((CORPUS / "papers.json").read_text(encoding="utf-8"))
    chunks = load_jsonl(CORPUS / "chunks.jsonl")
    labels = load_jsonl(CORPUS / "labels.jsonl")
    destination.mkdir(parents=True, exist_ok=True)
    plans, records = await prepare_plans(labels, destination)
    if plans_only:
        return
    reports = {}
    variants = [
        ("bm25", "bm25", None, 30, 60),
        ("dense", "dense", None, 30, 60),
        ("single_hybrid", "hybrid", None, 30, 60),
        ("multi_hybrid", "hybrid", plans, 30, 60),
    ]
    for name, mode, query_plans, top_k, rrf_k in variants:
        report = await evaluate_bm25(
            papers,
            chunks,
            labels,
            mode=mode,
            corpus_version="qasper-v0.3-dev-40",
            corpus_public=True,
            trace_output=destination / f"trace_{name}.jsonl",
            query_plans=query_plans,
            per_query_k=top_k,
            rrf_k=rrf_k,
        )
        (destination / f"{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        reports[name] = report
        print(f"Completed {name}: {report['metrics']}", flush=True)
    for name, top_k, rrf_k in [("multi_k10", 10, 60), ("multi_rrf20", 30, 20)]:
        reports[name] = replay_fusion(
            labels,
            load_jsonl(destination / "trace_multi_hybrid.jsonl"),
            top_k=top_k,
            rrf_k=rrf_k,
        )
        (destination / f"{name}.json").write_text(
            json.dumps(reports[name], indent=2), encoding="utf-8"
        )
    await write_summary(destination, reports, plans, records, chunks, judge=judge)


async def write_summary(destination, reports, plans, records, chunks, *, judge):
    config_version, config = configuration_snapshot()
    code_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )
    summary = {
        "code_head": code_head,
        "working_tree_dirty": dirty,
        "config_version": config_version,
        "config": config,
        "dataset_hashes": HASHES,
        "label_count": len(records),
        "answerable_count": 32,
        "no_answer_count": 8,
        "query_count_distribution": {
            str(count): sum(len(p.queries) == count for p in plans.values())
            for count in (1, 2, 3)
        },
        "rejected_variant_count": sum(p.rejected_count for p in plans.values()),
        "planning_ms_total": sum(r["planning_ms"] for r in records),
        "planning_llm_successful_calls": sum(r["trace"]["llm_calls"] for r in records),
        "planning_token_usage": {
            key: sum(r["trace"]["token_usage"][key] or 0 for r in records)
            for key in ("input_tokens", "output_tokens")
        },
        "comparison": paired_comparison(
            reports["single_hybrid"], reports["multi_hybrid"]
        ),
        "modes": {
            name: {
                key: report[key]
                for key in (
                    "metrics",
                    "latency_p50_ms",
                    "latency_p95_ms",
                    "vector_dataset_id",
                )
            }
            for name, report in reports.items()
        },
        "judge": await judge_contexts(reports["multi_hybrid"], chunks)
        if judge
        else {"status": "not_evaluated", "reason": "not_requested"},
        "limitations": [
            "Frozen QASPER development40, used for tuning; "
            "not human Gold or significance evidence.",
            "QASPER legacy children retained for label comparability; "
            "actual-library structured generation tested separately.",
            "Retrieval timing excludes cached query planning; "
            "planning timing/calls reported separately. Sequential parameter "
            "runs are not controlled performance comparisons.",
            "Model-weight revisions unknown. No answer generation in this run.",
            "Two Contextual Relevancy smoke cases, one context each, "
            "do not establish overall quality; Judge API accounting unmeasured.",
        ],
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/evaluation/m3_20261024")
    )
    parser.add_argument("--plans-only", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.output_dir, plans_only=args.plans_only, judge=args.judge))


if __name__ == "__main__":
    main()
