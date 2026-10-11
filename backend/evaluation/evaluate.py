"""Offline retrieval benchmark using independently reviewed local JSONL labels."""

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import async_session_maker, engine
from app.models import PaperChunk
from app.workflows.source_query import answer_from_evidence, retrieve
from evaluation.labels import GROUPS, load_cases


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float | None:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else None


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float | None:
    if not relevant:
        return None
    gain = sum(
        1 / math.log2(position + 2)
        for position, chunk_id in enumerate(ranked[:k])
        if chunk_id in relevant
    )
    ideal = sum(
        1 / math.log2(position + 2) for position in range(min(k, len(relevant)))
    )
    return gain / ideal


def citations_valid(answer: str, evidence: list[dict]) -> bool:
    ids = set(re.findall(r"\[C:([^\]]+)\]", answer))
    return ids == {source["chunk_id"] for source in evidence}


async def run_case(case: dict, judge_model: str | None = None) -> dict:
    output = {"query": case["query"], "group": case.get("group", "other"), "modes": {}}
    relevant = set(case["relevant_chunk_ids"])
    for mode in ("dense", "bm25", "hybrid"):
        start = time.perf_counter()
        async with async_session_maker() as db:
            evidence, actual = await retrieve(db, case["query"], mode=mode)
        ranked = [source["chunk_id"] for source in evidence]
        output["modes"][mode] = {
            "actual_mode": actual,
            "recall_at_10": recall_at_k(ranked, relevant),
            "ndcg_at_10": ndcg_at_k(ranked, relevant),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "chunk_ids": ranked,
        }
        if mode == "hybrid":
            answer, cited = await answer_from_evidence(case["query"], evidence)
            output["answer"] = answer
            output["citation_ids_valid"] = citations_valid(answer, cited)
            if judge_model and cited:
                from deepeval.metrics import FaithfulnessMetric
                from deepeval.test_case import LLMTestCase

                metric = FaithfulnessMetric(model=judge_model)
                test_case = LLMTestCase(
                    input=case["query"],
                    actual_output=answer,
                    retrieval_context=[source["excerpt"] for source in cited],
                )
                metric.measure(test_case)
                output["faithfulness"] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "judge": judge_model,
                }
    return output


def _average(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def build_report(results: list[dict], human_test_labels_confirmed: bool) -> dict:
    """Build a report whose gate stays closed for incomplete comparisons."""
    groups = {case["group"] for case in results}
    mode_comparison_valid = all(
        case["modes"][mode]["actual_mode"] == mode
        for case in results
        for mode in ("dense", "bm25", "hybrid")
    )
    report = {
        "label_count": len(results),
        "minimum_40_met": len(results) >= 40,
        "human_test_labels_confirmed": human_test_labels_confirmed,
        "groups_present": sorted(groups),
        "missing_groups": sorted(GROUPS - groups),
        "no_answer_count": sum(case["group"] == "no_answer" for case in results),
        "mode_comparison_valid": mode_comparison_valid,
        "citations_valid": all(
            case.get("citation_ids_valid", False) for case in results
        ),
        "cases": results,
    }
    for mode in ("dense", "bm25", "hybrid"):
        report[mode] = {
            metric: _average([case["modes"][mode][metric] for case in results])
            for metric in ("recall_at_10", "ndcg_at_10", "latency_ms")
        }
    exact = [case for case in results if case["group"] == "exact_terms"]
    report["exact_terms_recall_improvement"] = (
        _average(
            [
                case["modes"]["hybrid"]["recall_at_10"]
                - case["modes"]["dense"]["recall_at_10"]
                for case in exact
            ]
        )
        if exact
        else None
    )
    blockers = []
    if len(results) < 40:
        blockers.append("Need at least 40 labelled queries")
    if not human_test_labels_confirmed:
        blockers.append("Human-reviewed independent test labels not confirmed")
    if report["missing_groups"]:
        blockers.append("Missing required query groups")
    if not mode_comparison_valid:
        blockers.append("One or more retrieval modes fell back")
    if not report["citations_valid"]:
        blockers.append("Missing or unresolvable citation IDs")
    if (report["exact_terms_recall_improvement"] or 0) < 0.05:
        blockers.append("Exact-term Recall@10 gain below 5 percentage points")
    dense_recall = report["dense"]["recall_at_10"]
    hybrid_recall = report["hybrid"]["recall_at_10"]
    if (
        dense_recall is None
        or hybrid_recall is None
        or hybrid_recall < dense_recall - 0.02
    ):
        blockers.append("Overall Recall@10 regressed by more than 2 points")
    report["gate_blockers"] = blockers
    report["quality_gate_met"] = not blockers
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/local/report.json")
    )
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--confirm-human-test-labels",
        action="store_true",
        help="Confirm labels were human reviewed and not used for tuning",
    )
    args = parser.parse_args()
    if args.judge and not args.judge_model:
        parser.error("--judge requires --judge-model to record the exact judge")
    if args.judge:
        if not settings.anthropic_api_key:
            parser.error("--judge requires ANTHROPIC_API_KEY for answer generation")
        if not (os.getenv("OPENAI_API_KEY") or settings.openai_api_key):
            parser.error("--judge requires OPENAI_API_KEY for the DeepEval judge")
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        try:
            deepeval_version = importlib.metadata.version("deepeval")
        except importlib.metadata.PackageNotFoundError:
            parser.error("Install deepeval before running --judge")
    try:
        async with async_session_maker() as db:
            valid_ids = set((await db.execute(select(PaperChunk.id))).scalars())
        try:
            cases = load_cases(args.labels, valid_ids)
        except ValueError as exc:
            parser.error(str(exc))
        results = [
            await run_case(case, args.judge_model if args.judge else None)
            for case in cases
        ]
    finally:
        await engine.dispose()
    report = build_report(results, args.confirm_human_test_labels)
    if args.judge:
        report["judge_model"] = args.judge_model
        report["deepeval_version"] = deepeval_version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}; quality gate met: {report['quality_gate_met']}")


if __name__ == "__main__":
    asyncio.run(main())
