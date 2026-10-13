"""Offline retrieval benchmark; supply independent human-labelled JSONL.

Usage: uv run python -m evaluation.evaluate labels.jsonl --output report.json
Each line: {"query": "...", "relevant_chunk_ids": ["..."], "group": "exact_terms"}
The optional --judge flag needs deepeval and a configured judge API key.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from app.db import async_session_maker
from app.workflows.source_query import answer_from_evidence, retrieve


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 1.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    gain = sum(
        1 / math.log2(position + 2)
        for position, chunk_id in enumerate(ranked[:k])
        if chunk_id in relevant
    )
    ideal = sum(
        1 / math.log2(position + 2) for position in range(min(k, len(relevant)))
    )
    return gain / ideal if ideal else 1.0


def citations_valid(
    answer: str, evidence: list[dict], *, answerable: bool = True
) -> bool:
    import re

    ids = set(re.findall(r"\[C:([^\]]+)\]", answer))
    if not ids:
        abstained = answer in {
            "Insufficient evidence in the local library. "
            "Discover or import relevant papers first.",
            "Insufficient verified evidence in the retrieved excerpts.",
            "Unable to verify answer support from the retrieved excerpts.",
        }
        return not answerable and abstained
    return ids <= {source["chunk_id"] for source in evidence}


async def run_case(case: dict, judge: bool = False) -> dict:
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
            output["citation_ids_valid"] = citations_valid(
                answer, cited, answerable=bool(relevant)
            )
            if judge and cited:
                from deepeval.metrics import FaithfulnessMetric
                from deepeval.test_case import LLMTestCase

                metric = FaithfulnessMetric()
                test_case = LLMTestCase(
                    input=case["query"],
                    actual_output=answer,
                    retrieval_context=[source["excerpt"] for source in cited],
                )
                metric.measure(test_case)
                output["faithfulness"] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "judge": type(metric.model).__name__,
                }
    return output


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation-report.json"))
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise SystemExit("No labelled queries provided")
    results = [await run_case(case, args.judge) for case in cases]
    report = {
        "label_count": len(cases),
        "minimum_40_met": len(cases) >= 40,
        "cases": results,
        "citations_valid": all(
            case.get("citation_ids_valid", False) for case in results
        ),
    }
    for mode in ("dense", "bm25", "hybrid"):
        report[mode] = {
            metric: sum(case["modes"][mode][metric] for case in results) / len(results)
            for metric in ("recall_at_10", "ndcg_at_10", "latency_ms")
        }
    exact = [case for case in results if case["group"] == "exact_terms"]
    if exact:
        report["exact_terms_recall_improvement"] = sum(
            case["modes"]["hybrid"]["recall_at_10"]
            - case["modes"]["dense"]["recall_at_10"]
            for case in exact
        ) / len(exact)
    report["quality_gate_met"] = bool(
        len(cases) >= 40
        and report["citations_valid"]
        and report.get("exact_terms_recall_improvement", -1) >= 0.05
        and report["hybrid"]["recall_at_10"] >= report["dense"]["recall_at_10"] - 0.02
        and all(case["modes"]["dense"]["actual_mode"] == "dense" for case in results)
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}; quality gate met: {report['quality_gate_met']}")


if __name__ == "__main__":
    asyncio.run(main())
