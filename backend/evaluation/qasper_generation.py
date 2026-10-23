"""Generate real QASPER answers and local DeepEval inputs from a retrieval run.

Run from backend/ after evaluation.isolated_corpus with --mode hybrid.
The input and output files are public-corpus artifacts under ignored data/evaluation/.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.config import settings
from app.services.local_execution import local_only_request, loopback_url
from app.services.retrieval_trace import RetrievalTrace
from app.workflows.source_query import answer_from_evidence
from evaluation.isolated_corpus import load_jsonl


def validate_generation_inputs(
    report: dict,
    traces: list[dict],
    labels: list[dict],
    chunks: dict,
    titles: dict,
    source: dict,
    limits: dict[str, int],
    evidence_limit: int,
) -> dict:
    """Reject mixed runs and scopes before generating any model answer."""
    if evidence_limit <= 0 or any(value < 0 for value in limits.values()):
        raise ValueError("Case counts must be nonnegative and evidence limit positive")
    if len(traces) != len(labels) or len(report["results"]) != len(labels):
        raise ValueError("Report, traces and labels must have the same case count")
    questions = {}
    counts = dict.fromkeys(limits, 0)
    for label, result, trace in zip(labels, report["results"], traces, strict=True):
        query = label["query"]
        if (
            result["query"] != query
            or trace.get("original_query_sha256")
            != hashlib.sha256(query.encode()).hexdigest()
        ):
            raise ValueError("Question mismatch between label, report and trace")
        scope = label["paper_ids"]
        if result.get("paper_ids") != scope or trace.get("scope", {}).get(
            "paper_id_hashes"
        ) != [hashlib.sha256(pid.encode()).hexdigest() for pid in scope]:
            raise ValueError("Paper scope mismatch")
        ranked = result["ranked_chunk_ids"]
        resolved = [
            item["child_id"]
            for item in trace.get("rankings", {}).get("sqlite_resolved", [])
        ]
        if ranked != resolved or len(ranked) != len(set(ranked)):
            raise ValueError("Ranked evidence mismatch between report and trace")
        if result["actual_mode"] != "hybrid" or trace.get("retrieval_mode") != "hybrid":
            raise ValueError("Generation requires a real hybrid retrieval report")
        for cid in ranked:
            if cid not in chunks or chunks[cid]["paper_id"] not in scope:
                raise ValueError("Ranked chunk missing or outside paper scope")
            if chunks[cid]["paper_id"] not in titles:
                raise ValueError("Missing paper title")
        source_id = label["source_paper_id"]
        qa = next(
            (
                qa
                for qa in source.get(source_id, {}).get("qas", [])
                if qa["question_id"] == label["id"]
            ),
            None,
        )
        if (
            qa is None
            or qa["question"] != query
            or scope != [f"qasper:{source_id}"]
            or label["id"] in questions
        ):
            raise ValueError("Published question does not match label")
        questions[label["id"]] = qa
        if label["group"] not in counts:
            raise ValueError("Unknown question group")
        counts[label["group"]] += 1
    if any(counts[group] < limit for group, limit in limits.items()):
        raise ValueError("Requested more cases than available")
    return questions


def reference_answer(qa: dict) -> str | None:
    """Choose one published QASPER annotation without manufacturing an answer."""
    for item in qa["answers"]:
        answer = item["answer"]
        if answer["unanswerable"]:
            continue
        if answer["free_form_answer"].strip():
            return answer["free_form_answer"].strip()
        if answer["yes_no"] is not None:
            return "Yes" if answer["yes_no"] else "No"
        spans = [span for span in answer["extractive_spans"] if span.strip()]
        if spans:
            return "; ".join(spans)
    return None


async def generate_answer(label: dict, evidence: list[dict], retrieval: dict) -> tuple:
    """Give generation its own ID/accounting while linking its retrieval run."""
    if settings.llm_provider != "ollama" or not loopback_url(settings.ollama_base_url):
        raise ValueError("QASPER generation requires a loopback Ollama endpoint")
    trace = RetrievalTrace(
        label["query"],
        {
            "paper_ids": label["paper_ids"],
            "requested_mode": "hybrid",
            "local_only": True,
        },
        corpus_version=retrieval["corpus_version"],
        corpus_public=True,
        index_version=retrieval["index_version"],
    )
    trace.record.update(
        retrieval_run_id=retrieval["run_id"],
        retrieval_config_version=retrieval.get("config_version"),
        index_locator=retrieval.get("index_locator", retrieval["index_version"]),
        attempted_index_locators=list(retrieval.get("attempted_index_locators", [])),
        physical_index_revision=retrieval.get("physical_index_revision"),
        retrieval_ms=retrieval["total_ms"],
        timing_scope="generation",
        rankings=retrieval["rankings"],
    )
    trace.set_mode("hybrid")
    token = local_only_request.set(True)
    try:
        answer, used = await answer_from_evidence(label["query"], evidence, trace)
    finally:
        local_only_request.reset(token)
    if evidence and not trace.record["llm_calls"]:
        raise ValueError("Generation produced no successful model response")
    trace.finish(
        "supported" if used else "abstain",
        [item["chunk_id"] for item in evidence],
        [item["chunk_id"] for item in used],
    )
    return answer, trace.as_dict()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answerable", type=int, default=3)
    parser.add_argument("--no-answer", type=int, default=1)
    parser.add_argument("--evidence-limit", type=int, default=5)
    parser.add_argument("--omit-reference", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    traces = load_jsonl(args.traces)
    labels = load_jsonl(args.labels)
    chunks = {row["id"]: row for row in load_jsonl(args.chunks)}
    titles = {
        row["paper_id"]: row["title"]
        for row in json.loads(args.papers.read_text(encoding="utf-8"))
    }
    source = json.loads(args.source.read_text(encoding="utf-8"))
    limits = {"paper_answerable": args.answerable, "paper_no_answer": args.no_answer}
    questions = validate_generation_inputs(
        report, traces, labels, chunks, titles, source, limits, args.evidence_limit
    )
    selected = {name: 0 for name in limits}
    output = []
    for label, result, trace in zip(labels, report["results"], traces, strict=True):
        group = label["group"]
        if selected[group] >= limits[group]:
            continue
        if result["actual_mode"] != "hybrid":
            raise ValueError("Generation requires a real hybrid retrieval report")
        evidence = []
        for chunk_id in result["ranked_chunk_ids"][: args.evidence_limit]:
            chunk = chunks[chunk_id]
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": chunk["paper_id"],
                    "title": titles[chunk["paper_id"]],
                    "content_scope": chunk.get("content_scope", "full_text"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source_url": chunk.get("source_url"),
                    "excerpt": chunk["text"],
                }
            )
        answer, trace = await generate_answer(label, evidence, trace)
        output.append(
            {
                "case_id": label["id"],
                "group": group,
                "trace": trace,
                "actual_output": answer,
                "evidence_by_id": {
                    item["chunk_id"]: item["excerpt"] for item in evidence
                },
                "expected_output": (
                    reference_answer(questions[label["id"]])
                    if group == "paper_answerable" and not args.omit_reference
                    else None
                ),
                "reference_omitted": args.omit_reference,
            }
        )
        selected[group] += 1
        print(f"Generated {len(output)} answers", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} real QASPER answers to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
