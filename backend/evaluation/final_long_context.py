"""Actual matched RAG/raw-full-paper pairs under the same context/model limits."""

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.services.llm_client import LLMClient
from evaluation.final_contracts import (
    load_jsonl,
    sha256,
    verify_freeze,
    verify_models,
    verify_runtime,
    write_jsonl,
)
from evaluation.final_generation import evaluate_case


def direct_context(paper, sources, budget):
    version = "sha256:" + sha256(paper["text"].encode())
    if any(
        s.get("paper_id") != paper["paper_id"] or s.get("document_version") != version
        for s in sources
    ):
        raise ValueError("Whole-paper source identity changed")
    context = json.dumps(
        {
            "paper_id": paper["paper_id"],
            "document_version": version,
            "full_paper": paper["text"],
            "citable_ranges": {
                s["chunk_id"]: [
                    [span["raw_start"], span["raw_end"]] for span in s["source_spans"]
                ]
                for s in sources
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(context.encode()) > budget:
        raise ValueError("Whole original does not fit; never silently truncate")
    return context


class DirectPaperClient:
    def __init__(self, context):
        self.context = context
        self.real = LLMClient(max_retries=1, timeout_seconds=30)

    @property
    def trace(self):
        return self.real.trace

    @trace.setter
    def trace(self, value):
        self.real.trace = value

    async def complete_json(self, **kwargs):
        payload = json.loads(kwargs["user_prompt"])
        if "evidence_blocks" in payload:
            payload["evidence_blocks"] = self.context
            kwargs["user_prompt"] = json.dumps(payload, ensure_ascii=False)
        return await self.real.complete_json(**kwargs)

    async def close(self):
        await self.real.client.close()


async def run(root):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    verify_runtime(manifest)
    verified = verify_models(root, manifest)
    output = root / "raw_long_context_pairs.jsonl"
    if output.exists():
        raise ValueError("Preserve original matched-pair results")
    papers = {r["paper_id"]: r for r in load_jsonl(root / "papers.jsonl")}
    refs = {r["case_id"]: r for r in load_jsonl(root / "development_reference.jsonl")}
    catalog = load_jsonl(root / "retrieval/source_catalog.jsonl")
    packed = {
        r["case_id"]: r
        for r in load_jsonl(root / "retrieval/contexts.jsonl")
        if r["variant"] == "multi_rerank_parent"
    }
    eligible = load_jsonl(root / "generation/long_context_eligibility.jsonl")
    engine = create_async_engine(
        "sqlite+aiosqlite:///" + (root / "retrieval/benchmark.db").resolve().as_posix()
    )
    rows = []
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            for eligibility in eligible:
                if not eligibility["eligible_all_source_children_fit"]:
                    continue
                ref = refs[eligibility["case_id"]]
                original_sources = [
                    s for s in catalog if s["paper_id"] == ref["paper_id"]
                ]
                try:
                    context = direct_context(
                        papers[ref["paper_id"]],
                        original_sources,
                        settings.context_token_budget,
                    )
                except ValueError:
                    rows.append(
                        {
                            "case_id": ref["case_id"],
                            "status": "not_applicable_raw_full_paper_exceeds_budget",
                        }
                    )
                    continue
                for mode in ("rag", "raw_full_paper"):
                    client = (
                        DirectPaperClient(context) if mode == "raw_full_paper" else None
                    )
                    try:
                        result = await evaluate_case(
                            db,
                            {
                                "case_id": ref["case_id"],
                                "question": ref["question"],
                                "provenance": (
                                    "matched_actual_full_paper_vs_RAG_"
                                    "reference_development"
                                ),
                            },
                            original_sources
                            if client
                            else packed[ref["case_id"]]["sources"],
                            client=client,
                            run_judge=False,
                        )
                    finally:
                        if client:
                            await client.close()
                    rows.append(
                        {
                            "case_id": ref["case_id"],
                            "mode": mode,
                            "status": "measured",
                            "context_budget": settings.context_token_budget,
                            "actual_models_verified_at_stage_start": verified,
                            "result": result,
                            "human_answer_completeness": None,
                            "deepeval_comparison": "not_selected_for_bounded_subset",
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "case_id": ref["case_id"],
                                "mode": mode,
                                "support_status": result["support_status"],
                            }
                        ),
                        flush=True,
                    )
                write_jsonl(output, rows)
    finally:
        await engine.dispose()
    write_jsonl(output, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(run(parser.parse_args().root))
