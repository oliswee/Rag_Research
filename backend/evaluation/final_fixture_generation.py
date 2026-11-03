"""Six controlled source scenarios and fixed-evidence ordering, not live Web gains."""

import argparse
import asyncio
import json
from pathlib import Path

import fitz
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, Paper, PaperType
from app.services.context_packing import pack_context
from app.services.evidence_registry import seal_web_sources
from app.services.parent_child import stage_generation
from app.services.pdf_parser import PDFParserService
from app.services.source_anchors import resolve_child
from evaluation.final_contracts import (
    verify_freeze,
    verify_models,
    verify_runtime,
    write_jsonl,
)
from evaluation.final_generation import evaluate_case
from evaluation.generation_benchmark import web_source


async def local(db, text, *, full_text=False):
    paper = Paper(
        title="Synthetic controlled memory study",
        type=PaperType.research_paper,
        content_scope="full_text" if full_text else "abstract",
    )
    db.add(paper)
    await db.flush()
    parser = PDFParserService()
    data = None
    if full_text:
        document = fitz.open()
        document.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750), text)
        data = document.tobytes()
        parsed = parser.parse(data)
        document.close()
    else:
        parsed = parser.parse_abstract(text)
    build = await stage_generation(db, paper, parsed, source_bytes=data)
    sources = [
        {
            **await resolve_child(db, child, allow_parent_failure=True),
            "title": paper.title,
        }
        for child in build.children
    ]
    # Match the JSON representation at the real HTTP/SQLite evidence boundary.
    # PDF bbox tuples become lists after persistence; do not score that fixture
    # serialization difference as an unavailable source or a model refusal.
    return json.loads(json.dumps(sources))


async def run(root, *, output_name="controlled_generation"):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    verify_runtime(manifest)
    verify_models(root, manifest)
    if output_name not in {"controlled_generation", "controlled_generation_fixed"}:
        raise ValueError("Unknown controlled output name")
    output = root / output_name
    output.mkdir(exist_ok=True)
    if (output / "results.jsonl").exists():
        raise ValueError("Preserve previous controlled diagnostics")
    database = output / "original_sources.db"
    if database.exists():
        raise ValueError("Preserve existing original source database")
    engine = create_async_engine("sqlite+aiosqlite:///" + database.resolve().as_posix())
    results = []
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            native = await local(db, "Memory accuracy is 95 percent.", full_text=True)
            abstract = await local(
                db, "The abstract reports memory accuracy of 95 percent."
            )
            conflict = abstract + await local(
                db, "The abstract reports memory accuracy of 85 percent."
            )
            oa = web_source("Memory accuracy is 95 percent.", full_text=True)
            oa["doi"] = None
            await db.commit()
            write_jsonl(output / "web_originals.jsonl", [oa])
            scenarios = [
                (
                    "local_pdf",
                    "What memory accuracy is reported?",
                    native,
                    (),
                    "95 percent",
                ),
                (
                    "web_oa",
                    "What memory accuracy is reported?",
                    [oa],
                    seal_web_sources([oa]),
                    "95 percent",
                ),
                (
                    "abstract_only",
                    "What memory accuracy and sample size are reported?",
                    abstract,
                    (),
                    "95 percent; sample size not provided",
                ),
                ("metadata_only", "What memory accuracy is reported?", [], (), None),
                ("web_fault", "What memory accuracy is reported?", [], (), None),
                (
                    "comparison_conflict",
                    "Report both memory accuracies. Can the studies be pooled?",
                    conflict,
                    (),
                    "95 and 85 percent; comparability remains unknown",
                ),
            ]
            for name, question, sources, sealed, expected in scenarios:
                row = await evaluate_case(
                    db,
                    {
                        "case_id": name,
                        "question": question,
                        "provenance": "synthetic_source_scenario_not_human_review",
                    },
                    sources,
                    web_snapshots=sealed,
                    expected=expected,
                )
                row["acquisition"] = (
                    "injected_licensed_fixture"
                    if name == "web_oa"
                    else "injected_failure_no_http"
                    if name == "web_fault"
                    else "controlled_original_no_external_fetch"
                )
                row["external_acquisition_attempts"] = 0
                row["expected_missing_information"] = (
                    "sample size remains unknown" if name == "abstract_only" else None
                )
                results.append(row)
                with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            "scenario": name,
                            "support_status": row["support_status"],
                            "judge_evaluated": sum(
                                m["status"] == "evaluated" for m in row["metrics"]
                            ),
                        }
                    ),
                    flush=True,
                )
            order_rows = []
            original_ids = {s["chunk_id"] for s in conflict}
            for name, evidence in (
                ("source_order", conflict),
                ("reversed_order", list(reversed(conflict))),
            ):
                packed = pack_context(evidence)
                if {s["chunk_id"] for s in packed.sources} != original_ids:
                    order_rows.append(
                        {"order": name, "status": "not_comparable_changed_evidence"}
                    )
                    continue
                result = await evaluate_case(
                    db,
                    {
                        "case_id": "order:" + name,
                        "question": "Report each accuracy separately; do not pool.",
                        "provenance": "fixed_synthetic_evidence_order_diagnostic",
                    },
                    evidence,
                    run_judge=False,
                )
                order_rows.append(
                    {
                        "order": name,
                        "same_evidence_set": True,
                        "input_ids": [s["chunk_id"] for s in evidence],
                        "result": result,
                        "human_completeness": None,
                    }
                )
            write_jsonl(output / "evidence_order.jsonl", order_rows)
    finally:
        await engine.dispose()
    summary = {
        "scenarios": len(results),
        "project_human_reviewed": 0,
        "pipeline_calls": sum(r["pipeline_llm_calls"] for r in results),
        "judge_attempts": sum(
            m["judge_attempts"] for r in results for m in r["metrics"]
        ),
        "evaluated_metrics": sum(
            m["status"] == "evaluated" for r in results for m in r["metrics"]
        ),
        "citation_integrity_passes": sum(
            r["deterministic"]["citation_integrity"] for r in results
        ),
        "fake_pdf_pages": sum(r["deterministic"]["fake_pdf_pages"] for r in results),
        "external_acquisition_attempts": 0,
        "notes": "Controlled Web fixtures; no live-Web answer gain inferred.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-name", default="controlled_generation")
    args = parser.parse_args()
    asyncio.run(run(args.root, output_name=args.output_name))
