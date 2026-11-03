"""Guarded Windows-safe public-Web smoke; no user-library sources or imports."""

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base
from app.routers.query import submit_query
from app.schemas.query import QueryRequest
from app.services.source_index import SourceVectorIndex
from app.workflows.source_query import retrieve
from evaluation.final_contracts import (
    load_jsonl,
    verify_freeze,
    verify_models,
    verify_runtime,
)


async def run(root):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    verify_runtime(manifest)
    verify_models(root, manifest)
    output = root / "actual_web_flow_verified.json"
    trace_path = root / "actual_web_flow_verified_trace.jsonl"
    if output.exists() or trace_path.exists():
        raise ValueError("Preserve previous actual Web observations")
    settings.retrieval_trace_path = str(trace_path)
    index = SourceVectorIndex(dataset_id_override="m8-empty-public-web-probe")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    rows = []

    async def isolated_retrieval(*args, **kwargs):
        return await retrieve(*args, **kwargs, vector_index=index)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            with patch("app.routers.query.retrieve", isolated_retrieval):
                for only in (True, False):
                    response = await submit_query(
                        QueryRequest(
                            query="What does retrieval-augmented generation combine?",
                            local_only=only,
                            rewrite_query=False,
                        ),
                        db,
                    )
                    matching = [
                        r
                        for r in load_jsonl(trace_path)
                        if r["run_id"] == response["run_id"]
                    ]
                    if len(matching) != 1:
                        raise ValueError("Run/trace identity mismatch")
                    trace = matching[0]
                    rows.append(
                        {
                            "scope": "actual_public_Web_smoke_not_human_gold",
                            "run_id": response["run_id"],
                            "local_only": only,
                            "support_status": response["support_status"],
                            "citation_count": len(response["citations"]),
                            "evidence_count": len(response["evidence"]),
                            "web_record_count": len(response["web_records"]),
                            "routing": response.get("routing"),
                            "external_attempts": trace["request_budget"][
                                "external_attempts"
                            ],
                            "web_events": trace["web_events"],
                            "pipeline_calls": trace["llm_calls"],
                            "pipeline_tokens": trace["token_usage"],
                            "human_correctness": None,
                        }
                    )
                    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                    print(json.dumps(rows[-1]), flush=True)
    finally:
        index.client.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(run(parser.parse_args().root))
