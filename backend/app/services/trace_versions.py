"""Deterministic provenance for runtime traces, without persisted source text."""

import hashlib
import json
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import (
    Paper,
    PaperChunk,
    PaperIdentifier,
    Pattern,
    SourceDocument,
    SourceIndexGeneration,
    SourceParent,
)


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@cache
def pipeline_revision() -> str:
    """Hash normalized app source once per process; no Git or local paths needed."""
    app_root = Path(__file__).resolve().parents[1]
    files = [
        (
            str(path.relative_to(app_root).as_posix()),
            hashlib.sha256(
                path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            ).hexdigest(),
        )
        for path in sorted(app_root.rglob("*.py"))
    ]
    return "sha256:" + digest(files)


@cache
def dependency_versions() -> dict[str, str | None]:
    result = {}
    for name in ("sqlalchemy", "qdrant-client", "fastembed", "openai", "anthropic"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def configuration_snapshot() -> tuple[str, dict]:
    """Use an allowlist: never include credentials, paths, emails or endpoints."""
    manifest = {
        "schema": "m1_runtime_v2",
        "pipeline_revision": pipeline_revision(),
        "dependencies": dependency_versions(),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.fastembed_model
        if settings.embedding_provider == "fastembed"
        else "text-embedding-3-small",
        "generation_provider": settings.llm_provider,
        "generation_model": {
            "ollama": settings.ollama_model,
            "anthropic": settings.anthropic_model,
            "deepseek": settings.deepseek_model,
        }.get(settings.llm_provider),
        "model_weight_revision": None,
        "ollama_endpoint_sha256": digest(settings.ollama_base_url),
        "qdrant_endpoint_sha256": digest([settings.qdrant_host, settings.qdrant_port]),
        "llm_max_attempts": settings.extraction_max_retries,
        "llm_timeout_seconds": settings.extraction_timeout_seconds,
        "retrieval": {
            "query_limit": settings.query_variant_limit,
            "rewrite_timeout_seconds": settings.query_rewrite_timeout_seconds,
            "branch_timeout_seconds": settings.retrieval_branch_timeout_seconds,
            "concurrency": settings.retrieval_concurrency,
            "per_query_k": settings.retrieval_per_query_k,
            "rrf_k": settings.retrieval_rrf_k,
            "final_k": 10,
            "fusion": "single_layer_rrf",
        },
        "ranking_packing": {
            "model": settings.reranker_model,
            "candidate_limit": settings.reranker_candidate_limit,
            "selected_limit": settings.reranker_selected_limit,
            "timeout_seconds": settings.reranker_timeout_seconds,
            "context_budget": settings.context_token_budget,
            "parent_budget": settings.context_parent_budget,
            "budget_units": "utf8_bytes_conservative_bpe_bound",
            "runtime_weights": "cached_only_cpu",
        },
    }
    return "sha256:" + digest(manifest), manifest


async def corpus_version(db: AsyncSession) -> str:
    """Read persisted rows in stable order, hashing text only in memory.

    A linear scan detects external edits and stale stored checksums. The caller
    records its duration. Physical vector contents are not certified by this hash.
    """
    fingerprint = hashlib.sha256()
    queries = (
        (
            "paper_identifiers",
            select(
                PaperIdentifier.paper_id, PaperIdentifier.kind, PaperIdentifier.value
            ).order_by(
                PaperIdentifier.paper_id, PaperIdentifier.kind, PaperIdentifier.value
            ),
        ),
        (
            "papers",
            select(
                Paper.id,
                Paper.title,
                Paper.source,
                Paper.published_date,
                Paper.status,
                Paper.import_source,
                Paper.open_access,
                Paper.content_scope,
            ).order_by(Paper.id),
        ),
        (
            "chunks",
            select(
                PaperChunk.id,
                PaperChunk.paper_id,
                PaperChunk.chunk_index,
                PaperChunk.text,
                PaperChunk.checksum,
                PaperChunk.section,
                PaperChunk.content_scope,
                PaperChunk.page_start,
                PaperChunk.page_end,
                PaperChunk.start_char,
                PaperChunk.end_char,
                PaperChunk.source_url,
                PaperChunk.cleaned_text,
                PaperChunk.index_version,
                PaperChunk.document_id,
                PaperChunk.parent_id,
                PaperChunk.source_spans,
            ).order_by(PaperChunk.id),
        ),
        (
            "source_documents",
            select(
                SourceDocument.id,
                SourceDocument.paper_id,
                SourceDocument.document_version,
                SourceDocument.parser_version,
                SourceDocument.snapshot_digest,
                SourceDocument.raw_pages,
                SourceDocument.content_scope,
                SourceDocument.source_origin,
                SourceDocument.source_url,
            ).order_by(SourceDocument.id),
        ),
        (
            "source_generations",
            select(
                SourceIndexGeneration.index_version,
                SourceIndexGeneration.document_id,
                SourceIndexGeneration.is_active,
                SourceIndexGeneration.vector_status,
                SourceIndexGeneration.cleaner_version,
                SourceIndexGeneration.chunker_version,
                SourceIndexGeneration.embedding_model_id,
                SourceIndexGeneration.physical_revision,
                SourceIndexGeneration.verified_at,
            ).order_by(SourceIndexGeneration.index_version),
        ),
        (
            "source_parents",
            select(
                SourceParent.id,
                SourceParent.generation_id,
                SourceParent.document_id,
                SourceParent.raw_text,
                SourceParent.cleaned_text,
                SourceParent.raw_spans,
            ).order_by(SourceParent.id),
        ),
        (
            "patterns",
            select(
                Pattern.id,
                Pattern.paper_id,
                Pattern.name,
                Pattern.claim,
                Pattern.evidence,
                Pattern.context,
                Pattern.status,
                Pattern.embedding_id,
                Pattern.tags,
                Pattern.custom_fields,
            ).order_by(Pattern.id),
        ),
    )
    for table, statement in queries:
        fingerprint.update(json.dumps(table).encode())
        rows = await db.stream(statement.execution_options(yield_per=256))
        try:
            async for row in rows.mappings():
                # Each row becomes one fixed-size digest, never a persisted text.
                fingerprint.update(bytes.fromhex(digest(dict(row))))
        finally:
            await rows.close()
    return "sqlite:sha256:" + fingerprint.hexdigest()
