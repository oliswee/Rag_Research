"""Retrieve SQLite evidence and compose answers from validated source IDs."""

import logging
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Paper, PaperChunk
from app.services.source_index import SourceVectorIndex, keyword_search, rrf_merge

logger = logging.getLogger(__name__)


async def has_source_chunks(db: AsyncSession) -> bool:
    return bool((await db.execute(select(func.count(PaperChunk.id)))).scalar())


async def retrieve(
    db: AsyncSession,
    query: str,
    paper_ids: list[str] | None = None,
    mode: str = "hybrid",
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    provider: str | None = None,
) -> tuple[list[dict], str]:
    lexical = (
        await keyword_search(
            db,
            query,
            paper_ids,
            limit=30,
            year_from=year_from,
            year_to=year_to,
            provider=provider,
        )
        if mode in {"hybrid", "bm25"}
        else []
    )
    dense = []
    actual_mode = mode
    if mode in {"hybrid", "dense"}:
        try:
            from app.services.embedding_service import EmbeddingService

            vector = (await EmbeddingService().generate_embedding(query)).embedding
            dense = SourceVectorIndex().search(
                vector,
                paper_ids,
                limit=30,
                year_from=year_from,
                year_to=year_to,
                provider=provider,
            )
        except Exception as exc:
            logger.warning("Dense source retrieval unavailable: %s", type(exc).__name__)
            if mode == "dense":
                lexical = await keyword_search(
                    db,
                    query,
                    paper_ids,
                    limit=30,
                    year_from=year_from,
                    year_to=year_to,
                    provider=provider,
                )
            actual_mode = "bm25_fallback"
    if mode == "dense" and not dense and not lexical:
        lexical = await keyword_search(
            db,
            query,
            paper_ids,
            limit=30,
            year_from=year_from,
            year_to=year_to,
            provider=provider,
        )
        actual_mode = "bm25_fallback"
    ids = (
        rrf_merge([dense, lexical], limit=limit)
        if dense and lexical
        else (dense or lexical)[:limit]
    )
    if not ids:
        return [], actual_mode
    statement = (
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(PaperChunk.id.in_(ids))
    )
    if year_from is not None:
        statement = statement.where(Paper.published_date >= date(year_from, 1, 1))
    if year_to is not None:
        statement = statement.where(Paper.published_date <= date(year_to, 12, 31))
    if provider:
        statement = statement.where(Paper.import_source == provider)
    rows = (await db.execute(statement)).all()
    by_id = {
        chunk.id: {
            "paper_id": paper.id,
            "chunk_id": chunk.id,
            "title": paper.title,
            "source_url": chunk.source_url or paper.source,
            "content_scope": chunk.content_scope,
            "excerpt": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        }
        for chunk, paper in rows
    }
    return [
        by_id[chunk_id]
        for chunk_id in ids
        if chunk_id in by_id
        and (paper_ids is None or by_id[chunk_id]["paper_id"] in paper_ids)
    ], actual_mode


async def answer_from_evidence(
    query: str, evidence: list[dict]
) -> tuple[str, list[dict]]:
    if not evidence:
        return (
            "Insufficient evidence in the local library. "
            "Discover or import relevant papers first.",
            [],
        )
    if not settings.anthropic_api_key:
        return (
            "Relevant source excerpts are shown below. Configure "
            "ANTHROPIC_API_KEY for a synthesized answer.",
            evidence,
        )
    from app.services.llm_client import LLMClient

    context = "\n\n".join(
        f"ID {source['chunk_id']} | {source['title']} | {source['content_scope']} | "
        f"page {source['page_start'] or 'none'}\n{source['excerpt']}"
        for source in evidence
    )
    result = await LLMClient().complete_json(
        system_prompt=(
            'Answer only from provided excerpts. Return JSON: {"claims":'
            '[{"text":"one factual sentence","source_ids":["exact excerpt ID"]}]}. '
            "Every claim must have at least one source ID. Say nothing unsupported."
        ),
        user_prompt=f"Question: {query}\n\nExcerpts:\n{context}",
    )
    allowed = {source["chunk_id"]: source for source in evidence}
    lines, used = [], {}
    for claim in result.get("claims", []):
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        ids = claim.get("source_ids", [])
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(source_id, str) or source_id not in allowed
                for source_id in ids
            )
        ):
            continue
        for source_id in ids:
            used[source_id] = allowed[source_id]
        lines.append(
            claim["text"].strip()
            + " "
            + " ".join(f"[C:{source_id}]" for source_id in ids)
        )
    return (
        "\n\n".join(lines)
        if lines
        else "Insufficient evidence in the retrieved excerpts."
    ), list(used.values())


async def compare_evidence(
    dimension: str, question: str | None, by_paper: dict[str, list[dict]]
) -> list[dict]:
    """Generate cross-paper cells while validating IDs against each paper."""
    if not settings.anthropic_api_key:
        return [
            {
                "paper_id": paper_id,
                "text": "Source excerpts are shown. Configure ANTHROPIC_API_KEY "
                "to synthesize differences."
                if evidence
                else "Insufficient evidence found",
                "evidence": evidence,
            }
            for paper_id, evidence in by_paper.items()
        ]
    from app.services.llm_client import LLMClient

    context = "\n\n".join(
        f"Paper {paper_id}; excerpt {source['chunk_id']}; title {source['title']}: "
        f"{source['excerpt']}"
        for paper_id, evidence in by_paper.items()
        for source in evidence
    )
    result = await LLMClient().complete_json(
        system_prompt=(
            "Compare the papers only using supplied excerpts. Return JSON: "
            '{"cells":[{"paper_id":"exact paper ID","text":"one sourced comparison",'
            '"source_ids":["exact excerpt ID"]}]}. '
            "Use no unsupported claims or invented pages. If evidence is insufficient, "
            "omit that cell. Each cell must cite excerpts belonging to its paper."
        ),
        user_prompt=(
            f"Dimension: {dimension}\nQuestion: {question or 'None'}\n{context}"
        ),
    )
    proposed = {
        cell.get("paper_id"): cell
        for cell in result.get("cells", [])
        if isinstance(cell, dict)
    }
    cells = []
    for paper_id, evidence in by_paper.items():
        allowed = {source["chunk_id"]: source for source in evidence}
        cell = proposed.get(paper_id, {})
        ids = cell.get("source_ids", [])
        if (
            not isinstance(cell.get("text"), str)
            or not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(source_id, str) or source_id not in allowed
                for source_id in ids
            )
        ):
            cells.append(
                {
                    "paper_id": paper_id,
                    "text": "Insufficient evidence found",
                    "evidence": [],
                }
            )
        else:
            cells.append(
                {
                    "paper_id": paper_id,
                    "text": cell["text"].strip(),
                    "evidence": [
                        allowed[source_id] for source_id in dict.fromkeys(ids)
                    ],
                }
            )
    return cells
