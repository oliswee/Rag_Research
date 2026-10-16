"""Authoritative SQLite excerpts and rebuildable keyword/vector indices."""

import hashlib
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.embedding_service import (
    configured_embedding_dimensions,
    embedding_collection,
)

CHUNK_COLLECTION = embedding_collection("source_chunks")


def dataset_id() -> str:
    """Scope vectors to the configured SQLite library on this machine."""
    identity = f"{Path.cwd().resolve()}|{settings.database_url}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def build_chunks(
    pages: list[tuple[str, int | None]],
    scope: str,
    source_url: str | None,
    size: int = 1200,
) -> list[dict]:
    """Split each source page independently so citations retain a real page."""
    chunks: list[dict] = []
    for content, page in pages:
        content = content.strip()
        if not content:
            continue
        position = 0
        while position < len(content):
            end = min(position + size, len(content))
            if end < len(content):
                split = content.rfind(" ", position + size // 2, end)
                if split > position:
                    end = split
            excerpt = content[position:end].strip()
            if excerpt:
                chunks.append(
                    dict(
                        id=str(uuid.uuid4()),
                        chunk_index=len(chunks),
                        text=excerpt,
                        content_scope=scope,
                        page_start=page,
                        page_end=page,
                        start_char=position,
                        end_char=end,
                        source_url=source_url,
                        checksum=hashlib.sha256(excerpt.encode()).hexdigest(),
                    )
                )
            position = end
            while position < len(content) and content[position].isspace():
                position += 1
    return chunks


async def ensure_fts(db: AsyncSession) -> None:
    """Create FTS in fresh test databases; production creates it in Alembic."""
    existed = (
        await db.execute(
            text(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='paper_chunks_fts'"
            )
        )
    ).scalar_one_or_none()
    await db.execute(
        text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunks_fts "
            "USING fts5(text, content='paper_chunks', content_rowid='rowid')"
        )
    )
    await db.execute(
        text(
            "CREATE TRIGGER IF NOT EXISTS paper_chunks_ai AFTER INSERT ON paper_chunks "
            "BEGIN INSERT INTO paper_chunks_fts(rowid,text) "
            "VALUES (new.rowid,new.text); END"
        )
    )
    await db.execute(
        text(
            "CREATE TRIGGER IF NOT EXISTS paper_chunks_ad AFTER DELETE ON paper_chunks "
            "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
            "VALUES ('delete',old.rowid,old.text); END"
        )
    )
    await db.execute(
        text(
            "CREATE TRIGGER IF NOT EXISTS paper_chunks_au AFTER UPDATE ON paper_chunks "
            "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
            "VALUES ('delete',old.rowid,old.text); "
            "INSERT INTO paper_chunks_fts(rowid,text) VALUES (new.rowid,new.text); END"
        )
    )
    if not existed:
        # Rebuild covers rows inserted before FTS was initialized (legacy/test DBs).
        await db.execute(
            text("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES ('rebuild')")
        )


async def keyword_search(
    db: AsyncSession,
    query: str,
    paper_ids: list[str] | None = None,
    limit: int = 30,
    year_from: int | None = None,
    year_to: int | None = None,
    provider: str | None = None,
    with_scores: bool = False,
) -> list[str] | list[tuple[str, float]]:
    await ensure_fts(db)
    tokens = re.findall(r"\w+", query, re.UNICODE)[:16]
    if not tokens:
        return []
    expression = " OR ".join('"' + token.replace('"', "") + '"' for token in tokens)
    params: dict = {"query": expression, "limit": limit}
    clause = ""
    if paper_ids is not None:
        if not paper_ids:
            return []
        markers = []
        for index, paper_id in enumerate(paper_ids):
            key = f"paper_{index}"
            params[key] = paper_id
            markers.append(f":{key}")
        clause = f" AND c.paper_id IN ({','.join(markers)})"
    if year_from is not None or year_to is not None or provider:
        clause += " AND EXISTS (SELECT 1 FROM papers p WHERE p.id=c.paper_id"
        if year_from is not None:
            clause += " AND p.published_date>=:year_from"
            params["year_from"] = f"{year_from}-01-01"
        if year_to is not None:
            clause += " AND p.published_date<=:year_to"
            params["year_to"] = f"{year_to}-12-31"
        if provider:
            clause += " AND p.import_source=:provider"
            params["provider"] = provider
        clause += ")"
    result = await db.execute(
        text(
            "SELECT c.id, bm25(paper_chunks_fts) AS score FROM paper_chunks_fts f "
            "JOIN paper_chunks c ON c.rowid=f.rowid "
            "WHERE paper_chunks_fts MATCH :query"
            + clause
            + " ORDER BY bm25(paper_chunks_fts) LIMIT :limit"
        ),
        params,
    )
    if with_scores:
        return [(chunk_id, float(score)) for chunk_id, score in result.all()]
    return list(result.scalars())


def rrf_merge(rankings: list[list[str]], k: int = 60, limit: int = 10) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


class SourceVectorIndex:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.dimensions = configured_embedding_dimensions()
        self.dataset_id = dataset_id()

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(CHUNK_COLLECTION):
            self.client.create_collection(
                CHUNK_COLLECTION,
                vectors_config=VectorParams(
                    size=self.dimensions, distance=Distance.COSINE
                ),
            )
        elif (
            self.client.get_collection(CHUNK_COLLECTION).config.params.vectors.size
            != self.dimensions
        ):
            raise ValueError("Existing source vector collection has wrong dimensions")

    def upsert(
        self,
        chunk_id: str,
        paper_id: str,
        vector: list[float],
        year: int | None = None,
        provider: str | None = None,
    ) -> None:
        self.ensure_collection()
        self.client.upsert(
            CHUNK_COLLECTION,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "paper_id": paper_id,
                        "year": year,
                        "provider": provider,
                        "dataset_id": self.dataset_id,
                    },
                )
            ],
        )

    def upsert_many(
        self, chunks: list[tuple[str, str, list[float], int | None, str | None]]
    ) -> None:
        if not chunks:
            return
        self.ensure_collection()
        self.client.upsert(
            CHUNK_COLLECTION,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "paper_id": paper_id,
                        "year": year,
                        "provider": provider,
                        "dataset_id": self.dataset_id,
                    },
                )
                for chunk_id, paper_id, vector, year, provider in chunks
            ],
        )

    def search(
        self,
        vector: list[float],
        paper_ids: list[str] | None = None,
        limit: int = 30,
        year_from: int | None = None,
        year_to: int | None = None,
        provider: str | None = None,
        with_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        self.ensure_collection()
        conditions = [
            FieldCondition(key="dataset_id", match=MatchValue(value=self.dataset_id))
        ]
        if paper_ids is not None:
            conditions.append(
                FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))
            )
        if year_from is not None or year_to is not None:
            conditions.append(
                FieldCondition(key="year", range=Range(gte=year_from, lte=year_to))
            )
        if provider:
            conditions.append(
                FieldCondition(key="provider", match=MatchValue(value=provider))
            )
        query_filter = Filter(must=conditions) if conditions else None
        points = self.client.query_points(
            CHUNK_COLLECTION, query=vector, query_filter=query_filter, limit=limit
        ).points
        if with_scores:
            return [(str(hit.id), float(hit.score)) for hit in points]
        return [str(hit.id) for hit in points]

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids and self.client.collection_exists(CHUNK_COLLECTION):
            self.client.delete(
                CHUNK_COLLECTION, points_selector=PointIdsList(points=chunk_ids)
            )
