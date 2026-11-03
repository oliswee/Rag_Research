"""Original source and citation endpoints, including retired generation IDs."""

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import PaperChunk, SourceDocument
from app.services.source_anchors import resolve_child, validate_document

router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("/chunks/{chunk_id}")
async def source_chunk(chunk_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    child = await db.get(PaperChunk, chunk_id)
    if child is None:
        raise HTTPException(404, "Source citation not found")
    try:
        return await resolve_child(db, child, allow_parent_failure=True)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(409, "Stored source provenance failed validation") from None


@router.get("/documents/{document_id}/pdf")
async def source_pdf(document_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    document = await db.get(SourceDocument, document_id)
    if document is None or document.original_pdf is None:
        raise HTTPException(404, "Original PDF is unavailable for this source")
    try:
        validate_document(document)
        if "sha256:" + hashlib.sha256(document.original_pdf).hexdigest() != (
            document.document_version
        ):
            raise ValueError("Original PDF version mismatch")
    except (ValueError, KeyError, TypeError):
        raise HTTPException(409, "Stored original PDF failed validation") from None
    return Response(
        content=document.original_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{document.id}.pdf"'},
    )
