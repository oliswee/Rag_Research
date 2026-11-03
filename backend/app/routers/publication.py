"""Explicit publication metadata checks; reading cached state never uses Web."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.publication_status import get_status, refresh_status

router = APIRouter(prefix="/api/papers", tags=["publication"])


class RefreshPublication(BaseModel):
    local_only: bool = False


@router.get("/{paper_id}/publication")
async def publication_status(paper_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await get_status(db, paper_id)
    except LookupError:
        raise HTTPException(
            404, detail={"code": "NOT_FOUND", "message": "Paper not found"}
        ) from None


@router.post("/{paper_id}/publication/refresh")
async def refresh_publication(
    paper_id: str, request: RefreshPublication, db: AsyncSession = Depends(get_db)
):
    try:
        return await refresh_status(db, paper_id, local_only=request.local_only)
    except LookupError:
        raise HTTPException(
            404, detail={"code": "NOT_FOUND", "message": "Paper not found"}
        ) from None
