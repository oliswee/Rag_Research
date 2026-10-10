"""Evidence-based comparisons across selected local papers."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Paper
from app.workflows.source_query import compare_evidence, retrieve

router = APIRouter(prefix="/api/research", tags=["research"])

ROWS = [
    "Common findings",
    "Different conclusions",
    "Methods and samples",
    "Limitations and open questions",
]
QUESTIONS = [
    "main results findings",
    "conclusions results disagreement",
    "methods study sample design",
    "limitations future research",
]


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    question: str | None = None


@router.post("/compare")
async def compare(request: CompareRequest, db: AsyncSession = Depends(get_db)) -> dict:
    if len(set(request.paper_ids)) != len(request.paper_ids):
        raise HTTPException(422, "Choose distinct papers")
    papers = [await db.get(Paper, paper_id) for paper_id in request.paper_ids]
    if any(paper is None for paper in papers):
        raise HTTPException(404, "One or more papers not found")
    rows = []
    for label, prompt in zip(ROWS, QUESTIONS, strict=True):
        by_paper = {}
        for paper in papers:
            evidence, _ = await retrieve(
                db,
                (request.question + " " if request.question else "") + prompt,
                [paper.id],
                "bm25",
                limit=3,
            )
            by_paper[paper.id] = evidence
        cells = await compare_evidence(label, request.question, by_paper)
        rows.append({"label": label, "cells": cells})
    return {
        "papers": [{"id": paper.id, "title": paper.title} for paper in papers],
        "rows": rows,
    }
