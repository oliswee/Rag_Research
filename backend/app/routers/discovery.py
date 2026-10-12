"""Search public academic APIs and queue selected records for local import."""

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import ImportItem, ImportJob, Paper
from app.services.discovery import DiscoveryService, normalize_title
from app.services.metadata_context import get_paper_context
from app.workflows.discovery_import import process_job

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


class CandidateRef(BaseModel):
    provider: str
    provider_id: str = Field(min_length=1, max_length=500)


class ImportRequest(BaseModel):
    candidates: list[CandidateRef] = Field(min_length=1, max_length=100)


@router.get("/search")
async def search_papers(
    query: str = Query(min_length=2),
    year_from: int | None = Query(None, ge=1800, le=2100),
    year_to: int | None = Query(None, ge=1800, le=2100),
    open_access: bool = False,
    limit: int = Query(20, ge=1, le=20),
    semantic_offset: int = Query(0, ge=0, le=9999),
    openalex_offset: int = Query(0, ge=0, le=9999),
    pubmed_offset: int = Query(0, ge=0, le=9999),
    arxiv_offset: int = Query(0, ge=0, le=9999),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if year_from and year_to and year_from > year_to:
        raise HTTPException(422, "year_from must not exceed year_to")
    service = DiscoveryService()
    try:
        result = await service.search(
            query,
            year_from,
            year_to,
            open_access,
            limit,
            {
                "semantic_scholar": semantic_offset,
                "openalex": openalex_offset,
                "pubmed": pubmed_offset,
                "arxiv": arxiv_offset,
            },
        )
    finally:
        await service.close()
    # Title similarity is advisory; only identifiers cause automatic merging.
    existing_titles = (await db.execute(select(Paper.id, Paper.title))).all()
    by_title: dict[str, list[str]] = {}
    for paper_id, title in existing_titles:
        by_title.setdefault(normalize_title(title), []).append(paper_id)
    for candidate in result["candidates"]:
        candidate["possible_duplicate_ids"] = by_title.get(
            normalize_title(candidate["title"]), []
        )[:3]
    return result


@router.get("/context")
async def paper_context(
    provider: str = Query(pattern="^(semantic_scholar|openalex|pubmed|arxiv)$"),
    provider_id: str = Query(min_length=1, max_length=500),
) -> dict:
    """Fetch citation details only when a visitor opens a paper's context."""
    service = DiscoveryService()
    try:
        return await get_paper_context(service, provider, provider_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = 404 if exc.response.status_code == 404 else 502
        raise HTTPException(status, "Paper source unavailable") from exc
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        raise HTTPException(502, "Paper source unavailable") from exc
    finally:
        await service.close()


@router.post("/import", status_code=202)
async def start_import(
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    if any(
        ref.provider not in {"semantic_scholar", "openalex", "pubmed", "arxiv"}
        for ref in request.candidates
    ):
        raise HTTPException(422, "Unsupported provider")
    job = ImportJob()
    db.add(job)
    await db.flush()
    seen = set()
    for ref in request.candidates:
        key = (ref.provider, ref.provider_id)
        if key not in seen:
            seen.add(key)
            db.add(
                ImportItem(
                    job_id=job.id,
                    provider=ref.provider,
                    provider_id=ref.provider_id,
                    candidate=ref.model_dump(),
                )
            )
    await db.commit()
    background_tasks.add_task(process_job, job.id)
    return {"job_id": job.id, "status": "queued", "estimated_api_calls": len(seen)}


@router.get("/imports/{job_id}")
async def get_import(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    job = await db.get(ImportJob, job_id)
    if job is None:
        raise HTTPException(404, "Import job not found")
    items = (
        (await db.execute(select(ImportItem).where(ImportItem.job_id == job_id)))
        .scalars()
        .all()
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "items": [
            {
                "provider": item.provider,
                "provider_id": item.provider_id,
                "status": item.status,
                "paper_id": item.paper_id,
                "content_scope": item.content_scope,
                "warning": item.warning,
                "error": item.error,
            }
            for item in items
        ],
    }


@router.post("/imports/{job_id}/retry", status_code=202)
async def retry_import(
    job_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> dict:
    job = await db.get(ImportJob, job_id)
    if job is None:
        raise HTTPException(404, "Import job not found")
    if job.status in {"queued", "running"}:
        raise HTTPException(409, "Import is already running")
    items = (
        (await db.execute(select(ImportItem).where(ImportItem.job_id == job_id)))
        .scalars()
        .all()
    )
    for item in items:
        if item.status == "failed":
            item.status, item.error = "queued", None
    job.status, job.completed_at = "queued", None
    await db.commit()
    background_tasks.add_task(process_job, job_id)
    return {"job_id": job_id, "status": "queued"}
