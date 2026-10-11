"""Read-only Australian researcher directory endpoints."""

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.services.researchers import ResearcherService

router = APIRouter(prefix="/api/researchers", tags=["researchers"])


def _provider_failure(error: httpx.HTTPError) -> HTTPException:
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
        return HTTPException(503, "OpenAlex rate limit reached. Try again later.")
    return HTTPException(503, "OpenAlex researcher data is temporarily unavailable.")


@router.get("/search")
async def search_researchers(
    name: str = Query("", max_length=100),
    institution: str = Query("", max_length=100),
    topic: str = Query("", max_length=100),
    discipline: str = Query("", max_length=100),
    limit: int = Query(20, ge=1, le=20),
) -> dict:
    service = ResearcherService()
    try:
        return await service.search(name, institution, topic, discipline, limit)
    except httpx.HTTPError as error:
        raise _provider_failure(error) from error
    finally:
        await service.close()


@router.get("/{author_id}")
async def get_researcher(author_id: str) -> dict:
    service = ResearcherService()
    try:
        try:
            profile = await service.profile(author_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise HTTPException(404, "Researcher not found") from error
            raise _provider_failure(error) from error
        except httpx.RequestError as error:
            raise _provider_failure(error) from error
        if profile is None:
            raise HTTPException(404, "No latest Australian affiliation found")
        return profile
    finally:
        await service.close()
