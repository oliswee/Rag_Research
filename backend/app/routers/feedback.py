"""Local feedback persistence; no external write or model action is exposed."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.feedback import FeedbackCreate, FixReference, RegressionCreate
from app.services import answer_feedback as service

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


async def checked(operation):
    try:
        return await operation
    except LookupError:
        raise HTTPException(
            404,
            detail={
                "code": "FEEDBACK_NOT_FOUND",
                "message": "Feedback record or run not found",
            },
        ) from None
    except ValueError:
        raise HTTPException(
            400,
            detail={
                "code": "INVALID_FEEDBACK",
                "message": "Feedback source, category or regression state is invalid",
            },
        ) from None


@router.get("/runs/{run_id}")
async def run_manifest(run_id: str, db: AsyncSession = Depends(get_db)):
    return await checked(service.get_run(db, run_id))


@router.post("", status_code=201)
async def create_feedback(request: FeedbackCreate, db: AsyncSession = Depends(get_db)):
    return await checked(service.create_feedback(db, **request.model_dump()))


@router.post("/{feedback_id}/regressions", status_code=201)
async def create_regression(
    feedback_id: str, request: RegressionCreate, db: AsyncSession = Depends(get_db)
):
    return await checked(
        service.create_regression(db, feedback_id, synthetic=request.synthetic)
    )


@router.post("/regressions/{regression_id}/run")
async def run_regression(regression_id: str, db: AsyncSession = Depends(get_db)):
    return await checked(service.run_regression(db, regression_id))


@router.post("/regressions/{regression_id}/fix")
async def record_fix(
    regression_id: str, request: FixReference, db: AsyncSession = Depends(get_db)
):
    return await checked(service.record_fix(db, regression_id, request.reference))
