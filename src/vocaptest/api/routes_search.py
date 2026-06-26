"""API routes for search-related endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from vocaptest.api.job_store import get_job
from vocaptest.api.schemas import JobStatusResponse

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an asynchronous analysis job."""
    return get_job(job_id)
