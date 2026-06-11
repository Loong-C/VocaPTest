"""API routes for search-related endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from vocaptest.api.schemas import JobStatusResponse

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an analysis job.

    In MVP, jobs are synchronous, so this always returns 'done' or 'not_found'.
    Future versions will support async job queues.
    """
    # In MVP, jobs are processed synchronously.
    # This endpoint exists for API completeness and future async support.
    return JobStatusResponse(
        job_id=job_id,
        status="not_found",
        error="Job not found. In MVP, jobs are processed synchronously via POST /api/analyze.",
    )
