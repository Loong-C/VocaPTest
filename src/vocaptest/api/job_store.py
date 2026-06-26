"""In-memory analysis job status store."""
from __future__ import annotations

import time
from threading import Lock

from vocaptest.api.schemas import AnalyzeResult, JobStatusResponse

MAX_JOBS = 100
JOB_TTL_SECONDS = 60 * 60

_jobs: dict[str, dict] = {}
_lock = Lock()


def _now() -> float:
    return time.time()


def _prune_locked() -> None:
    if len(_jobs) <= MAX_JOBS:
        return
    cutoff = _now() - JOB_TTL_SECONDS
    stale_ids = [
        job_id
        for job_id, job in _jobs.items()
        if job.get("updated_at", 0.0) < cutoff
    ]
    for job_id in stale_ids:
        _jobs.pop(job_id, None)
    if len(_jobs) <= MAX_JOBS:
        return
    for job_id, _job in sorted(
        _jobs.items(),
        key=lambda item: item[1].get("updated_at", 0.0),
    )[: len(_jobs) - MAX_JOBS]:
        _jobs.pop(job_id, None)


def create_job(job_id: str) -> JobStatusResponse:
    with _lock:
        _prune_locked()
        timestamp = _now()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "stage": "received",
            "progress": 0.05,
            "result": None,
            "error": None,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        return _to_response(_jobs[job_id])


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    result: AnalyzeResult | None = None,
    error: str | None = None,
) -> JobStatusResponse:
    with _lock:
        job = _jobs.setdefault(
            job_id,
            {
                "job_id": job_id,
                "status": "processing",
                "stage": "received",
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": _now(),
            },
        )
        if status is not None:
            job["status"] = status
        if stage is not None:
            job["stage"] = stage
        if progress is not None:
            job["progress"] = max(job.get("progress", 0.0), min(progress, 1.0))
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = _now()
        return _to_response(job)


def get_job(job_id: str) -> JobStatusResponse:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return JobStatusResponse(
                job_id=job_id,
                status="not_found",
                stage="failed",
                progress=0.0,
                error=(
                    "Job not found. Completed jobs are kept in memory for a "
                    "limited time and are lost when the API process restarts."
                ),
            )
        return _to_response(job)


def _to_response(job: dict) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job.get("status", "processing"),
        stage=job.get("stage", "received"),
        result=job.get("result"),
        error=job.get("error"),
        progress=job.get("progress", 0.0),
    )
