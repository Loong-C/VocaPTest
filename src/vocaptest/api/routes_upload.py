"""API routes for audio upload and analysis."""
from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from vocaptest.api.dependencies import get_search_engine, get_config
from vocaptest.api.job_store import create_job, update_job
from vocaptest.api.schemas import (
    AnalyzeResponse,
    AnalyzeResult,
    JobStatusResponse,
    SearchResultItem,
)
from vocaptest.data.producer_catalog import (
    load_producer_metadata,
    load_producer_style_tags,
    load_representative_song_catalog,
)
from vocaptest.utils.logging import setup_logging

logger = setup_logging()
router = APIRouter(prefix="/api", tags=["analyze"])

ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
READ_CHUNK_BYTES = 1024 * 1024
USER_ANALYSIS_ERROR = "分析失败，请稍后重试或换一段更清晰的音频。"
ProgressCallback = Callable[[str, float], None]


async def _save_upload_to_temp_file(file: UploadFile) -> tuple[str, str]:
    cfg = get_config()
    max_mb = cfg.api.get("max_upload_mb", 50)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    max_bytes = max_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        total = 0
        try:
            while chunk := await file.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum: {max_mb}MB",
                    )
                tmp.write(chunk)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return tmp_path, ext


def _build_result(
    results,
    diagnostics: dict | None,
) -> AnalyzeResult:
    producer_metadata = load_producer_metadata()
    style_tags = load_producer_style_tags()
    representative_songs = load_representative_song_catalog()
    top_k = [
        SearchResultItem(
            producer_slug=r.producer_slug,
            display_name=r.display_name,
            avatar_url=producer_metadata.get(r.producer_slug, {}).get("avatar_url"),
            score=r.score,
            rank=r.rank,
            style_tags=style_tags.get(r.producer_slug, {}).get("style_tags", []),
            representative_songs=representative_songs.get(r.producer_slug, []),
        )
        for r in results
    ]

    warnings = []
    if not results:
        warnings.append("No matching producers found. The reference library may be empty.")
    if len(top_k) < 3:
        warnings.append("Fewer than 3 results — reference data may be limited.")
    if diagnostics and not diagnostics["accepted"]:
        warnings.append(
            "Low-confidence result: the uploaded song is outside the "
            "model's calibrated acceptance region."
        )

    return AnalyzeResult(
        top_k=top_k,
        accepted=diagnostics["accepted"] if diagnostics else None,
        confidence=diagnostics["confidence"] if diagnostics else None,
        margin=diagnostics["margin"] if diagnostics else None,
        entropy=diagnostics["entropy"] if diagnostics else None,
        warnings=warnings,
    )


def _analyze_temp_file(
    tmp_path: str,
    job_id: str,
    progress_callback: ProgressCallback | None = None,
) -> AnalyzeResult:
    t0 = time.time()
    engine = get_search_engine()
    results, diagnostics = engine.search_file_detailed(
        tmp_path,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - t0
    logger.info("Job %s: analyzed in %.2fs, %d results", job_id, elapsed, len(results))
    return _build_result(results, diagnostics)


def _run_analysis_job(job_id: str, tmp_path: str) -> None:
    def progress_callback(stage: str, progress: float) -> None:
        update_job(job_id, stage=stage, progress=progress)

    try:
        update_job(job_id, status="processing", stage="received", progress=0.10)
        result = _analyze_temp_file(tmp_path, job_id, progress_callback)
        update_job(
            job_id,
            status="done",
            stage="done",
            progress=1.0,
            result=result,
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error=USER_ANALYSIS_ERROR,
        )
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Upload an audio file and get top-K similar producers."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    tmp_path, _ext = await _save_upload_to_temp_file(file)

    try:
        result = _analyze_temp_file(tmp_path, job_id)
        return AnalyzeResponse(
            job_id=job_id,
            status="done",
            result=result,
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        return AnalyzeResponse(
            job_id=job_id,
            status="failed",
            error=USER_ANALYSIS_ERROR,
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


@router.post("/analyze/jobs", response_model=JobStatusResponse)
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload an audio file and start an asynchronous analysis job."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    tmp_path, _ext = await _save_upload_to_temp_file(file)
    response = create_job(job_id)
    background_tasks.add_task(_run_analysis_job, job_id, tmp_path)
    return response
