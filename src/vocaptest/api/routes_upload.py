"""API routes for audio upload and analysis."""
from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from vocaptest.api.dependencies import get_search_engine, get_config
from vocaptest.api.schemas import AnalyzeResponse, AnalyzeResult, SearchResultItem
from vocaptest.utils.logging import setup_logging

logger = setup_logging()
router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Upload an audio file and get top-K similar producers."""
    cfg = get_config()
    max_mb = cfg.api.get("max_upload_mb", 50)

    # Validate file type
    allowed_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_exts)}",
        )

    # Read file content
    content = await file.read()
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {max_mb}MB",
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        engine = get_search_engine()
        results, diagnostics = engine.search_file_detailed(tmp_path)
        elapsed = time.time() - t0

        logger.info("Job %s: analyzed in %.2fs, %d results", job_id, elapsed, len(results))

        top_k = [
            SearchResultItem(
                producer_slug=r.producer_slug,
                display_name=r.display_name,
                score=r.score,
                rank=r.rank,
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

        return AnalyzeResponse(
            job_id=job_id,
            status="done",
            result=AnalyzeResult(
                top_k=top_k,
                accepted=diagnostics["accepted"] if diagnostics else None,
                confidence=diagnostics["confidence"] if diagnostics else None,
                margin=diagnostics["margin"] if diagnostics else None,
                entropy=diagnostics["entropy"] if diagnostics else None,
                warnings=warnings,
            ),
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        return AnalyzeResponse(
            job_id=job_id,
            status="failed",
            error=str(e),
        )
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
