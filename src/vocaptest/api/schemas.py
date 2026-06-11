"""Pydantic schemas for API request/response."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    producer_slug: str
    display_name: str
    score: float
    rank: int


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = "done"
    result: AnalyzeResult | None = None
    error: str | None = None


class AnalyzeResult(BaseModel):
    top_k: list[SearchResultItem]
    warnings: list[str] = []


class ProducerInfo(BaseModel):
    slug: str
    display_name: str
    song_count: int | None = None
    segment_count: int | None = None


class ProducerListResponse(BaseModel):
    producers: list[ProducerInfo]
    backend: str | None = None
    total_producers: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | processing | done | failed
    result: AnalyzeResult | None = None
    error: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    backend: str | None = None
    producers_loaded: int = 0
