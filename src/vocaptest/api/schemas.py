"""Pydantic schemas for API request/response."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ProducerSong(BaseModel):
    song_id: str
    title: str
    source_url: str | None = None


class SearchResultItem(BaseModel):
    producer_slug: str
    display_name: str
    avatar_url: str | None = None
    score: float
    rank: int
    style_tags: list[str] = Field(default_factory=list)
    representative_songs: list[ProducerSong] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = "done"
    result: AnalyzeResult | None = None
    error: str | None = None


class AnalyzeResult(BaseModel):
    top_k: list[SearchResultItem]
    accepted: bool | None = None
    confidence: float | None = None
    margin: float | None = None
    entropy: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ProducerInfo(BaseModel):
    slug: str
    display_name: str
    song_count: int | None = None
    segment_count: int | None = None
    avatar_url: str | None = None
    aliases: list[str] = Field(default_factory=list)
    profile_url: str | None = None
    style_tags: list[str] = Field(default_factory=list)
    style_tag_source: str | None = None
    style_tag_source_url: str | None = None
    songs: list[ProducerSong] = Field(default_factory=list)
    training_songs: list[ProducerSong] = Field(default_factory=list)
    dev_song_count: int = 0
    dev_songs: list[ProducerSong] = Field(default_factory=list)
    frozen_song_count: int = 0
    frozen_songs: list[ProducerSong] = Field(default_factory=list)
    test_song_count: int = 0
    test_songs: list[ProducerSong] = Field(default_factory=list)


class ProducerListResponse(BaseModel):
    producers: list[ProducerInfo]
    backend: str | None = None
    total_producers: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # processing | done | failed | not_found
    stage: str = "received"  # received | segmenting | embedding | classifying | done | failed
    result: AnalyzeResult | None = None
    error: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    backend: str | None = None
    producers_loaded: int = 0
