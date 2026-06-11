"""Data schemas for songs, segments, embeddings, and search results."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Producer:
    slug: str
    display_name: str
    vocadb_artist_id: Optional[int] = None
    aliases: list[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Song:
    song_id: str
    producer_slug: str
    title: str
    publish_date: Optional[str] = None
    source_urls: list[str] = field(default_factory=list)
    vocalists: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_cover: bool = False
    is_remix: bool = False
    is_collaboration: bool = False
    status: str = "pending"  # pending | accepted | rejected | pending_review
    status_reason: Optional[str] = None
    local_audio_path: Optional[str] = None


@dataclass
class Segment:
    segment_id: str
    song_id: str
    producer_slug: str
    path: str
    start_sec: float
    end_sec: float
    duration_sec: float
    rms_db: float


@dataclass
class EmbeddingRecord:
    segment_id: str
    song_id: str
    producer_slug: str
    model_backend: str
    embedding_path: str
    embedding_dim: int
    work_id: Optional[str] = None
    recording_id: Optional[str] = None
    title: Optional[str] = None


@dataclass
class SearchResult:
    producer_slug: str
    display_name: str
    score: float
    rank: int


@dataclass
class AudioManifestEntry:
    """Record of a downloaded audio file."""
    file_hash: str
    path: str
    duration_sec: float
    sample_rate: int
    channels: int
    source_url: str
    producer_slug: str
    song_id: str
