"""API routes for producer metadata."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from vocaptest.api.dependencies import get_reference_library
from vocaptest.api.schemas import ProducerInfo, ProducerListResponse, ProducerSong
from vocaptest.data.producer_catalog import (
    load_producer_metadata,
    load_training_song_catalog,
)

router = APIRouter(prefix="/api", tags=["producers"])


def _producer_info(
    slug: str,
    model_info: dict,
    *,
    include_songs: bool,
) -> ProducerInfo:
    metadata = load_producer_metadata().get(slug, {})
    songs = load_training_song_catalog().get(slug, []) if include_songs else []
    return ProducerInfo(
        slug=slug,
        display_name=metadata.get(
            "display_name",
            model_info.get("display_name", slug),
        ),
        song_count=len(songs) if include_songs else model_info.get("song_count"),
        segment_count=model_info.get("segment_count"),
        avatar_url=metadata.get("avatar_url"),
        aliases=metadata.get("aliases", []),
        profile_url=metadata.get("profile_url"),
        songs=[ProducerSong(**song) for song in songs],
    )


@router.get("/producers", response_model=ProducerListResponse)
async def list_producers():
    """Return the list of producers in the reference library."""
    profiles = get_reference_library()
    producers_dict = profiles.get("producers", {})

    producers = [
        _producer_info(slug, info, include_songs=False)
        for slug, info in sorted(producers_dict.items())
    ]

    return ProducerListResponse(
        producers=producers,
        backend=profiles.get("backend"),
        total_producers=len(producers),
    )


@router.get("/producers/{producer_slug}", response_model=ProducerInfo)
async def get_producer(producer_slug: str):
    """Return info for a specific producer."""
    profiles = get_reference_library()
    producers_dict = profiles.get("producers", {})

    info = producers_dict.get(producer_slug)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Producer not found: {producer_slug}")

    return _producer_info(producer_slug, info, include_songs=True)
