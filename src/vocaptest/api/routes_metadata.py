"""API routes for producer metadata."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from vocaptest.api.dependencies import get_profiles
from vocaptest.api.schemas import ProducerInfo, ProducerListResponse

router = APIRouter(prefix="/api", tags=["producers"])


@router.get("/producers", response_model=ProducerListResponse)
async def list_producers():
    """Return the list of producers in the reference library."""
    profiles = get_profiles()
    producers_dict = profiles.get("producers", {})

    producers = [
        ProducerInfo(
            slug=slug,
            display_name=info.get("display_name", slug),
            song_count=info.get("song_count"),
            segment_count=info.get("segment_count"),
        )
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
    profiles = get_profiles()
    producers_dict = profiles.get("producers", {})

    info = producers_dict.get(producer_slug)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Producer not found: {producer_slug}")

    return ProducerInfo(
        slug=producer_slug,
        display_name=info.get("display_name", producer_slug),
        song_count=info.get("song_count"),
        segment_count=info.get("segment_count"),
    )
