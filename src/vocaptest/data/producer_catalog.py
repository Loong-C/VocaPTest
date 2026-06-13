"""Load producer metadata and the songs represented in the deployed model."""
from __future__ import annotations

import json
from functools import lru_cache

import yaml

from vocaptest.utils.paths import project_root


@lru_cache(maxsize=1)
def load_producer_metadata() -> dict[str, dict]:
    """Return configured producer metadata keyed by slug."""
    path = project_root() / "configs" / "producers.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        item["slug"]: item
        for item in config.get("producers", [])
    }


def _youtube_url(song_id: str) -> str | None:
    prefix = "youtube_"
    if not song_id.startswith(prefix):
        return None
    return f"https://www.youtube.com/watch?v={song_id.removeprefix(prefix)}"


@lru_cache(maxsize=1)
def load_training_song_catalog() -> dict[str, list[dict]]:
    """Return one entry per unique song from the current P1 manifest."""
    path = (
        project_root()
        / "data"
        / "processed"
        / "curated"
        / "mert_95_p1"
        / "segments.jsonl"
    )
    if not path.exists():
        return {}

    songs_by_producer: dict[str, dict[str, dict]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            slug = record["producer_slug"]
            song_id = record["song_id"]
            songs_by_producer.setdefault(slug, {}).setdefault(
                song_id,
                {
                    "song_id": song_id,
                    "title": record.get("title") or song_id,
                    "source_url": _youtube_url(song_id),
                },
            )

    return {
        slug: sorted(
            songs.values(),
            key=lambda item: item["title"].casefold(),
        )
        for slug, songs in songs_by_producer.items()
    }
