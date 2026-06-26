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


@lru_cache(maxsize=1)
def load_producer_style_tags() -> dict[str, dict]:
    """Return cached VocaDB-backed display style tags keyed by producer slug."""
    path = project_root() / "configs" / "producer_style_tags.yaml"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    source = config.get("source", {})
    producers = {}
    for slug, item in (config.get("producers") or {}).items():
        raw_tags = item.get("style_tags") or []
        display_tags = []
        for tag in raw_tags:
            if isinstance(tag, str):
                display_tags.append(tag)
            elif isinstance(tag, dict):
                display_tags.append(tag.get("display_zh") or tag.get("label"))
        producers[slug] = {
            "style_tags": [tag for tag in display_tags if tag],
            "style_tag_source": source.get("name"),
            "style_tag_source_url": item.get("source_url") or source.get("url"),
        }
    return producers


def _source_url(song_id: str) -> str | None:
    if song_id.startswith("youtube_"):
        return f"https://www.youtube.com/watch?v={song_id.removeprefix('youtube_')}"
    if song_id.startswith("niconico_"):
        return f"https://www.nicovideo.jp/watch/{song_id.removeprefix('niconico_')}"
    return None


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
                    "source_url": record.get("source_url") or _source_url(song_id),
                },
            )

    return {
        slug: sorted(
            songs.values(),
            key=lambda item: item["title"].casefold(),
        )
        for slug, songs in songs_by_producer.items()
    }


@lru_cache(maxsize=1)
def load_representative_song_catalog(limit: int = 3) -> dict[str, list[dict]]:
    """Return representative training songs keyed by producer slug."""
    root = project_root()
    segment_path = (
        root
        / "data"
        / "processed"
        / "curated"
        / "mert_95_p1"
        / "segments.jsonl"
    )
    decisions_path = (
        root
        / "data"
        / "processed"
        / "curated"
        / "mert_95"
        / "song_decisions.jsonl"
    )
    if not segment_path.exists() or not decisions_path.exists():
        return {}

    current_song_ids: set[str] = set()
    with open(segment_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            current_song_ids.add(json.loads(line)["song_id"])

    songs_by_producer: dict[str, list[dict]] = {}
    seen_by_producer: dict[str, set[str]] = {}
    with open(decisions_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "accepted":
                continue
            song_id = record["song_id"]
            if song_id not in current_song_ids:
                continue
            slug = record["producer_slug"]
            seen = seen_by_producer.setdefault(slug, set())
            if song_id in seen:
                continue
            if len(songs_by_producer.get(slug, [])) >= limit:
                continue
            seen.add(song_id)
            songs_by_producer.setdefault(slug, []).append({
                "song_id": song_id,
                "title": record.get("title") or song_id,
                "source_url": record.get("source_url") or _source_url(song_id),
            })

    return songs_by_producer


def _load_jsonl_song_catalog(path_parts: tuple[str, ...]) -> dict[str, list[dict]]:
    path = (
        project_root()
        .joinpath(*path_parts)
    )
    if not path.exists():
        return {}

    songs_by_producer: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            songs_by_producer.setdefault(
                record["producer_slug"],
                [],
            ).append({
                "song_id": record["song_id"],
                "title": record.get("title") or record["song_id"],
                "source_url": record.get("source_url"),
            })
    return {
        slug: sorted(
            songs,
            key=lambda item: item["title"].casefold(),
        )
        for slug, songs in songs_by_producer.items()
    }


@lru_cache(maxsize=1)
def load_dev_holdout_song_catalog() -> dict[str, list[dict]]:
    """Return validation songs reserved for model development."""
    return _load_jsonl_song_catalog((
        "data",
        "processed",
        "dev_holdout",
        "catalog.jsonl",
    ))


@lru_cache(maxsize=1)
def load_frozen_test_song_catalog() -> dict[str, list[dict]]:
    """Return final-test songs that never participate in model development."""
    return _load_jsonl_song_catalog((
        "data",
        "processed",
        "frozen_test",
        "catalog.jsonl",
    ))
