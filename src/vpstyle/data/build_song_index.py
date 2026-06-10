"""Build unified song index from VocaDB raw metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vpstyle.data.metadata_schema import Song
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def _first_original_youtube_url(pvs: list[dict]) -> Optional[str]:
    """Return the first YouTube PV marked as Original."""
    for pv in pvs:
        service = pv.get("service", "").lower()
        pv_type = pv.get("pvType", "")
        if service in ("youtube", "youtu.be") and pv_type == "Original":
            url = pv.get("url", "")
            if url:
                return url
    return None


def _count_producers(artists: list[dict]) -> int:
    """Count the number of artists whose categories include 'Producer'."""
    count = 0
    for artist in artists:
        categories = artist.get("categories", [])
        if "Producer" in categories:
            count += 1
    return count


def _parse_song_type(song_type: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return (is_cover, is_remix, is_instrumental, is_live, is_other)."""
    st = song_type
    return (
        st == "Cover",
        st == "Remix",
        st == "Instrumental",
        st == "Live",
        st not in ("Original", "Cover", "Remix", "Instrumental", "Live", "MusicPV"),
    )


def _determine_status(
    song_type: str,
    producer_count: int,
    length_seconds: int,
) -> tuple[str, Optional[str]]:
    """Determine accept/reject/pending_review status."""
    is_cover, is_remix, is_instrumental, is_live, is_other = _parse_song_type(song_type)

    if is_cover:
        return "rejected", "cover"
    if is_remix:
        return "rejected", "remix"
    if is_instrumental:
        return "rejected", "instrumental"
    if is_live:
        return "rejected", "live"
    if is_other:
        return "rejected", "other_song_type"
    if length_seconds < 60:
        return "rejected", "short_preview"
    if producer_count > 1:
        return "pending_review", "multiple_producers_need_check"
    return "accepted", None


def build_song_index(
    artists_jsonl: Path,
    producer_slug: str,
    cleaning_config: Optional[dict] = None,
) -> list[Song]:
    """Convert raw VocaDB artists JSONL into a list of Song dataclasses."""
    songs: list[Song] = []
    if not artists_jsonl.exists():
        logger.warning("File not found: %s", artists_jsonl)
        return songs

    with open(artists_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            song_id = str(item.get("id", ""))
            title = item.get("name", item.get("defaultName", ""))
            publish_date = item.get("publishDate")
            song_type = item.get("songType", "Original")
            length_seconds = item.get("lengthSeconds", 0)
            artists = item.get("artists", [])
            vocalists = [
                a["artist"]["name"]
                for a in artists
                if "Vocalist" in a.get("categories", [])
                and "artist" in a
            ]
            tags = [t.get("tag", {}).get("name", "") for t in item.get("tags", [])]
            pvs = item.get("pvs", [])

            source_url = _first_original_youtube_url(pvs)
            source_urls = [source_url] if source_url else []
            producer_count = _count_producers(artists)
            is_cover, is_remix, is_instrumental, is_live, _ = _parse_song_type(
                song_type
            )
            is_collaboration = producer_count > 1

            status, reason = _determine_status(song_type, producer_count, length_seconds)

            song = Song(
                song_id=song_id,
                producer_slug=producer_slug,
                title=title,
                publish_date=str(publish_date) if publish_date else None,
                source_urls=source_urls,
                vocalists=vocalists,
                tags=tags,
                is_cover=is_cover,
                is_remix=is_remix,
                is_collaboration=is_collaboration,
                status=status,
                status_reason=reason,
            )
            songs.append(song)

    accepted = [s for s in songs if s.status == "accepted"]
    pending = [s for s in songs if s.status == "pending_review"]
    rejected = [s for s in songs if s.status == "rejected"]
    logger.info(
        "[%s] total=%d accepted=%d pending=%d rejected=%d",
        producer_slug,
        len(songs),
        len(accepted),
        len(pending),
        len(rejected),
    )
    return songs


def save_song_index(songs: list[Song], path: str | Path) -> None:
    """Save songs as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for song in songs:
            f.write(json.dumps(song.__dict__, ensure_ascii=False) + "\n")
    logger.info("Saved song index to %s (%d songs)", path, len(songs))
